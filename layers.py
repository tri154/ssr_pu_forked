import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def rms_norm(hidden_states, variance_epsilon=1e-5):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor

def _find_multiple(a, b):
    return (-(a // -b)) * b

class CastedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features)))

    def forward(self, input):
        return F.linear(input, self.weight.to(input.dtype), self.bias.to(input.dtype) if self.bias is not None else None)

class Attention(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.qkv_proj = CastedLinear(hidden_size, hidden_size * 3, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, cos_sin=None):
        bs, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(bs, seq_len, self.num_heads * 3, self.head_size)
        q = qkv[:, :, :self.num_heads]
        k = qkv[:, :, self.num_heads:self.num_heads * 2]
        v = qkv[:, :, self.num_heads * 2:]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )
        attn_output = attn_output.reshape(bs, seq_len, -1)
        return self.o_proj(attn_output)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion):
        super().__init__()
        self.hidden_size = hidden_size
        inter = _find_multiple(round(self.hidden_size * expansion * 2 / 3), 256)
        self.proj_up = CastedLinear(hidden_size, inter * 2, bias=False)
        self.proj_down = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, hidden_states):
        a, b = self.proj_up(hidden_states).chunk(2, dim=-1)
        return self.proj_down(F.silu(a) * b)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

if __name__ =="__main__":
    # attn = Attention(8, 512)
    # BS, SEQ_LEN, HIDDEN_SIZE = 64, 2, 512
    # size = [BS, SEQ_LEN, HIDDEN_SIZE]
    # hs = torch.rand(size)
    # out = attn(hs)
    # print(out.shape)

    sl = SwiGLU(512, 4)
    BS, SEQ_LEN, HIDDEN_SIZE = 64, 2, 512
    size = [BS, SEQ_LEN, HIDDEN_SIZE]
    hs = torch.rand(size)
    out = sl(hs)
    print(out.shape)
