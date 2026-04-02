import torch
import torch.nn as nn
from layers import (
    Attention,
    CastedLinear,
    SwiGLU,
    rms_norm,
    RotaryEmbedding
)
import torch.nn.functional as F

SMALL_NEGATIVE = -1e10

class RRBlock(nn.Module):
    def __init__(self, num_heads, hidden_size, expansion):
        super().__init__()
        self.attn = Attention(num_heads, hidden_size)
        self.swiglu = SwiGLU(hidden_size, expansion)

    def forward(self, hidden_states, cos_sin=None):
        hidden_states = rms_norm(
            hidden_states + self.attn(hidden_states, cos_sin=cos_sin)
        )
        hidden_states = rms_norm(
            hidden_states + self.swiglu(hidden_states)
        )
        return hidden_states

class RRBlock_mlp(nn.Module):
    def __init__(self, num_tokens,  hidden_size, expansion):
        super().__init__()
        self.mlp = SwiGLU(num_tokens, expansion)
        self.swiglu = SwiGLU(hidden_size, expansion)

    def forward(self, hidden_states, cos_sin=None):
        hidden_states = hidden_states.transpose(1,2)
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out)
        hidden_states = hidden_states.transpose(1,2)

        hidden_states = rms_norm(
            hidden_states + self.swiglu(hidden_states)
        )
        return hidden_states

class RRModule(nn.Module):
    def __init__(self, n_layers, num_heads, hidden_size, expansion, use_mlp, num_tokens):
        super().__init__()
        if use_mlp:
            self.layers = nn.ModuleList([
                RRBlock_mlp(num_tokens, hidden_size, expansion) for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                RRBlock(num_heads, hidden_size, expansion) for _ in range(n_layers)
            ])

    def forward(self, hidden_states, input_injection=None, cos_sin=None):
        if input_injection is not None:
            hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)
        return hidden_states

class RRModel(nn.Module):
    # num_class: int
    # H_layers: int
    # L_layers: int
    # H_cycles: int
    # L_cycles: int
    # num_heads: int
    # expansion: int
    # num_tokens: int
    # rope_theta: float
    # single_net: bool
    # use_mlp: bool
    # dropout_rate: float
    # topk: int
    # halt_max_steps: int
    # halt_exploration_prob: float

    def __init__(
        self,
        *,
        hidden_size,
        num_class,
        L_layers=1,
        H_layers=2, # doesn't matter if use single net
        L_cycles=1,
        H_cycles=2,
        num_heads=8, # doesn't matter if use mlp
        expansion=4,
        num_tokens=64,
        rope_theta=10000.0,
        single_net=True,
        use_mlp=True,
        dropout_rate=0.1,
    ):
        super().__init__()

        # Core configs
        self.hidden_size = hidden_size
        self.num_class = num_class

        # Layer / cycle configs
        self.L_layers = L_layers
        self.H_layers = H_layers
        self.L_cycles = L_cycles
        self.H_cycles = H_cycles

        # Architecture configs
        self.num_heads = num_heads
        self.expansion = expansion
        self.num_tokens = num_tokens
        self.rope_theta = rope_theta

        # Flags
        self.single_net = single_net
        self.use_mlp = use_mlp

        # Regularization
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        # =========================================
        self.token_hidden_size = int((self.hidden_size * 3) / self.num_tokens)

        self.rotary_embedding = RotaryEmbedding(
            dim=(self.token_hidden_size) // self.num_heads,
            max_position_embeddings=self.num_tokens,
            base=self.rope_theta
        )
        if not self.single_net:
            self.H_level = RRModule(
                n_layers=self.H_layers,
                num_heads=self.num_heads,
                hidden_size=self.token_hidden_size,
                expansion=self.expansion,
                use_mlp=self.use_mlp,
                num_tokens=self.num_tokens,
            )

        self.L_level = RRModule(
            n_layers=self.L_layers,
            num_heads=self.num_heads,
            hidden_size=self.token_hidden_size,
            expansion=self.expansion,
            use_mlp=self.use_mlp,
            num_tokens=self.num_tokens,
        )

        dims = [self.hidden_size * 3 , self.hidden_size * 2, self.hidden_size]
        self.down_proj = nn.ModuleList(
            [CastedLinear(dims[i], dims[i + 1], bias=False) for i in range(len(dims) - 1)]
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        self.classify_head = CastedLinear(self.hidden_size, self.num_class, bias=False)


    def forward_down_proj(self, zh):
        for idx, layer in enumerate(self.down_proj):
            zh = self.dropout(F.relu(layer(zh)))
        return zh

    def recursive_reasoning_single_net(self, pairs):
        zl = torch.zeros_like(pairs).to(pairs.device)
        zh = torch.zeros_like(pairs).to(pairs.device)
        cos_sin = self.rotary_embedding()

        with torch.no_grad():
            for _H_step in range(self.H_cycles-1):
                for _L_step in range(self.L_cycles):
                    zl = self.L_level(zl, zh + pairs, cos_sin=cos_sin)
                zh = self.L_level(zh, zl, cos_sin=cos_sin)
        # 1 with grad
        for _L_step in range(self.L_cycles):
            zl = self.L_level(zl, zh + pairs, cos_sin=cos_sin)
        zh = self.L_level(zh, zl, cos_sin=cos_sin)

        return zh, zl

    def recursive_reasoning(self, pairs):
        if self.single_net:
            return self.recursive_reasoning_single_net(pairs)
        # from TRM
        zl = torch.zeros_like(pairs).to(pairs.device)
        zh = torch.zeros_like(pairs).to(pairs.device)
        cos_sin = self.rotary_embedding()

        with torch.no_grad():
            for _H_step in range(self.H_cycles-1):
                for _L_step in range(self.L_cycles):
                    zl = self.L_level(zl, zh + pairs, cos_sin=cos_sin)
                zh = self.H_level(zh, zl, cos_sin=cos_sin)
        # 1 with grad
        for _L_step in range(self.L_cycles):
            zl = self.L_level(zl, zh + pairs, cos_sin=cos_sin)
        zh = self.H_level(zh, zl, cos_sin=cos_sin)

        return zh, zl

    def forward(
        self,
        hs,
        ts,
        rs
    ):
        rel_emb = torch.cat([hs, ts, rs], dim=1)
        rel_emb = rel_emb.view(rel_emb.shape[0], self.num_tokens, -1)

        zh, zl = self.recursive_reasoning(rel_emb)
        zh = zh.view(zh.shape[0], -1)

        features = self.forward_down_proj(zh.view(zh.shape[0], -1))
        logits = self.classify_head(features)

        return logits
