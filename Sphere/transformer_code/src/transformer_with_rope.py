import torch
import torch.nn as nn
from transformer_code.src.rope import get_rotary_embedding, apply_rotary_pos_emb
from transformer_code.src.rope import get_rotary_embedding, apply_rotary_pos_emb

class RotarySelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_heads, T, head_dim)

        sin, cos = get_rotary_embedding(T, self.head_dim, x.device)
        q = apply_rotary_pos_emb(q, sin, cos)
        k = apply_rotary_pos_emb(k, sin, cos)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        out = attn_output.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.attn = RotarySelfAttention(dim, n_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * dim), dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RotaryTransformer(nn.Module):
    def __init__(self, num_layers, dim, n_heads):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
