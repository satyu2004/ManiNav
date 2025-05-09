import torch
import math

def apply_rotary_pos_emb(x, sin, cos):
    # Applies RoPE to queries or keys
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated

def get_rotary_embedding(seq_len, dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float().to(device) / dim))
    positions = torch.arange(0, seq_len, dtype=torch.float, device=device)
    sinusoid_inp = torch.einsum("i , j -> i j", positions, inv_freq)
    sin = torch.sin(sinusoid_inp)[None, :, :]
    cos = torch.cos(sinusoid_inp)[None, :, :]
    return sin, cos

