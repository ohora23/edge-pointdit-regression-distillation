"""NPU-friendly transformer building blocks for the point-map DiT denoiser.

Every block is static-shape and control-flow-free in forward so the module exports
cleanly to ONNX -> RKNN. The softmax/layernorm/gelu variants are for PC training;
the relu_linear/batchnorm/relu variants are the RKNN-friendly counterparts.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# positional embedding: fixed 2D sin-cos (no learnable table -> resolution-flexible,
# export-safe, and naturally supports the two deploy grids).
# --------------------------------------------------------------------------------------
def get_2d_sincos_pos_embed(dim: int, gh: int, gw: int) -> torch.Tensor:
    """Return [gh*gw, dim] fixed 2D sin-cos positional embedding."""
    assert dim % 4 == 0, "pos-embed dim must be divisible by 4"
    omega = np.arange(dim // 4, dtype=np.float64) / (dim // 4)
    omega = 1.0 / (10000 ** omega)  # [dim/4]
    y = np.arange(gh, dtype=np.float64)
    x = np.arange(gw, dtype=np.float64)
    grid = np.meshgrid(x, y)  # (x, y), each [gh, gw]
    out = []
    for g in grid:  # x then y
        g = g.reshape(-1)  # [gh*gw]
        ang = np.einsum("m,d->md", g, omega)  # [gh*gw, dim/4]
        out.append(np.sin(ang))
        out.append(np.cos(ang))
    emb = np.concatenate(out, axis=1)  # [gh*gw, dim]
    return torch.from_numpy(emb).float()


# --------------------------------------------------------------------------------------
# norm / activation factories (toggleable)
# --------------------------------------------------------------------------------------
class TokenBatchNorm(nn.Module):
    """BatchNorm over channels for [B, N, C] tokens (RKNN-friendly, folds into linear)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, N, C]
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


def make_norm(dim: int, norm_type: str) -> nn.Module:
    if norm_type == "layernorm":
        # elementwise_affine=False: AdaLN supplies scale/shift.
        return nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    if norm_type == "batchnorm":
        return TokenBatchNorm(dim)
    raise ValueError(f"unknown norm_type {norm_type}")


def make_act(act: str) -> nn.Module:
    return {"gelu": nn.GELU(), "relu": nn.ReLU()}[act]


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# --------------------------------------------------------------------------------------
# timestep embedding
# --------------------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t in [0,1], shape [B]
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        return self.mlp(emb)


# --------------------------------------------------------------------------------------
# attention (softmax or ReLU-linear)
# --------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_type: str = "softmax",
                 attn_norm: str = "data"):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_type = attn_type
        self.attn_norm = attn_norm
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.attn_type == "softmax":
            out = F.scaled_dot_product_attention(q, k, v)
        elif self.attn_type == "relu_linear":
            # linear attention: phi(q)=ReLU(q). O(N), softmax-free, and EINSUM-FREE
            # (plain matmul/transpose/sum) so RKNN converts it without Einsum ops.
            # ReLU features are unbounded, so the k^T@v / q@kv sums overflow fp16 under AMP
            # (inf -> NaN). Upcast the attention core to fp32 ONLY when it is low precision,
            # so fp32 export (RKNN) stays Cast-free.
            in_dtype = q.dtype
            if in_dtype != torch.float32:
                q, k, v = q.float(), k.float(), v.float()
            q = F.relu(q)                                  # [B,H,N,d]
            k = F.relu(k)                                  # [B,H,N,d]
            kv = k.transpose(-2, -1) @ v                   # [B,H,d,d]
            if self.attn_norm == "const":
                # fixed 1/N scale -> a constant Mul (no Div/Reciprocal -> stays on NPU).
                # N is fixed per deploy resolution; the network learns to absorb the scale.
                out = (q @ kv) * (1.0 / float(N))          # [B,H,N,d]
            else:
                k_sum = k.sum(dim=2, keepdim=True)             # [B,H,1,d]
                z = 1.0 / ((q * k_sum).sum(dim=-1, keepdim=True) + 1e-6)  # [B,H,N,1]
                out = (q @ kv) * z                             # [B,H,N,d]
            out = out.to(in_dtype)
        else:
            raise ValueError(self.attn_type)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, act: str):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = make_act(act)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# --------------------------------------------------------------------------------------
# DiT block with AdaLN-Zero conditioning on the timestep
# --------------------------------------------------------------------------------------
class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 norm_type: str, attn_type: str, act: str, attn_norm: str = "data"):
        super().__init__()
        self.norm1 = make_norm(dim, norm_type)
        self.attn = Attention(dim, num_heads, attn_type, attn_norm)
        self.norm2 = make_norm(dim, norm_type)
        self.mlp = Mlp(dim, mlp_ratio, act)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.ada(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Linear head: tokens -> [B, N, p*p*C]. Paired with pixel_shuffle unpatchify
    (exports as DepthToSpace, which falls to CPU on RKNN)."""

    def __init__(self, dim: int, patch_size: int, out_channels: int, norm_type: str):
        super().__init__()
        self.norm = make_norm(dim, norm_type)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class ConvTransposeHead(nn.Module):
    """NPU-friendly output head: AdaLN-modulated point tokens -> [B, D, gh, gw] ->
    ConvTranspose2d(kernel=stride=patch) -> [B, C, gh*p, gw*p]. This is a single Conv op
    (stays on the RK3588 NPU) instead of the linear+DepthToSpace path that falls to CPU."""

    def __init__(self, dim: int, patch_size: int, out_channels: int, norm_type: str):
        super().__init__()
        self.norm = make_norm(dim, norm_type)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.deconv = nn.ConvTranspose2d(dim, out_channels, kernel_size=patch_size,
                                         stride=patch_size)

    def forward(self, x: torch.Tensor, c: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)     # [B, N, D]
        B, _, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, gh, gw)  # [B, D, gh, gw]
        return self.deconv(x)                        # [B, C, gh*p, gw*p]
