"""Point-map DiT denoiser.

Input : noisy point map [B, 3, H, W], timestep t [B], image tokens [B, Ne, D]
Output: velocity field  [B, 3, H, W]  (rectified-flow target)

The noisy point map is patchified into XYZ tokens (exactly like an image but with XYZ
channels instead of RGB, per the PointDiT formulation), concatenated with the frozen
image tokens for joint self-attention (in-context conditioning), and denoised by a stack
of AdaLN-Zero DiT blocks conditioned on the timestep.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import (
    ConvTransposeHead,
    DiTBlock,
    FinalLayer,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
)


class PointMapDiT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        p = cfg.patch_size

        # patchify XYZ point map -> tokens
        self.x_embed = nn.Conv2d(cfg.out_channels, d, kernel_size=p, stride=p)
        # project frozen image tokens into the denoiser width
        self.img_proj = nn.Linear(cfg.encoder_dim, d)

        self.t_embed = TimestepEmbedder(d)

        self.blocks = nn.ModuleList([
            DiTBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.norm_type, cfg.attn_type,
                     cfg.act, cfg.attn_norm)
            for _ in range(cfg.depth)
        ])
        if cfg.head == "convtranspose":
            self.final = ConvTransposeHead(d, p, cfg.out_channels, cfg.norm_type)
        else:
            self.final = FinalLayer(d, p, cfg.out_channels, cfg.norm_type)

        # separate learnable "stream" embeddings for point vs image tokens
        self.point_type = nn.Parameter(torch.zeros(1, 1, d))
        self.image_type = nn.Parameter(torch.zeros(1, 1, d))

        self._pos_cache: dict = {}
        self.initialize_weights()

    # ---- pos-embed (fixed 2D sincos, cached per grid) ----
    def _pos(self, gh: int, gw: int, device, dtype) -> torch.Tensor:
        key = (gh, gw)
        if key not in self._pos_cache:
            self._pos_cache[key] = get_2d_sincos_pos_embed(self.cfg.hidden_size, gh, gw)
        return self._pos_cache[key].to(device=device, dtype=dtype).unsqueeze(0)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # AdaLN-Zero: zero the modulation outputs so blocks start as identity.
        for b in self.blocks:
            nn.init.zeros_(b.ada[-1].weight)
            nn.init.zeros_(b.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        # zero the output projection (AdaLN-Zero style) for either head type
        out_proj = self.final.deconv if hasattr(self.final, "deconv") else self.final.linear
        nn.init.zeros_(out_proj.weight)
        nn.init.zeros_(out_proj.bias)

    def unpatchify(self, x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        # x: [B, gh*gw, C*p*p] -> [B, C, gh*p, gw*p]
        # Done as a 4D permute + pixel_shuffle (ONNX DepthToSpace) so it stays on the
        # RK3588 NPU. The RKNN rehearsal showed the naive 5D reshape/transpose fell to
        # CPU; DepthToSpace does not. Channel order is learned, so the exact (c,ph,pw)
        # interpretation is fixed by training and need not match any external convention.
        p, c = self.cfg.patch_size, self.cfg.out_channels
        B = x.shape[0]
        x = x.reshape(B, gh, gw, c * p * p).permute(0, 3, 1, 2)  # [B, C*p*p, gh, gw]
        return F.pixel_shuffle(x, p)                              # [B, C, gh*p, gw*p]

    def forward(self, x: torch.Tensor, t: torch.Tensor, img_tokens: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        pt = self.x_embed(x)                         # [B, d, gh, gw]
        gh, gw = pt.shape[-2], pt.shape[-1]
        pt = pt.flatten(2).transpose(1, 2)           # [B, Np, d]
        pos = self._pos(gh, gw, x.device, x.dtype)   # [1, Np, d]
        pt = pt + pos + self.point_type

        it = self.img_proj(img_tokens) + self.image_type  # [B, Ne, d]
        # image tokens share the same grid (encoder stride == patch_size) -> add same pos
        if it.shape[1] == pt.shape[1]:
            it = it + pos

        tokens = torch.cat([pt, it], dim=1)          # [B, Np+Ne, d]
        c = self.t_embed(t)                          # [B, d]
        for blk in self.blocks:
            tokens = blk(tokens, c)

        pt_out = tokens[:, : gh * gw, :]             # take point-token stream
        if self.cfg.head == "convtranspose":
            return self.final(pt_out, c, gh, gw)     # [B, 3, H, W] directly (Conv, NPU)
        v = self.final(pt_out, c)                    # [B, Np, p*p*C]
        return self.unpatchify(v, gh, gw)            # [B, 3, H, W]
