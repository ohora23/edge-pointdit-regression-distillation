"""Image encoders. Pluggable so the pipeline always runs.

- ConvTokenizer: NPU-friendly conv stem (stride patch_size). Always available, no download.
                 Default for the runnable PC demo + ONNX/RKNN export.
- DinoEncoder:   frozen pretrained ViT (DINOv2 ungated / DINOv3 gated). For the accuracy phase.

All encoders return (tokens [B, N, C], (gh, gw)) with C == cfg.encoder_dim.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class ConvTokenizer(nn.Module):
    """Conv stem that maps an RGB image to patch tokens at stride patch_size.

    NPU-friendly (conv + BN + ReLU), static shape, exports cleanly.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        p = cfg.patch_size
        assert p == 16, "ConvTokenizer currently assumes patch_size=16 (4 stride-2 convs)"
        d = cfg.encoder_dim
        chans = [64, 128, 256, d]
        layers = []
        in_c = 3
        for out_c in chans:
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            in_c = out_c
        self.stem = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # img: [B, 3, H, W], H,W multiples of patch_size
        f = self.stem(img)              # [B, d, gh, gw]
        gh, gw = f.shape[-2], f.shape[-1]
        tokens = f.flatten(2).transpose(1, 2)  # [B, gh*gw, d]
        return tokens, (gh, gw)


class DinoEncoder(nn.Module):
    """Frozen pretrained ViT encoder via HuggingFace transformers.

    Falls back with a clear error if the model can't be loaded (e.g. gated DINOv3
    without HF access); callers should catch and use ConvTokenizer instead.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from transformers import AutoModel  # local import: only needed on this path
        name = cfg.default_encoder_name()
        try:
            self.model = AutoModel.from_pretrained(name)
        except OSError as e:
            if "gated" in str(e).lower():
                raise OSError(
                    f"'{name}' is a gated HF repo. Accept the license at "
                    f"https://huggingface.co/{name} and run `huggingface-cli login`, "
                    f"or use --encoder dinov2 (ungated) for now."
                ) from e
            raise
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        hidden = getattr(self.model.config, "hidden_size", cfg.encoder_dim)
        self.proj = nn.Identity() if hidden == cfg.encoder_dim else nn.Linear(hidden, cfg.encoder_dim)
        self.patch = getattr(self.model.config, "patch_size", 16)
        self.target_patch = cfg.patch_size  # resample tokens onto the point-map (patch-16) grid

    @torch.no_grad()
    def _backbone(self, img: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=img)
        return out.last_hidden_state  # [B, 1(+reg)+N, hidden]

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # resize to a multiple of the encoder patch size (e.g. DINOv2 patch=14 needs
        # 240x320 -> 238x308), run the backbone, then RESAMPLE the patch tokens onto the
        # point-map (patch-16) grid so image tokens align 1:1 spatially with point tokens
        # and share their positional embedding. Without this the conditioning is spatially
        # unlocalized and the model learns nothing but the mean depth (noise output).
        p = self.patch
        H, W = img.shape[-2], img.shape[-1]
        gh, gw = H // p, W // p
        Hs, Ws = gh * p, gw * p
        if (Hs, Ws) != (H, W):
            img = F.interpolate(img, size=(Hs, Ws), mode="bilinear", align_corners=False)
        feats = self._backbone(img)
        n = gh * gw
        patch_tokens = self.proj(feats[:, feats.shape[1] - n:, :])  # drop CLS/register prefix
        # resample encoder grid (gh,gw) -> point grid (tgh,tgw)
        tgh, tgw = H // self.target_patch, W // self.target_patch
        if (tgh, tgw) != (gh, gw):
            C = patch_tokens.shape[-1]
            g = patch_tokens.transpose(1, 2).reshape(patch_tokens.shape[0], C, gh, gw)
            g = F.interpolate(g, size=(tgh, tgw), mode="bilinear", align_corners=False)
            patch_tokens = g.flatten(2).transpose(1, 2)
        return patch_tokens, (tgh, tgw)


def build_encoder(cfg: ModelConfig) -> nn.Module:
    if cfg.encoder == "conv":
        return ConvTokenizer(cfg)
    if cfg.encoder in ("dinov2", "dinov2b", "dinov3"):
        return DinoEncoder(cfg)
    raise ValueError(f"unknown encoder {cfg.encoder}")
