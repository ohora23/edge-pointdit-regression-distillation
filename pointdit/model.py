"""Full PointDiT model: frozen image encoder + point-map DiT denoiser + flow sampler."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig
from .denoiser import PointMapDiT
from .encoders import build_encoder


class RefineHead(nn.Module):
    """RGB-guided conv refinement applied to the coarse point map.

    The denoiser output is a token-grid map upsampled by pixel_shuffle, so its detail is
    capped by the token grid (blocky). This small residual conv stack sharpens it using the
    full-res input image's high-frequency edges, breaking that block ceiling WITHOUT adding
    tokens (no O(N^2) attention blow-up). Conv+ReLU only -> NPU-friendly. Zero-init last
    conv so it starts as identity (warm-starting from a no-refine ckpt is a no-op, then learns).
    """

    def __init__(self, out_ch: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(out_ch + 3, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coarse: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        return coarse + self.net(torch.cat([coarse, img], dim=1))


class PointDiT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_encoder(cfg)
        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.denoiser = PointMapDiT(cfg)
        self.refine = RefineHead(cfg.out_channels) if getattr(cfg, "refine", False) else None

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.encoder(img)
        return tokens

    def velocity(self, x: torch.Tensor, t: torch.Tensor, img_tokens: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x, t, img_tokens)

    def predict(self, img: torch.Tensor, img_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        # deterministic regression: image -> point map in ONE forward (zero point input,
        # t=0). No sampling noise -> cleaner + faster than 1-step diffusion. Deploy-friendly.
        if img_tokens is None:
            img_tokens = self.encode(img)
        B, _, H, W = img.shape
        z = torch.zeros(B, self.cfg.out_channels, H, W, device=img.device, dtype=img.dtype)
        t = torch.zeros(B, device=img.device, dtype=img.dtype)
        out = self.denoiser(z, t, img_tokens)
        if self.refine is not None:
            out = self.refine(out, img)          # RGB-guided sharpening
        return out

    @torch.no_grad()
    def sample(self, img: torch.Tensor, steps: int = 1,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Rectified-flow sampling from noise x1 ~ N(0,I) back to the clean point map x0.

        Integrates dx/dt = v(x,t) from t=1 -> t=0 with `steps` Euler steps.
        steps=1 is the real-time deploy path.
        """
        cfg = self.cfg
        B, _, H, W = img.shape
        img_tokens = self.encode(img)
        x = torch.randn(B, cfg.out_channels, H, W, device=img.device, dtype=img.dtype,
                        generator=generator)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=img.device)
        for i in range(steps):
            t = ts[i].expand(B)
            v = self.velocity(x, t, img_tokens)
            dt = ts[i + 1] - ts[i]          # negative
            x = x + v * dt
        return x  # predicted clean point map [B, 3, H, W]


class ExportWrapperRegression(nn.Module):
    """Deploy graph for the regression path: image -> point map in ONE forward.

    Single 4D input (image only); the zero point-map input and t=0 are constants folded
    into the graph. This is the cleanest RKNN graph (no noise/scalar-t inputs) and matches
    how the model is actually used (predict()). Includes the refine head if the model has one.
    """

    def __init__(self, model: "PointDiT"):
        super().__init__()
        self.model = model

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model.predict(img)


class ExportWrapper(nn.Module):
    """Fixed-shape single-step forward for ONNX/RKNN export.

    Takes (image, noise, t) as explicit inputs so the whole graph is static and
    control-flow-free. On device you run the encoder+denoiser once (steps=1).
    """

    def __init__(self, model: PointDiT):
        super().__init__()
        self.model = model

    def forward(self, img: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        img_tokens = self.model.encode(img)
        v = self.model.velocity(noise, t, img_tokens)
        # one Euler step from t=1 to t=0: x0 = x1 - v  (dt = -1)
        return noise - v


class ExportWrapperFixedT(nn.Module):
    """RKNN-friendly 2-input (image, noise) graph with the timestep folded to t=1.

    The single-step deploy path always starts at t=1, so the timestep embedding is a
    constant and both remaining inputs are 4D image-like tensors (no 1-D scalar input
    that can trip up RKNN conversion). Equivalent to ExportWrapper with t==1.
    """

    def __init__(self, model: PointDiT, t_value: float = 1.0):
        super().__init__()
        self.model = model
        self.t_value = t_value

    def forward(self, img: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]
        t = torch.full((B,), self.t_value, device=img.device, dtype=img.dtype)
        img_tokens = self.model.encode(img)
        v = self.model.velocity(noise, t, img_tokens)
        return noise - v
