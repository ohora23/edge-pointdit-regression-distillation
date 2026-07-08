"""Model / deployment configuration.

Design goals:
- One weight set serves both deploy resolutions (multi-res training).
- NPU-friendly toggles so the SAME architecture exports cleanly to RKNN later:
    norm_type: "layernorm" (PC/training) | "batchnorm" (NPU-friendly, folds into conv)
    attn_type: "softmax"   (PC/training) | "relu_linear" (avoids softmax CPU-fallback)
    act:       "gelu"      (PC/training) | "relu"  (RKNN native)
- Static shapes only; no dynamic control flow in forward (see models/denoiser.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Deploy resolutions as patch-16-aligned (H, W).
#   high = 320x240 input  -> (240, 320)                 -> grid 15x20 = 300 tokens
#   low  = 160x120 input  -> H 120 padded to 128 -> (128, 160) -> grid 8x10 = 80 tokens
RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "high": (240, 320),
    "low": (128, 160),
    "xhigh": (384, 512),   # quality-tuning res (bigger point/image token grid); deploy compresses later
}


@dataclass
class ModelConfig:
    # --- geometry / io ---
    patch_size: int = 16
    out_channels: int = 3  # XYZ point map

    # --- image encoder ---
    # "conv"  : always-available NPU-friendly conv tokenizer (default; needs no download)
    # "dinov2": ungated pretrained ViT-S/14 (facebook/dinov2-small)   [accuracy phase]
    # "dinov3": gated pretrained ViT-S/16 (facebook/dinov3-vits16-...) [needs HF access]
    encoder: str = "conv"
    encoder_name: str = ""            # HF id override; empty -> sensible default per encoder
    encoder_dim: int = 384            # token dim produced by the conv tokenizer
    freeze_encoder: bool = True

    # --- denoiser (point-map DiT) ---
    hidden_size: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0

    # --- NPU-friendly toggles ---
    norm_type: str = "layernorm"      # layernorm | batchnorm
    attn_type: str = "softmax"        # softmax | relu_linear
    attn_norm: str = "data"           # data (reciprocal, CPU on RKNN) | const (fixed 1/N, NPU)
    act: str = "gelu"                 # gelu | relu
    head: str = "pixelshuffle"        # pixelshuffle (DepthToSpace, 1 CPU op) | convtranspose (Conv, NPU)
    refine: bool = False              # RGB-guided conv refinement head after unpatchify (breaks token-grid block ceiling; NPU-friendly conv)

    # --- diffusion / flow ---
    flow: str = "rectified"           # rectified-flow (velocity prediction)

    def default_encoder_name(self) -> str:
        if self.encoder_name:
            return self.encoder_name
        return {
            "conv": "",
            "dinov2": "facebook/dinov2-small",
            "dinov2b": "facebook/dinov2-base",
            "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
        }.get(self.encoder, "")

    def npu_profile(self) -> "ModelConfig":
        """Return a copy with NPU-friendly toggles flipped on (for export/deploy).

        attn_type='softmax': the RKNN rehearsal showed softmax attention converts fully to the
        RK3588 NPU at our token count (~600) with a single output-boundary CPU op (same as the
        linear-attention variant) — AND it has no Div fallback, is more accurate, and (unlike
        BatchNorm+relu_linear) trains stably under fp16 AMP. The relu_linear+const-norm path is
        kept as a config option for on-device A/B (see RKNN_REHEARSAL.md).

        head stays 'pixelshuffle': both pixelshuffle(DepthToSpace) and convtranspose leave
        exactly 1 output-boundary CPU op; DepthToSpace (pure rearrange) is cheaper.
        """
        from dataclasses import replace
        return replace(self, norm_type="batchnorm", attn_type="softmax", act="relu")
