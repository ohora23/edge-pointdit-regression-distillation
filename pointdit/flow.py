"""Rectified-flow training objective + affine-invariant point-map loss.

Rectified flow:
    x_t = (1 - t) * x0 + t * x1,   x1 ~ N(0, I),   target velocity v* = x1 - x0
The denoiser predicts v*; loss is (masked) MSE, optionally in an affine-invariant
frame so monocular scale/shift ambiguity is not penalized (cf. MoGe / DUSt3R).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def sample_flow_pair(x0: torch.Tensor, generator: Optional[torch.Generator] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_t, t, target_velocity) for a batch of clean point maps x0 [B,3,H,W]."""
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device, generator=generator)
    x1 = torch.randn_like(x0)
    tb = t.view(B, 1, 1, 1)
    x_t = (1 - tb) * x0 + tb * x1
    v_target = x1 - x0
    return x_t, t, v_target


def _affine_align(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
                  ) -> torch.Tensor:
    """Per-sample least-squares scale+shift aligning pred to gt over valid pixels.

    Solves min_{a,b} || a*pred + b - gt ||^2 on masked entries, returns a*pred+b.
    Operates per-channel on flattened spatial dims.
    """
    B, C = pred.shape[0], pred.shape[1]
    p = pred.reshape(B, C, -1)
    g = gt.reshape(B, C, -1)
    m = mask.reshape(B, 1, -1).float()
    w = m.sum(dim=2, keepdim=True).clamp_min(1.0)
    pm = (p * m).sum(2, keepdim=True) / w
    gm = (g * m).sum(2, keepdim=True) / w
    pc = (p - pm) * m
    gc = (g - gm) * m
    a = (pc * gc).sum(2, keepdim=True) / (pc * pc).sum(2, keepdim=True).clamp_min(1e-6)
    b = gm - a * pm
    return (a * p + b).reshape_as(pred)


def flow_loss(v_pred: torch.Tensor, v_target: torch.Tensor,
              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(v_pred, v_target)
    m = mask.float()
    if m.dim() == 3:
        m = m.unsqueeze(1)
    diff = (v_pred - v_target) ** 2 * m
    return diff.sum() / m.sum().clamp_min(1.0) / v_pred.shape[1]


def gradient_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: Optional[torch.Tensor] = None, scales: int = 4) -> torch.Tensor:
    """Multi-scale masked L1 loss on spatial gradients of depth.

    The plane (dominant, smooth) has near-constant gradient; objects have large local
    gradients. Matching gradients forces the model to reproduce local/object structure
    instead of collapsing to the table plane (standard sharpness fix in monocular depth).
    """
    if mask is None:
        mask = torch.ones_like(pred[:, :1])
    m = mask.float()
    if m.dim() == 3:
        m = m.unsqueeze(1)
    p, t, mm = pred, target, m
    total = pred.new_zeros(())
    for _ in range(scales):
        dxp = p[..., :, 1:] - p[..., :, :-1]
        dxt = t[..., :, 1:] - t[..., :, :-1]
        dyp = p[..., 1:, :] - p[..., :-1, :]
        dyt = t[..., 1:, :] - t[..., :-1, :]
        mx = mm[..., :, 1:] * mm[..., :, :-1]
        my = mm[..., 1:, :] * mm[..., :-1, :]
        total = total + ((dxp - dxt).abs() * mx).sum() / mx.sum().clamp_min(1.0) \
                      + ((dyp - dyt).abs() * my).sum() / my.sum().clamp_min(1.0)
        if p.shape[-1] < 4 or p.shape[-2] < 4:
            break
        p = F.avg_pool2d(p, 2); t = F.avg_pool2d(t, 2); mm = F.avg_pool2d(mm, 2)
    return total / scales


def pointmap_loss(pred_x0: torch.Tensor, gt_x0: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  affine_invariant: bool = True) -> torch.Tensor:
    """Auxiliary reconstruction loss on the predicted clean point map."""
    if mask is None:
        mask = torch.ones_like(pred_x0[:, :1])
    m = mask.float()
    if m.dim() == 3:
        m = m.unsqueeze(1)
    pred = _affine_align(pred_x0, gt_x0, m) if affine_invariant else pred_x0
    diff = (pred - gt_x0).abs() * m
    return diff.sum() / m.sum().clamp_min(1.0) / pred.shape[1]
