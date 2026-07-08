"""Held-out depth evaluation + visualization for training monitoring.

The model outputs an affine-invariant (normalized) point map, so before comparing to metric
GT depth we align the predicted depth to GT with a per-sample least-squares scale+shift
(standard for scale/shift-invariant depth eval, cf. MiDaS/MoGe). Metrics are computed over
valid (masked) pixels only — for TransCG that is the transparent region.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD


def align_scale_shift(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample least-squares a*pred+b ~ gt over masked pixels. Shapes [B,H,W]."""
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    g = gt.reshape(B, -1)
    m = mask.reshape(B, -1).float()
    w = m.sum(1, keepdim=True).clamp_min(1.0)
    pm = (p * m).sum(1, keepdim=True) / w
    gm = (g * m).sum(1, keepdim=True) / w
    pc = (p - pm) * m
    gc = (g - gm) * m
    a = (pc * gc).sum(1, keepdim=True) / (pc * pc).sum(1, keepdim=True).clamp_min(1e-8)
    b = gm - a * pm
    return (a * p + b).reshape_as(pred)


def depth_metrics(pred_z: torch.Tensor, gt_z: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """AbsRel, delta<1.25, RMSE over masked pixels (pred already scale-shift aligned)."""
    m = mask.reshape(-1) > 0
    pa = pred_z.reshape(-1)[m].clamp_min(1e-3)
    ga = gt_z.reshape(-1)[m].clamp_min(1e-3)
    absrel = (torch.abs(pa - ga) / ga).mean().item()
    ratio = torch.maximum(pa / ga, ga / pa)
    delta1 = (ratio < 1.25).float().mean().item()
    rmse = torch.sqrt(((pa - ga) ** 2).mean()).item()
    return {"absrel": absrel, "delta1": delta1, "rmse": rmse}


@torch.no_grad()
def evaluate(model, loader, device, n_batches: int = 8, seed: int = 0,
             regression: bool = False) -> Dict[str, float]:
    """Held-out metrics over up to n_batches batches (fixed-noise sampling).

    Returns depth metrics (absrel/delta1/rmse, scale-shift aligned) AND `val_loss` — the same
    rectified-flow objective as training. val_loss is the most discriminative progress signal
    here, because TransCG's transparent regions have a very narrow depth range that saturates
    scale-shift-aligned AbsRel/delta1.
    """
    from .data import normalize_pointmap
    from .flow import flow_loss, pointmap_loss, sample_flow_pair
    was_training = model.training
    model.eval()
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    acc = {"absrel": [], "delta1": [], "rmse": [], "val_loss": [], "corr": []}
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        img = batch["image"].to(device)
        gt = batch["points"].to(device)              # metric point map
        mask3 = batch["mask"].to(device)             # [B,1,H,W]
        mask = mask3[:, 0]                            # [B,H,W]
        img_n = (img - mean) / std
        tokens = model.encode(img_n)

        # prediction: deterministic regression OR 1-step diffusion sample (fixed noise)
        if regression:
            pred = model.predict(img_n, tokens)
        else:
            g = torch.Generator(device=device).manual_seed(seed + i)
            pred = model.sample(img_n, steps=1, generator=g)
        pred_z = align_scale_shift(pred[:, 2], gt[:, 2], mask)
        m = depth_metrics(pred_z, gt[:, 2], mask)
        # pred-GT structural correlation over valid pixels (the honest learning signal:
        # unlike AbsRel/delta1 it is NOT saturated by the narrow depth range)
        mm = mask.reshape(-1) > 0
        pz = pred_z.reshape(-1)[mm]
        gz = gt[:, 2].reshape(-1)[mm]
        pz = pz - pz.mean()
        gz = gz - gz.mean()
        denom = (pz.norm() * gz.norm()).clamp_min(1e-8)
        m["corr"] = ((pz * gz).sum() / denom).item()

        # held-out loss (same objective as training)
        pts_n = normalize_pointmap(gt, mask3)
        if regression:
            vloss = flow_loss(pred.float(), pts_n.float(), mask3)
        else:
            gf = torch.Generator(device=device).manual_seed(1000 + seed + i)
            x_t, t, v_target = sample_flow_pair(pts_n, generator=gf)
            v_pred = model.velocity(x_t, t, tokens).float()
            vloss = flow_loss(v_pred, v_target.float(), mask3)
            vloss = vloss + 0.5 * pointmap_loss(x_t.float() - t.view(-1, 1, 1, 1) * v_pred,
                                                pts_n.float(), mask3)
        m["val_loss"] = vloss.item()
        for k in acc:
            acc[k].append(m[k])
    if was_training:
        model.train()
    return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}


@torch.no_grad()
def save_pred_gt_figure(model, batch, device, path: str, seed: int = 0, regression: bool = False):
    """Save a side-by-side [input | pred depth | gt depth] PNG for one fixed sample."""
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    img = batch["image"][:1].to(device)
    gt = batch["points"][:1].to(device)
    mask = batch["mask"][:1].to(device)[:, 0]
    img_n = (img - mean) / std
    g = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    pred = model.predict(img_n) if regression else model.sample(img_n, steps=1, generator=g)
    if was_training:
        model.train()
    pred_z = align_scale_shift(pred[:, 2], gt[:, 2], mask)[0].cpu().numpy()
    gt_z = gt[0, 2].cpu().numpy()
    m = mask[0].cpu().numpy() > 0

    def colorize(z, valid):
        z = z.copy()
        if valid.any():
            lo, hi = np.percentile(z[valid], [2, 98])
            z = np.clip((z - lo) / (hi - lo + 1e-6), 0, 1)
        z8 = (z * 255).astype(np.uint8)
        try:
            import cv2
            return cv2.cvtColor(cv2.applyColorMap(z8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        except Exception:
            return np.stack([z8] * 3, -1)

    rgb = (img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    panel = np.concatenate([rgb, colorize(pred_z, m), colorize(gt_z, m)], axis=1)
    from PIL import Image
    Image.fromarray(panel).save(path)
