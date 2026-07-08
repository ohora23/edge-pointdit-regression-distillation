"""Data utilities: fixed-resolution preprocessing, point-map normalization, datasets.

- `prepare_image` resizes/pads an RGB image to a patch-16-aligned deploy resolution
  (120 -> 128 padding for the 'low' profile, matching the report).
- `TransCGDataset` is a loader stub for the primary transparent-object benchmark.
- `SyntheticPointMaps` lets the training loop and demos run before real data exists.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import RESOLUTIONS

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prepare_image(img: torch.Tensor, profile: str = "high") -> torch.Tensor:
    """img: [B,3,H,W] float in [0,1] -> resized/padded+normalized to RESOLUTIONS[profile]."""
    import torch.nn.functional as F
    Ht, Wt = RESOLUTIONS[profile]
    img = F.interpolate(img, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return (img - IMAGENET_MEAN.to(img)) / IMAGENET_STD.to(img)


def normalize_pointmap(pts: torch.Tensor, mask: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
    """Center + scale a point map to a roughly unit affine-invariant frame (per sample).

    Statistics are computed over VALID (masked) pixels only, and invalid pixels are zeroed
    afterwards, so a small valid region (e.g. TransCG's ~10% transparent mask) does not get
    polluted by the 90% zero background.
    """
    B, C = pts.shape[0], pts.shape[1]
    p = pts.reshape(B, C, -1)
    if mask is not None:
        m = mask.reshape(B, 1, -1).float()
    else:
        m = torch.ones(B, 1, p.shape[-1], device=pts.device, dtype=pts.dtype)
    w = m.sum(2, keepdim=True).clamp_min(1.0)
    mean = (p * m).sum(2, keepdim=True) / w
    p = p - mean
    scale = ((p.abs() * m).sum(dim=(1, 2), keepdim=True) / (C * w)).clamp_min(1e-6)
    p = (p / scale) * m  # zero the invalid background
    return p.reshape_as(pts)


class SyntheticPointMaps(Dataset):
    """Random-but-structured (image, point map, mask) triples for smoke tests."""

    def __init__(self, n: int = 64, profile: str = "high", seed: int = 0):
        self.n = n
        self.H, self.W = RESOLUTIONS[profile]
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed + idx)
        img = torch.rand(3, self.H, self.W, generator=g)
        ys = torch.linspace(-1, 1, self.H).view(self.H, 1).expand(self.H, self.W)
        xs = torch.linspace(-1, 1, self.W).view(1, self.W).expand(self.H, self.W)
        z = torch.sin(3 * xs) * torch.cos(3 * ys)
        pts = torch.stack([xs, ys, z], 0)
        mask = torch.ones(1, self.H, self.W)
        return {"image": img, "points": pts, "mask": mask}


class TransCGDataset(Dataset):
    """TransCG loader (https://arxiv.org/abs/2202.08471), faithful to the official
    layout (github.com/Galaxies99/TransCG datasets/transcg.py):

        root/metadata.json                     # total_scenes, train/test scene ids, *_samples
        root/camera_intrinsics/1-camIntrinsics-D435.npy   # 3x3 K (cam 1)
        root/camera_intrinsics/2-camIntrinsics-L515.npy   # 3x3 K (cam 2)
        root/scene{i}/metadata.json            # type, split, D435/L515_valid_perspective_list
        root/scene{i}/{pid}/rgb{cam}.png       # cam in {1:D435, 2:L515}
        root/scene{i}/{pid}/depth{cam}-gt.png  # uint16 depth in mm
        root/scene{i}/{pid}/depth{cam}-gt-mask.png

    Returns {image [3,H,W] in [0,1], points [3,H,W] (XYZ, meters), mask [1,H,W]} resized
    to RESOLUTIONS[profile]. Point map = backproject(depth_gt, K).
    Raises with a download hint if the root/metadata is missing.
    """

    def __init__(self, root: str, profile: str = "high", split: str = "train"):
        md = os.path.join(root, "metadata.json")
        if not os.path.isfile(md):
            raise FileNotFoundError(
                f"TransCG metadata not found at {md}. Download from "
                "https://graspnet.net/transcg and point --data at the extracted root. "
                "Until then use --synthetic for scaffold runs."
            )
        self.root = root
        self.H, self.W = RESOLUTIONS[profile]
        self.split = split
        with open(md) as fp:
            meta = json.load(fp)
        self.K = [None,
                  np.load(os.path.join(root, "camera_intrinsics", "1-camIntrinsics-D435.npy")),
                  np.load(os.path.join(root, "camera_intrinsics", "2-camIntrinsics-L515.npy"))]
        self.samples = []
        skipped = 0
        for sid in meta[split]:
            base = os.path.join(root, f"scene{sid}")
            sm_path = os.path.join(base, "metadata.json")
            if not os.path.isfile(sm_path):
                skipped += 1  # subset download: scene not present on disk -> skip
                continue
            with open(sm_path) as fp:
                sm = json.load(fp)
            for pid in sm.get("D435_valid_perspective_list", []):
                self.samples.append((os.path.join(base, str(pid)), 1, pid))
            for pid in sm.get("L515_valid_perspective_list", []):
                self.samples.append((os.path.join(base, str(pid)), 2, pid))
        if skipped:
            print(f"[TransCG] {split}: skipped {skipped} scene(s) not present on disk "
                  f"(subset), using {len(self.samples)} samples")
        if not self.samples:
            raise RuntimeError(
                f"no TransCG samples for split={split} on disk. If you downloaded a subset, "
                "the available scenes may be in the other split — try split='test'."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch.nn.functional as F
        path, cam, pid = self.samples[idx]
        rgb = np.array(Image.open(os.path.join(path, f"rgb{cam}.png")), dtype=np.float32) / 255.0
        depth_gt = np.array(Image.open(os.path.join(path, f"depth{cam}-gt.png")), dtype=np.float32)
        depth_m = depth_gt / 1000.0  # uint16 mm -> meters (depth_gt is the completed GT)
        pts = backproject_depth(depth_m, self.K[cam])  # [3, Hn, Wn]
        # Supervise on ALL valid GT depth (~99% of the frame), NOT just the transparent-object
        # mask (~10%). depth_gt is the completed full-scene depth, so for depth-from-RGB we
        # learn the whole geometry; sparse 10% supervision leaves the output as noise.
        # (depth{cam}-gt-mask.png marks transparent regions, kept for transparent-specific eval.)
        mask = (depth_m > 0).astype(np.float32)

        img_t = torch.from_numpy(rgb).permute(2, 0, 1)[None]          # [1,3,Hn,Wn]
        pts_t = torch.from_numpy(pts)[None]                            # [1,3,Hn,Wn]
        mask_t = torch.from_numpy(mask)[None, None]                    # [1,1,Hn,Wn]
        img_t = F.interpolate(img_t, (self.H, self.W), mode="bilinear", align_corners=False)
        pts_t = F.interpolate(pts_t, (self.H, self.W), mode="nearest")
        mask_t = F.interpolate(mask_t, (self.H, self.W), mode="nearest")
        return {"image": img_t[0], "points": pts_t[0], "mask": mask_t[0]}


def backproject_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """depth [H,W], K 3x3 -> XYZ point map [3,H,W] (helper for real datasets)."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], 0).astype(np.float32)


def read_exr_depth(path: str) -> np.ndarray:
    """Read an EXR depth map as a 2D float32 array, trying multiple backends.

    pip opencv-python usually ships WITHOUT EXR support, so we prefer the OpenEXR package.
    Order: OpenEXR (v3) -> imageio -> cv2. Picks a depth-like channel (Z/R/Y/first).
    """
    # OpenEXR v3
    try:
        import OpenEXR
        with OpenEXR.File(path) as f:
            chs = f.channels()
            for pref in ("Z", "R", "Y", "depth"):
                if pref in chs:
                    px = np.asarray(chs[pref].pixels, dtype=np.float32)
                    return px[..., 0] if px.ndim == 3 else px
            px = np.asarray(chs[list(chs)[0]].pixels, dtype=np.float32)
            return px[..., 0] if px.ndim == 3 else px
    except ImportError:
        pass
    # imageio (needs an EXR plugin)
    try:
        import imageio.v2 as iio
        d = np.asarray(iio.imread(path), dtype=np.float32)
        return d[..., 0] if d.ndim == 3 else d
    except Exception:
        pass
    # cv2 (needs an EXR-enabled build)
    import cv2
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if d is None:
        raise RuntimeError(
            f"cannot read EXR {path}. Install EXR support: `pip install OpenEXR` "
            "(recommended) or an opencv build with OpenEXR enabled."
        )
    return (d[..., 0] if d.ndim == 3 else d).astype(np.float32)


# DREDS/STD hardcoded intrinsics (fx, fy) at native resolution; cx,cy = center (see loader).
DREDS_INTRINSICS = {
    "sim": (446.31, 446.31),                       # DREDS synthetic, native 640x360
    "real": (918.295227050781, 917.5439453125),    # STD real (RealSense), native 1280x720
}


class DREDSDataset(Dataset):
    """DREDS / STD transparent-&-specular object depth (https://arxiv.org/abs/2208.03792,
    PKU-EPIC), faithful to SwinDRNet/datasets/datasets.py.

    Folder-based, no split file: frames are '<root>/<scene>/<prefix>_<suffix>'. Files:
        rgb        : '<prefix>_color.png'
        GT depth   : '<prefix>_gt_depth.exr' | '_depth_120.exr' | '_depth_0.exr'  (EXR, METERS)
        mask       : '<prefix>_mask.png' | '_mask.exr'   (instance ids; 255 = background)
        meta       : '<prefix>_meta.txt'  (per-object; material 2=transparent, 3=specular)

    Intrinsics are hardcoded per domain (fx/fy in DREDS_INTRINSICS); cx=W/2-0.5, cy=H/2-0.5.
    Depth EXR is already metric (no /1000). Point map = backproject(gt_depth, K).
    Point --data at a split root such as '.../DREDS-CatKnown/train'.
    """

    RGB = "_color.png"
    GT_CANDIDATES = ("_gt_depth.exr", "_depth_120.exr", "_depth_0.exr")
    MASK_CANDIDATES = ("_mask.png", "_mask.exr")

    def __init__(self, root: str, profile: str = "high", domain: str = "sim"):
        import glob
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"DREDS root not found: {root}. Download from "
                "https://mirrors.pku.edu.cn/dl-release/DREDS_ECCV2022/ and point --data at a "
                "split root (e.g. DREDS-CatKnown/train). Until then use --synthetic."
            )
        if domain not in DREDS_INTRINSICS:
            raise ValueError(f"domain must be one of {list(DREDS_INTRINSICS)}")
        self.root = root
        self.H, self.W = RESOLUTIONS[profile]
        self.domain = domain
        self.fx, self.fy = DREDS_INTRINSICS[domain]
        self.frames = sorted(
            p[: -len(self.RGB)] for p in glob.glob(os.path.join(root, "*", "*" + self.RGB))
        )
        if not self.frames:
            raise FileNotFoundError(f"no '*/*{self.RGB}' frames under {root}")

    @staticmethod
    def _first_existing(prefix: str, suffixes) -> Optional[str]:
        for s in suffixes:
            if os.path.isfile(prefix + s):
                return prefix + s
        return None

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        import torch.nn.functional as F
        prefix = self.frames[idx]

        rgb = np.array(Image.open(prefix + self.RGB).convert("RGB"), dtype=np.float32) / 255.0
        gt_path = self._first_existing(prefix, self.GT_CANDIDATES)
        if gt_path is None:
            raise FileNotFoundError(f"no GT depth for frame {prefix} (tried {self.GT_CANDIDATES})")
        depth = read_exr_depth(gt_path)  # [H,W] float32, meters

        mpath = self._first_existing(prefix, self.MASK_CANDIDATES)
        if mpath and mpath.endswith(".png"):
            mask = np.array(Image.open(mpath)) != 255
        elif mpath:
            mask = read_exr_depth(mpath) > 0
        else:
            mask = np.ones_like(depth, dtype=bool)
        mask = (mask & (depth > 0)).astype(np.float32)

        Hn, Wn = depth.shape
        K = np.array([[self.fx, 0, Wn * 0.5 - 0.5],
                      [0, self.fy, Hn * 0.5 - 0.5],
                      [0, 0, 1]], dtype=np.float32)
        pts = backproject_depth(depth, K)  # [3, Hn, Wn]

        img_t = torch.from_numpy(rgb).permute(2, 0, 1)[None]
        pts_t = torch.from_numpy(pts)[None]
        mask_t = torch.from_numpy(mask)[None, None]
        img_t = F.interpolate(img_t, (self.H, self.W), mode="bilinear", align_corners=False)
        pts_t = F.interpolate(pts_t, (self.H, self.W), mode="nearest")
        mask_t = F.interpolate(mask_t, (self.H, self.W), mode="nearest")
        return {"image": img_t[0], "points": pts_t[0], "mask": mask_t[0]}
