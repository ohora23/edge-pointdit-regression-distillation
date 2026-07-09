#!/usr/bin/env python
"""Local webcam -> PointDiT live viewer: browser sends frames, server runs model.predict().

Usage:
  .venv/bin/python webapp/server.py
  .venv/bin/python webapp/server.py --ckpt artifacts/reg_130_long/pointdit.pt --resolution high
Then open http://localhost:5050
"""
import argparse
import base64
import io
import os

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "static", "samples")

from pointdit.config import ModelConfig, RESOLUTIONS
from pointdit.model import PointDiT


def auto_correct(rgb):
    """Gray-world white balance + CLAHE contrast — for raw-UVC camera feeds (e.g. a
    RealSense color stream grabbed without the vendor SDK's AE/AWB) that come in
    dim/color-cast relative to the properly ISP-processed images the model trained on."""
    mean = rgb.reshape(-1, 3).mean(axis=0)
    gray = mean.mean()
    gains = np.clip(gray / np.clip(mean, 1, None), 0.5, 2.5)
    balanced = np.clip(rgb.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(balanced, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

app = Flask(__name__, static_folder="static", static_url_path="")
model = device = H = W = None
STRIDE = 4  # point-cloud subsampling; full res is used for the depth colormap


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/samples")
def samples():
    names = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    return jsonify(names)


@app.route("/infer", methods=["POST"])
def infer():
    data_url = request.json["image"]
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((W, H))
    rgb = np.array(img)  # H,W,3 uint8 — reused as point-cloud vertex colors
    if request.json.get("correct"):
        rgb = auto_correct(rgb)

    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t = ((t - MEAN) / STD).to(device)

    with torch.no_grad():
        pts = model.predict(t)[0].float().cpu().numpy()  # [3, H, W] XYZ point map

    z = pts[2]
    zn = (z - z.min()) / (np.ptp(z) + 1e-6)
    cm_bgr = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    _, buf = cv2.imencode(".jpg", cm_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    depth_jpg = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    # affine-invariant training -> scale/offset are arbitrary; normalize into a unit cube for viewing
    sub = pts[:, ::STRIDE, ::STRIDE].reshape(3, -1).T  # [N, 3]
    center = sub.mean(axis=0)
    scale = float(np.abs(sub - center).max()) + 1e-6
    positions = ((sub - center) / scale).astype(np.float32)
    colors = (rgb[::STRIDE, ::STRIDE].reshape(-1, 3) / 255.0).astype(np.float32)

    return jsonify(
        depth_jpg=depth_jpg,
        positions=positions.round(4).tolist(),
        colors=colors.round(4).tolist(),
    )


def main():
    global model, device, H, W, STRIDE
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/reg_xhigh130_more/pointdit.pt",
                     help="regression checkpoint (must match --resolution)")
    ap.add_argument("--resolution", default="xhigh", choices=list(RESOLUTIONS))
    ap.add_argument("--stride", type=int, default=4, help="point-cloud subsampling stride")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    STRIDE = args.stride
    H, W = RESOLUTIONS[args.resolution]
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    sd = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig(**sd["cfg"])
    model = PointDiT(cfg).to(device).eval()
    model.load_state_dict(sd["model"])
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded {args.ckpt} encoder={cfg.encoder} params={nparam:.1f}M "
          f"resolution={args.resolution} {H}x{W} device={device}")

    app.run(host="0.0.0.0", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
