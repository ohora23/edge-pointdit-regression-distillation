#!/usr/bin/env python
"""Train PointDiT with a rectified-flow objective.

Scaffold: runs on synthetic data out of the box so the loop is verifiable now; point
`--data` at a real transparent-object dataset (TransCG) once downloaded. The frozen
encoder is not updated; only the denoiser trains.

Examples:
  python scripts/train.py --synthetic --steps 50 --profile high        # smoke test
  python scripts/train.py --data /path/to/TransCG --profile high --steps 100000
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from pointdit.config import ModelConfig
from pointdit.data import SyntheticPointMaps, normalize_pointmap
from pointdit.flow import flow_loss, gradient_loss, pointmap_loss, sample_flow_pair
from pointdit.model import PointDiT


def build_dataset(args):
    if args.synthetic or not args.data:
        return SyntheticPointMaps(n=max(256, args.batch * 8), profile=args.profile)
    if args.dataset == "transcg":
        from pointdit.data import TransCGDataset
        return TransCGDataset(args.data, profile=args.profile, split=args.split)
    if args.dataset == "dreds":
        from pointdit.data import DREDSDataset
        return DREDSDataset(args.data, profile=args.profile, domain=args.domain)
    raise ValueError(f"unknown dataset {args.dataset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--dataset", default="transcg", choices=["transcg", "dreds"])
    ap.add_argument("--split", default="train", choices=["train", "test"], help="TransCG split")
    ap.add_argument("--domain", default="sim", choices=["sim", "real"], help="DREDS domain")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--profile", default="high")
    ap.add_argument("--encoder", default="conv", choices=["conv", "dinov2", "dinov2b", "dinov3"])
    ap.add_argument("--npu", action="store_true",
                    help="train the RK3588 deploy config (BatchNorm/ReLU/const-norm linear "
                         "attn) so train == deploy, avoiding a train/deploy architecture gap")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--aux", type=float, default=0.5, help="affine-invariant point-map loss weight")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--regression", action="store_true",
                    help="deterministic image->depth regression (no diffusion noise, cleaner+faster)")
    ap.add_argument("--grad_weight", type=float, default=0.0,
                    help="multi-scale gradient loss weight (emphasize object structure over the plane)")
    ap.add_argument("--refine", action="store_true",
                    help="add RGB-guided conv refinement head (breaks token-grid detail ceiling)")
    ap.add_argument("--ckpt", default="artifacts/pointdit.pt")
    ap.add_argument("--init_ckpt", default=None, help="resume weights + step from this checkpoint")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=0, help="held-out AbsRel/delta1 every N steps (0=off)")
    ap.add_argument("--viz_every", type=int, default=0, help="save pred|gt depth figure every N steps")
    ap.add_argument("--ckpt_every", type=int, default=0, help="checkpoint every N steps")
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--outdir", default="artifacts/run", help="dir for periodic figures/ckpts")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(encoder=args.encoder)
    if args.npu:
        cfg = cfg.npu_profile()
    if args.refine:
        from dataclasses import replace
        cfg = replace(cfg, refine=True)
    model = PointDiT(cfg).to(device)
    model.encoder.eval()  # frozen
    start_step = 0
    if args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(sd["model"], strict=False)
        start_step = sd.get("step", 0)
        print(f"resumed from {args.init_ckpt} @ step {start_step}")

    # denoiser + refine head (everything except the frozen encoder)
    trainable = [p for n, p in model.named_parameters()
                 if p.requires_grad and not n.startswith("encoder.")]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    ds = build_dataset(args)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=4)
    print(f"dataset={type(ds).__name__} size={len(ds)} device={device} "
          f"trainable={sum(p.numel() for p in trainable)/1e6:.1f}M encoder={cfg.encoder}")

    # held-out eval/viz loader (TransCG test split); fixed batch for consistent figures
    eval_dl = fixed_batch = None
    if (args.eval_every or args.viz_every) and args.dataset == "transcg" and args.data:
        from pointdit.data import TransCGDataset
        from pointdit import eval as ev
        try:
            eds = TransCGDataset(args.data, profile=args.profile, split="test")
            eval_dl = DataLoader(eds, batch_size=args.batch, shuffle=False, num_workers=4)
            fixed_batch = next(iter(eval_dl))
            print(f"held-out eval: {len(eds)} samples (TransCG test split)")
        except Exception as e:
            print(f"eval disabled: {str(e)[:80]}")
    os.makedirs(args.outdir, exist_ok=True)

    step, ema = start_step, None
    while step < args.steps:
        for batch in dl:
            img = batch["image"].to(device)
            pts = normalize_pointmap(batch["points"].to(device), batch["mask"].to(device))
            mask = batch["mask"].to(device)
            from pointdit.data import IMAGENET_MEAN, IMAGENET_STD
            img_n = (img - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)

            with torch.no_grad():
                tokens = model.encode(img_n)

            with torch.amp.autocast("cuda", enabled=args.amp):
                if args.regression:
                    x0_hat = model.predict(img_n, tokens)
                else:
                    x_t, t, v_target = sample_flow_pair(pts)
                    v_pred = model.velocity(x_t, t, tokens)
            # losses in fp32 (large spatial sums overflow fp16 under AMP)
            if args.regression:
                x0_hat = x0_hat.float()
                loss = flow_loss(x0_hat, pts.float(), mask)
                loss = loss + args.aux * pointmap_loss(x0_hat, pts.float(), mask)
                if args.grad_weight > 0:
                    loss = loss + args.grad_weight * gradient_loss(x0_hat, pts.float(), mask)
            else:
                v_pred = v_pred.float()
                loss = flow_loss(v_pred, v_target.float(), mask)
                if args.aux > 0:
                    tb = t.view(-1, 1, 1, 1)
                    x0_hat = x_t.float() - tb * v_pred  # x0 = x_t - t*(x1-x0)
                    loss = loss + args.aux * pointmap_loss(x0_hat, pts.float(), mask)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            scaler.step(opt)
            scaler.update()

            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            if step % args.log_every == 0:
                print(f"step {step:6d}  loss {lv:.4f}  ema {ema:.4f}  gnorm {gnorm:.2f}", flush=True)

            if args.eval_every and eval_dl is not None and step % args.eval_every == 0:
                from pointdit import eval as ev
                m = ev.evaluate(model, eval_dl, device, n_batches=args.eval_batches,
                                regression=args.regression)
                print(f"  [eval @ {step}] val_loss {m['val_loss']:.4f}  corr {m['corr']:.3f}  |  "
                      f"AbsRel {m['absrel']:.4f}  delta1 {m['delta1']:.4f}  (held-out test)", flush=True)
            if args.viz_every and fixed_batch is not None and step % args.viz_every == 0:
                from pointdit import eval as ev
                p = os.path.join(args.outdir, f"viz_{step:06d}.png")
                ev.save_pred_gt_figure(model, fixed_batch, device, p, regression=args.regression)
                print(f"  [viz  @ {step}] {p}  (input | pred | gt)", flush=True)
            if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
                cp = os.path.join(args.outdir, f"ckpt_{step:06d}.pt")
                torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step}, cp)

            step += 1
            if step >= args.steps:
                break

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, args.ckpt)
    print(f"saved checkpoint -> {args.ckpt}  final_ema={ema:.4f}")


if __name__ == "__main__":
    main()
