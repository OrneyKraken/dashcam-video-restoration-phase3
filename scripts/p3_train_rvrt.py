"""Fine-tune RVRT (video denoising) on the Phase 3 dashcam Track A training split.

Continues from the public '006_RVRT_videodenoising_DAVIS_16frames' checkpoint
(same weights/config rvrt_wrapper.py uses for evaluation). Only Track A
(gaussian / poisson-gaussian noise) is used: RVRT is a non-blind denoiser and
takes a known per-pixel noise sigma as an extra input channel, which has no
clean equivalent for Track B's low-light degradation - that's left to a
low-light-specific baseline instead.

Usage (Kaggle or local, with THESIS_P3_ROOT / RVRT_DIR env vars set):
    python p3_train_rvrt.py --steps 2000 --batch-size 1 --seq-len 8 --crop 128
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
RVRT_DIR = Path(os.environ.get("RVRT_DIR", str(SCRIPT_DIR.parent / "external" / "RVRT")))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(RVRT_DIR))

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")

from p3_dataloader import P3TrainSequences          # noqa: E402
from p3_degrade import POISSON_GAUSSIAN             # noqa: E402
from models.network_rvrt import RVRT as net         # noqa: E402


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps * eps))


def sigma_channel(lq: torch.Tensor, kind: str, sigma255) -> torch.Tensor:
    """(T,1,H,W) plane holding this sample's noise sigma in [0,1] units.

    gaussian: the true sigma the dataloader drew. poisson_gaussian: the
    analytically equivalent sigma from the noisy frame's own mean intensity
    (identical formula to rvrt_wrapper.py's _sigma255, for consistency with
    how the model is evaluated).
    """
    if kind == "gaussian":
        sigma = float(sigma255) / 255.0
    else:
        a, b = POISSON_GAUSSIAN["a"], POISSON_GAUSSIAN["b"]
        mean_x = float(lq.mean())
        sigma = float(np.sqrt(mean_x / a + b * b))
    T, _, H, W = lq.shape
    return torch.full((T, 1, H, W), sigma, dtype=lq.dtype)


def collate_fn(batch):
    lq = torch.stack([b["lq"] for b in batch], 0)
    gt = torch.stack([b["gt"] for b in batch], 0)
    sigma = torch.stack(
        [sigma_channel(b["lq"], b["kind"], b.get("sigma255", 0.0)) for b in batch], 0
    )
    return {"lq": lq, "gt": gt, "sigma": sigma}


def build_model(device: torch.device, ckpt_path: Path) -> torch.nn.Module:
    model = net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8],
                num_blocks=[1, 2, 1], depths=[2, 2, 2], embed_dims=[192, 192, 192],
                num_heads=[6, 6, 6], inputconv_groups=[1, 3, 4, 6, 8, 4],
                deformable_groups=12, attention_heads=12, attention_window=[3, 3],
                nonblind_denoising=True, cpu_cache_length=100)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["params"] if "params" in ckpt else ckpt, strict=True)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=8, help="must be a multiple of 2")
    ap.add_argument("--crop", type=int, default=128, help="must be a multiple of 8")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None,
                     help="pretrained checkpoint to start from (default: RVRT's 006 DAVIS denoising)")
    ap.add_argument("--resume", type=str, default=None,
                     help="resume training from a p3_train_rvrt.py checkpoint (has step count)")
    args = ap.parse_args()

    assert args.seq_len % 2 == 0, "--seq-len must be a multiple of 2 (RVRT window_size[0]=2)"
    assert args.crop % 8 == 0, "--crop must be a multiple of 8 (RVRT window_size[-1]=8)"

    out_dir = Path(args.out_dir or (SCRIPT_DIR.parent / "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device found, this will be extremely slow.")
    else:
        print(f"device: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    ds = P3TrainSequences(split="train", seq_len=args.seq_len, crop=args.crop,
                          tracks=("a",), kinds=("gaussian", "poisson_gaussian"))
    print(f"training clips available: {len(ds)}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0, collate_fn=collate_fn)

    ckpt_path = Path(args.ckpt) if args.ckpt else (
        RVRT_DIR / "model_zoo" / "rvrt" / "006_RVRT_videodenoising_DAVIS_16frames.pth")
    model = build_model(device, ckpt_path)

    start_step = 0
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd["model"])
        start_step = sd.get("step", 0)
        print(f"resumed from {args.resume} at step {start_step}")
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    step = start_step
    running = 0.0
    t0 = time.time()
    data_iter = iter(loader)

    def save(tag: str):
        payload = {"model": model.state_dict(), "step": step}
        torch.save(payload, out_dir / f"rvrt_dashcam_{tag}.pth")
        torch.save(payload, out_dir / "rvrt_dashcam_latest.pth")
        print(f"  saved rvrt_dashcam_{tag}.pth")

    try:
        while step < args.steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            sigma = batch["sigma"].to(device, non_blocking=True)
            x = torch.cat([lq, sigma], dim=2)   # (B,T,4,H,W)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(x)
                loss = charbonnier(pred, gt)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                sec_per_step = elapsed / max(1, step - start_step)
                eta_min = sec_per_step * (args.steps - step) / 60
                print(f"step {step}/{args.steps}  loss {running / args.log_every:.5f}  "
                      f"{sec_per_step:.2f}s/step  eta {eta_min:.1f}min")
                running = 0.0

            if step % args.save_every == 0 or step == args.steps:
                save(f"step{step}")
    except KeyboardInterrupt:
        print("interrupted - saving current state")
        save(f"interrupt{step}")
        raise

    print(f"done. final checkpoint: {out_dir / 'rvrt_dashcam_latest.pth'}")


if __name__ == "__main__":
    main()
