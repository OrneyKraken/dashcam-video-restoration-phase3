"""Two-stage training driver for DashMambaNet.

  stage 1 (pretrain): DAVIS 2017, blind random degradation (matches how
                       RVRT/FastDVDnet's own official checkpoints were
                       trained - see docs/Phase3_Baseline_Weakness_Analysis
                       Finding A: task-matched pretraining is what makes a
                       checkpoint actually useful, not architecture alone).
  stage 2 (finetune):  Phase 3 dashcam training split (Track A), same recipe.

Resumable: saves a checkpoint every --save-every steps plus one on Ctrl-C /
crash-safe exit, and --resume picks the run back up from the last checkpoint
without losing progress - same reasoning as p3_evaluate.py's incremental
per-clip writes (see that script's own comments): long unattended runs on
this project need to survive interruption.

Usage:
    python train_dashmamba.py --stage pretrain --steps 2000 --run-name pretrain_v1
    python train_dashmamba.py --stage finetune --steps 2000 --run-name finetune_v1 \
        --init-from runs/pretrain_v1/last.pt
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from dashmamba import DashMambaNet  # noqa: E402

from p3_config import P3_ROOT  # noqa: E402

RUNS_DIR = P3_ROOT / "runs"
DAVIS_ROOT = P3_ROOT / "external" / "pretrain_data" / "DAVIS"


def charbonnier(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps * eps))


def cosine_lr(step: int, base_lr: float, total_steps: int, eta_min: float = 1e-6) -> float:
    """Stateless closed-form cosine schedule, purely a function of (step, total_steps).
    Deliberately not torch.optim.lr_scheduler.CosineAnnealingLR: that scheduler is
    recursive (each step computed from the current lr), so once it reaches eta_min
    it's a fixed point it can never leave - including within a single run that
    keeps training past its original T_max, resume or not. A stateless formula
    sidesteps that whole bug class and needs no separate state to save/restore."""
    t = min(step, total_steps)
    return eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t / total_steps))


def collate_lq_gt(batch):
    """P3TrainSequences returns DIFFERENT metadata keys depending on which
    degradation it randomly picked per sample (gaussian adds 'sigma255',
    low_light adds 'level', poisson_gaussian adds neither). torch's
    default_collate requires identical keys across every sample in a batch
    and raises KeyError on the first mismatch. The training loop only needs
    lq/gt, so collate exactly those and pass the varying metadata through as
    a plain list instead of trying to batch it."""
    return {
        "lq": torch.stack([b["lq"] for b in batch]),
        "gt": torch.stack([b["gt"] for b in batch]),
        "meta": [{k: v for k, v in b.items() if k not in ("lq", "gt")} for b in batch],
    }


# Train ONLY on the degradations Track A actually evaluates.
#
# The first training run used the full three-kind mix (gaussian,
# poisson_gaussian, low_light) and produced a model that learned ONLY the
# low-light brightness restoration and ignored denoising entirely
# (measured: low_light +8.73 dB with a large residual, gaussian +0.01 dB with
# a ~0.001 residual, i.e. identity). Low-light inputs are near-black - about
# 3 dB PSNR - so their loss dwarfs the denoising cases and the optimizer
# spends all its capacity brightening images. RVRT and FastDVDnet are each
# trained for one task, which is exactly why they reach 28+ dB here.
# If a low-light model is wanted later (Track B), train it as a SEPARATE
# checkpoint rather than mixing the objectives.
TRAIN_KINDS = ("gaussian", "poisson_gaussian")


def build_dataset(stage: str, seq_len: int, crop: int):
    if stage == "pretrain":
        from pretrain_dataloader import DavisTrainSequences
        return DavisTrainSequences(DAVIS_ROOT, seq_len=seq_len, crop=crop, kinds=TRAIN_KINDS)
    from p3_dataloader import P3TrainSequences
    return P3TrainSequences(split="train", seq_len=seq_len, crop=crop, tracks=("a",),
                            kinds=TRAIN_KINDS)


def save_ckpt(path: Path, model, optim, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "step": step}, tmp)
    tmp.replace(path)  # atomic on the same filesystem - avoids a half-written ckpt on interrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mid-channels", type=int, default=64)
    ap.add_argument("--num-res-blocks", type=int, default=5)
    ap.add_argument("--state-dim", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--init-from", default=None, help="checkpoint to initialize weights from (e.g. pretrain's last.pt for finetune)")
    ap.add_argument("--resume", action="store_true", help="resume THIS run from its own last checkpoint")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUNS_DIR / args.run_name
    ckpt_path = run_dir / "last.pt"

    print(f"stage        : {args.stage}")
    print(f"run          : {args.run_name}  ({run_dir})")
    print(f"device       : {device}")

    ds = build_dataset(args.stage, args.seq_len, args.crop)
    print(f"dataset      : {len(ds)} clips/videos")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0, collate_fn=collate_lq_gt)

    model = DashMambaNet(mid_channels=args.mid_channels, num_res_blocks=args.num_res_blocks,
                         state_dim=args.state_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params : {n_params:,}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_step = ck["step"]
        print(f"resumed from {ckpt_path} at step {start_step}")
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"initialized weights from {args.init_from} (step {ck.get('step', '?')} of that run)")

    # save on Ctrl-C / termination too, not just on the --save-every cadence
    state = {"step": start_step}

    def _save_and_exit(signum, frame):
        save_ckpt(ckpt_path, model, optim, state["step"])
        print(f"\ninterrupted - saved checkpoint at step {state['step']} to {ckpt_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    model.train()
    data_iter = iter(loader)
    t0 = time.time()
    loss_ema = None

    for step in range(start_step, args.steps):
        state["step"] = step
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        lq = batch["lq"].to(device, non_blocking=True).float()
        gt = batch["gt"].to(device, non_blocking=True).float()

        lr = cosine_lr(step, args.lr, args.steps)
        for g in optim.param_groups:
            g["lr"] = lr

        pred = model(lq)
        loss = charbonnier(pred, gt)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()

        loss_v = loss.item()
        loss_ema = loss_v if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_v

        if step % args.log_every == 0 or step == args.steps - 1:
            el = time.time() - t0
            done = step - start_step + 1
            rate = el / done
            eta_min = rate * (args.steps - step - 1) / 60
            print(f"step {step:6d}/{args.steps}  loss={loss_v:.4f}  ema={loss_ema:.4f}  "
                 f"lr={lr:.2e}  {rate:.2f}s/it  eta={eta_min:.1f}min")

        if step % args.save_every == 0 or step == args.steps - 1:
            state["step"] = step + 1
            save_ckpt(ckpt_path, model, optim, step + 1)

    print(f"done. final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
