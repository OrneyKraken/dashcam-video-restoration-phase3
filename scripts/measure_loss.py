"""Measure DashMamba's training loss (Charbonnier) on held-out data, with no
gradient update. No training-loss curve was logged during the original runs
(see HANDOFF_FOR_NEW_CHAT.md), so this is the substitute: load a checkpoint,
run it forward over windows drawn the same way training did (P3TrainSequences
/ DavisTrainSequences: random seq_len-frame window, random crop, random
degradation from TRAIN_KINDS), and average the same loss train_dashmamba.py
optimized - charbonnier(pred, gt), eps=1e-3 - but on a split never trained on.

Track A test split (--data track-a-test) never appears in P3TrainSequences'
"train" split, so it is held-out for BOTH checkpoints. DAVIS test-dev
(--data davis-test-dev, once downloaded - see the DAVIS-2017-test-dev-480p.zip
fetch) is held out for Stage-1 too: Stage-1 pretraining used whatever the
downloaded DAVIS folder contained (see pretrain_dataloader.py), which on the
original machine was the trainval package, not test-dev - a disjoint official
DAVIS 2017 split, so evaluating on it is leak-free without needing to know
exactly which videos the original machine's folder held.

Usage:
    python measure_loss.py --ckpt ../checkpoints/finetune_stage2.pt --data track-a-test
    python measure_loss.py --ckpt ../checkpoints/pretrain_stage1.pt --data davis-test-dev \
        --davis-root D:/thesis_p3/external/pretrain_data/DAVIS_test_dev/DAVIS
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from dashmamba import DashMambaNet  # noqa: E402

TRAIN_KINDS = ("gaussian", "poisson_gaussian")  # matches train_dashmamba.py


def charbonnier(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps * eps))


def build_dataset(which: str, seq_len: int, crop: int, seed: int, davis_root: str | None):
    if which == "track-a-test":
        from p3_dataloader import P3TrainSequences
        return P3TrainSequences(split="test", seq_len=seq_len, crop=crop, tracks=("a",),
                                kinds=TRAIN_KINDS, augment=False, seed=seed)
    from pretrain_dataloader import DavisTrainSequences
    return DavisTrainSequences(davis_root, seq_len=seq_len, crop=crop,
                               kinds=TRAIN_KINDS, augment=False, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", choices=["track-a-test", "davis-test-dev"], required=True)
    ap.add_argument("--davis-root", default=None, help="required for --data davis-test-dev")
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--passes", type=int, default=3,
                    help="re-draw the dataset this many times with different seeds "
                         "(each pass = one random window+degradation per clip)")
    args = ap.parse_args()

    if args.data == "davis-test-dev" and not args.davis_root:
        raise SystemExit("--davis-root is required for --data davis-test-dev")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DashMambaNet(mid_channels=64, num_res_blocks=5, state_dim=16, num_mamba_blocks=2)
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    print(f"loaded {args.ckpt} (trained to step {ck.get('step', '?')})")

    losses, by_kind = [], {}
    for p in range(args.passes):
        ds = build_dataset(args.data, args.seq_len, args.crop, seed=1000 + p, davis_root=args.davis_root)
        if p == 0:
            print(f"dataset: {args.data}  ({len(ds)} clips/videos per pass x {args.passes} passes "
                  f"= {len(ds) * args.passes} windows, seq_len={args.seq_len}, crop={args.crop})")
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                lq = s["lq"].unsqueeze(0).float().to(device)
                gt = s["gt"].unsqueeze(0).float().to(device)
                pred = model(lq)
                loss = charbonnier(pred, gt).item()
                losses.append(loss)
                by_kind.setdefault(s["kind"], []).append(loss)

    mean, sd = statistics.mean(losses), statistics.pstdev(losses)
    se = sd / len(losses) ** 0.5
    print(f"\ncharbonnier loss, {args.data}, n={len(losses)} windows (never used in this "
          f"checkpoint's training): mean={mean:.5f}  std={sd:.5f}  SE={se:.5f}  "
          f"95% CI=[{mean - 1.96 * se:.5f}, {mean + 1.96 * se:.5f}]")
    for k, vs in sorted(by_kind.items()):
        print(f"  {k:18s} n={len(vs):4d}  mean={statistics.mean(vs):.5f}")


if __name__ == "__main__":
    main()
