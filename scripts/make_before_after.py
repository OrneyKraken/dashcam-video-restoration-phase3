"""Before/after visuals for the two new held-out tests (Track C real noise,
DAVIS test-dev generalization) - the same idea as make_qualitative.py used for
Track A, but scoped to DashMamba alone (no baselines run on these two sets;
see results/complexity/README.md and the Track C / DAVIS run configs for why)
and without RVRT's subprocess isolation dance, which existed only because that
script also loaded RVRT and FastDVDnet in the same process.

Clip/video selection is NOT cherry-picked: it stride-samples evenly across the
full eligible list (same reasoning as QUALITATIVE_RESULTS.md's median-clip
rule) rather than hand-picking the best-looking results.

Track C (real footage, no ground truth):
    before = the raw noisy input frame itself
    after  = DashMamba Stage-2's restoration (matches results/dashmamba_stage2_track_c_real)

DAVIS test-dev (synthetic degradation, ground truth available):
    before = degraded input at the given axis
    after  = DashMamba Stage-1's restoration (matches results/dashmamba_stage1_davis_testdev)
    gt     = the original clean DAVIS frame

Every clip gets: <name>_before.png, <name>_after.png, [<name>_gt.png],
and <name>_compare.png (labeled side-by-side) - so there are both solo
"after" shots and a paired proof.

Usage:
    set THESIS_P3_ROOT=D:\\thesis_p3
    set THESIS_P3_DASHMAMBA_CKPT=..\\checkpoints\\finetune_stage2.pt
    python make_before_after.py --dataset track-c --n 5 --frame 40 ^
        --out ..\\figures\\track_c

    set THESIS_P3_DASHMAMBA_CKPT=..\\checkpoints\\pretrain_stage1.pt
    python make_before_after.py --dataset davis --n 5 --kind gaussian --level high --frame 20 ^
        --davis-root D:\\thesis_p3\\external\\pretrain_data\\DAVIS_test_dev\\DAVIS ^
        --out ..\\figures\\davis
"""
from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path

import cv2
import numpy as np

import p3_degrade
from p3_config import METADATA_DIR, DATASET_DIR

REFPOOL = DATASET_DIR / "_refpool"
FRAME_WINDOW = 40  # frames of temporal context fed to the model around the saved frame


def stride_sample(items: list, n: int) -> list:
    """Evenly spaced picks across the sorted list - not the best/worst n, the
    whole range, so the sample isn't cherry-picked."""
    if n >= len(items):
        return items
    idx = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)] if n > 1 else [len(items) // 2]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


def save_png(path: Path, rgb01: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    img_bgr = img_bgr.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img_bgr, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(img_bgr, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img_bgr


def save_compare(path: Path, panels: list[tuple[str, np.ndarray]]):
    imgs = []
    for name, rgb01 in panels:
        bgr = cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        imgs.append(label(bgr, name))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(imgs, axis=1))


def load_frames(dir_: Path, glob: str, n: int) -> tuple[np.ndarray, list[str]]:
    paths = sorted(dir_.glob(glob))[:n]
    seq = np.empty((len(paths), *cv2.imread(str(paths[0])).shape[:2], 3), np.float32)
    names = []
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        seq[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        names.append(p.name)
    return seq, names


def run_track_c(args, restore):
    rows = list(csv.DictReader((METADATA_DIR / "p3_eval_index.csv").open(newline="", encoding="utf-8-sig")))
    clips = sorted(set(r["clip_stem"] for r in rows if r["track"] == "C" and r["split"] == "test"))
    picked = stride_sample(clips, args.n)
    print(f"Track C: {len(clips)} eligible clips, picked {len(picked)}: {picked}")

    for stem in picked:
        clean, names = load_frames(REFPOOL / stem, "frame_*.png", args.frame + FRAME_WINDOW)
        lq = clean  # kind=none, level=real: the raw frames ARE the noisy input
        sr = np.clip(np.asarray(restore(lq), np.float32), 0, 1)
        i = min(args.frame, len(sr) - 1)
        out = Path(args.out)
        save_png(out / f"{stem}_before.png", lq[i])
        save_png(out / f"{stem}_after.png", sr[i])
        save_compare(out / f"{stem}_compare.png",
                    [("BEFORE (real noise)", lq[i]), ("AFTER (DashMamba)", sr[i])])
        print(f"  {stem}: saved before/after/compare (frame {names[i]})")


def run_track_a(args, restore):
    rows = list(csv.DictReader((METADATA_DIR / "p3_eval_index.csv").open(newline="", encoding="utf-8-sig")))
    clips = sorted(set(r["clip_stem"] for r in rows if r["track"] == "A" and r["split"] == "test"
                       and r["kind"] == args.kind and r["level"] == args.level))
    # the eval index lists all 142 Track A test clips, but a portable copy of
    # the dataset (this repo's dataset/) may hold only the stride-2 subset
    # (71) actually used for scoring - restrict the pick to what's present.
    have = [c for c in clips if (REFPOOL / c).is_dir() and any((REFPOOL / c).glob("frame_*.png"))]
    if len(have) < len(clips):
        print(f"  ({len(clips) - len(have)} of {len(clips)} eligible clips have no local frames - "
              f"picking from the {len(have)} that do)")
    picked = stride_sample(have, args.n)
    print(f"Track A test, {args.kind}/{args.level}: {len(clips)} eligible clips, picked {len(picked)}: {picked}")

    for stem in picked:
        clean, names = load_frames(REFPOOL / stem, "frame_*.png", args.frame + FRAME_WINDOW)
        lq = np.empty_like(clean)
        for i, nm in enumerate(names):
            bgr_u8 = cv2.cvtColor((clean[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            d = p3_degrade.degrade(bgr_u8, args.kind, args.level, stem, nm)
            lq[i] = cv2.cvtColor(d, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sr = np.clip(np.asarray(restore(lq), np.float32), 0, 1)
        i = min(args.frame, len(sr) - 1)
        out = Path(args.out)
        save_png(out / f"{stem}_before.png", lq[i])
        save_png(out / f"{stem}_after.png", sr[i])
        save_png(out / f"{stem}_gt.png", clean[i])
        save_compare(out / f"{stem}_compare.png",
                    [("GROUND TRUTH", clean[i]), (f"BEFORE ({args.kind}/{args.level})", lq[i]),
                     ("AFTER (DashMamba)", sr[i])])
        print(f"  {stem}: saved before/after/gt/compare (frame {names[i]})")


def run_davis(args, restore):
    img_root = Path(args.davis_root) / "JPEGImages" / "480p"
    videos = sorted(p.name for p in img_root.iterdir() if p.is_dir())
    picked = stride_sample(videos, args.n)
    print(f"DAVIS test-dev: {len(videos)} videos, picked {len(picked)}: {picked}")

    for video in picked:
        clean, names = load_frames(img_root / video, "*.jpg", args.frame + FRAME_WINDOW)
        lq = np.empty_like(clean)
        for i, nm in enumerate(names):
            bgr_u8 = cv2.cvtColor((clean[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            d = p3_degrade.degrade(bgr_u8, args.kind, args.level, video, nm)
            lq[i] = cv2.cvtColor(d, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sr = np.clip(np.asarray(restore(lq), np.float32), 0, 1)
        i = min(args.frame, len(sr) - 1)
        out = Path(args.out)
        save_png(out / f"{video}_before.png", lq[i])
        save_png(out / f"{video}_after.png", sr[i])
        save_png(out / f"{video}_gt.png", clean[i])
        save_compare(out / f"{video}_compare.png",
                    [("GROUND TRUTH", clean[i]), (f"BEFORE ({args.kind}/{args.level})", lq[i]),
                     ("AFTER (DashMamba)", sr[i])])
        print(f"  {video}: saved before/after/gt/compare (frame {names[i]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["track-c", "track-a", "davis"], required=True)
    ap.add_argument("--n", type=int, default=5, help="clips/videos to render, stride-sampled")
    ap.add_argument("--frame", type=int, default=20, help="frame index (within the loaded window) to save")
    ap.add_argument("--kind", default="gaussian", help="DAVIS only")
    ap.add_argument("--level", default="high", help="DAVIS only")
    ap.add_argument("--davis-root", default=None, help="required for --dataset davis")
    ap.add_argument("--model-import", default="dashmamba_wrapper:restore")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.dataset == "davis" and not args.davis_root:
        raise SystemExit("--davis-root is required for --dataset davis")

    mod_name, _, attr = args.model_import.partition(":")
    restore = getattr(importlib.import_module(mod_name), attr)

    if args.dataset == "track-c":
        run_track_c(args, restore)
    elif args.dataset == "track-a":
        run_track_a(args, restore)
    else:
        run_davis(args, restore)
    print(f"\nwrote images to {args.out}")


if __name__ == "__main__":
    main()
