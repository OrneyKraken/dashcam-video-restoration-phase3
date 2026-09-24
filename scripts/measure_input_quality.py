"""Mean NIQE/BRISQUE of the RAW (un-restored) frames for a track/axis, so the
"after" scores in a results summary.csv (computed on the model's output only -
see p3_evaluate.py's per-clip loop) have a "before" to compare against. No
model inference - just the two no-reference metrics on the input frames
themselves, reusing p3_evaluate.py's _Optional exactly.

Usage:
    python measure_input_quality.py --track C --level real --max-frames 80
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

from p3_evaluate import _Optional, load_clip_rgb01, degrade_clip
from p3_config import METADATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--kind", default=None, help="e.g. gaussian (omit for Track C's 'none')")
    ap.add_argument("--level", required=True)
    ap.add_argument("--max-frames", type=int, default=80)
    args = ap.parse_args()

    rows = list(csv.DictReader((METADATA_DIR / "p3_eval_index.csv").open(newline="", encoding="utf-8-sig")))
    items = [r for r in rows if r["track"] == args.track and r["split"] == args.split
            and r["level"] == args.level and (args.kind is None or r["kind"] == args.kind)]
    clips = sorted(set(r["clip_stem"] for r in items))
    kind = items[0]["kind"]
    print(f"{args.track}/{kind}/{args.level}: {len(clips)} clips")

    opt = _Optional()
    if not opt.brisque:
        raise SystemExit("pyiqa not available")

    niqe_vals, brisque_vals = [], []
    for i, stem in enumerate(clips, 1):
        clean, names = load_clip_rgb01(stem, args.max_frames)
        lq = degrade_clip(clean, names, stem, kind, args.level)
        niqe_vals.append(opt.niqe(lq))
        brisque_vals.append(opt.brisque(lq))
        if i % 20 == 0 or i == len(clips):
            print(f"  {i}/{len(clips)}  niqe={statistics.mean(niqe_vals):.3f}  "
                  f"brisque={statistics.mean(brisque_vals):.3f}")

    print(f"\nINPUT (before) quality, {args.track}/{kind}/{args.level}, n={len(clips)} clips:")
    print(f"  niqe    mean={statistics.mean(niqe_vals):.4f}  std={statistics.pstdev(niqe_vals):.4f}")
    print(f"  brisque mean={statistics.mean(brisque_vals):.4f}  std={statistics.pstdev(brisque_vals):.4f}")


if __name__ == "__main__":
    main()
