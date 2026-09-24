"""Zoomed center-crop comparisons from already-saved before/after PNGs (no
model rerun - pure image ops, seconds not minutes).

Same crop convention as make_qualitative.py: a centre box W/4 x H/4,
nearest-neighbour resized 2x, so detail-level noise/grain is actually visible
- full-frame comparisons compress real (mild) sensor noise below what's
legible at thumbnail scale, exactly what make_qualitative.py's own crops_*.png
was built to fix for the Track A figures.

Usage:
    python make_crops.py --dir ..\\figures\\track_c
    python make_crops.py --dir ..\\figures\\davis --labels "BEFORE (degraded)" "AFTER (DashMamba)" "GROUND TRUTH"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

LABEL_H = 30


def label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bar = np.full((LABEL_H, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img_bgr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--suffixes", nargs="+", default=["before", "after"],
                    help="file suffixes to include, in order, e.g. before after gt")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="one label per suffix; default = suffix name, upper-cased")
    ap.add_argument("--zoom", type=int, default=2)
    args = ap.parse_args()

    d = Path(args.dir)
    labels = args.labels or [s.upper() for s in args.suffixes]
    stems = sorted(set(p.name[:-(len(args.suffixes[0]) + 5)] for p in d.glob(f"*_{args.suffixes[0]}.png")))
    print(f"{len(stems)} sets found in {d}")

    for stem in stems:
        imgs = [cv2.imread(str(d / f"{stem}_{suf}.png")) for suf in args.suffixes]
        if any(im is None for im in imgs):
            print(f"  [skip] {stem}: missing a file")
            continue
        h, w = imgs[0].shape[:2]
        cw, ch = w // 4, h // 4
        cx, cy = (w - cw) // 2, (h - ch) // 2
        crops = [label(cv2.resize(im[cy:cy + ch, cx:cx + cw], (cw * args.zoom, ch * args.zoom),
                                  interpolation=cv2.INTER_NEAREST), lab)
                for im, lab in zip(imgs, labels)]
        out = d / f"{stem}_crop.png"
        cv2.imwrite(str(out), np.hstack(crops))
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
