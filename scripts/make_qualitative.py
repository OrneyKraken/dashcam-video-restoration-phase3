"""Produce before/after visual comparisons across all four models.

The evaluation runs (p3_evaluate.py) only record metrics - they never write
images. This script re-runs a SMALL set of clips through every model and
saves the visual material a thesis/paper needs:

  <out>/<clip>_<kind>_<level>/
      frame_XXXX_1_clean.png        ground truth
      frame_XXXX_2_noisy.png        degraded input ("before")
      frame_XXXX_3_<model>.png      each model's restoration ("after")
      compare_frame_XXXX.png        all of the above side by side, labelled
      video_<model>.mp4             before|after split-screen clip
      crops_frame_XXXX.png          zoomed detail crop, same layout

Degradation uses the exact same seeded p3_degrade path as p3_evaluate.py, so
the "before" image here is bit-identical to what was actually scored.

Usage (after the metric runs have finished, so the GPU is free):
    python make_qualitative.py --clips raw_video_008_clip_0000 raw_video_022_clip_0001 \
        --kind gaussian --level high --frames 40
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

import p3_evaluate as ev  # reuses load_clip_rgb01 / degrade_clip / psnr

MODELS = {
    "rvrt": "rvrt_wrapper:restore",
    "basicvsrpp": "basicvsrpp_wrapper:restore",
    "fastdvdnet": "fastdvdnet_wrapper:restore",
    "dashmamba": "dashmamba_wrapper:restore",
}

LABEL_H = 34


def _to_bgr8(img01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(img01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bar = np.full((LABEL_H, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img_bgr])


def load_model(spec: str):
    mod_name, _, attr = spec.partition(":")
    import importlib
    fn = getattr(importlib.import_module(mod_name), attr)
    wants_meta = len(inspect.signature(fn).parameters) >= 2
    return fn, wants_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--kind", default="gaussian")
    ap.add_argument("--level", default="high")
    ap.add_argument("--frames", type=int, default=40, help="frames per clip to process")
    ap.add_argument("--save-frames", nargs="+", type=int, default=[0, 20],
                    help="which frame indices to save as stills / comparison grids")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--crop", nargs=4, type=int, default=None,
                    metavar=("X", "Y", "W", "H"), help="detail-crop box; default is a centre box")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    out_root = Path(args.out) if args.out else (ev.P3_ROOT / "results" / "qualitative")
    out_root.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for name in args.models:
        try:
            loaded[name] = load_model(MODELS[name])
            print(f"[ok]   {name}")
        except Exception as e:
            print(f"[skip] {name}: {e}")

    for stem in args.clips:
        print(f"\n=== {stem}  ({args.kind}/{args.level}) ===")
        clean, names = ev.load_clip_rgb01(stem, args.frames)
        lq = ev.degrade_clip(clean, names, stem, args.kind, args.level)

        outs = {}
        for name, (fn, wants_meta) in loaded.items():
            meta = {"kind": args.kind, "level": args.level, "clip_stem": stem}
            sr = fn(lq, meta) if wants_meta else fn(lq)
            sr = np.clip(np.asarray(sr, np.float32), 0, 1)
            outs[name] = sr
            p = float(np.mean([ev.psnr(sr[i], clean[i]) for i in range(len(sr))]))
            print(f"    {name:12s} psnr={p:.2f} dB")

        d = out_root / f"{stem}_{args.kind}_{args.level}"
        d.mkdir(parents=True, exist_ok=True)

        H, W = clean.shape[1:3]
        if args.crop:
            cx, cy, cw, ch = args.crop
        else:
            cw, ch = W // 4, H // 4
            cx, cy = (W - cw) // 2, (H - ch) // 2

        for fi in args.save_frames:
            if fi >= len(clean):
                continue
            panels, crops = [], []
            for label, img in [("1_clean (ground truth)", clean[fi]),
                               ("2_noisy (input)", lq[fi])] + \
                              [(f"3_{n}", outs[n][fi]) for n in outs]:
                bgr = _to_bgr8(img)
                cv2.imwrite(str(d / f"frame_{fi:04d}_{label.replace(' ', '_').replace('(', '').replace(')', '')}.png"), bgr)
                panels.append(_label(bgr, label))
                crops.append(_label(cv2.resize(bgr[cy:cy + ch, cx:cx + cw], (cw * 2, ch * 2),
                                               interpolation=cv2.INTER_NEAREST), label))
            cv2.imwrite(str(d / f"compare_frame_{fi:04d}.png"), np.hstack(panels))
            cv2.imwrite(str(d / f"crops_frame_{fi:04d}.png"), np.hstack(crops))
            print(f"    wrote compare_frame_{fi:04d}.png / crops_frame_{fi:04d}.png")

        # split-screen before|after videos, one per model
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for name, sr in outs.items():
            vp = d / f"video_{name}.mp4"
            vw = cv2.VideoWriter(str(vp), fourcc, args.fps, (W * 2, H + LABEL_H))
            if not vw.isOpened():
                print(f"    [warn] could not open {vp}")
                continue
            for i in range(len(sr)):
                left = _label(_to_bgr8(lq[i]), "BEFORE (noisy input)")
                right = _label(_to_bgr8(sr[i]), f"AFTER ({name})")
                vw.write(np.hstack([left, right]))
            vw.release()
            print(f"    wrote {vp.name}")

    print(f"\nall qualitative output under: {out_root}")


if __name__ == "__main__":
    main()
