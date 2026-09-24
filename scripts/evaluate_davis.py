"""Evaluate a DashMamba checkpoint on DAVIS 2017 test-dev - held out from
Stage-1 pretraining (which used the downloaded DAVIS folder's videos; see
pretrain_dataloader.py) and from Track A's dashcam data entirely, to show
generalization beyond the training domain.

Reuses p3_evaluate.py's own metric functions (psnr, ssim, temporal_warping_error,
_Optional for LPIPS/NIQE/BRISQUE) unchanged, and p3_degrade.py's degradation
functions unchanged, so numbers are computed identically to the Track A runs -
only the source frames and the loader differ.

Same 4 axes as Track A test (gaussian low/medium/high, poisson_gaussian), same
metrics, same per_clip/summary.csv shape (with 'clip_stem' = the DAVIS video
name and 'lighting' left blank - DAVIS has no lighting label).

Usage:
    set THESIS_P3_DASHMAMBA_CKPT=..\checkpoints\pretrain_stage1.pt
    python evaluate_davis.py --davis-root D:/thesis_p3/external/pretrain_data/DAVIS_test_dev/DAVIS \
        --run-name dashmamba_stage1_davis_testdev --max-frames 80
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

import p3_degrade
from p3_evaluate import psnr, ssim, temporal_warping_error, _Optional
from p3_config import P3_ROOT

AXES = [("gaussian", "low"), ("gaussian", "medium"), ("gaussian", "high"),
       ("poisson_gaussian", "realistic")]


def load_video(video_dir: Path, max_frames: int | None) -> tuple[np.ndarray, list[str]]:
    frames = sorted(video_dir.glob("*.jpg"))
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise FileNotFoundError(f"no frames in {video_dir}")
    seq = np.empty((len(frames), *cv2.imread(str(frames[0])).shape[:2], 3), np.float32)
    names = []
    for i, fp in enumerate(frames):
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        seq[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        names.append(fp.stem)
    return seq, names


def degrade_video(clean: np.ndarray, names: list[str], video: str, kind: str, level: str) -> np.ndarray:
    out = np.empty_like(clean)
    for i, nm in enumerate(names):
        bgr_u8 = cv2.cvtColor((clean[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        d = p3_degrade.degrade(bgr_u8, kind, level, video, nm)
        out[i] = cv2.cvtColor(d, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis-root", required=True, help="path to .../DAVIS (holds JPEGImages/480p/<video>)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--model-import", default="dashmamba_wrapper:restore")
    ap.add_argument("--no-tof", action="store_true")
    args = ap.parse_args()

    import importlib
    mod_name, _, attr = args.model_import.partition(":")
    restore = getattr(importlib.import_module(mod_name), attr)

    img_root = Path(args.davis_root) / "JPEGImages" / "480p"
    videos = sorted(p.name for p in img_root.iterdir() if p.is_dir())
    if not videos:
        raise SystemExit(f"no videos under {img_root}")
    print(f"DAVIS test-dev: {len(videos)} videos from {img_root}")

    opt = _Optional()
    print(f"optional      : LPIPS={'y' if opt.lpips else 'n'}  BRISQUE/NIQE={'y' if opt.brisque else 'n'}")

    out_dir = P3_ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_clip_path = out_dir / "per_clip.csv"
    fields = ["kind", "level", "clip_stem", "frames", "psnr", "ssim", "lpips",
             "psnr_in", "tof", "tof_in", "brisque", "niqe"]

    done = set()
    rows = []
    if per_clip_path.exists():
        with per_clip_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k in list(r):
                    if k in ("clip_stem", "kind", "level"):
                        continue
                    if r[k] == "":
                        del r[k]
                    elif k == "frames":
                        r[k] = int(r[k])
                    else:
                        r[k] = float(r[k])
                rows.append(r)
                done.add((r["kind"], r["level"], r["clip_stem"]))
        print(f"resuming: {len(done)} already done")

    f = per_clip_path.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fields, restval="")
    if not done:
        w.writeheader()

    t0 = time.time()
    for kind, level in AXES:
        print(f"\n[{kind}/{level}]  {len(videos)} videos")
        for k, video in enumerate(videos, 1):
            if (kind, level, video) in done:
                continue
            clean, names = load_video(img_root / video, args.max_frames)
            lq = degrade_video(clean, names, video, kind, level)
            sr = np.ascontiguousarray(np.clip(np.asarray(restore(lq), np.float32), 0, 1))

            row = {"kind": kind, "level": level, "clip_stem": video, "frames": len(sr)}
            row["psnr"] = round(np.mean([psnr(sr[i], clean[i]) for i in range(len(sr))]), 4)
            row["ssim"] = round(np.mean([ssim(sr[i], clean[i]) for i in range(len(sr))]), 5)
            row["psnr_in"] = round(np.mean([psnr(lq[i], clean[i]) for i in range(len(sr))]), 4)
            if opt.lpips:
                row["lpips"] = round(opt.lpips(sr, clean), 5)
            if opt.brisque:
                row["brisque"] = round(opt.brisque(sr), 4)
                row["niqe"] = round(opt.niqe(sr), 4)
            if not args.no_tof:
                row["tof"] = round(temporal_warping_error(sr, clean), 5)
                row["tof_in"] = round(temporal_warping_error(lq, clean), 5)
            rows.append(row)
            w.writerow(row)
            f.flush()
            if k % 5 == 0 or k == len(videos):
                el = time.time() - t0
                print(f"   {k}/{len(videos)}  psnr={row['psnr']}  ssim={row['ssim']}  ({el:.0f}s)")
    f.close()

    groups = defaultdict(list)
    for r in rows:
        groups[(r["kind"], r["level"])].append(r)
    sfields = ["kind", "level", "clips", "psnr", "ssim", "lpips", "psnr_in", "tof", "tof_in", "brisque", "niqe"]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as sf:
        sw = csv.DictWriter(sf, fieldnames=sfields, extrasaction="ignore")
        sw.writeheader()
        for (kind, level), rs in sorted(groups.items()):
            e = {"kind": kind, "level": level, "clips": len(rs)}
            for m in ("psnr", "ssim", "lpips", "psnr_in", "tof", "tof_in", "brisque", "niqe"):
                vals = [r[m] for r in rs if m in r and r[m] == r[m]]
                if vals:
                    e[m] = round(float(np.mean(vals)), 5)
            sw.writerow(e)
            print(f"  {kind:16s} {level:10s} psnr={e.get('psnr','-')} ssim={e.get('ssim','-')} "
                  f"lpips={e.get('lpips','-')} niqe={e.get('niqe','-')} brisque={e.get('brisque','-')}")

    (out_dir / "config.json").write_text(json.dumps({
        "run_name": args.run_name, "model": args.model_import, "dataset": "DAVIS 2017 test-dev",
        "davis_root": args.davis_root, "videos": len(videos), "max_frames": args.max_frames,
        "finished": datetime.now().isoformat(timespec="seconds"), "elapsed_sec": round(time.time() - t0, 1),
    }, indent=2))
    print(f"\nwrote {out_dir}/per_clip.csv, summary.csv, config.json")


if __name__ == "__main__":
    main()
