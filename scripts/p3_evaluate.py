"""Step 3 - evaluate a restoration model on the Phase 3 benchmark.

Reads  : thesis_p3/metadata/p3_eval_index.csv   (which clip / degradation / level)
         thesis_p3/dataset/_refpool/<clip>/frame_*.png   (clean frames, by name)
Applies: p3_degrade.py   (deterministic, seeded degradation)
Writes : thesis_p3/results/<run>/per_clip.csv
         thesis_p3/results/<run>/summary.csv
         thesis_p3/results/<run>/config.json

Nothing is read by directory listing of _refpool/ - clips are looked up by name
from the CSV - so the NTFS duplicate-directory-entry issue cannot affect results.

Model interface
---------------
A model is a callable:  restore(lq) -> sr
  lq, sr : float32 ndarray, shape (T, H, W, 3), range 0..1, RGB
Built-in:  --model identity          (sr = lq; the "no restoration" floor)
External:  --model-import pkg.mod:fn  (fn is the callable above; used on Kaggle)

A model that needs to know the true degradation (e.g. a non-blind denoiser
that takes a known noise level) may instead define restore(lq, meta) -> sr,
where meta = {"kind": ..., "level": ..., "clip_stem": ...}. Detected
automatically from the function's signature - no flag needed. This mirrors
how such models are evaluated in their own papers (given the true noise
level, not forced to guess it blind).

Metrics
-------
full-reference (tracks A, B):  PSNR, SSIM, [LPIPS if torch+lpips available]
all tracks                  :  tOF  (temporal optical-flow warping error, lower=better)
no-reference (track C, opt.) :  [BRISQUE/NIQE if pyiqa available]
"""

from __future__ import annotations

import argparse
import csv
import inspect as _inspect
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

import p3_degrade
from p3_config import DATASET_DIR, METADATA_DIR, P3_ROOT

REFPOOL = DATASET_DIR / "_refpool"
EVAL_INDEX = METADATA_DIR / "p3_eval_index.csv"
RESULTS_DIR = P3_ROOT / "results"


# --------------------------------------------------------------------------- #
# frame IO  (always by explicit clip name - never lists _refpool/)
# --------------------------------------------------------------------------- #
def load_clip_rgb01(clip_stem: str, max_frames: int | None = None) -> tuple[np.ndarray, list[str]]:
    d = REFPOOL / clip_stem
    frames = sorted(d.glob("frame_*.png"))
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise FileNotFoundError(f"no frames for {clip_stem} at {d}")
    seq = np.empty((len(frames), *cv2.imread(str(frames[0])).shape[:2], 3), np.float32)
    names = []
    for i, fp in enumerate(frames):
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        seq[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        names.append(fp.name)
    return seq, names


def degrade_clip(clean: np.ndarray, names: list[str], clip_stem: str,
                 kind: str, level: str) -> np.ndarray:
    if kind == "none":
        return clean.copy()
    out = np.empty_like(clean)
    for i, nm in enumerate(names):
        bgr_u8 = cv2.cvtColor((clean[i] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        d = p3_degrade.degrade(bgr_u8, kind, level, clip_stem, nm)
        out[i] = cv2.cvtColor(d, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return out


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse == 0 else 10.0 * np.log10(1.0 / mse)


_SSIM_C1 = 0.01 ** 2
_SSIM_C2 = 0.03 ** 2


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over channels, Gaussian window 11 / sigma 1.5 (Wang et al.)."""
    def _one(x, y):
        x = x.astype(np.float64); y = y.astype(np.float64)
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        s_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        s_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        s_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        num = (2 * mu_xy + _SSIM_C1) * (2 * s_xy + _SSIM_C2)
        den = (mu_x2 + mu_y2 + _SSIM_C1) * (s_x + s_y + _SSIM_C2)
        return float(np.mean(num / den))
    return float(np.mean([_one(a[..., c], b[..., c]) for c in range(a.shape[-1])]))


def temporal_warping_error(seq: np.ndarray, guide: np.ndarray) -> float:
    """Mean warped-frame MSE: warp seq[t-1] -> t using flow estimated on `guide`
    (the clean sequence, so motion is the same for every model), compare to seq[t].
    Lower = more temporally consistent. Returns value x1e3 for readability."""
    if len(seq) < 2:
        return float("nan")
    errs = []
    g_prev = cv2.cvtColor((guide[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    H, W = g_prev.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    for t in range(1, len(seq)):
        g_cur = cv2.cvtColor((guide[t] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g_prev, g_cur, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        map_x = (gx + flow[..., 0]).astype(np.float32)
        map_y = (gy + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(seq[t - 1], map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        valid = ((map_x >= 0) & (map_x < W) & (map_y >= 0) & (map_y < H))[..., None]
        diff = (warped - seq[t]) ** 2 * valid
        errs.append(float(diff.sum() / max(valid.sum() * seq.shape[-1], 1)))
        g_prev = g_cur
    return float(np.mean(errs) * 1e3)


# optional metrics ---------------------------------------------------------- #
class _Optional:
    def __init__(self):
        self.lpips = None
        self.brisque = None
        self.niqe = None
        try:
            import torch, lpips  # noqa
            self._torch = torch
            self._lpips_net = lpips.LPIPS(net="alex", verbose=False)
            self._lpips_net.eval()
            self.lpips = self._lpips_run
        except Exception:
            self._torch = None
        try:
            import pyiqa  # noqa
            dev = "cuda" if (self._torch and self._torch.cuda.is_available()) else "cpu"
            self._brisque_m = pyiqa.create_metric("brisque", device=dev)
            self._niqe_m = pyiqa.create_metric("niqe", device=dev)
            self.brisque = lambda s: self._iqa_run(self._brisque_m, s)
            self.niqe = lambda s: self._iqa_run(self._niqe_m, s)
        except Exception:
            pass

    def _lpips_run(self, a, b):
        t = self._torch
        with t.no_grad():
            xa = t.from_numpy(a.transpose(0, 3, 1, 2)).float() * 2 - 1
            xb = t.from_numpy(b.transpose(0, 3, 1, 2)).float() * 2 - 1
            return float(self._lpips_net(xa, xb).mean())

    def _iqa_run(self, metric, seq):
        t = self._torch
        vals = []
        for fr in seq:
            x = t.from_numpy(fr.transpose(2, 0, 1)[None]).float()
            vals.append(float(metric(x)))
        return float(np.mean(vals))


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load_model(spec: str, import_spec: str | None):
    if import_spec:
        mod_name, _, attr = import_spec.partition(":")
        import importlib
        fn = getattr(importlib.import_module(mod_name), attr)
        wants_meta = len(_inspect.signature(fn).parameters) >= 2
        return fn, import_spec, wants_meta
    if spec == "identity":
        return (lambda lq: lq), "identity", False
    raise SystemExit(f"unknown --model {spec!r} (use 'identity' or --model-import)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def load_axes(tracks: set[str], splits: set[str]):
    with EVAL_INDEX.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["track"] in tracks and r["split"] in splits]
    axes = defaultdict(list)
    for r in rows:
        axes[(r["track"], r["split"], r["kind"], r["level"])].append(r)
    return axes


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a restoration model on the Phase 3 benchmark.")
    ap.add_argument("--model", default="identity", help="'identity' or a name handled by --model-import")
    ap.add_argument("--model-import", default=None, metavar="pkg.mod:fn",
                    help="import a restore(lq)->sr callable (used on Kaggle with a real model)")
    ap.add_argument("--run-name", default=None, help="results/<run-name>/ (default: model + timestamp)")
    ap.add_argument("--tracks", default="ABC")
    ap.add_argument("--splits", default="test", help="comma list: test,val")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames per clip (debug/speed)")
    ap.add_argument("--limit-clips", type=int, default=None, help="only N clips per axis (debug)")
    ap.add_argument("--stride", type=int, default=None,
                    help="take every Nth clip per axis (spreads evenly across source videos, "
                         "unlike --limit-clips which takes a prefix) - for time-budgeted runs")
    ap.add_argument("--no-tof", action="store_true", help="skip temporal warping error (slow)")
    args = ap.parse_args()

    tracks = set(args.tracks.upper())
    splits = set(s.strip() for s in args.splits.split(","))
    restore, model_id, wants_meta = load_model(args.model, args.model_import)
    opt = _Optional()
    run_name = args.run_name or f"{Path(model_id).name.replace(':', '_')}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run           : {run_name}")
    print(f"model         : {model_id}")
    print(f"tracks/splits : {sorted(tracks)} / {sorted(splits)}")
    print(f"optional      : LPIPS={'y' if opt.lpips else 'n'}  BRISQUE/NIQE={'y' if opt.brisque else 'n'}")
    print(f"tOF           : {'off' if args.no_tof else 'on'}")

    axes = load_axes(tracks, splits)
    t0 = time.time()

    # Fixed superset of columns (independent of any single row) so per_clip.csv
    # can be written incrementally - one flush per clip - and resumed after a
    # crash/interruption without losing already-computed clips (long runs on
    # this hardware can take hours; see handoff notes on drive fragility).
    per_clip_path = out_dir / "per_clip.csv"
    fields = ["track", "split", "kind", "level", "clip_stem", "lighting", "condition",
             "frames", "psnr", "ssim", "lpips", "psnr_in", "tof", "tof_in", "brisque", "niqe"]

    done = set()
    per_clip_rows = []
    if per_clip_path.exists():
        with per_clip_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for key in list(r):
                    if key in ("track", "split", "kind", "level", "clip_stem", "lighting", "condition"):
                        continue
                    if r[key] == "":
                        del r[key]
                    elif key == "frames":
                        r[key] = int(r[key])
                    else:
                        r[key] = float(r[key])
                per_clip_rows.append(r)
                done.add((r["track"], r["split"], r["kind"], r["level"], r["clip_stem"]))
        print(f"resuming: {len(done)} clips already done in {per_clip_path}")

    csv_f = per_clip_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=fields, restval="")
    if not done:
        writer.writeheader()

    for (track, split, kind, level), items in sorted(axes.items()):
        if args.stride:
            items = items[::args.stride]
        if args.limit_clips:
            items = items[:args.limit_clips]
        need_ref = items[0]["needs_reference"] == "1"
        print(f"\n[{track} {split} {kind}/{level}]  {len(items)} clips  (reference={'yes' if need_ref else 'no'})")
        for k, it in enumerate(items, 1):
            stem = it["clip_stem"]
            if (track, split, kind, level, stem) in done:
                continue
            clean, names = load_clip_rgb01(stem, args.max_frames)
            lq = degrade_clip(clean, names, stem, kind, level)
            if wants_meta:
                sr_raw = restore(lq, {"kind": kind, "level": level, "clip_stem": stem})
            else:
                sr_raw = restore(lq)
            sr = np.ascontiguousarray(np.clip(np.asarray(sr_raw, np.float32), 0, 1))

            row = {"track": track, "split": split, "kind": kind, "level": level,
                   "clip_stem": stem, "lighting": it["lighting"], "condition": it["condition"],
                   "frames": len(sr)}
            if need_ref:
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
            per_clip_rows.append(row)
            writer.writerow(row)
            csv_f.flush()
            if k % 10 == 0 or k == len(items):
                el = time.time() - t0
                print(f"   {k}/{len(items)}  last psnr={row.get('psnr','-')} "
                      f"ssim={row.get('ssim','-')} tof={row.get('tof','-')}  ({el:.0f}s)")

    csv_f.close()

    # aggregate: per axis, and per axis x lighting
    def agg(keyfn):
        groups = defaultdict(list)
        for r in per_clip_rows:
            groups[keyfn(r)].append(r)
        out = []
        for key, rs in sorted(groups.items()):
            e = dict(zip(("track", "split", "kind", "level", "lighting"), list(key) + [""] * 5))
            e["clips"] = len(rs)
            for m in ("psnr", "ssim", "lpips", "psnr_in", "tof", "tof_in", "brisque", "niqe"):
                vals = [r[m] for r in rs if m in r and r[m] == r[m]]
                if vals:
                    e[m] = round(float(np.mean(vals)), 5)
            out.append(e)
        return out

    summary = agg(lambda r: (r["track"], r["split"], r["kind"], r["level"]))
    by_light = agg(lambda r: (r["track"], r["split"], r["kind"], r["level"], r["lighting"]))
    sfields = ["track", "split", "kind", "level", "lighting", "clips",
               "psnr", "ssim", "lpips", "psnr_in", "tof", "tof_in", "brisque", "niqe"]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sfields, extrasaction="ignore")
        w.writeheader()
        for e in summary: w.writerow(e)
        w.writerow({})
        for e in by_light: w.writerow(e)

    (out_dir / "config.json").write_text(json.dumps({
        "run_name": run_name, "model": model_id, "tracks": sorted(tracks),
        "splits": sorted(splits), "max_frames": args.max_frames,
        "limit_clips": args.limit_clips, "tof": not args.no_tof,
        "lpips": bool(opt.lpips), "brisque_niqe": bool(opt.brisque),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - t0, 1),
    }, indent=2))

    print(f"\n=== summary ({run_name}) ===")
    for e in summary:
        print(f"  {e['track']} {e['kind']:16s} {e['level']:10s} "
              f"psnr={e.get('psnr','-')} ssim={e.get('ssim','-')} "
              f"lpips={e.get('lpips','-')} tof={e.get('tof','-')} "
              f"(in: psnr={e.get('psnr_in','-')} tof={e.get('tof_in','-')})")
    print(f"\nwrote {out_dir}/per_clip.csv, summary.csv, config.json")


if __name__ == "__main__":
    main()
