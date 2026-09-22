"""Step 2 - extract clean 1280x720 @ 15 fps frames for the Phase 3 benchmark.

Runtime-degradation design: only CLEAN frames are stored, once, in _refpool/.
Gaussian / Poisson-Gaussian / synthetic-low-light degradations are applied on
the fly by the training dataloader and the evaluation script, both importing
p3_degrade.py, seeded per (clip, frame, kind, level) so every number is
bit-exact reproducible. Split / track membership lives in the metadata CSVs,
so no frame is ever copied or hardlinked.

Reads : thesis_p3/metadata/p3_clip_splits.csv
        thesis_dataset/clips/<source>/<clip>.mp4        (existing clips, read only)
Writes: thesis_p3/dataset/_refpool/<clip_stem>/frame_000001.png
        thesis_p3/dataset/preview/<kind>_<level>/<clip_stem>/   (only with --preview N)
        thesis_p3/metadata/p3_frame_manifest.csv
        thesis_p3/metadata/p3_eval_index.csv
        thesis_p3/metadata/p3_dataset_summary.csv

Resumable: every frame folder gets a _done.txt marker; rerun skips finished work.
"""

import argparse
import csv
import os
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

cv2.setNumThreads(1)

from p3_config import (
    OLD_ROOT, CLIP_SPLITS_CSV, DATASET_DIR, REFPOOL_DIR, METADATA_DIR,
    FRAME_MANIFEST_CSV, DATASET_SUMMARY_CSV,
    FPS, FRAME_WIDTH, FRAME_HEIGHT, PNG_COMPRESSION, DONE_MARKER,
)
import p3_degrade

PREVIEW_DIR = DATASET_DIR / "preview"
EVAL_INDEX_CSV = METADATA_DIR / "p3_eval_index.csv"

# evaluation axes -> written into p3_eval_index.csv, consumed later by the eval script
TRACK_A_AXES = [("gaussian", "low"), ("gaussian", "medium"), ("gaussian", "high"),
                ("poisson_gaussian", "realistic")]
TRACK_B_AXES = [("low_light", "low"), ("low_light", "medium"), ("low_light", "high")]


# --------------------------------------------------------------------------- #
def safe_inside(path: Path) -> None:
    r = path.resolve()
    if DATASET_DIR.resolve() not in r.parents and r != DATASET_DIR.resolve():
        raise RuntimeError(f"refusing to touch path outside dataset dir: {path}")


def count_frames(d: Path) -> int:
    return sum(1 for _ in d.glob("frame_*.png")) if d.exists() else 0


def marker_ok(d: Path, expected: int | None = None) -> bool:
    if not (d / DONE_MARKER).exists():
        return False
    n = count_frames(d)
    return n > 0 and (expected is None or n == expected)


def write_marker(d: Path, **kv) -> None:
    lines = [f"completed_at={datetime.now().isoformat(timespec='seconds')}"]
    lines += [f"{k}={v}" for k, v in kv.items()]
    (d / DONE_MARKER).write_text("\n".join(lines) + "\n", encoding="utf-8")


def clear_dir(d: Path) -> None:
    safe_inside(d)
    if not d.exists():
        return
    for p in d.glob("frame_*.png"):
        p.unlink()
    if (d / DONE_MARKER).exists():
        (d / DONE_MARKER).unlink()


# --------------------------------------------------------------------------- #
# frame extraction (worker)
# --------------------------------------------------------------------------- #
def _extract_worker(payload: tuple) -> tuple:
    r, force = payload
    stem = Path(r["clip_name"]).stem
    out = REFPOOL_DIR / stem
    if marker_ok(out) and not force:
        return r, stem, "ok", count_frames(out), ""
    clear_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    clip_path = OLD_ROOT / r["clip_path"]
    if not clip_path.exists():
        return r, stem, "error", 0, f"missing clip {clip_path}"
    vf = f"fps={FPS},scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=lanczos"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(clip_path),
           "-vf", vf, "-start_number", "1", "-compression_level", str(PNG_COMPRESSION),
           str(out / "frame_%06d.png")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    n = count_frames(out)
    if res.returncode != 0:
        return r, stem, "error", n, res.stderr.strip()[:300]
    if n == 0:
        return r, stem, "error", 0, "ffmpeg produced zero frames"
    write_marker(out, fps=FPS, width=FRAME_WIDTH, height=FRAME_HEIGHT, frame_count=n)
    return r, stem, "ok", n, ""


# --------------------------------------------------------------------------- #
# preview render (worker) - tiny pre-rendered set for paper figures only
# --------------------------------------------------------------------------- #
def _preview_worker(payload: tuple) -> tuple:
    stem, kind, level = payload
    src = REFPOOL_DIR / stem
    frames = sorted(src.glob("frame_*.png"))
    if not frames:
        return stem, kind, level, "error", 0
    dst = PREVIEW_DIR / f"{kind}_{level}" / stem
    if marker_ok(dst, len(frames)):
        return stem, kind, level, "ok", count_frames(dst)
    clear_dir(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for fp in frames:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        out = p3_degrade.degrade(img, kind, level, stem, fp.name)
        cv2.imwrite(str(dst / fp.name), out, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    write_marker(dst, kind=kind, level=level, frame_count=len(frames))
    return stem, kind, level, "ok", len(frames)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def load_split_rows() -> list[dict]:
    with CLIP_SPLITS_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract the Phase 3 clean-frame dataset.")
    ap.add_argument("--limit", type=int, default=None, help="process at most N clips (debug)")
    ap.add_argument("--force", action="store_true", help="re-extract even if _done.txt exists")
    ap.add_argument("--extract-jobs", type=int, default=3,
                    help="concurrent ffmpeg extractions (default 3)")
    ap.add_argument("--preview", type=int, default=0,
                    help="also pre-render this many test clips x every axis into dataset/preview/")
    ap.add_argument("--preview-jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_split_rows()
    need = [r for r in rows if r["in_track_a"] == "1" or r["in_track_b"] == "1" or r["in_track_c"] == "1"]

    ref_by_stem = {Path(r["clip_name"]).stem: r for r in need}
    ref_list = list(ref_by_stem.values())
    if args.limit:
        ref_list = ref_list[:args.limit]
    keep = {Path(r["clip_name"]).stem for r in ref_list}

    # ---- 1. extract clean reference frames (parallel ffmpeg) --------------
    print(f"reference clips to extract: {len(ref_list)}  (extract-jobs={args.extract_jobs})", flush=True)
    frame_rows, done = [], 0
    with ProcessPoolExecutor(max_workers=args.extract_jobs) as ex:
        for r, stem, status, n, err in ex.map(
                _extract_worker, [(r, args.force) for r in ref_list], chunksize=1):
            done += 1
            print(f"[ref {done}/{len(ref_list)}] {stem}  {status} {n}" + (f"  {err}" if err else ""),
                  flush=True)
            frame_rows.append({
                "clip_stem": stem, "clip_name": r["clip_name"], "source_video": r["source_video"],
                "split": r["split"], "lighting_final": r["lighting_final"],
                "condition_final": r["condition_final"],
                "in_track_a": r["in_track_a"], "in_track_b": r["in_track_b"], "in_track_c": r["in_track_c"],
                "fps": FPS, "width": FRAME_WIDTH, "height": FRAME_HEIGHT,
                "frame_count": n, "status": status, "error": err})
    frame_rows.sort(key=lambda x: x["clip_stem"])
    have = {fr["clip_stem"] for fr in frame_rows if fr["status"] == "ok"}

    # ---- 2. eval index (drives the later evaluation script) --------------
    _write_eval_index(rows, have)

    # ---- 3. optional preview render for paper figures -------------------
    if args.preview:
        a_test = [Path(r["clip_name"]).stem for r in need
                  if r["in_track_a"] == "1" and r["split"] == "test"
                  and Path(r["clip_name"]).stem in have][:args.preview]
        b_test = [Path(r["clip_name"]).stem for r in need
                  if r["in_track_b"] == "1" and r["split"] == "test"
                  and Path(r["clip_name"]).stem in have][:args.preview]
        jobs = ([(s, k, l) for s in a_test for k, l in TRACK_A_AXES]
                + [(s, k, l) for s in b_test for k, l in TRACK_B_AXES])
        print(f"\npreview render jobs: {len(jobs)}  (jobs={args.preview_jobs})", flush=True)
        with ProcessPoolExecutor(max_workers=args.preview_jobs) as ex:
            for s, k, l, st, n in ex.map(_preview_worker, jobs, chunksize=1):
                print(f"  preview {k}_{l} {s}  {st} {n}", flush=True)

    _write_manifests(frame_rows)
    _write_summary(frame_rows)


def _write_eval_index(rows, have):
    """One row per evaluation unit: (track, split, clip, kind, level, needs_reference).
    Track A eval  = val+test clips, gaussian(3 levels) + poisson_gaussian(realistic).
    Track B eval  = val+test day/good/clear+shadow clips, low_light(3 levels).
    Track C eval  = test evening/night clips, no degradation, no reference.
    Clean GT for A and B is dataset/_refpool/<clip_stem>/<frame>.
    """
    out = []
    for r in rows:
        stem = Path(r["clip_name"]).stem
        if stem not in have:
            continue
        if r["in_track_a"] == "1" and r["split"] in ("val", "test"):
            for kind, level in TRACK_A_AXES:
                out.append(("A", r["split"], stem, r["lighting_final"], r["condition_final"],
                            kind, level, 1))
        if r["in_track_b"] == "1":
            for kind, level in TRACK_B_AXES:
                out.append(("B", r["split"], stem, r["lighting_final"], r["condition_final"],
                            kind, level, 1))
        if r["in_track_c"] == "1":
            out.append(("C", r["split"], stem, r["lighting_final"], r["condition_final"],
                        "none", "real", 0))
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with EVAL_INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track", "split", "clip_stem", "lighting", "condition",
                    "kind", "level", "needs_reference"])
        w.writerows(out)
    print(f"\neval index: {len(out)} rows -> {EVAL_INDEX_CSV}", flush=True)


def _write_manifests(frame_rows):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with FRAME_MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        w.writeheader(); w.writerows(frame_rows)


def _write_summary(frame_rows):
    ok = [r for r in frame_rows if r["status"] == "ok"]
    by_split = Counter(r["split"] for r in ok)
    by_light = Counter(r["lighting_final"] for r in ok)
    total_frames = sum(r["frame_count"] for r in ok)
    lines = [
        ("clean_clips_ok", len(ok)),
        ("clean_clips_failed", len(frame_rows) - len(ok)),
        ("clean_frames_total", total_frames),
        ("clips_train", by_split["train"]), ("clips_val", by_split["val"]), ("clips_test", by_split["test"]),
        ("clips_day", by_light["day"]), ("clips_evening", by_light["evening"]), ("clips_night", by_light["night"]),
        ("fps", FPS), ("width", FRAME_WIDTH), ("height", FRAME_HEIGHT),
        ("degradation", "runtime via p3_degrade.py (nothing pre-noised on disk)"),
    ]
    with DATASET_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["key", "value"]); w.writerows(lines)
    print("\nsummary:", flush=True)
    for k, v in lines:
        print(f"  {k:22s} {v}")
    fails = [r for r in frame_rows if r["status"] != "ok"]
    if fails:
        print(f"\n{len(fails)} FAILED clips:")
        for r in fails[:20]:
            print(f"  {r['clip_stem']}: {r['error']}")


if __name__ == "__main__":
    main()
