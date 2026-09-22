"""Build a portable package so the DashMamba evaluation can run on another PC.

Produces two folders under --out:

  dashmamba_eval_core/   (~40 MB)  code + checkpoints + metadata + baseline
                                    results + README. Transfer this always.
  dashmamba_eval_data/   (~7.5 GB) ONLY the clip frames the evaluation
                                    actually reads (the stride-2 subset of
                                    Track A test, first --max-frames frames
                                    of each clip). Skip this if the target PC
                                    already has the full thesis_p3 dataset.

Why only a subset is needed: p3_evaluate.py reads clean frames by name from
dataset/_refpool/<clip>/ and regenerates the degraded input deterministically
(seeded per clip+frame+kind+level), so as long as the frame FILENAMES match,
the degraded inputs - and therefore the scores - are bit-identical to a run
against the full dataset. load_clip_rgb01 takes the first N frames, so a
folder holding only those N frames yields exactly the same result.

Usage:
    python package_for_eval.py --out F:/dashmamba_package
    python package_for_eval.py --out F:/dashmamba_package --skip-data
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from p3_config import P3_ROOT, METADATA_DIR, DATASET_DIR

EVAL_INDEX = METADATA_DIR / "p3_eval_index.csv"

CORE_FILES = [
    ("scripts/p3_config.py", "scripts/p3_config.py"),
    ("scripts/p3_degrade.py", "scripts/p3_degrade.py"),
    ("scripts/p3_dataloader.py", "scripts/p3_dataloader.py"),
    ("scripts/p3_evaluate.py", "scripts/p3_evaluate.py"),
    ("scripts/dashmamba_wrapper.py", "scripts/dashmamba_wrapper.py"),
    ("scripts/make_qualitative.py", "scripts/make_qualitative.py"),
    ("scripts/train_dashmamba.py", "scripts/train_dashmamba.py"),
    ("scripts/pretrain_dataloader.py", "scripts/pretrain_dataloader.py"),
    ("models/dashmamba.py", "models/dashmamba.py"),
    ("runs/pretrain_stage1/last.pt", "runs/pretrain_stage1/last.pt"),
    ("runs/finetune_stage2/last.pt", "runs/finetune_stage2/last.pt"),
]

BASELINE_RESULTS = ["rvrt_track_a_test", "bvrpp_track_a_test", "fastdvdnet_track_a_test"]


def eval_clip_stems(stride: int) -> list[str]:
    """Exactly the clips p3_evaluate.py --stride N would touch, across all axes."""
    with EVAL_INDEX.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    axes: dict[tuple, list[str]] = {}
    for r in rows:
        if r["track"] != "A" or r["split"] != "test":
            continue
        axes.setdefault((r["kind"], r["level"]), []).append(r["clip_stem"])
    stems: set[str] = set()
    for items in axes.values():
        stems.update(items[::stride])
    return sorted(stems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--skip-data", action="store_true",
                    help="skip the frame subset (target PC already has the dataset)")
    args = ap.parse_args()

    out = Path(args.out)
    core = out / "dashmamba_eval_core"
    core.mkdir(parents=True, exist_ok=True)

    # ---- code, checkpoints -------------------------------------------------
    print("== core files ==")
    missing = []
    for src_rel, dst_rel in CORE_FILES:
        src, dst = P3_ROOT / src_rel, core / dst_rel
        if not src.exists():
            missing.append(src_rel)
            print(f"  MISSING  {src_rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {src_rel}  ({src.stat().st_size/1e6:.1f} MB)")

    # ---- metadata ----------------------------------------------------------
    (core / "metadata").mkdir(exist_ok=True)
    for f in METADATA_DIR.glob("*.csv"):
        shutil.copy2(f, core / "metadata" / f.name)
    print(f"  metadata/*.csv ({len(list(METADATA_DIR.glob('*.csv')))} files)")

    # ---- baseline results (for the comparison table) -----------------------
    for name in BASELINE_RESULTS:
        src = P3_ROOT / "results" / name
        if src.exists():
            shutil.copytree(src, core / "results" / name, dirs_exist_ok=True)
            print(f"  results/{name}")

    # ---- docs --------------------------------------------------------------
    docs = P3_ROOT / "docs"
    if docs.exists():
        shutil.copytree(docs, core / "docs", dirs_exist_ok=True)
        print(f"  docs/ ({len(list(docs.glob('*')))} files)")

    # ---- frame subset ------------------------------------------------------
    stems = eval_clip_stems(args.stride)
    print(f"\n== data subset: {len(stems)} clips x {args.max_frames} frames ==")
    if args.skip_data:
        print("  skipped (--skip-data)")
    else:
        data = out / "dashmamba_eval_data" / "dataset" / "_refpool"
        data.mkdir(parents=True, exist_ok=True)
        total = 0
        for i, stem in enumerate(stems, 1):
            src_dir = DATASET_DIR / "_refpool" / stem
            dst_dir = data / stem
            dst_dir.mkdir(exist_ok=True)
            frames = sorted(src_dir.glob("frame_*.png"))[:args.max_frames]
            for fp in frames:
                shutil.copy2(fp, dst_dir / fp.name)
                total += fp.stat().st_size
            if i % 10 == 0 or i == len(stems):
                print(f"  {i}/{len(stems)} clips  ({total/1e9:.2f} GB)")

    write_readme(core, stems, args)

    core_size = sum(f.stat().st_size for f in core.rglob("*") if f.is_file())
    print(f"\ncore package: {core_size/1e6:.0f} MB at {core}")
    if missing:
        print(f"WARNING - missing files: {missing}")


def write_readme(core: Path, stems, args):
    txt = f"""# DashMamba evaluation package

Everything needed to run the DashMamba evaluation on another machine and
produce the 4-way comparison against RVRT, BasicVSR++ and FastDVDnet.

## What is already done (do NOT redo)

- The three BASELINES are already evaluated. Their per-clip numbers are in
  `results/rvrt_track_a_test/`, `results/bvrpp_track_a_test/`,
  `results/fastdvdnet_track_a_test/`. Do not re-run them.
- DashMamba is already TRAINED. Two checkpoints ship with this package:
    runs/pretrain_stage1/last.pt   Stage 1, DAVIS pretrain, 28,800 steps
    runs/finetune_stage2/last.pt   Stage 2, dashcam fine-tune, 11,000 steps

## What still needs running

1. Evaluate Stage-2 DashMamba  -> the headline 4-way comparison  (~5 h GPU)
2. Evaluate Stage-1 DashMamba  -> the FAIR comparison, because the three
   baselines are pretrained-only/zero-shot and Stage-1 DashMamba is likewise
   public-data-only  (~5 h GPU)
3. Qualitative before/after images and clips  (~30 min)

## Setup on the new PC

Requires: Python 3.10+, an NVIDIA GPU, PyTorch with CUDA.

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install opencv-python numpy requests

No compiler or CUDA toolkit is needed - DashMamba is pure PyTorch, with no
custom CUDA extensions. (The baselines needed those; they are already run.)

Then point the code at wherever you unpacked this:

    Windows:  set THESIS_P3_ROOT=C:\\path\\to\\dashmamba_eval_core
    Linux:    export THESIS_P3_ROOT=/path/to/dashmamba_eval_core

If you transferred `dashmamba_eval_data/`, merge its `dataset/` folder into
the core folder so that `<THESIS_P3_ROOT>/dataset/_refpool/<clip>/frame_*.png`
exists. If the target PC already holds the full thesis_p3 dataset, just point
THESIS_P3_ROOT at that instead.

## Commands

From the `scripts/` folder:

    # 1. headline result (Stage-2, fine-tuned)
    python -u p3_evaluate.py --model-import dashmamba_wrapper:restore \\
        --tracks A --splits test --stride {args.stride} --max-frames {args.max_frames} \\
        --run-name dashmamba_stage2_track_a_test

    # 2. fair comparison (Stage-1, public data only)
    #    Windows: set THESIS_P3_DASHMAMBA_CKPT=...\\runs\\pretrain_stage1\\last.pt
    python -u p3_evaluate.py --model-import dashmamba_wrapper:restore \\
        --tracks A --splits test --stride {args.stride} --max-frames {args.max_frames} \\
        --run-name dashmamba_stage1_track_a_test

    # 3. before/after visuals (needs the baseline model code too - optional)
    python make_qualitative.py --clips raw_video_008_clip_0000 raw_video_022_clip_0001 \\
        --kind gaussian --level high --frames 40 --models dashmamba

DO NOT change --stride or --max-frames. The baselines were evaluated with
exactly these settings ({len(stems)} clips, {args.max_frames} frames, 4 degradation axes);
changing them breaks the comparison.

p3_evaluate.py writes per_clip.csv incrementally, so an interrupted run is
resumed simply by re-issuing the same command.

## Reading the results

`results/<run-name>/summary.csv` has one row per (kind, level) axis, plus a
per-lighting breakdown below a blank line. Compare `psnr`/`ssim`/`tof`
against the baseline folders. `psnr_in` is the degraded input's own score -
the "before" number.

Baseline reference (71 clips, 80 frames, Track A test, pretrained-only):

    axis                        input     RVRT   BasicVSR++  FastDVDnet
    gaussian/low               25.14    39.13      26.48       38.75
    gaussian/medium            20.96    35.53      22.42       35.31
    gaussian/high              15.50    28.79      16.54       28.64
    poisson_gaussian/realistic 23.67    35.54      24.98       35.37

## Important context

- DashMamba is BLIND: it never receives the true noise level. RVRT and
  FastDVDnet were both GIVEN the true sigma. That asymmetry favours the
  baselines and should be stated in the paper.
- An earlier training run was discarded: it mixed low_light into the training
  degradations, and the model learned only brightness restoration while
  ignoring denoising entirely (+0.01 dB on gaussian). Training now uses only
  gaussian + poisson_gaussian, matching what Track A evaluates. Do not
  re-add low_light to this model's training mix - train a separate checkpoint
  for Track B if low-light is wanted.
- See `docs/` for the full architecture, weakness analysis, dataset scope and
  session handoff documents.
"""
    (core / "README_START_HERE.md").write_text(txt, encoding="utf-8")
    print("  README_START_HERE.md")


if __name__ == "__main__":
    main()
