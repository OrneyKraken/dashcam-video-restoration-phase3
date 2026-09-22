"""Shared configuration for the Thesis Phase 3 (thesis_p3) dataset build.

This is a fresh, self-contained pipeline. It only READS from the existing
thesis_dataset folder (raw curation manifest + clips) and writes everything
new under /media/ishrak/THESIS_DATA/thesis_p3, so the Phase 2 work is untouched.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Override with env vars THESIS_P3_ROOT / THESIS_P2_ROOT on machines where the
# data lives somewhere else (Kaggle, another PC). Defaults match this repo's
# original Linux box; the training machine (Windows, RTX 4080) sets THESIS_P3_ROOT.
P3_ROOT = Path(os.environ.get("THESIS_P3_ROOT", "/media/ishrak/THESIS_DATA/thesis_p3"))
OLD_ROOT = Path(os.environ.get("THESIS_P2_ROOT", "/media/ishrak/THESIS_DATA/thesis_dataset"))

CLIPS_MANIFEST = OLD_ROOT / "logs" / "clips_manifest.csv"   # existing curation (1425 kept)
CLIPS_DIR = OLD_ROOT / "clips"                              # existing 20s clips (mp4)

METADATA_DIR = P3_ROOT / "metadata"
LOGS_DIR = P3_ROOT / "logs"
DATASET_DIR = P3_ROOT / "dataset"
REFPOOL_DIR = DATASET_DIR / "_refpool"                      # every reference frame, extracted once

CLIP_SPLITS_CSV = METADATA_DIR / "p3_clip_splits.csv"
SPLIT_SUMMARY_CSV = METADATA_DIR / "p3_split_summary.csv"
FRAME_MANIFEST_CSV = METADATA_DIR / "p3_frame_manifest.csv"
DEGRADATION_MANIFEST_CSV = METADATA_DIR / "p3_degradation_manifest.csv"
FRAME_PAIRS_CSV = METADATA_DIR / "p3_frame_pairs.csv"
DATASET_SUMMARY_CSV = METADATA_DIR / "p3_dataset_summary.csv"

# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
# Source clips are 1920x1080 @ ~30 fps. Phase 3 stores 1280x720 (same 16:9).
FPS = 15                 # native motion preserved (~67 ms between frames); Phase 2's 3 fps was too sparse
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
PNG_COMPRESSION = 3
DONE_MARKER = "_done.txt"

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_NAMES = ["train", "val", "test"]
SPLIT_SEED = 20260904

# ---------------------------------------------------------------------------
# Track selection caps (0 or None = take everything eligible)
# ---------------------------------------------------------------------------
TRACK_A_MAX_CLIPS = 600   # full benchmark pool: all lighting, kept clips, all splits
TRACK_B_MAX_CLIPS = 0     # held-out low-light eval: val+test day/good/clear+shadow (take all)
TRACK_C_MAX_CLIPS = 0     # held-out real-world eval: test evening+night clips (take all)

# ---------------------------------------------------------------------------
# Degradation parameters
# ---------------------------------------------------------------------------
# Track B: synthetic low-light sensor degradation (identical to Phase 2 script 06/08
# so the two phases stay comparable).
DAY_TO_NIGHT_PARAMS = {
    "low": {
        "exposure": 0.65, "gamma": 1.15, "photon_peak": 120.0,
        "read_sigma": 0.012, "bgr_gains": (1.02, 1.00, 0.97),
    },
    "medium": {
        "exposure": 0.42, "gamma": 1.35, "photon_peak": 65.0,
        "read_sigma": 0.022, "bgr_gains": (1.05, 1.00, 0.94),
    },
    "high": {
        "exposure": 0.28, "gamma": 1.55, "photon_peak": 35.0,
        "read_sigma": 0.035, "bgr_gains": (1.08, 1.00, 0.90),
    },
}
LOWLIGHT_LEVELS = ["low", "medium", "high"]

# Track A test set: pre-rendered fixed degradations for reproducible scoring.
# Training/val for Track A store references only; Gaussian/Poisson noise is added
# on the fly in the dataloader (standard practice for BasicVSR++/RVRT/FastDVDnet).
GAUSSIAN_SIGMA_255 = {"low": 15.0, "medium": 25.0, "high": 50.0}

# Heteroscedastic (signal-dependent) Poisson-Gaussian model for the "realistic"
# Track A test level:  y = Poisson(x * a) / a + N(0, b)
POISSON_GAUSSIAN = {"a": 60.0, "b": 0.018}

TRACK_A_TEST_LEVELS = ["low", "medium", "high", "realistic"]
