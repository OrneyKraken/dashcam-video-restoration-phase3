"""Reference PyTorch dataset for Phase 3 training / evaluation.

Reads CLEAN frames from thesis_p3/dataset/_refpool/<clip_stem>/ and applies
degradations on the fly via p3_degrade.py. Split / track membership comes from
thesis_p3/metadata/p3_clip_splits.csv (training) and p3_eval_index.csv (eval).

Nothing is pre-noised on disk; a given (clip, frame, kind, level) always yields
the same noisy frame because p3_degrade seeds from those names.

Typical use
-----------
train:
    ds = P3TrainSequences(split="train", seq_len=16, crop=256)
    #   -> dict(lq=[T,3,H,W] float 0..1, gt=[T,3,H,W], info=...)

eval (one axis):
    ds = P3EvalSequences(track="A", kind="gaussian", level="high", split="test")
    #   -> full clips, no crop; gt is None for track C
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:                      # allow import without torch for inspection
    Dataset = object
    torch = None

import p3_degrade
from p3_config import DATASET_DIR, METADATA_DIR

cv2.setNumThreads(1)

REFPOOL = DATASET_DIR / "_refpool"
CLIP_SPLITS = METADATA_DIR / "p3_clip_splits.csv"
EVAL_INDEX = METADATA_DIR / "p3_eval_index.csv"


# --------------------------------------------------------------------------- #
def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _frame_paths(clip_stem: str) -> list[Path]:
    return sorted((REFPOOL / clip_stem).glob("frame_*.png"))


def _read_rgb01(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _to_tensor(seq_hwc: list[np.ndarray]):
    arr = np.stack(seq_hwc, 0).transpose(0, 3, 1, 2)          # T,C,H,W
    return torch.from_numpy(np.ascontiguousarray(arr)) if torch else arr


# --------------------------------------------------------------------------- #
class P3TrainSequences(Dataset):
    """Random consecutive-frame windows with a random degradation per window."""

    def __init__(self, split="train", seq_len=16, crop=256,
                 tracks=("a",), kinds=("gaussian", "poisson_gaussian", "low_light"),
                 min_clip_frames=None, augment=True, seed=0):
        rows = _load_csv(CLIP_SPLITS)
        want = set(t.lower() for t in tracks)
        self.clips = []
        for r in rows:
            if r["split"] != split:
                continue
            if not (("a" in want and r["in_track_a"] == "1") or
                    ("b" in want and r["in_track_b"] == "1")):
                continue
            fp = _frame_paths(Path(r["clip_name"]).stem)
            need = min_clip_frames or seq_len
            if len(fp) >= need:
                self.clips.append((Path(r["clip_name"]).stem, fp, r))
        if not self.clips:
            raise RuntimeError(f"no usable clips for split={split} tracks={tracks}")
        self.seq_len, self.crop, self.kinds = seq_len, crop, kinds
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        stem, fp, meta = self.clips[idx]
        start = self._rng.randint(0, len(fp) - self.seq_len)
        window = fp[start:start + self.seq_len]

        # Crop on the raw uint8 BGR frames BEFORE the colour/float conversion.
        # Numerically identical to converting first (BGR2RGB is a per-pixel
        # channel permutation and /255 is elementwise, so both commute with a
        # spatial slice), but ~25x less conversion work: a 1280x720 float32
        # RGB frame is an 11MB array, of which a 192x192 crop keeps ~1.5%.
        # This was the dominant cost in Stage-2's dataloader.
        raw = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in window]
        if raw[0] is None:
            raise FileNotFoundError(window[0])
        H, W = raw[0].shape[:2]
        if self.crop:
            top = self._rng.randint(0, max(0, H - self.crop))
            left = self._rng.randint(0, max(0, W - self.crop))
            raw = [r[top:top + self.crop, left:left + self.crop] for r in raw]
        clean = [cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for r in raw]

        # one random degradation, same params for every frame in the window
        drng = np.random.default_rng(self._rng.randrange(2**32))
        kind = self.kinds[drng.integers(len(self.kinds))]
        if kind == "gaussian":
            lo, hi = p3_degrade.GAUSSIAN_BLIND_RANGE_255
            sigma = float(drng.uniform(lo, hi))
            noisy = [p3_degrade.gaussian((c * 255).astype(np.uint8),
                     np.random.default_rng(drng.integers(2**32)), sigma).astype(np.float32) / 255.0
                     for c in clean]
            info = {"kind": kind, "sigma255": round(sigma, 2)}
        elif kind == "poisson_gaussian":
            noisy = [p3_degrade.poisson_gaussian((c * 255).astype(np.uint8),
                     np.random.default_rng(drng.integers(2**32))).astype(np.float32) / 255.0
                     for c in clean]
            info = {"kind": kind}
        else:
            level = p3_degrade.LEVELS[drng.integers(len(p3_degrade.LEVELS))]
            noisy = [p3_degrade.low_light((c * 255).astype(np.uint8),
                     np.random.default_rng(drng.integers(2**32)), level).astype(np.float32) / 255.0
                     for c in clean]
            info = {"kind": kind, "level": level}

        if self.augment:
            if self._rng.random() < 0.5:
                clean = [np.ascontiguousarray(c[:, ::-1]) for c in clean]
                noisy = [np.ascontiguousarray(n[:, ::-1]) for n in noisy]
            if self._rng.random() < 0.5:
                clean = [np.ascontiguousarray(c[::-1]) for c in clean]
                noisy = [np.ascontiguousarray(n[::-1]) for n in noisy]

        return {"lq": _to_tensor(noisy), "gt": _to_tensor(clean),
                "clip": stem, "start": start, **info}


class P3EvalSequences(Dataset):
    """Whole-clip sequences for one evaluation axis (deterministic degradation)."""

    def __init__(self, track, kind, level, split="test"):
        rows = _load_csv(EVAL_INDEX)
        self.items = [r for r in rows
                      if r["track"] == track and r["kind"] == kind
                      and r["level"] == level and r["split"] == split]
        if not self.items:
            raise RuntimeError(f"no eval items for track={track} kind={kind} "
                               f"level={level} split={split} (has the build run?)")
        self.track, self.kind, self.level = track, kind, level

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        stem = it["clip_stem"]
        fp = _frame_paths(stem)
        clean = [_read_rgb01(p) for p in fp]
        names = [p.name for p in fp]
        if self.kind == "none":
            lq = clean
            gt = None
        else:
            lq = []
            for c, nm in zip(clean, names):
                bgr_u8 = cv2.cvtColor((c * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                d = p3_degrade.degrade(bgr_u8, self.kind, self.level, stem, nm)
                lq.append(cv2.cvtColor(d, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            gt = clean
        return {"lq": _to_tensor(lq),
                "gt": _to_tensor(gt) if gt is not None else None,
                "clip": stem, "lighting": it["lighting"], "condition": it["condition"],
                "track": self.track, "kind": self.kind, "level": self.level}


if __name__ == "__main__":
    # smoke test against whatever is already extracted
    try:
        ds = P3TrainSequences(split="train", seq_len=8, crop=128)
        s = ds[0]
        shp = tuple(s["lq"].shape) if torch else s["lq"].shape
        print(f"train ok: {len(ds)} clips, sample lq {shp}, {s.get('kind')}")
    except Exception as e:
        print("train dataset:", e)
