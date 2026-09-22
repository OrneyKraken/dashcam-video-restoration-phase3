"""Stage-1 pretraining dataset: DAVIS 2017 (JPEGImages/480p/<video>/*.jpg).

Same random-window / random-crop / random-degradation recipe as
P3TrainSequences (scripts/p3_dataloader.py) so Stage-1 and Stage-2 feed the
model identically-shaped batches - only the source frames differ.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import p3_degrade

cv2.setNumThreads(1)


def _read_rgb01(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _to_tensor(seq_hwc: list[np.ndarray]) -> torch.Tensor:
    arr = np.stack(seq_hwc, 0).transpose(0, 3, 1, 2)  # T,C,H,W
    return torch.from_numpy(np.ascontiguousarray(arr))


class DavisTrainSequences(Dataset):
    def __init__(self, root: str | Path, seq_len: int = 16, crop: int = 256,
                kinds=("gaussian", "poisson_gaussian", "low_light"),
                augment: bool = True, seed: int = 0):
        root = Path(root)
        img_root = root / "JPEGImages" / "480p"
        if not img_root.exists():
            raise RuntimeError(f"DAVIS not found at {img_root} - has it been downloaded/extracted?")

        self.videos = []
        for vdir in sorted(img_root.iterdir()):
            if not vdir.is_dir():
                continue
            frames = sorted(vdir.glob("*.jpg"))
            if len(frames) >= seq_len:
                self.videos.append(frames)
        if not self.videos:
            raise RuntimeError(f"no DAVIS videos with >= {seq_len} frames under {img_root}")

        self.seq_len, self.crop, self.kinds = seq_len, crop, kinds
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frames = self.videos[idx]
        start = self._rng.randint(0, len(frames) - self.seq_len)
        window = frames[start:start + self.seq_len]

        clean = [_read_rgb01(p) for p in window]
        H, W, _ = clean[0].shape
        if self.crop and (H > self.crop and W > self.crop):
            top = self._rng.randint(0, H - self.crop)
            left = self._rng.randint(0, W - self.crop)
            clean = [c[top:top + self.crop, left:left + self.crop] for c in clean]

        drng = np.random.default_rng(self._rng.randrange(2**32))
        kind = self.kinds[drng.integers(len(self.kinds))]
        if kind == "gaussian":
            lo, hi = p3_degrade.GAUSSIAN_BLIND_RANGE_255
            sigma = float(drng.uniform(lo, hi))
            noisy = [p3_degrade.gaussian((c * 255).astype(np.uint8),
                     np.random.default_rng(drng.integers(2**32)), sigma).astype(np.float32) / 255.0
                     for c in clean]
        elif kind == "poisson_gaussian":
            noisy = [p3_degrade.poisson_gaussian((c * 255).astype(np.uint8),
                     np.random.default_rng(drng.integers(2**32))).astype(np.float32) / 255.0
                     for c in clean]
        else:
            level = p3_degrade.LEVELS[drng.integers(len(p3_degrade.LEVELS))]
            noisy = [p3_degrade.low_light((c * 255).astype(np.uint8),
                     np.random.default_rng(drng.integers(2**32)), level).astype(np.float32) / 255.0
                     for c in clean]

        if self.augment:
            if self._rng.random() < 0.5:
                clean = [np.ascontiguousarray(c[:, ::-1]) for c in clean]
                noisy = [np.ascontiguousarray(n[:, ::-1]) for n in noisy]
            if self._rng.random() < 0.5:
                clean = [np.ascontiguousarray(c[::-1]) for c in clean]
                noisy = [np.ascontiguousarray(n[::-1]) for n in noisy]

        return {"lq": _to_tensor(noisy), "gt": _to_tensor(clean), "kind": kind}


if __name__ == "__main__":
    import sys
    ds = DavisTrainSequences(sys.argv[1] if len(sys.argv) > 1 else
                             "F:/user4/thesis_p3/external/pretrain_data/DAVIS",
                             seq_len=8, crop=128)
    s = ds[0]
    print(f"ok: {len(ds)} videos, sample lq {tuple(s['lq'].shape)} kind={s['kind']}")
