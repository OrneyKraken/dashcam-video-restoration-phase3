"""Deterministic degradation models for the Thesis Phase 3 benchmark.

This is the SINGLE source of truth for how a clean frame is turned into a
degraded input. The training dataloader and the evaluation script must both
import from here so that every reported number is bit-exact reproducible.

A degraded frame is fully determined by:
    (clean frame, kind, level, seed)
where `seed` is derived from (clip_stem, frame_name, kind, level) via `frame_seed`.

Kinds
  gaussian          additive white Gaussian noise, sigma in 8-bit units
  poisson_gaussian  signal-dependent shot noise + read noise ("realistic" sensor)
  low_light         Phase-2 synthetic low-light sensor pipeline (exposure drop,
                    gamma, colour-gain, Poisson shot, Gaussian read) - identical
                    parameters to thesis_dataset/scripts/06 & 08 so Phase 2 and
                    Phase 3 stay directly comparable.
"""

from __future__ import annotations

import hashlib

import numpy as np

# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
GAUSSIAN_SIGMA_255 = {"low": 15.0, "medium": 25.0, "high": 50.0}
GAUSSIAN_BLIND_RANGE_255 = (0.0, 50.0)        # used for training (random sigma)

POISSON_GAUSSIAN = {"a": 60.0, "b": 0.018}    # y = Poisson(x*a)/a + N(0,b)

DAY_TO_NIGHT_PARAMS = {
    "low":    {"exposure": 0.65, "gamma": 1.15, "photon_peak": 120.0,
               "read_sigma": 0.012, "bgr_gains": (1.02, 1.00, 0.97)},
    "medium": {"exposure": 0.42, "gamma": 1.35, "photon_peak": 65.0,
               "read_sigma": 0.022, "bgr_gains": (1.05, 1.00, 0.94)},
    "high":   {"exposure": 0.28, "gamma": 1.55, "photon_peak": 35.0,
               "read_sigma": 0.035, "bgr_gains": (1.08, 1.00, 0.90)},
}

LEVELS = ("low", "medium", "high")
KINDS = ("gaussian", "poisson_gaussian", "low_light")


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def frame_seed(*parts) -> int:
    """Stable 32-bit seed from any string parts (order matters)."""
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(),
                          "little") % (2**32)


def rng_for(clip_stem: str, frame_name: str, kind: str, level: str) -> np.random.Generator:
    return np.random.default_rng(frame_seed(kind, level, clip_stem, frame_name))


# --------------------------------------------------------------------------- #
# models  (input/output: uint8 BGR HxWx3, same convention as cv2.imread)
# --------------------------------------------------------------------------- #
def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def gaussian(img: np.ndarray, rng: np.random.Generator, sigma255: float) -> np.ndarray:
    x = _to_float(img)
    return _to_uint8(x + rng.normal(0.0, sigma255 / 255.0, x.shape).astype(np.float32))


def poisson_gaussian(img: np.ndarray, rng: np.random.Generator,
                     a: float = POISSON_GAUSSIAN["a"],
                     b: float = POISSON_GAUSSIAN["b"]) -> np.ndarray:
    x = _to_float(img)
    shot = rng.poisson(np.clip(x * a, 0.0, None)).astype(np.float32) / a
    return _to_uint8(shot + rng.normal(0.0, b, x.shape).astype(np.float32))


def low_light(img: np.ndarray, rng: np.random.Generator, level: str) -> np.ndarray:
    p = DAY_TO_NIGHT_PARAMS[level]
    x = _to_float(img)
    x = np.clip(x * p["exposure"], 0.0, 1.0)
    x = np.power(x, p["gamma"])
    x = np.clip(x * np.array(p["bgr_gains"], np.float32).reshape(1, 1, 3), 0.0, 1.0)
    peak = p["photon_peak"]
    shot = rng.poisson(np.clip(x * peak, 0.0, peak)).astype(np.float32) / peak
    read = rng.normal(0.0, p["read_sigma"], x.shape).astype(np.float32)
    return _to_uint8(shot + read)


# --------------------------------------------------------------------------- #
# unified entry points
# --------------------------------------------------------------------------- #
def degrade(img: np.ndarray, kind: str, level: str,
            clip_stem: str, frame_name: str) -> np.ndarray:
    """Deterministic degradation for a NAMED frame (used by the eval script and
    the pre-render preview builder)."""
    rng = rng_for(clip_stem, frame_name, kind, level)
    if kind == "gaussian":
        return gaussian(img, rng, GAUSSIAN_SIGMA_255[level])
    if kind == "poisson_gaussian":
        return poisson_gaussian(img, rng)
    if kind == "low_light":
        return low_light(img, rng, level)
    raise ValueError(f"unknown kind: {kind}")


def degrade_random(img: np.ndarray, rng: np.random.Generator,
                   kinds=("gaussian", "poisson_gaussian", "low_light")) -> tuple[np.ndarray, dict]:
    """Random degradation for TRAINING. Returns (noisy_img, info dict).
    Gaussian uses a blind sigma range; low_light/poisson_gaussian pick a level."""
    kind = kinds[rng.integers(len(kinds))]
    if kind == "gaussian":
        lo, hi = GAUSSIAN_BLIND_RANGE_255
        sigma = float(rng.uniform(lo, hi))
        return gaussian(img, rng, sigma), {"kind": kind, "sigma255": round(sigma, 2)}
    if kind == "poisson_gaussian":
        return poisson_gaussian(img, rng), {"kind": kind}
    level = LEVELS[rng.integers(len(LEVELS))]
    return low_light(img, rng, level), {"kind": kind, "level": level}
