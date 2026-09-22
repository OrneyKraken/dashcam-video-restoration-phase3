"""restore(lq, meta) -> sr adapter so p3_evaluate.py can score FastDVDnet.

Like RVRT's checkpoint, FastDVDnet is non-blind (takes the true noise sigma
as an extra input, not baked into the frames) - see fastdvdnet_wrapper's
sibling rvrt_wrapper.py for why we give it the true value rather than
guessing blind. Pure PyTorch, no compiled ops, so this is the simplest of
the three wrappers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FASTDVDNET_DIR = Path(__file__).resolve().parent.parent / "external" / "fastdvdnet"
if str(FASTDVDNET_DIR) not in sys.path:
    sys.path.insert(0, str(FASTDVDNET_DIR))

CKPT_PATH = FASTDVDNET_DIR / "model.pth"   # trained for AWGN, sigma in [5, 55]

_model = None
_device = None


def _load():
    global _model, _device
    if _model is not None:
        return
    import torch
    from models import FastDVDnet

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastDVDnet(num_input_frames=5)
    state = torch.load(CKPT_PATH, map_location="cpu")
    # checkpoint was saved from nn.DataParallel - keys are prefixed "module."
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval().to(_device)
    _model = model


def _sigma255(meta: dict, lq: np.ndarray) -> float:
    kind = meta.get("kind")
    if kind == "gaussian":
        from p3_degrade import GAUSSIAN_SIGMA_255
        return GAUSSIAN_SIGMA_255[meta["level"]]
    if kind == "poisson_gaussian":
        from p3_degrade import POISSON_GAUSSIAN
        a, b = POISSON_GAUSSIAN["a"], POISSON_GAUSSIAN["b"]
        mean_x = float(np.mean(lq))
        return float(np.sqrt(mean_x / a + b * b) * 255.0)
    return 25.0


def restore(lq: np.ndarray, meta: dict) -> np.ndarray:
    """lq: (T,H,W,3) float32 RGB [0,1]. Returns sr in the same format."""
    import torch
    from fastdvdnet import denoise_seq_fastdvdnet

    _load()
    sigma = _sigma255(meta, lq) / 255.0

    seq = torch.from_numpy(lq).permute(0, 3, 1, 2).float().to(_device)  # (T,3,H,W)
    noise_std = torch.tensor([sigma], device=_device, dtype=torch.float32)

    with torch.no_grad():
        out = denoise_seq_fastdvdnet(seq=seq, noise_std=noise_std,
                                     temp_psz=5, model_temporal=_model)

    sr = out.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    return sr
