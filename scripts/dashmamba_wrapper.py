"""restore(lq) -> sr adapter so p3_evaluate.py can score DashMamba.

NOTE the single-argument signature. Unlike rvrt_wrapper and fastdvdnet_wrapper
(which take `meta` and are handed the TRUE noise sigma, because their public
checkpoints are non-blind), DashMamba is fully BLIND: its Signal B reliability
estimator infers noise/exposure from the raw frame itself and never sees a
ground-truth sigma. This is a deliberate, reportable asymmetry IN THE
BASELINES' FAVOUR - if DashMamba matches them, it does so with strictly less
information. Say so explicitly in the paper rather than burying it.

Checkpoint is taken from THESIS_P3_DASHMAMBA_CKPT if set, else the Stage-2
fine-tune run's last.pt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

P3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P3_ROOT / "models"))

# Look in both layouts: the training machine writes runs/<name>/last.pt, while
# the git repo / portable package ships checkpoints/<name>.pt. Checking both
# means the same wrapper works unmodified in either place.
CKPT_CANDIDATES = [
    P3_ROOT / "runs" / "finetune_stage2" / "last.pt",
    P3_ROOT / "checkpoints" / "finetune_stage2.pt",
]


def _default_ckpt() -> Path:
    for c in CKPT_CANDIDATES:
        if c.exists():
            return c
    return CKPT_CANDIDATES[0]  # report the canonical path in the error message


CKPT_PATH = Path(os.environ["THESIS_P3_DASHMAMBA_CKPT"]) if os.environ.get(
    "THESIS_P3_DASHMAMBA_CKPT") else _default_ckpt()

# Model config must match how the checkpoint was TRAINED. Overridable by env
# var so an ablation-cell checkpoint (trained with different flags) can be
# scored through this same wrapper without editing code.
MID_CHANNELS = int(os.environ.get("DASHMAMBA_MID_CHANNELS", 64))
NUM_RES_BLOCKS = int(os.environ.get("DASHMAMBA_NUM_RES_BLOCKS", 5))
STATE_DIM = int(os.environ.get("DASHMAMBA_STATE_DIM", 16))
NUM_MAMBA_BLOCKS = int(os.environ.get("DASHMAMBA_NUM_MAMBA_BLOCKS", 2))
USE_FLOW_ALIGN = os.environ.get("DASHMAMBA_USE_FLOW_ALIGN", "1") == "1"
USE_MOTION_GATE = os.environ.get("DASHMAMBA_USE_MOTION_GATE", "1") == "1"
USE_RELIABILITY_DELTA = os.environ.get("DASHMAMBA_USE_RELIABILITY_DELTA", "1") == "1"

# Spatial tiling: the selective scan folds every spatial position into its
# batch dimension, so a full 1280x720 frame at once is far heavier than the
# 192x192 crops used in training. Tile spatially (same approach as
# basicvsrpp_wrapper), keeping the FULL temporal extent inside each tile so
# the temporal scan still sees the whole clip.
TILE = 256
TILE_OVERLAP = 32
# Temporal chunking: cap frames processed in one pass; long clips are split
# into overlapping chunks and blended, so memory stays bounded regardless of
# clip length.
MAX_FRAMES_PER_CHUNK = int(os.environ.get("DASHMAMBA_MAX_FRAMES", 40))
FRAME_OVERLAP = 4

_model = None
_device = None


def _load():
    global _model, _device
    if _model is not None:
        return
    import torch
    from dashmamba import DashMambaNet

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DashMambaNet(mid_channels=MID_CHANNELS, num_res_blocks=NUM_RES_BLOCKS,
                         state_dim=STATE_DIM, num_mamba_blocks=NUM_MAMBA_BLOCKS,
                         use_flow_align=USE_FLOW_ALIGN, use_motion_gate=USE_MOTION_GATE,
                         use_reliability_delta=USE_RELIABILITY_DELTA)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"no DashMamba checkpoint at {CKPT_PATH} - has Stage-2 fine-tuning finished? "
            f"(set THESIS_P3_DASHMAMBA_CKPT to point elsewhere)")
    ck = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval().to(_device)
    _model = model
    print(f"[dashmamba_wrapper] loaded {CKPT_PATH} (trained to step {ck.get('step', '?')})")


def _forward_tiled(clip):
    """clip: (1,T,3,H,W) tensor on device -> (1,T,3,H,W) on CPU."""
    import torch
    b, t, c, h, w = clip.shape
    if h <= TILE and w <= TILE:
        with torch.no_grad():
            return _model(clip).cpu()

    stride = TILE - TILE_OVERLAP
    h_idx = list(range(0, max(h - TILE, 0), stride)) + [max(0, h - TILE)]
    w_idx = list(range(0, max(w - TILE, 0), stride)) + [max(0, w - TILE)]
    E = torch.zeros(b, t, c, h, w)
    W = torch.zeros_like(E)
    half = TILE_OVERLAP // 2

    with torch.no_grad():
        for hi in h_idx:
            for wi in w_idx:
                patch = clip[..., hi:hi + TILE, wi:wi + TILE]
                out_patch = _model(patch).cpu()
                mask = torch.ones_like(out_patch)
                if hi < h_idx[-1]:
                    out_patch[..., -half:, :] *= 0
                    mask[..., -half:, :] *= 0
                if wi < w_idx[-1]:
                    out_patch[..., :, -half:] *= 0
                    mask[..., :, -half:] *= 0
                if hi > h_idx[0]:
                    out_patch[..., :half, :] *= 0
                    mask[..., :half, :] *= 0
                if wi > w_idx[0]:
                    out_patch[..., :, :half] *= 0
                    mask[..., :, :half] *= 0
                E[..., hi:hi + TILE, wi:wi + TILE] += out_patch
                W[..., hi:hi + TILE, wi:wi + TILE] += mask
    return E / W.clamp(min=1e-8)


def restore(lq: np.ndarray) -> np.ndarray:
    """lq: (T,H,W,3) float32 RGB [0,1]. Returns sr in the same format.
    No noise level is taken - DashMamba estimates its own (Signal B)."""
    import torch
    _load()

    t_total = lq.shape[0]
    x_all = torch.from_numpy(lq).permute(0, 3, 1, 2).unsqueeze(0).float()  # (1,T,3,H,W)

    if t_total <= MAX_FRAMES_PER_CHUNK:
        out = _forward_tiled(x_all.to(_device))
    else:
        # overlapping temporal chunks, averaged in the overlap region
        stride = MAX_FRAMES_PER_CHUNK - FRAME_OVERLAP
        starts = list(range(0, max(t_total - MAX_FRAMES_PER_CHUNK, 0), stride))
        starts += [max(0, t_total - MAX_FRAMES_PER_CHUNK)]
        acc = torch.zeros_like(x_all)
        cnt = torch.zeros(1, t_total, 1, 1, 1)
        for s in starts:
            e = min(s + MAX_FRAMES_PER_CHUNK, t_total)
            chunk_out = _forward_tiled(x_all[:, s:e].to(_device))
            acc[:, s:e] += chunk_out
            cnt[:, s:e] += 1
        out = acc / cnt.clamp(min=1)

    sr = out.squeeze(0).clamp(0, 1).permute(0, 2, 3, 1).numpy().astype(np.float32)
    return sr
