"""restore(lq) -> sr adapter so p3_evaluate.py can score BasicVSR++.

Uses the 'ntire-decompress-track1' checkpoint (compressed-video quality
enhancement) with is_low_res_input=False - i.e. the same-resolution
restoration configuration of BasicVSR++, not its 4x super-resolution
configuration. This is the correct way to run BasicVSR++ as a general video
restorer (matching how Phase 2 generalization-tested it): no upscale/
downscale hack, native weights, native resolution.

BasicVSR++ is blind (no noise-level input), so unlike rvrt_wrapper this only
needs restore(lq) - no meta.

Only imports mmagic's specific submodules directly (not `import mmagic`
wholesale) - see editors/__init__.py and archs/__init__.py in the vendored
mmagic checkout, which were trimmed to drop hard dependencies on diffusers/
transformers/mediapipe that this project's BasicVSR++ evaluation never uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MMAGIC_DIR = Path(__file__).resolve().parent.parent / "external" / "mmagic"
if str(MMAGIC_DIR) not in sys.path:
    sys.path.insert(0, str(MMAGIC_DIR))

CKPT_URL = ("https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
           "basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth")
CKPT_PATH = MMAGIC_DIR / "checkpoints" / "basicvsr_plusplus_c128n25_ntire_decompress_track1.pth"

_model = None
_device = None
TILE = 320       # spatial tile size (multiple of 4, required by the two stride-2 convs)
TILE_OVERLAP = 32


def _load():
    global _model, _device
    if _model is not None:
        return
    import torch
    from mmagic.models.editors.basicvsr_plusplus_net.basicvsr_plusplus_net import BasicVSRPlusPlusNet

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicVSRPlusPlusNet(mid_channels=128, num_blocks=25,
                                is_low_res_input=False, cpu_cache_length=100,
                                spynet_pretrained=None)

    if not CKPT_PATH.exists():
        import requests
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CKPT_PATH.write_bytes(requests.get(CKPT_URL, allow_redirects=True).content)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # mmagic checkpoints prefix the generator's own weights with "generator."
    state = {k[len("generator."):]: v for k, v in state.items() if k.startswith("generator.")}
    model.load_state_dict(state, strict=True)
    model.eval().to(_device)
    _model = model


def _tiled_forward(x, model):
    """Spatial-tile the whole clip (all frames at once per tile - the model's
    own cpu_cache_length handles the temporal/memory tradeoff internally,
    same as BasicVSR++'s own recurrent design intends)."""
    import torch
    n, t, c, h, w = x.shape
    if h <= TILE and w <= TILE:
        with torch.no_grad():
            return model(x)

    stride = TILE - TILE_OVERLAP
    h_idx_list = list(range(0, h - TILE, stride)) + [max(0, h - TILE)]
    w_idx_list = list(range(0, w - TILE, stride)) + [max(0, w - TILE)]
    E = torch.zeros(n, t, c, h, w)
    W = torch.zeros_like(E)
    half = TILE_OVERLAP // 2
    with torch.no_grad():
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                patch = x[..., h_idx:h_idx + TILE, w_idx:w_idx + TILE]
                out_patch = model(patch).detach().cpu()
                mask = torch.ones_like(out_patch)
                if h_idx < h_idx_list[-1]:
                    out_patch[..., -half:, :] *= 0
                    mask[..., -half:, :] *= 0
                if w_idx < w_idx_list[-1]:
                    out_patch[..., :, -half:] *= 0
                    mask[..., :, -half:] *= 0
                if h_idx > h_idx_list[0]:
                    out_patch[..., :half, :] *= 0
                    mask[..., :half, :] *= 0
                if w_idx > w_idx_list[0]:
                    out_patch[..., :, :half] *= 0
                    mask[..., :, :half] *= 0
                E[..., h_idx:h_idx + TILE, w_idx:w_idx + TILE] += out_patch
                W[..., h_idx:h_idx + TILE, w_idx:w_idx + TILE] += mask
    return E / W


def restore(lq: np.ndarray) -> np.ndarray:
    """lq: (T,H,W,3) float32 RGB [0,1]. Returns sr in the same format."""
    import torch
    _load()

    x = torch.from_numpy(lq).permute(0, 3, 1, 2).unsqueeze(0).float().to(_device)  # (1,T,3,H,W)
    out = _tiled_forward(x, _model)
    sr = out.squeeze(0).clamp(0, 1).permute(0, 2, 3, 1).numpy().astype(np.float32)
    return sr
