"""restore(lq, meta) -> sr adapter so p3_evaluate.py can score RVRT.

RVRT's public '006_RVRT_videodenoising_DAVIS_16frames' checkpoint is a
NON-BLIND denoiser: it takes the true per-pixel noise sigma as an extra
input channel (this is how its own paper evaluates it - see
RVRT/data/dataset_video_test.py, which injects noise itself from a known
sigma). Since we control our own degradation exactly (p3_degrade.py), we
give RVRT its true noise level for the 'gaussian' axis, and an analytically
equivalent local noise level for 'poisson_gaussian' (derived from the same
a/b sensor-noise parameters we used to create it) rather than guessing
blind - matching how a non-blind baseline is fairly evaluated.

Only imported lazily (model + CUDA extension load on first call) so that
importing this module for --model-import doesn't cost anything until an
axis that actually needs it runs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

RVRT_DIR = Path(__file__).resolve().parent.parent / "external" / "RVRT"
if str(RVRT_DIR) not in sys.path:
    sys.path.insert(0, str(RVRT_DIR))

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

_model = None
_device = None


def _load():
    global _model, _device
    if _model is not None:
        return
    import torch
    from models.network_rvrt import RVRT as net

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8],
               num_blocks=[1, 2, 1], depths=[2, 2, 2], embed_dims=[192, 192, 192],
               num_heads=[6, 6, 6], inputconv_groups=[1, 3, 4, 6, 8, 4],
               deformable_groups=12, attention_heads=12, attention_window=[3, 3],
               nonblind_denoising=True, cpu_cache_length=100)

    ckpt_path = RVRT_DIR / "model_zoo" / "rvrt" / "006_RVRT_videodenoising_DAVIS_16frames.pth"
    if not ckpt_path.exists():
        import requests
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/JingyunLiang/RVRT/releases/download/v0.0/{ckpt_path.name}"
        ckpt_path.write_bytes(requests.get(url, allow_redirects=True).content)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["params"] if "params" in ckpt else ckpt, strict=True)
    model.eval().to(_device)
    _model = model


# same tiled inference as RVRT/main_test_rvrt.py test_video()/test_clip(), factored
# out so we can call it directly on our own frames instead of through their dataset.
class _Args:
    scale = 1
    window_size = [2, 8, 8]
    nonblind_denoising = True
    tile = [0, 256, 256]        # [temporal, h, w]; 0 = whole clip at once (recurrent + cpu_cache_length handles memory)
    tile_overlap = [2, 20, 20]


def _test_clip(lq, model, args):
    import torch
    sf = args.scale
    window_size = args.window_size
    size_patch_testing = args.tile[1]
    assert size_patch_testing % window_size[-1] == 0

    if size_patch_testing:
        overlap_size = args.tile_overlap[1]
        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h - size_patch_testing, stride)) + [max(0, h - size_patch_testing)]
        w_idx_list = list(range(0, w - size_patch_testing, stride)) + [max(0, w - size_patch_testing)]
        E = torch.zeros(b, d, c, h * sf, w * sf)
        W = torch.zeros_like(E)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]
                out_patch = model(in_patch).detach().cpu()
                out_patch_mask = torch.ones_like(out_patch)
                if h_idx < h_idx_list[-1]:
                    out_patch[..., -overlap_size // 2:, :] *= 0
                    out_patch_mask[..., -overlap_size // 2:, :] *= 0
                if w_idx < w_idx_list[-1]:
                    out_patch[..., :, -overlap_size // 2:] *= 0
                    out_patch_mask[..., :, -overlap_size // 2:] *= 0
                if h_idx > h_idx_list[0]:
                    out_patch[..., :overlap_size // 2, :] *= 0
                    out_patch_mask[..., :overlap_size // 2, :] *= 0
                if w_idx > w_idx_list[0]:
                    out_patch[..., :, :overlap_size // 2] *= 0
                    out_patch_mask[..., :, :overlap_size // 2] *= 0
                E[..., h_idx * sf:(h_idx + size_patch_testing) * sf,
                     w_idx * sf:(w_idx + size_patch_testing) * sf].add_(out_patch)
                W[..., h_idx * sf:(h_idx + size_patch_testing) * sf,
                     w_idx * sf:(w_idx + size_patch_testing) * sf].add_(out_patch_mask)
        return E.div_(W)

    _, _, _, h_old, w_old = lq.size()
    h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
    w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]
    lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
    lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq
    output = model(lq).detach().cpu()
    return output[:, :, :, :h_old * sf, :w_old * sf]


def _sigma255(meta: dict, lq: np.ndarray) -> float:
    """The true (or, for poisson_gaussian, analytically equivalent) noise sigma
    in 8-bit units, using the exact parameters from p3_degrade.py."""
    kind = meta.get("kind")
    if kind == "gaussian":
        from p3_degrade import GAUSSIAN_SIGMA_255
        return GAUSSIAN_SIGMA_255[meta["level"]]
    if kind == "poisson_gaussian":
        from p3_degrade import POISSON_GAUSSIAN
        a, b = POISSON_GAUSSIAN["a"], POISSON_GAUSSIAN["b"]
        mean_x = float(np.mean(lq))          # already-degraded frame's own mean intensity
        return float(np.sqrt(mean_x / a + b * b) * 255.0)
    return 25.0  # fallback (e.g. Track C real footage: no known model, use a mid default)


def restore(lq: np.ndarray, meta: dict) -> np.ndarray:
    """lq: (T,H,W,3) float32 RGB [0,1]. Returns sr in the same format."""
    import torch
    _load()

    sigma = _sigma255(meta, lq) / 255.0
    x = torch.from_numpy(lq).permute(0, 3, 1, 2).unsqueeze(0).float()  # (1,T,3,H,W)
    noise_ch = torch.full((1, x.shape[1], 1, x.shape[3], x.shape[4]), sigma, dtype=torch.float32)
    x = torch.cat([x, noise_ch], dim=2).to(_device)

    # pad temporal dim to a multiple of window_size[0] (RVRT's test_video, tile[0]==0 branch)
    d_old = x.size(1)
    d_pad = (_Args.window_size[0] - d_old % _Args.window_size[0]) % _Args.window_size[0]
    if d_pad:
        x = torch.cat([x, torch.flip(x[:, -d_pad:, ...], [1])], 1)

    with torch.no_grad():
        out = _test_clip(x, _model, _Args)
    out = out[:, :d_old, :, :, :]

    sr = out.squeeze(0).clamp(0, 1).permute(0, 2, 3, 1).numpy().astype(np.float32)
    return sr
