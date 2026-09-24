"""Parameters, FLOPs, inference time and peak GPU memory for DashMamba and the
three Track A baselines, under one protocol.

Four measurements, each reported per output frame:

  params    parameter count of each network, with and without buffers.
  flops     network-only FLOPs on a fixed input (--std-frames x --std-size^2),
            then scaled to 1280x720 by pixel count. Every op in these models is
            local (convs, windowed attention, per-pixel scans), so cost is linear
            in pixel count; --check-scaling re-measures at 2x size to verify.
  latency   network-only GPU time on that same fixed input (median of --repeats).
  pipeline  the exact inference path used for the reported results: an
            80-frame 1280x720 clip through each scripts/*_wrapper.py, including
            its spatial tiling, temporal chunking and CPU<->GPU copies. Reports
            wall time, FLOPs (counted over the whole call) and peak memory.

FLOPs convention: torch.utils.flop_counter, i.e. 2 x multiply-accumulates of
conv / matmul / einsum. Elementwise ops (activations, softmax, grid_sample,
the exp/multiply updates inside DashMamba's scan) are not counted, as with
fvcore/thop. GMACs = GFLOPs / 2 is also written, since papers mix the two.

Baselines are built by baseline_shims.py from upstream source under external/,
with random weights (cost does not depend on weight values). DashMamba's
pipeline run uses the committed Stage-2 checkpoint through dashmamba_wrapper.

Usage (from scripts/):
    python model_complexity.py                       # everything, all models
    python model_complexity.py --parts params flops  # quick, no pipeline run
    python model_complexity.py --models DashMamba RVRT
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision  # noqa: F401  registers torch.ops.torchvision.deform_conv2d
from torch.utils.flop_counter import FlopCounterMode

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(SCRIPTS))

import baseline_shims  # noqa: E402
from dashmamba import DashMambaNet  # noqa: E402

FRAME_H, FRAME_W = 720, 1280
MODEL_NAMES = ["DashMamba", "RVRT", "BasicVSR++", "FastDVDnet"]
# the Gaussian sigma the non-blind baselines are handed; value does not affect cost
SIGMA = 50.0 / 255.0


def deform_conv2d_flop(x_shape, weight_shape, offset_shape, mask_shape, bias_shape,
                       *args, out_shape=None, **kwargs):
    """torchvision deform_conv2d: same multiply-accumulates as the dense conv it
    replaces (sampling is elementwise and not counted, like grid_sample)."""
    _, cin_per_group, kh, kw = weight_shape
    return 2 * math.prod(out_shape) * cin_per_group * kh * kw


CUSTOM_FLOPS = {torch.ops.torchvision.deform_conv2d.default: deform_conv2d_flop}


def count_flops(fn) -> int:
    counter = FlopCounterMode(display=False, custom_mapping=CUSTOM_FLOPS)
    with counter, torch.no_grad():
        fn()
    return counter.get_total_flops()


def build(name: str):
    return {
        "DashMamba": lambda: DashMambaNet(mid_channels=64, num_res_blocks=5, state_dim=16,
                                          num_mamba_blocks=2),
        "RVRT": baseline_shims.build_rvrt,
        "BasicVSR++": baseline_shims.build_basicvsrpp,
        "FastDVDnet": baseline_shims.build_fastdvdnet,
    }[name]()


def network_call(name: str, model, frames: int, size: int, device):
    """Return (fn, output_frames): one forward on the fixed input.
    FastDVDnet restores one frame per call from a 5-frame window; the recurrent
    models restore every frame of the clip in one call."""
    g = torch.Generator(device="cpu").manual_seed(0)
    if name == "FastDVDnet":
        x = torch.rand(1, 15, size, size, generator=g).to(device)
        nmap = torch.full((1, 1, size, size), SIGMA, device=device)
        return (lambda: model(x, nmap)), 1
    x = torch.rand(1, frames, 3, size, size, generator=g).to(device)
    if name == "RVRT":  # non-blind: noise level is a 4th input channel
        x = torch.cat([x, torch.full((1, frames, 1, size, size), SIGMA, device=device)], dim=2)
    return (lambda: model(x)), frames


def time_fn(fn, repeats: int, warmup: int, device) -> list[float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


# --------------------------------------------------------------------------- #
# pipeline: the evaluation wrappers, unchanged
# --------------------------------------------------------------------------- #
def load_clip(clip_dir: Path, frames: int) -> np.ndarray:
    import cv2
    paths = sorted(clip_dir.glob("frame_*.png"))[:frames]
    if len(paths) < frames:
        raise FileNotFoundError(f"need {frames} frames in {clip_dir}, found {len(paths)}")
    imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
    clean = np.stack(imgs).astype(np.float32) / 255.0
    rng = np.random.default_rng(0)
    return np.clip(clean + rng.normal(0, SIGMA, clean.shape).astype(np.float32), 0, 1)


def pipeline_call(name: str, device, cpu_cache: bool = False):
    """Return restore(lq) through the same wrapper p3_evaluate.py used.

    cpu_cache: turn on RVRT/BasicVSR++'s own long-video mode, which parks
    per-frame features in host RAM between recurrent steps. Same computation,
    lower VRAM. Needed on an 8 GB GPU: RVRT holds ~13 GB for a whole 80-frame
    tile, and Windows' sysmem fallback would otherwise silently spill it to
    shared memory over PCIe, making the timing meaningless."""
    meta = {"kind": "gaussian", "level": "high"}
    if name == "DashMamba":
        import dashmamba_wrapper as w
        w._load()  # real Stage-2 checkpoint
        return lambda lq: w.restore(lq), w
    if name == "RVRT":
        import rvrt_wrapper as w
        fn = lambda lq: w.restore(lq, meta)  # noqa: E731
    elif name == "BasicVSR++":
        import basicvsrpp_wrapper as w
        fn = lambda lq: w.restore(lq)  # noqa: E731
    else:
        import fastdvdnet_wrapper as w
        fn = lambda lq: w.restore(lq, meta)  # noqa: E731
    w._model = build(name).eval().to(device)  # skips the wrapper's checkpoint load
    w._device = device
    if cpu_cache and hasattr(w._model, "cpu_cache_length"):
        w._model.cpu_cache_length = 0  # forward() enables caching when t > cpu_cache_length
    return fn, w


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--parts", nargs="+", default=["params", "flops", "latency", "pipeline"],
                    choices=["params", "flops", "latency", "pipeline"])
    ap.add_argument("--std-size", type=int, default=256)
    ap.add_argument("--std-frames", type=int, default=16)
    ap.add_argument("--check-scaling", action="store_true",
                    help="also count FLOPs at 2x --std-size to confirm linear-in-pixels scaling")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--clip", default="raw_video_013_clip_0009")
    ap.add_argument("--clip-frames", type=int, default=80)
    ap.add_argument("--pipeline-repeats", type=int, default=2)
    ap.add_argument("--cpu-cache", nargs="*", default=["RVRT"], choices=["RVRT", "BasicVSR++"],
                    help="models whose pipeline run uses their built-in CPU feature cache")
    ap.add_argument("--out", default=str(ROOT / "results" / "complexity"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "std_input": f"{args.std_frames} frames x {args.std_size}x{args.std_size}",
        "pipeline_input": f"{args.clip} ({args.clip_frames} frames x {FRAME_W}x{FRAME_H})",
    }
    print(json.dumps(env, indent=2))
    scale_720p = FRAME_H * FRAME_W / args.std_size ** 2

    lq = None
    if "pipeline" in args.parts:
        lq = load_clip(ROOT / "dataset" / "_refpool" / args.clip, args.clip_frames)
        print(f"loaded clip {args.clip}: {lq.shape}")

    rows = []
    for name in args.models:
        row = {"model": name}
        print(f"\n=== {name} ===")
        try:
            model = build(name).eval().to(device)
            if "params" in args.parts:
                row["params"] = sum(p.numel() for p in model.parameters())
                # the figures in RESULTS_AND_DIAGNOSIS.md are state_dict totals,
                # i.e. params + buffers (BN statistics, RVRT position indices)
                row["params_plus_buffers"] = sum(v.numel() for v in model.state_dict().values())
                print(f"params            : {row['params']:,}  "
                      f"(+buffers: {row['params_plus_buffers']:,})")

            if "flops" in args.parts:
                fn, nout = network_call(name, model, args.std_frames, args.std_size, device)
                f = count_flops(fn) / nout
                row["gflops_per_frame_std"] = f / 1e9
                row["gflops_per_frame_720p"] = f * scale_720p / 1e9
                row["gmacs_per_frame_720p"] = f * scale_720p / 2e9
                print(f"GFLOPs/frame      : {f / 1e9:.2f} @ {args.std_size}^2  ->  "
                      f"{row['gflops_per_frame_720p']:.1f} @ 720p")
                if args.check_scaling:
                    s2 = args.std_size * 2
                    fn2, nout2 = network_call(name, model, args.std_frames, s2, device)
                    f2 = count_flops(fn2) / nout2
                    row["scaling_ratio_2x"] = f2 / f
                    print(f"scaling check     : {s2}^2 / {args.std_size}^2 = {f2 / f:.3f} (4.000 = linear)")
                    del fn2

            if "latency" in args.parts:
                fn, nout = network_call(name, model, args.std_frames, args.std_size, device)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                t = time_fn(fn, args.repeats, args.warmup, device)
                row["net_ms_per_frame_std"] = statistics.median(t) / nout * 1e3
                if device.type == "cuda":
                    row["net_peak_mem_gb_std"] = torch.cuda.max_memory_allocated() / 1e9
                print(f"network latency   : {row['net_ms_per_frame_std']:.2f} ms/frame @ "
                      f"{args.std_size}^2 (median of {args.repeats})")
                del fn
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if "pipeline" in args.parts:
                row["pipe_cpu_cache"] = name in args.cpu_cache
                fn, w = pipeline_call(name, device, cpu_cache=row["pipe_cpu_cache"])
                fn(lq[:8])  # warm-up: cuDNN algorithm selection, allocator
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(args.pipeline_repeats):
                    t0 = time.perf_counter()
                    fn(lq)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                n = lq.shape[0]
                row["pipe_s_per_frame"] = min(times) / n
                row["pipe_fps"] = n / min(times)
                row["pipe_s_per_clip"] = min(times)
                if device.type == "cuda":
                    row["pipe_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    # With sysmem fallback, allocations past VRAM succeed silently
                    # and run over PCIe; a timing taken like that is not valid.
                    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                    row["pipe_vram_spill"] = row["pipe_peak_mem_gb"] > 0.95 * vram
                    print(f"pipeline peak mem : {row['pipe_peak_mem_gb']:.2f} GB of {vram:.2f} GB"
                          + ("  ** EXCEEDS VRAM - timing invalid **" if row["pipe_vram_spill"] else ""))
                print(f"pipeline time     : {row['pipe_s_per_frame'] * 1e3:.0f} ms/frame "
                      f"({row['pipe_fps']:.2f} fps, {min(times):.1f} s per {n}-frame clip, "
                      f"best of {args.pipeline_repeats})")
                pf = count_flops(lambda: fn(lq)) / n
                row["pipe_gflops_per_frame"] = pf / 1e9
                print(f"pipeline GFLOPs   : {pf / 1e9:.1f} per 720p frame (incl. tile/chunk overlap)")
                w._model = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as e:  # keep going: one model failing should not lose the others
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"FAILED: {row['error']}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "complexity.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    (out_dir / "complexity.json").write_text(json.dumps({"env": env, "rows": rows}, indent=2))
    print(f"\nwrote {out_dir / 'complexity.csv'} and complexity.json")


if __name__ == "__main__":
    main()
