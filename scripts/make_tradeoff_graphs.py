"""PSNR-vs-FLOPs and SSIM-vs-FLOPs tradeoff graphs: DashMamba (Stage-1, the fair
comparison - see RESULTS_AND_DIAGNOSIS.md) against RVRT, BasicVSR++, FastDVDnet.

Reads results/<run>/summary.csv (already-computed Track A test metrics - no
new inference) and results/complexity/complexity.csv (network-only FLOPs from
model_complexity.py) and writes two static PNGs for the thesis.

FLOPs convention: network-only cost at a fixed 720p frame (complexity.csv's
gflops_per_frame_720p), NOT the evaluation wrapper's tiled/chunked cost -
architecture comparisons in the literature report the former; the latter
carries each wrapper's own tiling overhead, which is an implementation detail,
not a property of the model (see results/complexity/README.md §2-3).

Each model contributes one point: PSNR/SSIM averaged across the 4 Track A test
axes (gaussian low/medium/high + poisson_gaussian), with a vertical bar
spanning that model's min-max across the 4 axes, since the ranking is not
uniform across axes (DashMamba wins at high noise, loses at low noise - see
RESULTS_AND_DIAGNOSIS.md "the noise-level trend").

Usage: python make_tradeoff_graphs.py [--out ../results/complexity]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent

# name -> (results run-name, complexity.csv model name)
MODELS = {
    "DashMamba": ("dashmamba_stage1_track_a_test", "DashMamba"),
    "RVRT": ("rvrt_track_a_test", "RVRT"),
    "BasicVSR++": ("bvrpp_track_a_test", "BasicVSR++"),
    "FastDVDnet": ("fastdvdnet_track_a_test", "FastDVDnet"),
}
# fixed categorical order + hex, from the dataviz skill's default palette
# (references/palette.md): slots 1/2/3/7 - the three that validate all-pairs
# plus violet, skipping slot 4 (yellow), which the palette doc flags as
# failing the all-pairs floor against orange past three series.
COLORS = {
    "DashMamba": "#2a78d6",   # blue
    "RVRT": "#eb6834",        # orange
    "BasicVSR++": "#1baf7a",  # aqua
    "FastDVDnet": "#4a3aa7",  # violet
}
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_axis_metrics(run_name: str) -> dict[str, list[float]]:
    """Return {'psnr': [4 axis values], 'ssim': [...]} from the top-level
    (lighting-blank) rows of summary.csv."""
    path = ROOT / "results" / run_name / "summary.csv"
    out = {"psnr": [], "ssim": []}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["lighting"] or not row["track"]:
                continue  # skip the per-lighting breakdown and the blank separator row
            for m in ("psnr", "ssim"):
                if row[m]:
                    out[m].append(float(row[m]))
    return out


def load_flops() -> dict[str, float]:
    path = ROOT / "results" / "complexity" / "complexity.csv"
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("gflops_per_frame_720p"):
                out[row["model"]] = float(row["gflops_per_frame_720p"])
    return out


def make_chart(metric: str, ylabel: str, gathered: dict, flops: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    points = {name: (flops[MODELS[name][1]], sum(gathered[name][metric]) / len(gathered[name][metric]))
              for name in MODELS}
    # Label placement: two points close in log-FLOPs (RVRT/FastDVDnet, ~1.2-1.5x
    # apart) collide if every label sits at the same offset. Alternate the
    # vertical offset whenever a point's log-x is within 0.15 dex of an
    # already-placed one, so close pairs separate instead of overlapping.
    placed: list[tuple[float, int]] = []  # (log10 x, dy in points) already used
    label_dy: dict[str, int] = {}
    for name in sorted(MODELS, key=lambda n: points[n][0]):
        lx = math.log10(points[name][0])
        dy = 9
        for plx, pdy in placed:
            if abs(lx - plx) < 0.15:
                dy = -16 if pdy > 0 else 9
        placed.append((lx, dy))
        label_dy[name] = dy

    for name in MODELS:  # fixed draw order, never re-sorted by value
        vals = gathered[name][metric]
        x, mean_v = points[name]
        lo, hi = min(vals), max(vals)
        color = COLORS[name]
        ax.plot([x, x], [lo, hi], color=color, linewidth=2, alpha=0.35, zorder=2, solid_capstyle="round")
        ax.scatter([x], [mean_v], s=64, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=name)
        # direct label: identity never relies on hue alone (dataviz skill, "never color-alone")
        ax.annotate(name, (x, mean_v), xytext=(10, label_dy[name]), textcoords="offset points",
                    color=INK, fontsize=9.5, va="center", fontweight="medium")

    ax.set_xscale("log")
    ax.set_xlabel("Network FLOPs per 720p frame (GFLOPs, log scale)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel(ylabel, color=SECONDARY_INK, fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(f"{ylabel} vs. compute — Stage-1 (fair comparison), Track A test, n=71",
                fontsize=11, color=INK, pad=12, loc="left")
    ax.text(0.0, -0.16, "Bars span the 4 Track A axes (gaussian low/medium/high, poisson-gaussian); "
           "dot = mean. Lower-left is better.", transform=ax.transAxes,
           fontsize=8, color=MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "complexity"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    flops = load_flops()
    gathered = {name: load_axis_metrics(run) for name, (run, _) in MODELS.items()}

    missing = [MODELS[n][1] for n in MODELS if MODELS[n][1] not in flops]
    if missing:
        raise SystemExit(f"missing FLOPs for {missing} - run model_complexity.py --parts flops first")

    make_chart("psnr", "PSNR (dB, higher is better)", gathered, flops, out_dir / "psnr_vs_flops.png")
    make_chart("ssim", "SSIM (higher is better)", gathered, flops, out_dir / "ssim_vs_flops.png")


if __name__ == "__main__":
    main()
