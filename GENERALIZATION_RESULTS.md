# Generalization, efficiency and visual proof — beyond the Track A benchmark

Everything in `RESULTS_AND_DIAGNOSIS.md` is measured one way: synthetic noise added by
this project's own code, scored against a clean reference, on this project's own
dashcam dataset. This document adds three things that test the model differently:

1. **Real noise, not synthetic** — DashMamba on Track C, 116 real night/evening dashcam
   clips that were never touched during training, validation or testing.
2. **A dataset the model has never seen** — DashMamba on DAVIS 2017 test-dev, a public
   video benchmark with nothing to do with dashcams (sports, animals, vehicles of every
   kind), on a split disjoint from Stage-1's own training data.
3. **Compute cost and visual proof** — FLOPs/inference time (full detail in
   `results/complexity/README.md`) and before/after images for all three of Track A,
   Track C and DAVIS.

All of it is **DashMamba only** — RVRT/BasicVSR++/FastDVDnet were not re-run on Track C
or DAVIS (a deliberate scope decision; see `HANDOFF_FOR_NEW_CHAT.md`). Produced
2026-09-24/25 on a Windows laptop (RTX 5050 Laptop GPU, 8 GB), not the RTX 4080 SUPER
the rest of the project was run on.

---

## 1. Track C — real noise, held out by construction

**What it is:** 116 real night/evening dashcam clips (~302 frames each, capped at 80 for
this run to match the rest of the project's protocol). `kind=none, level=real` in
`metadata/p3_eval_index.csv` — these are the camera's own raw frames, no synthetic
degradation applied. Track C was built into the dataset but never evaluated on any
model before this. Because there is no clean reference, only no-reference quality
metrics (NIQE, BRISQUE) and temporal consistency (tOF) apply — there is no PSNR/SSIM.

**Model:** `finetune_stage2.pt` (the deployed, dashcam-fine-tuned checkpoint).

**Command:** `python p3_evaluate.py --model-import dashmamba_wrapper:restore --tracks C --splits test --max-frames 80 --run-name dashmamba_stage2_track_c_real`

### Results

| Metric | Before (raw input) | After (DashMamba) | Reading |
|---|---|---|---|
| NIQE (lower better) | **4.97** | 5.18 | marginally worse (+0.21) |
| BRISQUE (lower better) | **34.25** | 34.02 | marginally better (−0.23) |
| tOF (lower better) | 6.329 | 6.329 | unchanged |

Both NIQE/BRISQUE deltas are small next to their per-clip spread (std 0.73 / 12.36) —
**not a clear win either way.** Per-lighting (from `summary.csv`): evening (29 clips)
scores better than night (87 clips) on both metrics after restoration (NIQE 4.78 vs
5.31, BRISQUE 31.95 vs 34.71) — night is the harder condition, as expected.

**Honest reading:** real dashcam sensor noise at typical ISO is far milder than the
synthetic σ50 noise the headline PSNR numbers are measured on. DashMamba's visible
effect on real footage is **subtle, not dramatic** — this is evidence the model doesn't
damage real footage and stays reasonable outside its synthetic-noise training
distribution, not a standalone quality win. Do not present it as one.

**A genuine limitation, found and disclosed, not hidden:** of 5 sampled clips rendered
for the visual proof, **one** (`raw_video_013_clip_0000`) shows a clear artifact — the
model invents a distorted, ghost-like shape near a market-stall awning that isn't in the
input at all. The other 4 are clean. It does not look like a fixed tiling-seam bug (the
same distortion doesn't reappear at the same position in other clips); it more likely
comes from the flow-alignment step misfiring on that frame's complex motion (moving
canopy, strong light flares) — exactly the kind of busy, high-motion scene
`docs/Phase3_Baseline_Weakness_Analysis.docx` already flags as hardest for temporal
alignment. Root cause not yet confirmed. **Do not use this clip as your primary
"it works" figure** — use one of the four clean ones — but keep it as a disclosed
failure case; this project's own practice (Signal A/B, the day-clip loss in the
qualitative figures) is to report limitations found, not select around them.

### Where the Track C results live

| What | Path |
|---|---|
| Per-clip stats (116 rows: tOF, NIQE, BRISQUE) | [`results/dashmamba_stage2_track_c_real/per_clip.csv`](results/dashmamba_stage2_track_c_real/per_clip.csv) |
| Summary (by axis, then by lighting) | [`results/dashmamba_stage2_track_c_real/summary.csv`](results/dashmamba_stage2_track_c_real/summary.csv) |
| Run config/timing | [`results/dashmamba_stage2_track_c_real/config.json`](results/dashmamba_stage2_track_c_real/config.json) |
| Before/after NIQE-BRISQUE baseline (the "before" row above) | [`results/dashmamba_stage2_track_c_real/input_quality_baseline.json`](results/dashmamba_stage2_track_c_real/input_quality_baseline.json) |
| **Visuals** — 5 clips, full-frame + 2× zoomed crop, incl. the artifact case | [`figures/track_c/`](figures/track_c/) — `<clip>_before.png`, `_after.png`, `_compare.png`, `_crop.png` |

Reproduce: `scripts/measure_input_quality.py` (the "before" baseline),
`scripts/make_before_after.py --dataset track-c`, `scripts/make_crops.py`.

---

## 2. DAVIS 2017 test-dev — a dataset the model has never seen

**What it is:** the official DAVIS 2017 **test-dev** split (30 videos, 31–127 frames
each, 854×480). Stage-1 pretraining used whatever the original machine's downloaded
DAVIS folder held (the standard trainval package, per `pretrain_dataloader.py`) — a
different, disjoint official split, so evaluating on test-dev is leak-free without
needing to know exactly which videos that folder contained. This tests generalization
to content that looks nothing like dashcam video: sports, animals, vehicles, people,
outdoors and indoors.

**Model:** `pretrain_stage1.pt` (DAVIS-trainval + no dashcam data — the fair-comparison
checkpoint used throughout this project). Degraded with the same 4 axes as Track A test
(`p3_degrade.py`, unchanged).

**Command:** `python evaluate_davis.py --davis-root <path>/DAVIS --run-name dashmamba_stage1_davis_testdev --max-frames 80`

### Results

| Axis | Input PSNR | **DAVIS PSNR** | SSIM | LPIPS | NIQE | BRISQUE | tOF (in) |
|---|---|---|---|---|---|---|---|
| gaussian / high (σ50) | 15.12 | **27.79** | 0.755 | 0.313 | 4.04 | 18.60 | 16.98 (62.65) |
| gaussian / medium (σ25) | 20.64 | 31.79 | 0.872 | 0.164 | 3.82 | 19.60 | 18.67 (31.69) |
| gaussian / low (σ15) | 24.91 | 34.41 | 0.917 | 0.094 | 3.59 | 17.12 | 19.17 (24.02) |
| poisson_gaussian | 21.87 | 32.63 | 0.899 | 0.134 | 3.69 | 19.24 | 18.88 (29.17) |

### Cross-domain comparison (same Stage-1 checkpoint, dashcam vs. DAVIS)

| Axis | Track A test (dashcam, in-domain) | DAVIS test-dev (out-of-domain) | Gap |
|---|---|---|---|
| gaussian / high | 30.58 | 27.79 | **−2.79 dB** |
| gaussian / medium | 35.02 | 31.79 | −3.23 dB |
| gaussian / low | 37.56 | 34.41 | −3.15 dB |
| poisson_gaussian | 36.35 | 32.63 | −3.72 dB |

**Honest reading:** a consistent ~2.8–3.7 dB drop moving to a completely unrelated
dataset is a normal, expected generalization gap — not a collapse. The model clearly
learned to denoise generally, not to memorize dashcam-specific patterns, and it still
recovers 27.8–34.4 dB PSNR on arbitrary video content it was never designed around.
**A genuine weak point:** tOF (temporal consistency) is markedly worse on DAVIS (17–19)
than on dashcam footage (6.0–6.3 at the same axes) — consistent with the already-known
diagnosis that Signal A (motion-aware fusion) never activated (`RESULTS_AND_DIAGNOSIS.md`
§2): DAVIS's more varied and often faster motion exposes that inactivity more than a
forward-facing dashcam's relatively directional motion does.

### Where the DAVIS results live

| What | Path |
|---|---|
| Per-clip stats (120 rows: 30 videos × 4 axes) | [`results/dashmamba_stage1_davis_testdev/per_clip.csv`](results/dashmamba_stage1_davis_testdev/per_clip.csv) |
| Summary (per axis) | [`results/dashmamba_stage1_davis_testdev/summary.csv`](results/dashmamba_stage1_davis_testdev/summary.csv) |
| Run config/timing | [`results/dashmamba_stage1_davis_testdev/config.json`](results/dashmamba_stage1_davis_testdev/config.json) |
| **Visuals** — 5 videos, ground truth / noisy / restored, full-frame + crop | [`figures/davis/`](figures/davis/) — `<video>_gt.png`, `_before.png`, `_after.png`, `_compare.png`, `_crop.png` |

Reproduce: `scripts/evaluate_davis.py`, `scripts/make_before_after.py --dataset davis`,
`scripts/make_crops.py`. The DAVIS test-dev frames themselves are **not in git**
(~300 MB of public data) — download from
`https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip`.

---

## 3. Model size and held-out training loss

### Checkpoint footprint

| Model | Parameters (trainable) | Params + buffers | File size on disk |
|---|---|---|---|
| **DashMamba** | **886,840** | 886,840 | **10.23 MB** (either checkpoint) |
| FastDVDnet | 2,479,096 | 2,483,578 | 9.51 MB |
| RVRT | 12,786,919 | 13,065,453 | 56.78 MB (51.28 denoising + 5.50 SpyNet flow init) |
| BasicVSR++ | 44,075,631 | 44,075,637 | 168.24 MB |

(Full parameter/FLOPs detail: `results/complexity/README.md` §1. Note the docs'
previously-quoted "14.7× smaller than RVRT" used params+buffers; on trainable
parameters alone DashMamba is 14.4× smaller — DashMamba has no buffers either way.)

### Charbonnier loss on held-out data (no training-loss curve was logged originally)

Measured with `scripts/measure_loss.py`: the exact loss `train_dashmamba.py` optimizes
(`eps=1e-3`), no gradient update, on Track A **test** — held out from both checkpoints'
training — using the same random-window/crop/degradation recipe training used
(seq_len=12, crop=192, `TRAIN_KINDS = (gaussian, poisson_gaussian)`), 3 passes × 71
clips = 213 windows.

| Checkpoint | Mean loss | 95% CI | gaussian | poisson_gaussian |
|---|---|---|---|---|
| Stage-1 (`pretrain_stage1.pt`) | 0.01310 | [0.01233, 0.01388] | 0.01473 | 0.01169 |
| Stage-2 (`finetune_stage2.pt`) | 0.01259 | [0.01184, 0.01334] | 0.01400 | 0.01137 |

Fine-tuning lowers held-out loss by about 4%, consistent with (and independent
confirmation of) the +0.17 to +0.46 dB PSNR gain already reported for Stage-2 over
Stage-1 in `RESULTS_AND_DIAGNOSIS.md`.

**Where these live:** [`results/dashmamba_stage1_track_a_test/held_out_loss.json`](results/dashmamba_stage1_track_a_test/held_out_loss.json),
[`results/dashmamba_stage2_track_a_test/held_out_loss.json`](results/dashmamba_stage2_track_a_test/held_out_loss.json).

---

## 4. Efficiency — FLOPs, inference time, memory

Full writeup, caveats and reproduction command: **[`results/complexity/README.md`](results/complexity/README.md)**.
Headline: DashMamba needs **7–30× fewer FLOPs** and runs **2–9× faster** (network only)
than RVRT/BasicVSR++/FastDVDnet on identical input. Two tradeoff graphs (PSNR vs. FLOPs,
SSIM vs. FLOPs, Stage-1 fair comparison) are at
[`results/complexity/psnr_vs_flops.png`](results/complexity/psnr_vs_flops.png) and
[`results/complexity/ssim_vs_flops.png`](results/complexity/ssim_vs_flops.png).

---

## 5. Everything in this document, indexed by folder

```
results/
  dashmamba_stage2_track_c_real/     Track C: 116 real clips, no reference
    per_clip.csv summary.csv config.json  input_quality_baseline.json
  dashmamba_stage1_davis_testdev/    DAVIS test-dev: 30 videos x 4 axes
    per_clip.csv summary.csv config.json
  dashmamba_stage1_track_a_test/     (existing Track A run) + held_out_loss.json
  dashmamba_stage2_track_a_test/     (existing Track A run) + held_out_loss.json
  complexity/                        FLOPs, inference time, memory, tradeoff graphs
    README.md complexity.csv complexity.json psnr_vs_flops.png ssim_vs_flops.png

figures/
  track_a/    5 clips, GROUND TRUTH / noisy (gaussian/high, sigma50) / DashMamba
              <clip>_gt.png _before.png _after.png _compare.png _crop.png
  track_c/    5 clips, real noise / DashMamba (no ground truth)
              <clip>_before.png _after.png _compare.png _crop.png
  davis/      5 videos, GROUND TRUTH / noisy (gaussian/high) / DashMamba
              <video>_gt.png _before.png _after.png _compare.png _crop.png

scripts/
  measure_input_quality.py   NIQE/BRISQUE on raw (un-restored) input, any track/axis
  evaluate_davis.py          Full DAVIS test-dev eval (reuses p3_evaluate.py's metrics)
  measure_loss.py            Held-out Charbonnier loss, either checkpoint, either dataset
  make_before_after.py       Before/after/gt PNGs, --dataset track-a|track-c|davis
  make_crops.py               Zoomed 2x center-crop comparisons from saved PNGs
  model_complexity.py         Params/FLOPs/latency/pipeline cost, all 4 models
  baseline_shims.py           Builds RVRT/BasicVSR++/FastDVDnet with no compiled ops
  make_tradeoff_graphs.py     The two PSNR/SSIM-vs-FLOPs charts above
```
