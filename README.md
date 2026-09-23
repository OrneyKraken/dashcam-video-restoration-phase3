# Dashcam Video Restoration — Phase 3 (DashMamba)

BRAC University undergraduate CS thesis, Phase 3. A Mamba-based video restoration
architecture for dashcam footage, plus a benchmark of three published baselines on a new
755-clip dashcam dataset.

> **New chat or new machine? Read [`HANDOFF_FOR_NEW_CHAT.md`](HANDOFF_FOR_NEW_CHAT.md) first**,
> then [`RESULTS_AND_DIAGNOSIS.md`](RESULTS_AND_DIAGNOSIS.md). This README is the map;
> those two are the detail.

---

## 1. Current status at a glance

| Item | State |
|---|---|
| Dataset (755 clips / 226,236 frames) | ✅ Built — frames **not** in git, see §4 |
| Baselines: RVRT, BasicVSR++, FastDVDnet | ✅ Evaluated on Track A — `results/` |
| Weakness analysis | ✅ Complete — `docs/` |
| DashMamba architecture | ✅ Designed + implemented — `models/dashmamba.py` |
| DashMamba training (Stage-1 + Stage-2) | ✅ Complete — `checkpoints/` |
| **Stage-2 evaluation (headline result)** | ✅ **DONE** — see `RESULTS_AND_DIAGNOSIS.md` |
| Statistical significance testing | ✅ Done |
| **Stage-1 evaluation (fairness comparison)** | ✅ **DONE — architecture win confirmed** |
| Qualitative before/after visuals | ⬜ Not started (~30 min) |
| Ablation study | ⬜ Low value now — signals measured inactive, see below |
| Tracks B and C | ⬜ Never evaluated — deliberate scope decision |

---

## 2. What has been done, in order

1. **Built the dataset** — 755 clips, 226,236 frames of real driving footage
   (Chittagong, Bangladesh), 1280×720 @ 15 fps, manually curated, stratified by lighting
   (day 465 / evening 87 / night 203), organised into three evaluation tracks.
2. **Benchmarked three published baselines** on Track A under one identical protocol
   (71 clips × 80 frames × 4 degradation axes = 284 clip-runs each).
3. **Wrote a weakness analysis** — found that daytime footage is the *hardest* for
   temporal consistency across **every** baseline, and that RVRT/FastDVDnet lose 4–5 dB
   specifically when severe noise meets evening/night lighting.
4. **Designed DashMamba** around those two findings, with two decoupled control signals.
5. **Trained it in two stages** — DAVIS pretrain (28,800 steps) then dashcam fine-tune
   (11,000 steps).
6. **Evaluated Stage-2** and ran paired significance tests.
7. **Measured the trained model's own mechanisms** — and found both proposed signals
   inactive (see below).

### Headline numbers — the FAIR comparison (Stage-1: no dashcam training)

All four models trained on public data only. None has seen dashcam footage.
This is the comparison to lead with.

| Axis | Input | **DashMamba S1** | RVRT | BasicVSR++ | FastDVDnet | Significance vs RVRT |
|---|---|---|---|---|---|---|
| gaussian / high (σ50) | 15.50 | **30.58** | 28.79 | 16.54 | 28.64 | **+1.79 dB** [+1.05, +2.52], significant |
| poisson_gaussian | 23.67 | **36.35** | 35.54 | 24.98 | 35.37 | **+0.81 dB** [+0.38, +1.25], significant |
| gaussian / medium (σ25) | 20.96 | 35.02 | 35.53 | 22.42 | 35.31 | −0.51 dB, tie (CI spans 0) |
| gaussian / low (σ15) | 25.14 | 37.56 | **39.13** | 26.48 | 38.75 | **−1.57 dB**, significant loss |

### What fine-tuning added (Stage-2 − Stage-1, paired, n = 71)

| Axis | Gain | Stage-2 PSNR |
|---|---|---|
| gaussian / high | **+0.46 dB** (t = +12.0) | 31.04 |
| gaussian / medium | **+0.38 dB** (t = +10.6) | 35.40 |
| gaussian / low | **+0.21 dB** (t = +4.6) | 37.77 |
| poisson_gaussian | **+0.17 dB** (t = +10.0) | 36.52 |

Fine-tuning is a consistent but modest lift. At σ50 the architectural gain (+1.79 dB) is
**~4×** the domain-adaptation gain (+0.46 dB) — the win is not an artifact of training
on the target domain.

**Parameters:** DashMamba **886,840** vs FastDVDnet 2.48 M, RVRT 13.07 M, BasicVSR++ 44.08 M.
DashMamba also runs **blind** — it is never given the true noise level, while RVRT and
FastDVDnet are.

### ⚠️ The critical finding

**Both proposed "novel" signals are inactive in the trained model.**
Signal B (blind reliability) outputs a constant 1.0000 — its sigmoid is saturated
(logits ≈ +44). Signal A (motion confidence) sits at a constant 0.993 because the flow
network collapsed to ~0.002 px displacement.

The gains are real but come from the **bidirectional selective-scan temporal core**, not
the adaptive mechanisms. This also explains the low-noise loss and why tOF temporal
consistency is *worse* than both working baselines. **Do not claim the decoupled-signal
design works.** Root causes and fixes: `RESULTS_AND_DIAGNOSIS.md` §2.

---

## 3. What to run next

If you can only do one thing, do **#2** (visuals) — #1 and #3 are now complete.

| # | Task | Time (RTX 4080 SUPER) | Command / file |
|---|---|---|---|
| ~~1~~ | ~~**Stage-1 evaluation**~~ — ✅ **done**, see above | — | §5 below |
| 2 | **Qualitative visuals** | ~30 min | `scripts/make_qualitative.py` |
| ~~3~~ | ~~Update docs with Stage-1 numbers~~ — ✅ **done** | — | `RESULTS_AND_DIAGNOSIS.md` |
| 4 | *(Optional)* Fix both signals and retrain | ~7 h train + 5 h eval | `RESULTS_AND_DIAGNOSIS.md` §2 |
| 5 | *(Low value)* Ablation | ~6 h | Signals are inert; would confirm, not inform |

**Why #1 mattered (and what it showed):** Stage-2 DashMamba was fine-tuned on dashcam
data while the three baselines were not, so an examiner would ask what the *architecture*
contributed versus what domain adaptation contributed. The Stage-1 run answers it:
on identical training data the architecture still wins by **+1.79 dB** at σ50, and
fine-tuning accounts for only +0.46 dB of the original +2.25 dB figure.

**If you cannot run anything right now:** you already have a complete, defensible result.
`RESULTS_AND_DIAGNOSIS.md` contains the numbers, the significance tests, the diagnosis,
the claim to make and the three claims to avoid. Stage-1, visuals and the ablation all
strengthen it but none is required for it to stand.

---

## 4. What you need to run anything — files and data

### Two pieces

**A. This repository (~22 MB)** — everything except the frames:

```
models/dashmamba.py        The architecture
checkpoints/
  pretrain_stage1.pt       Stage-1 weights (DAVIS, 28,800 steps)
  finetune_stage2.pt       Stage-2 weights (dashcam, 11,000 steps)
scripts/
  p3_config.py             Paths — reads the THESIS_P3_ROOT env var
  p3_degrade.py            Degradation models (seeded, reproducible)
  p3_evaluate.py           Evaluation harness — PSNR/SSIM/tOF
  dashmamba_wrapper.py     Adapts DashMamba to the harness (blind — takes no σ)
  make_qualitative.py      Before/after stills, grids, split-screen clips
  train_dashmamba.py       Two-stage training driver
  p3_dataloader.py         Dataset over dashcam frames
  pretrain_dataloader.py   DAVIS loader for Stage-1
  package_for_eval.py      Rebuilds the portable data subset
metadata/*.csv             Clip splits, evaluation index
results/                   All baseline + DashMamba Stage-2 results
docs/                      Architecture, weakness analysis, dataset scope, handoff
```

**B. The frame data (~7.16 GB)** — **not in git.**

The full dataset is 257 GB. The evaluation only reads **71 clips × 80 frames**, which was
packaged separately as `dashmamba_eval_data/`. Copy its `dataset/` folder into your clone
so this path exists:

```
<repo>/dataset/_refpool/<clip_stem>/frame_000001.png ...
```

> **Why a subset is valid, not an approximation:** degraded inputs are regenerated
> deterministically from clean frames, seeded per (clip, frame, kind, level). A folder
> holding only the first 80 frames produces **bit-identical scores** to the full dataset.

If the machine already has the full `thesis_p3` dataset, point `THESIS_P3_ROOT` at that
instead and skip the 7.16 GB entirely.

### Deliberately not included

- **`dataset/`** — 257 GB of frames.
- **`external/`** — RVRT, mmagic (BasicVSR++), FastDVDnet are third-party code under their
  own licences (**RVRT is CC-BY-NC**). Only needed if you want to re-run baselines, which
  you should not — their results are already in `results/`. `docs/Phase3_Session_Handoff_2.docx`
  lists the two patches mmagic needs if you ever do.

---

## 5. Setup on a new PC

DashMamba is **pure PyTorch — no custom CUDA extensions.** No compiler, no CUDA toolkit.
(The baselines needed both; they are already evaluated, so you never have to build them.)

```bash
git clone https://github.com/OrneyKraken/dashcam-video-restoration-phase3.git
cd dashcam-video-restoration-phase3
# copy dashmamba_eval_data/dataset/ into this folder

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python numpy requests

# Windows
set THESIS_P3_ROOT=C:\path\to\dashcam-video-restoration-phase3
# Linux / macOS
export THESIS_P3_ROOT=/path/to/dashcam-video-restoration-phase3

cd scripts
```

### Commands

```bash
# Stage-1 evaluation — the fairness comparison (task #1 above)
#   Windows: set THESIS_P3_DASHMAMBA_CKPT=C:\path\to\repo\checkpoints\pretrain_stage1.pt
export THESIS_P3_DASHMAMBA_CKPT=/path/to/repo/checkpoints/pretrain_stage1.pt
python -u p3_evaluate.py --model-import dashmamba_wrapper:restore \
    --tracks A --splits test --stride 2 --max-frames 80 \
    --run-name dashmamba_stage1_track_a_test

# Stage-2 evaluation — already done, but this is the command that produced it
python -u p3_evaluate.py --model-import dashmamba_wrapper:restore \
    --tracks A --splits test --stride 2 --max-frames 80 \
    --run-name dashmamba_stage2_track_a_test

# Before/after visuals
python make_qualitative.py \
    --clips raw_video_008_clip_0000 raw_video_022_clip_0001 raw_video_013_clip_0001 \
    --kind gaussian --level high --frames 40 --models dashmamba
```

> **Never change `--stride` or `--max-frames`.** The baselines were evaluated with exactly
> these settings. Changing them silently invalidates every comparison in this repo.

### Moving a run between machines

`p3_evaluate.py` writes `per_clip.csv` incrementally. To resume anywhere: copy
`results/<run-name>/per_clip.csv` to the new machine and re-issue the **same command**.
Completed clips are skipped.

### Expected runtime elsewhere

On an entry-tier laptop GPU (e.g. RTX 5050, 16 GB), expect roughly **3-4× slower** —
~12-16 h per evaluation. VRAM is not a constraint: evaluation peaks around 3.5 GB.
Avoid *training* on a laptop; Stage-2 is dataloader-bound (PNG decode), so it suffers far
more than the GPU gap suggests.

---

## 6. Reading results

`results/<run-name>/summary.csv` — one row per (kind, level) axis, then a per-lighting
breakdown below a blank line.

- `psnr`, `ssim` — restored quality (higher better)
- `psnr_in` — the degraded input's own score, i.e. the "before" number
- `tof`, `tof_in` — temporal warping error, output and input (**lower** better)

---

## 7. Traps already paid for — do not repeat

1. **Never re-add `low_light` to DashMamba's training mix.** An earlier run included it;
   near-black inputs (~3 dB) dominated the loss and the model learned *only* brightness
   restoration, scoring **+0.01 dB** on the denoising it was being evaluated on — a silent
   identity collapse that looked fine in the training loss. Train a separate checkpoint
   for Track B.
2. **Use `--num-workers 0`** for dashcam training on Windows. Workers spawn and pickle
   ~42 MB/batch through IPC: 1.05 s/batch with 0 workers vs 2.47 s/batch with 4.
3. **Don't zero-initialize both weight and bias** of a layer meant to start near zero — a
   zero weight matrix blocks backprop through that layer entirely.
4. **Don't rely on `CosineAnnealingLR` across a resume** — it's recursive, so restoring
   only optimizer state pins it at its minimum permanently. This project uses a stateless
   closed-form cosine instead.
5. **Watch for saturated sigmoids in estimator heads** — this is exactly how Signal B died.
6. **This machine's antivirus intermittently locks freshly written binaries**, causing git
   "unable to index file" and pip SSL errors. Retrying works.

---

## 8. Documentation index

| File | Contents |
|---|---|
| [`HANDOFF_FOR_NEW_CHAT.md`](HANDOFF_FOR_NEW_CHAT.md) | Full context for a fresh assistant — thesis, history, commands, traps |
| [`RESULTS_AND_DIAGNOSIS.md`](RESULTS_AND_DIAGNOSIS.md) | All results, significance tests, the signal-inactivity diagnosis, what to claim and what not to |
| `docs/Phase3_DashMamba_Architecture.docx` | Design rationale + **verified prior-art positioning**. One citation proposed externally ("MVSSM") could not be verified and appears fabricated — **do not cite it** |
| `docs/Phase3_Baseline_Weakness_Analysis.docx` | Per-lighting findings that motivated the design |
| `docs/Phase3_Baseline_Dataset_Scope.docx` | Exactly which data produced which numbers |
| `docs/Phase3_Session_Handoff_2.docx` | Environment setup and every gotcha encountered |

---

## 9. Licence

Not yet chosen. Note that third-party baseline code is excluded partly because **RVRT is
CC-BY-NC** (non-commercial); keep that in mind when selecting a licence and when writing
reproduction instructions.
