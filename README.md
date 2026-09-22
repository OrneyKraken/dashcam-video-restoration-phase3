# Dashcam Video Restoration — Phase 3 (DashMamba)

BRAC University undergraduate CS thesis, Phase 3. A Mamba-based video restoration
architecture for dashcam footage, plus a benchmark of three published baselines on a
new 755-clip dashcam dataset.

**Status: training complete, evaluation in progress.** See [What still needs running](#what-still-needs-running).

---

## What this project contains

| Contribution | State |
|---|---|
| **Dataset** — 755 clips / 226,236 frames of real dashcam footage (Chittagong, Bangladesh), 1280×720 @ 15fps, curated and stratified by lighting, three evaluation tracks | Built (frames not in this repo — see below) |
| **Baseline benchmark** — RVRT, BasicVSR++, FastDVDnet evaluated on Track A under one protocol | Complete, results in `results/` |
| **Weakness analysis** — per-lighting breakdown identifying where baselines fail | Complete, see `docs/` |
| **DashMamba** — custom architecture, each component motivated by a measured weakness | Trained; evaluation pending |

---

## Repository layout

```
models/dashmamba.py        The architecture (DashMambaNet)
scripts/
  p3_config.py             Paths — reads THESIS_P3_ROOT env var
  p3_degrade.py            Degradation models (single source of truth, seeded/reproducible)
  p3_dataloader.py         Training/eval datasets over the dashcam frames
  pretrain_dataloader.py   DAVIS loader for Stage-1
  p3_evaluate.py           Evaluation harness — PSNR/SSIM/tOF, pluggable models
  train_dashmamba.py       Two-stage training driver
  dashmamba_wrapper.py     Adapts DashMamba to the evaluation harness
  make_qualitative.py      Before/after stills, comparison grids, split-screen clips
  package_for_eval.py      Builds a portable evaluation bundle for another machine
metadata/*.csv             Clip splits, evaluation index, dataset summary
results/                   Baseline results (per-clip CSVs + summaries)
checkpoints/               Trained DashMamba weights
docs/                      Architecture, weakness analysis, dataset scope, session handoff
```

### Deliberately **not** in this repo

- **`dataset/`** — 262 GB of frames. The evaluation only reads 71 clips × 80 frames
  (~7.5 GB); use `scripts/package_for_eval.py` to extract exactly that subset.
- **`external/`** — RVRT, mmagic (BasicVSR++) and FastDVDnet are third-party code under
  their own licences (**RVRT is CC-BY-NC**). Clone them yourself; the session handoff
  doc lists the two patches mmagic needs (trimmed `__init__` imports to avoid a
  diffusers/transformers dependency, and an `np.bool8` fix for modern NumPy).

---

## Running the evaluation

DashMamba is **pure PyTorch — no custom CUDA extensions**, so no compiler or CUDA
toolkit is needed (the baselines did need those; they are already evaluated).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python numpy requests

export THESIS_P3_ROOT=/path/to/this/repo      # Windows: set THESIS_P3_ROOT=C:\path\to\repo
cd scripts
```

**Headline result** (Stage-2, fine-tuned on dashcam data):

```bash
python -u p3_evaluate.py --model-import dashmamba_wrapper:restore \
    --tracks A --splits test --stride 2 --max-frames 80 \
    --run-name dashmamba_stage2_track_a_test
```

**Fair comparison** (Stage-1, public data only — matching the baselines' pretrained-only condition):

```bash
export THESIS_P3_DASHMAMBA_CKPT=/path/to/repo/checkpoints/pretrain_stage1.pt
python -u p3_evaluate.py --model-import dashmamba_wrapper:restore \
    --tracks A --splits test --stride 2 --max-frames 80 \
    --run-name dashmamba_stage1_track_a_test
```

> **Do not change `--stride` or `--max-frames`.** The baselines were evaluated with
> exactly these settings (71 clips × 80 frames × 4 degradation axes). Changing them
> invalidates the comparison.

Runtime is ~5 h on an RTX 4080 SUPER. The harness writes `per_clip.csv` incrementally,
so an interrupted run resumes by re-issuing the same command.

### Visual before/after material

```bash
python make_qualitative.py \
    --clips raw_video_008_clip_0000 raw_video_022_clip_0001 raw_video_013_clip_0001 \
    --kind gaussian --level high --frames 40
```

Writes per-frame stills (clean / noisy / each model), labelled side-by-side comparison
grids, zoomed detail crops, and split-screen before|after MP4s. Include `--models dashmamba`
to skip the baselines if their code isn't set up.

---

## Baseline reference numbers

Track A test, 71 clips × 80 frames, **pretrained-only (no fine-tuning)**, PSNR in dB:

| Degradation | Input | RVRT | BasicVSR++ | FastDVDnet |
|---|---|---|---|---|
| gaussian / low (σ15) | 25.14 | 39.13 | 26.48 | 38.75 |
| gaussian / medium (σ25) | 20.96 | 35.53 | 22.42 | 35.31 |
| gaussian / high (σ50) | 15.50 | 28.79 | 16.54 | 28.64 |
| poisson_gaussian / realistic | 23.67 | 35.54 | 24.98 | 35.37 |

BasicVSR++ scores poorly for a known reason: its only public checkpoint is trained for
**compressed-video artifact removal**, not sensor noise. That is a domain mismatch, not
an architectural verdict — and it is itself a finding, since BasicVSR++ has the most
elaborate alignment machinery of the three.

---

## Things that will bite you

- **DashMamba runs blind.** It never receives the true noise level. RVRT and FastDVDnet
  are both *given* the true σ. The asymmetry favours the baselines and should be stated
  explicitly in any write-up.
- **Never re-add `low_light` to DashMamba's training mix.** An earlier run trained on
  gaussian + poisson_gaussian + low_light. Low-light inputs are near-black (~3 dB), so
  their loss dominated and the model learned *only* brightness restoration — scoring
  +0.01 dB on the denoising it was being evaluated on, i.e. a silent identity collapse.
  Train a separate checkpoint if Track B (low-light) is wanted.
- **Use `--num-workers 0` for dashcam-data training on Windows.** DataLoader workers use
  spawn and pickle ~42 MB per batch through IPC; measured 1.05 s/batch with 0 workers vs
  2.47 s/batch with 4.
- **Track B and Track C were never evaluated.** Deliberate scope decision under time
  constraints, documented in `docs/`. Not an oversight, but expect the question.

---

## What still needs running

1. Evaluate Stage-2 DashMamba → the 4-way comparison (~5 h)
2. Evaluate Stage-1 DashMamba → the fair, pretrained-only comparison (~5 h)
3. Qualitative visuals (~30 min)
4. Statistical significance testing over the 71 per-clip values (~5 min)
5. Ablation study — the architecture exposes `use_flow_align`, `use_motion_gate`,
   `use_reliability_delta` for exactly this (2 cells ≈ 6 h, full 5 cells ≈ 16 h)

---

## Documentation

- `docs/Phase3_DashMamba_Architecture.docx` — design, component-by-component rationale,
  verified prior-art positioning (which claims are defensible and which are not)
- `docs/Phase3_Baseline_Weakness_Analysis.docx` — per-lighting findings driving the design
- `docs/Phase3_Baseline_Dataset_Scope.docx` — exactly which data produced which numbers
- `docs/Phase3_Session_Handoff_2.docx` — environment setup, every gotcha and its fix

---

## Licence

Not yet chosen. Note that third-party baseline code is **not** included here partly
because RVRT is CC-BY-NC (non-commercial); keep that in mind when selecting a licence
and when describing reproduction steps.
