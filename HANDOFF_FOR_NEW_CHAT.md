# Handoff — read this first

You are picking up a **BRAC University undergraduate CS thesis, Phase 3**. This file
is written for a fresh assistant/chat with no prior context. Everything needed to
continue is either in this repo or described below.

---

## 1. What the thesis is

**Goal:** build a video restoration model for dashcam footage, and a benchmark to
evaluate it on. Two contributions:

1. **A new dashcam benchmark** — 755 clips / 226,236 frames of real driving footage
   (Chittagong, Bangladesh; Google Pixel 6), 1280×720 @ 15 fps, manually curated,
   stratified by lighting (day 465 / evening 87 / night 203), with three evaluation
   tracks.
2. **DashMamba** — a custom Mamba (state-space) architecture whose every design choice
   is motivated by a *measured* weakness in existing published models.

**Team:** Shuaib Zulkarnain, Irfan Karim, Ishrak Howlader, Rafidul Islam.
**Supervisor:** Dr. Chowdhury Mofizur Rahman. **Co-supervisor:** Sanjida Tasnim.
**Publication target:** a Q2 image/video-processing journal (candidates discussed:
IET Image Processing, Signal/Image/Video Processing, J. Real-Time Image Processing,
EURASIP JIVP). Re-verify quartiles before submission.

### The three evaluation tracks

| Track | What | Status |
|---|---|---|
| **A** | General denoising — Gaussian (σ15/25/50) + Poisson-Gaussian sensor noise | **The only track evaluated.** All results are Track A |
| **B** | Synthetic low-light (day→night) | Built, **never evaluated** — deliberate scope decision |
| **C** | Real-world night/evening footage, no reference | Built, **never evaluated** — deliberate scope decision |

Tracks B and C being unevaluated is a **known, deliberate** scope decision under time
pressure, documented in `docs/Phase3_Baseline_Dataset_Scope.docx` §9.2. It is not an
oversight — but expect an examiner to ask.

---

## 2. What has been done

### Baselines — complete
Three published models evaluated on Track A under one identical protocol
(71 clips × 80 frames × 4 degradation axes = 284 clip-runs each). Results are in
`results/`. **Do not re-run these.**

| Degradation | Input | RVRT | BasicVSR++ | FastDVDnet |
|---|---|---|---|---|
| gaussian / low (σ15) | 25.14 | 39.13 | 26.48 | 38.75 |
| gaussian / medium (σ25) | 20.96 | 35.53 | 22.42 | 35.31 |
| gaussian / high (σ50) | 15.50 | 28.79 | 16.54 | 28.64 |
| poisson_gaussian / realistic | 23.67 | 35.54 | 24.98 | 35.37 |

*(PSNR in dB. All three are **pretrained-only** — none was fine-tuned on dashcam data.)*

BasicVSR++ scores poorly for a **known reason**: its only public checkpoint is trained
for compressed-video artifact removal, not sensor noise. That is a domain mismatch, not
an architectural verdict — and it is itself a finding, since BasicVSR++ has the most
elaborate alignment machinery of the three.

### Weakness analysis — complete
See `docs/Phase3_Baseline_Weakness_Analysis.docx`. Two findings drive the architecture:

- **Temporal axis:** tOF (frame-to-frame warping error) is worst on **daytime** footage
  for *every* baseline without exception — including the broken one. A motion/scene-
  complexity effect.
- **Spatial axis:** RVRT and FastDVDnet lose 4–5 dB PSNR specifically when severe
  Gaussian noise combines with **evening/night** lighting. A noise×illumination effect,
  orthogonal to motion.

These peak under *opposite* conditions, which is why DashMamba keeps them decoupled.

### DashMamba — designed, implemented, trained
`models/dashmamba.py`. Five stages; two **decoupled** control signals:

- **Signal A** (motion confidence, from forward/backward flow consistency) gates *only*
  the feature-fusion stage.
- **Signal B** (blind noise/exposure reliability, from a small CBDNet-style estimator)
  modulates *only* Δ, the state-space discretization step.

Training complete:

| Stage | Data | Steps | Final loss (ema) | Checkpoint |
|---|---|---|---|---|
| 1 — pretrain | DAVIS 2017 (public) | 28,800 | 0.0197 | `checkpoints/pretrain_stage1.pt` |
| 2 — fine-tune | 360 dashcam train clips | 11,000 | 0.0136 | `checkpoints/finetune_stage2.pt` |

**DashMamba is blind** — it never receives the true noise level. RVRT and FastDVDnet are
both *given* the true σ. This asymmetry favours the baselines and must be stated in the paper.

---

## 3. What still needs running

| # | Task | Time (RTX 4080 SUPER) | Why it matters |
|---|---|---|---|
| 1 | Evaluate **Stage-2** DashMamba | ~5 h | The headline 4-way comparison. **This is the only thing blocking a result.** |
| 2 | Evaluate **Stage-1** DashMamba | ~5 h | Fairness: the baselines are pretrained-only, and Stage-1 DashMamba is likewise public-data-only. Makes it apples-to-apples |
| 3 | Qualitative before/after visuals | ~30 min | `scripts/make_qualitative.py` |
| 4 | Statistical significance over the 71 per-clip values | ~5 min | Cheap; closes a gap examiners probe |
| 5 | Ablation (flags already exist in the model) | 2 cells ≈ 6 h, 5 cells ≈ 16 h | Shows *which* signal earns the gain |

---

## 4. What you need to actually run it

### Two pieces

1. **This repo** (~22 MB) — code, checkpoints, metadata, baseline results, docs.
2. **The frame data** (~7.2 GB) — **NOT in git.** The full dataset is 257 GB; the
   evaluation only reads 71 clips × 80 frames. That subset was packaged separately as
   `dashmamba_eval_data/`. Copy its `dataset/` folder into your clone so this path exists:

```
<repo>/dataset/_refpool/<clip_stem>/frame_000001.png ...
```

**Why a subset is valid:** degraded inputs are regenerated deterministically from clean
frames, seeded per (clip, frame, kind, level). A folder holding only the first 80 frames
gives **bit-identical scores** to the full dataset. This is not an approximation.

If the machine already has the full `thesis_p3` dataset, point `THESIS_P3_ROOT` at that
instead and ignore the 7.2 GB package.

### Setup

DashMamba is **pure PyTorch — no custom CUDA extensions**. No compiler, no CUDA toolkit.
(The baselines needed those; they're already evaluated, so you never have to build them.)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python numpy requests

# Windows
set THESIS_P3_ROOT=C:\path\to\repo
# Linux/Mac
export THESIS_P3_ROOT=/path/to/repo
```

### Commands

From `scripts/`:

```bash
# 1. headline result (Stage-2, fine-tuned)
python -u p3_evaluate.py --model-import dashmamba_wrapper:restore \
    --tracks A --splits test --stride 2 --max-frames 80 \
    --run-name dashmamba_stage2_track_a_test

# 2. fair comparison (Stage-1, public data only)
#    Windows: set THESIS_P3_DASHMAMBA_CKPT=C:\path\to\repo\checkpoints\pretrain_stage1.pt
python -u p3_evaluate.py --model-import dashmamba_wrapper:restore \
    --tracks A --splits test --stride 2 --max-frames 80 \
    --run-name dashmamba_stage1_track_a_test

# 3. before/after visuals
python make_qualitative.py --clips raw_video_008_clip_0000 raw_video_022_clip_0001 \
    --kind gaussian --level high --frames 40 --models dashmamba
```

> **Never change `--stride` or `--max-frames`.** The baselines were run with exactly
> these settings. Changing them silently invalidates the entire comparison.

### Resuming an interrupted run

`p3_evaluate.py` writes `per_clip.csv` incrementally. To resume — even on a different
machine — copy `results/<run-name>/per_clip.csv` across and re-issue the **same command**.
It skips completed clips.

---

## 5. Reading the results

`results/<run-name>/summary.csv`: one row per (kind, level) axis, then a per-lighting
breakdown below a blank line.

- `psnr`, `ssim` — restored quality (higher better)
- `psnr_in` — the degraded input's own score, i.e. the "before" number
- `tof`, `tof_in` — temporal warping error, output and input (**lower** better)

Compare against `results/rvrt_track_a_test/`, `results/bvrpp_track_a_test/`,
`results/fastdvdnet_track_a_test/`.

---

## 6. Traps that have already cost time — do not repeat

1. **Never re-add `low_light` to DashMamba's training mix.** An earlier run trained on
   gaussian + poisson_gaussian + low_light. Low-light inputs are near-black (~3 dB PSNR),
   so their loss dominated; the model learned *only* brightness restoration and scored
   **+0.01 dB** on the denoising it was being evaluated on — a silent identity collapse
   that looked fine in the training loss. Train a **separate** checkpoint for Track B.
2. **Use `--num-workers 0`** for dashcam-data training on Windows. DataLoader workers use
   spawn and pickle ~42 MB per batch through IPC: measured 1.05 s/batch with 0 workers vs
   2.47 s/batch with 4. More workers is *slower*.
3. **Don't zero-initialize both weight and bias** of a layer you want to start near-zero.
   A fully-zero weight matrix blocks backprop through that layer entirely (the backward
   Jacobian *is* the weight matrix), starving everything upstream of gradient.
4. **Don't rely on `CosineAnnealingLR` across a resume.** It's recursive (next lr computed
   from current lr), so restoring only optimizer state pins it permanently at its minimum.
   The training script uses a stateless closed-form cosine instead.
5. **This machine's antivirus intermittently locks freshly written binaries**, causing
   git "unable to index file" and pip SSL errors. Retrying works.

---

## 7. Honest status

- **Enough for an undergrad defense?** Likely yes, once task 1 above is done — the dataset,
  the benchmark, the measured weakness analysis, and a custom architecture is well above
  typical scope.
- **Enough for a Q2 journal?** Not yet. Needs the full 5-cell ablation, and either
  fine-tuned baselines or the Stage-1 fairness framing argued carefully. Roughly 45 h more.
- **Is DashMamba actually better?** **Unknown.** Training converged well, but that only
  proves it learned the objective — not that it beats RVRT or FastDVDnet. Task 1 answers
  this. If it loses, a negative result with a thorough post-mortem is still legitimate
  thesis work; frame it deliberately rather than being surprised by it.

---

## 8. Other documents

- `README.md` — repo layout and quick start
- `docs/Phase3_DashMamba_Architecture.docx` — design rationale, **verified prior-art
  positioning** (which novelty claims are defensible and which are not — read §6 and §8
  before writing the paper; one citation proposed by an external document, "MVSSM",
  could not be verified and appears fabricated — **do not cite it**)
- `docs/Phase3_Baseline_Weakness_Analysis.docx` — the per-lighting findings
- `docs/Phase3_Baseline_Dataset_Scope.docx` — exactly which data produced which numbers
- `docs/Phase3_Session_Handoff_2.docx` — environment setup and every gotcha encountered
