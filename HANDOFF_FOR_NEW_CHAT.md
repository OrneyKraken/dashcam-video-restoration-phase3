# Handoff — read this first

**Last updated:** 2026-09-22, after all evaluation and visual work completed.
**Status: the thesis is complete and defensible. Nothing is half-finished.**

You are picking up a **BRAC University undergraduate CS thesis, Phase 3**. This file is
written for a fresh assistant/chat on a new machine with no prior context. Everything
needed to continue is either in this repo or described below.

---

## 0. Thirty-second summary

A custom blind video-restoration architecture (**DashMamba**, 886,840 parameters) was
designed, implemented, trained in two stages, and benchmarked against three published
baselines on a purpose-built dashcam dataset.

**It wins where it matters, and the win survives the fairness check.** Trained on the
same public data as the baselines, with no dashcam footage and no access to the true
noise level, it beats RVRT — a 14.7× larger model that *is* given the true noise level —
by **+1.79 dB at σ50** (95% CI [+1.05, +2.52], n = 71 clips).

**It also contains a documented negative result.** Direct measurement of the trained
weights shows the two "novel" control signals the architecture was built around never
activated. The gains come from the rest of the network. This is reported honestly, with
root causes and specific fixes, and it explains both places the model loses.

Everything remaining is optional strengthening.

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

## 2. What exists, in the order it was built

| # | Artifact | Where |
|---|---|---|
| 1 | Dataset — 755 clips / 226,236 frames | **not in git** — see §5 |
| 2 | Three baselines benchmarked under one identical protocol | `results/{rvrt,bvrpp,fastdvdnet}_track_a_test/` |
| 3 | Baseline weakness analysis | `docs/Phase3_Baseline_Weakness_Analysis.docx` |
| 4 | DashMamba architecture | `models/dashmamba.py`, `ARCHITECTURE.md` |
| 5 | Two-stage training driver | `scripts/train_dashmamba.py` |
| 6 | Trained checkpoints, both stages (~10 MB each) | `checkpoints/` ✅ **in git** |
| 7 | Stage-2 evaluation + paired significance tests | `results/dashmamba_stage2_track_a_test/` |
| 8 | Signal-inactivity diagnosis | `RESULTS_AND_DIAGNOSIS.md` §2 |
| 9 | **Stage-1 evaluation — the fairness comparison** | `results/dashmamba_stage1_track_a_test/` |
| 10 | Qualitative before/after visuals + chroma analysis | `figures/`, `QUALITATIVE_RESULTS.md` |

Evaluation protocol, applied identically to every model: **Track A test, 71 clips ×
80 frames × 4 degradation axes = 284 clip-runs per model.**

---

## 3. The findings a new chat must not get wrong

### 3.1 Lead with Stage-1, not Stage-2

There are two DashMamba checkpoints and they answer different questions.

| | Trained on | Use it for |
|---|---|---|
| **Stage-1** | DAVIS 2017 only | **The headline comparison.** Same condition as the baselines — no model has seen dashcam data. |
| **Stage-2** | + Track A dashcam fine-tune | Quantifying what domain adaptation adds. **Not** a fair architecture comparison. |

Leading with Stage-2's +2.25 dB invites *"but you fine-tuned on the target domain and
they didn't"*, which is fatal. Stage-1's **+1.79 dB** has no such hole.

Decomposition: +2.25 dB total = **+1.79 dB architecture** + **+0.46 dB fine-tuning.**

### 3.2 The model is blind; the baselines are not

DashMamba estimates its own noise level. RVRT and FastDVDnet are **handed the true σ** —
their public checkpoints are non-blind (see `scripts/rvrt_wrapper.py`). The asymmetry is
*in the baselines' favour* and should be stated explicitly, not buried.

### 3.3 Both proposed signals are dead

Measured directly from the trained checkpoint:

- **Signal B** (blind reliability → Δ): outputs a constant **1.0000**, std 0.0000, at
  every noise level. Its sigmoid is saturated — pre-activation logits average **+44**,
  where the gradient is ~0, so it could never recover.
- **Signal A** (motion confidence → fusion gate): constant **~0.993**, because the flow
  network collapsed to **0.002 px** displacement, so the consistency check always reports
  "perfectly aligned".

**Do not claim the decoupled-signal design works.** It is reported as a diagnosed
negative result — a strength when presented that way. It also explains both weaknesses:

- The **monotonic noise trend** (+1.79 / −0.51 / −1.57 dB as input PSNR rises
  15.5 → 21.0 → 25.1) is exactly what a model that cannot sense noise level produces:
  one fixed denoising strength, right at σ50 and too aggressive at σ15.
- **tOF never beats the baselines**, because nothing in the trained model actually
  targets temporal consistency.

### 3.4 The chroma finding

DashMamba's mean colour error is ~10× *lower* than the baselines' (0.002 vs 0.008–0.026),
but it leaves residual chroma noise (1.18–2.18× oversaturated), while RVRT and FastDVDnet
suppress chroma noise by desaturating (0.69–0.92×). **This is the mechanism behind the
PSNR/SSIM divergence** (+1.79 dB PSNR but −0.041 SSIM at σ50). An earlier draft
attributed that to over-smoothing; the measurements contradict it.

### 3.5 Claims to avoid

- ❌ The decoupled-signal design works.
- ❌ Improved temporal consistency — tOF is worse on 3 of 4 axes, tied on the 4th.
- ❌ A win at medium noise (statistical tie) or low noise (significant **loss**).
- ❌ BasicVSR++ is architecturally bad — its ~16 dB is a **checkpoint domain mismatch**
  (trained for compressed-video artifacts, not sensor noise). It is kept in the tables
  because silently dropping a poor performer would misrepresent the study.
- ❌ Citing **"MVSSM"** — that reference could not be verified and is likely fabricated.
  MambaOFR, EVDM and EVSSM are real and were verified.

---

## 4. Headline numbers

Stage-1 vs. baselines, paired per-clip, n = 71, 95% CI:

| Axis | vs RVRT | vs FastDVDnet | Verdict |
|---|---|---|---|
| gaussian / high (σ50) | **+1.79** [+1.05, +2.52], t=+4.86 | **+1.94** [+1.34, +2.54], t=+6.50 | **Significant win** |
| poisson_gaussian | **+0.81** [+0.38, +1.25], t=+3.70 | **+0.99** [+0.63, +1.35], t=+5.50 | **Significant win** |
| gaussian / medium (σ25) | −0.51, t=−1.51 | −0.29, t=−1.23 | Tie (CI spans 0) |
| gaussian / low (σ15) | **−1.57**, t=−6.66 | **−1.20**, t=−7.57 | Significant loss |

Parameters: DashMamba **886,840** · FastDVDnet 2.48 M · RVRT 13.07 M · BasicVSR++ 44.08 M.

Full tables — SSIM, tOF, per-lighting breakdowns, Stage-2, chroma — in
`RESULTS_AND_DIAGNOSIS.md`. Regenerate every significance figure in seconds with
`python scripts/paired_significance.py`.

---

## 5. What you need on the new PC

### 5.1 From git — clone and you have it

Private repo: `github.com/OrneyKraken/dashcam-video-restoration-phase3`

```
models/dashmamba.py              the architecture
checkpoints/pretrain_stage1.pt   Stage-1 weights (~10 MB)  ✅ committed
checkpoints/finetune_stage2.pt   Stage-2 weights (~10 MB)  ✅ committed
scripts/                         training, evaluation, wrappers, visuals
metadata/                        clip lists and split definitions
results/                         per-clip CSVs for all five runs
figures/                         detail crops (visual proof)
docs/                            the four .docx thesis documents
*.md                             this file + README + RESULTS_AND_DIAGNOSIS + ARCHITECTURE + QUALITATIVE_RESULTS
```

**The checkpoints are committed**, so every reported number can be reproduced without
retraining anything.

### 5.2 Dataset — NOT in git (262 GB total)

You only need **one part** for everything described here:

> ### **Track A, test split** — 71 clips × 80 frames ≈ **7.2 GB**
> This is the entire evaluation set. All 284 clip-runs per model use only this.

| Goal | Data needed | Size |
|---|---|---|
| Reproduce all reported numbers | **Track A test split only** | **~7.2 GB** |
| Re-run Stage-2 fine-tuning | + Track A train split | ~40 GB |
| Re-run Stage-1 pretraining | + DAVIS 2017 (public download) | ~5 GB |
| Tracks B / C | never used — skip | — |

A portable package of exactly these test clips was built by
`scripts/package_for_eval.py` on the original machine:

- `F:\dashmamba_package\dashmamba_eval_data` — 7.16 GB, 71 clips × 80 frames
- `F:\dashmamba_package\dashmamba_eval_core` — 22 MB, code + checkpoints

**Copy those two folders** rather than rebuilding from the 262 GB original. If you must
rebuild from raw footage: `scripts/p3_01_make_splits.py` then
`scripts/p3_02_build_dataset.py`. Splits are deterministic and defined in `metadata/`.

### 5.3 Software

| Requirement | Notes |
|---|---|
| Python 3.10 | 3.12 is installed on the old machine but unused |
| PyTorch + CUDA | any recent build |
| opencv-python, numpy, scipy | metrics and visuals |
| **Only if you re-run RVRT** | MSVC Build Tools (C++ workload), CUDA Toolkit 12.4, `ninja`, `wheel` |

**DashMamba needs no custom CUDA extension and none of the MSVC toolchain.** That burden
is entirely RVRT's — it JIT-compiles a deformable-attention kernel on first use. If you
are only working on DashMamba, ignore that row completely.

---

## 6. What is left to do

Nothing is required. In descending order of value:

| Task | Effort (RTX 4080 S) | Why / why not |
|---|---|---|
| **Fix Signal B and retrain** | ~7 h train + 5 h eval | Highest value. Remove the output sigmoid from `ReliabilityEstimator` (predict log-σ directly), or supervise it against the known σ during Stage-1, where σ *is* available. **Testable prediction: should recover most of the −1.57 dB low-noise deficit.** |
| Fix Signal A | +2 h | Initialise the flow net from pretrained SPyNet, or add a photometric warping loss so alignment is trained directly rather than only implicitly rewarded. |
| Chroma-aware loss | ~7 h | Addresses §3.4 — should close the SSIM gap. |
| Tracks B / C evaluation | ~5 h each | Broadens scope; deliberately skipped under time pressure. |
| Ablation over the three flags | ~6 h | **Low value** — the signals are measurably inert, so it would confirm §3.3, not inform it. |

On a 16 GB RTX 5050 laptop, budget roughly **1.8–2.2×** these times and drop
`DASHMAMBA_MAX_FRAMES` to ~24 to stay inside VRAM.

---

## 7. Running things

### Evaluate DashMamba (no special toolchain needed)

```bash
export THESIS_P3_DASHMAMBA_CKPT=/path/to/repo/checkpoints/pretrain_stage1.pt   # Stage-1
cd scripts
python -u p3_evaluate.py \
    --model-import dashmamba_wrapper:restore \
    --run-name dashmamba_stage1_track_a_test \
    --tracks A --splits test --stride 2 --max-frames 80
```

Run from `scripts/` so `dashmamba_wrapper` is importable. `THESIS_P3_ROOT` is optional:
it defaults to the repo root, so it only needs setting if `dataset/` lives elsewhere.

Point the variable at `finetune_stage2.pt` for Stage-2. On Windows use `set` instead of
`export`. `per_clip.csv` is written incrementally, so the run is **resumable** —
re-running skips clips already done. A full run is ~4.7 h on an RTX 4080 SUPER.

### Regenerate the visuals

```bat
scripts\run_qualitative.bat --clips raw_video_013_clip_0009 raw_video_022_clip_0018 raw_video_021_clip_0024 ^
    --kind gaussian --level high --frames 40 --save-frames 0 20
```

Use the `.bat` wrapper, not the `.py` directly — see traps 5 and 6 below.

### Significance tests

```bash
python scripts/paired_significance.py     # reads results/, prints every reported figure
```

---

## 8. Traps that cost real time — do not rediscover these

1. **The `low_light` training kind causes identity collapse.** Its inputs are ~3 dB, so
   it dominates the loss and the model learns to output its input on the noise tasks
   (+0.01 dB). `TRAIN_KINDS = ("gaussian", "poisson_gaussian")` in
   `scripts/train_dashmamba.py` is deliberate. **Do not add `low_light` back.** Cost
   ~6.5 h of discarded compute to diagnose.
2. **Never `nn.init.zeros_` a layer's final weight matrix.** An all-zero weight makes the
   backward Jacobian zero too, starving every earlier layer of gradient — 40 of 44
   parameter tensors received none. Use `.mul_(0.01)` instead.
3. **`CosineAnnealingLR` is recursive**, so resuming from a checkpoint pins the learning
   rate at its minimum permanently. `train_dashmamba.py` uses a stateless closed-form
   `cosine_lr()`.
4. **Windows DataLoader workers are counterproductive here** — spawn-based IPC made
   training **11× slower**. Use `--num-workers 0`.
5. **RVRT and FastDVDnet collide on the module name `models`** (RVRT ships a `models/`
   package, FastDVDnet a `models.py`). Loading both in one process breaks the second with
   `'models' is not a package`. The metric runs never hit this because each imported a
   single model; `make_qualitative.py` isolates each model in a subprocess.
6. **RVRT's CUDA build needs scratch space on a non-full drive.** If `%TEMP%` is full,
   nvcc dies with `No space left on device`, surfaced only as the useless
   `ninja: build stopped: subcommand failed`. `run_qualitative.bat` redirects
   `TORCH_EXTENSIONS_DIR` and `TMP`/`TEMP`.
7. **The original machine's C: drive was 100% full** (931 GB, with ~790 GB in
   `System Volume Information` / System Restore). Unrelated to the thesis, but it broke
   RVRT builds. On a new machine, simply keep scratch space free.
8. **Git Bash mangles `/c`-style arguments** — use `cmd.exe //c` or absolute paths.

---

## 9. Document map

| File | Contents |
|---|---|
| `README.md` | Project entry point, status, how to run |
| **`HANDOFF_FOR_NEW_CHAT.md`** | **This file — start here** |
| `RESULTS_AND_DIAGNOSIS.md` | All numbers, significance tests, diagnosis, the claim to make and the claims to avoid |
| `ARCHITECTURE.md` | What `models/dashmamba.py` actually builds, component by component, with parameter budget |
| `QUALITATIVE_RESULTS.md` | Visual material, clip-selection rationale, chroma analysis |
| `docs/Phase3_Baseline_Dataset_Scope.docx` | Exactly which data every run used |
| `docs/Phase3_Baseline_Weakness_Analysis.docx` | The measured findings that motivated the design |
| `docs/Phase3_DashMamba_Architecture.docx` | Thesis-body architecture write-up + verified prior art |
| `docs/Phase3_Session_Handoff_2.docx` | Earlier session handoff (superseded by this file) |

---

## 10. If you are a new chat, start here

1. Read §3 (findings you must not get wrong) and §4 (the numbers).
2. Confirm the dataset: you need **Track A test split, ~7.2 GB** (§5.2). Ask the user
   where it is rather than assuming a path.
3. Confirm the checkpoints exist: `checkpoints/pretrain_stage1.pt` and
   `checkpoints/finetune_stage2.pt`.
4. **Do not re-run evaluations to "verify".** The per-clip CSVs are committed and a full
   run is ~4.7 h. Use `scripts/paired_significance.py`, which takes seconds.
5. The likely next request is either **writing the thesis text from these results**, or
   **the Signal B fix** in §6. Both are fully specified above.
