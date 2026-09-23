# DashMamba — architecture

The implementation of record is [`models/dashmamba.py`](models/dashmamba.py) (411 lines,
self-contained, no custom CUDA). This document describes what that file actually builds,
not what was originally designed — where the two differ, the code wins and the difference
is flagged.

A prose/figure version for the thesis body is in
[`docs/Phase3_DashMamba_Architecture.docx`](docs/Phase3_DashMamba_Architecture.docx).

---

## 1. Design motivation

Two findings from the baseline weakness analysis drove the design
(`docs/Phase3_Baseline_Weakness_Analysis.docx`):

1. **Daytime footage is the hardest for temporal consistency across every baseline** —
   counter-intuitively, not night. Fast ego-motion and dense moving traffic break
   frame-to-frame alignment.
2. **RVRT and FastDVDnet lose 4–5 dB when severe noise meets evening/night lighting**,
   and both are *non-blind* — they must be handed the true noise σ.

So the model targets: (a) motion-aware temporal fusion that degrades gracefully when
alignment fails, and (b) fully **blind** operation, estimating its own noise level.

These become **two decoupled control signals**. The decoupling is the claimed novelty:
each signal enters the network at exactly one place, and they never mix.

| | Signal A — motion confidence | Signal B — blind reliability |
|---|---|---|
| Estimates | is temporal alignment trustworthy here? | how noisy is this pixel? |
| Computed from | forward/backward optical-flow consistency | raw input frame, CBDNet-style FCN |
| Enters the network at | **the fusion gate** in the reconstruction head | **Δ (dt) only** inside the selective scan |
| Never touches | Δ, B, C | B, C, the fusion gate |

> ⚠️ **Measurement shows neither signal activates in the trained model.** See §6 and
> `RESULTS_AND_DIAGNOSIS.md` §2. The architecture description below is what the code
> builds; the diagnosis is what it actually learned to do.

---

## 2. Data flow

```
clip (B, T, 3, H, W) in [0,1]
  │
  ├─► FeatureExtractor              conv stride-2 ×2 + 5 ResBlocks
  │     └─► feats (B, T, 64, H/4, W/4)
  │
  ├─► CoarseToFineFlowNet           per non-reference frame i, vs reference t//2
  │     ├─► flow_to_ref   ──► warp(feats[i]) ──► aligned
  │     └─► flow_from_ref ──► flow_consistency_confidence ──► GATE   ◄── Signal A
  │
  ├─► ReliabilityEstimator(raw frame)  ──► avg_pool2d(·,4) ──► RELIABILITY  ◄── Signal B
  │
  ├─► TemporalStack (2 × BidirectionalMambaCore)
  │     └─► SelectiveScan1D(aligned, RELIABILITY)   ── Signal B feeds Δ only
  │           forward scan + backward scan, fused
  │     └─► temporal (B, T, 64, H/4, W/4)
  │
  └─► ReconstructionHead
        blended = GATE · temporal + (1 − GATE) · aligned     ── Signal A used here only
        conv → PixelShuffle ×2 → PixelShuffle ×2 → conv
        return  frame + residual                             ── global residual
```

Output: `(B, T, 3, H, W)`. Every frame in the clip is restored, not just the reference.

---

## 3. Components

### FeatureExtractor — 408,000 params (46.0%)
Two stride-2 convolutions to `H/4 × W/4`, then 5 residual blocks at 64 channels.
Downsampling matters disproportionately here: the selective scan folds **every spatial
position into the batch dimension**, so its cost scales with `H·W`. Operating at 1/16
resolution is what makes the temporal core affordable.

### CoarseToFineFlowNet — 93,444 params (10.5%)
Estimates flow at 1/2 feature resolution, upsamples ×2 and refines. Flow is applied with
`warp()` (bilinear `grid_sample`). Run **twice** per non-reference frame — `i→ref` for
alignment and `ref→i` for the consistency check that produces Signal A.

`flow_consistency_confidence()` composes the two flows; where the round trip returns to
its starting point, alignment is trustworthy and confidence → 1.

### ReliabilityEstimator — 5,233 params (0.6%)
Four 3×3 convolutions at 16 hidden channels, sigmoid output, applied to the **raw input
frame** rather than extracted features, so it sees the true sensor signal before feature
extraction can mask it. Trained **purely through the reconstruction loss** — no σ
supervision at any point. This is what makes the model blind.

### SelectiveScan1D / BidirectionalMambaCore / TemporalStack — 46,080 params (5.2%)
The S6 recurrence, applied along time independently per spatial location:

```
h_t = exp(Δ_t · A) · h_{t−1} + (Δ_t · B_t · x_t)
y_t = C_t · h_t + D · x_t
```

with `A = −exp(A_log)` (negative, for stability), `state_dim = 16`.

The decoupling is enforced by the projection signatures:

```python
self.to_dt = nn.Linear(channels + 1, channels)   # x + Signal B (reliability)
self.to_B  = nn.Linear(channels, state_dim)      # content only
self.to_C  = nn.Linear(channels, state_dim)      # content only
nn.init.constant_(self.to_dt.bias, -2.0)         # start with small dt = slow forgetting
```

Only `to_dt` receives the reliability channel. `Δ` controls how fast state is forgotten,
so the intended behaviour is: noisy pixel → low reliability → small Δ → longer temporal
memory → more aggressive averaging across frames.

`BidirectionalMambaCore` runs one forward and one time-reversed scan and fuses them, so
each frame sees both past and future. `TemporalStack` stacks 2 such blocks.

### ReconstructionHead — 334,083 params (37.7%)
Fuses by Signal A, then two `PixelShuffle(2)` stages back to full resolution, and adds
the result to the input frame as a **global residual** — the network predicts only what
to change.

---

## 4. Configuration

| Parameter | Value |
|---|---|
| `mid_channels` | 64 |
| `num_res_blocks` | 5 |
| `state_dim` | 16 |
| `num_mamba_blocks` | 2 |
| **Total parameters** | **886,840** |

Against the baselines: FastDVDnet 2.48 M (2.8×), RVRT 13.07 M (14.7×),
BasicVSR++ 44.08 M (49.7×).

### Ablation flags

`DashMambaNet` takes three booleans, wired for an ablation that was never run because
the signals were measured inert:

| Flag | `False` behaviour |
|---|---|
| `use_flow_align` | no alignment at all; gate forced to 0.5 (also disables `use_motion_gate`) |
| `use_motion_gate` | alignment kept, but gate forced to a constant 0.5 |
| `use_reliability_delta` | reliability forced to 0 — Δ becomes content-only |

They are settable through the wrapper without editing code:
`DASHMAMBA_USE_FLOW_ALIGN`, `DASHMAMBA_USE_MOTION_GATE`, `DASHMAMBA_USE_RELIABILITY_DELTA`.

---

## 5. Training

Two stages, both Charbonnier loss, both driven by `scripts/train_dashmamba.py`.

| | Stage 1 | Stage 2 |
|---|---|---|
| Data | DAVIS 2017 (public) | Track A dashcam train split |
| Steps | 28,800 | 11,000 |
| Checkpoint | `checkpoints/pretrain_stage1.pt` | `checkpoints/finetune_stage2.pt` |
| Purpose | general restoration prior | domain adaptation |

**Stage-1 is the checkpoint that makes the baseline comparison fair** — it has seen no
dashcam data, exactly like the three pretrained baselines.

Three implementation details that are load-bearing:

```python
TRAIN_KINDS = ("gaussian", "poisson_gaussian")
```
Training originally included a `low_light` degradation. Its inputs are ~3 dB, so it
dominated the loss and the model collapsed to a near-identity function on the noise
tasks (+0.01 dB). Dropping it was the single most important fix in the project.

```python
self.net[-1].weight.data.mul_(0.01)   # NOT zeros_
nn.init.zeros_(self.net[-1].bias)
```
The flow head is initialised near-zero, not exactly zero: an all-zero weight matrix makes
the backward Jacobian through the layer zero too, starving every earlier layer of
gradient. With `zeros_`, only 40 of 44 parameter tensors received gradient.

```python
def cosine_lr(step, base_lr, total_steps, eta_min=1e-6): ...
```
A stateless closed-form schedule. PyTorch's `CosineAnnealingLR` is recursive, so resuming
from a checkpoint pins the LR at its minimum permanently.

---

## 6. What the trained model actually learned

Measured directly from `checkpoints/finetune_stage2.pt`:

| Signal | Intended | Measured |
|---|---|---|
| **B** (reliability) | varies with noise level | constant **1.0000**, std 0.0000, at every σ — sigmoid saturated, pre-activations ≈ **+44** |
| **A** (motion confidence) | varies with alignment quality | constant **~0.993**, flow magnitude **0.002 px** — the flow network never learned motion |

Both are inert. The measured gains therefore come from the parts that *did* train — the
CNN encoder, the bidirectional selective-scan core, and the residual head — which is
still a meaningful architectural result, since Stage-1 beats RVRT by **+1.79 dB** at σ50
on identical training data with 14.7× fewer parameters and no access to σ.

Root causes and the specific fixes to try are in `RESULTS_AND_DIAGNOSIS.md` §2. The
highest-value single experiment: remove the output sigmoid from `ReliabilityEstimator`
(predict log-σ directly), which should recover most of the −1.57 dB low-noise deficit.

---

## 7. Prior art

Verified to exist: **MambaOFR**, **EVDM**, **EVSSM** (video restoration with state-space
models). A citation for "MVSSM" appeared in an externally-supplied draft and **could not
be verified — do not cite it.** Details in `docs/Phase3_DashMamba_Architecture.docx` §8.

The novelty claim is the *decoupling* of two control signals into disjoint entry points
(Δ vs. fusion gate), combined with fully blind operation. Given §6, the honest framing is
that this was **proposed and tested, and the mechanism did not activate** — reported with
a diagnosis rather than quietly dropped.
