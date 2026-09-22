# Results and diagnosis — DashMamba vs. three baselines

Track A test, 71 clips × 80 frames × 4 degradation axes (284 clip-runs per model).
All numbers are means over the 71 clips. Baselines are **pretrained-only**;
DashMamba here is **Stage-2 (fine-tuned on dashcam data)** — see the fairness caveat.

---

## 1. Headline results

### PSNR (dB) — higher is better

| Axis | Input | **DashMamba** | RVRT | BasicVSR++ | FastDVDnet |
|---|---|---|---|---|---|
| gaussian / high (σ50) | 15.50 | **31.04** | 28.79 | 16.54 | 28.64 |
| gaussian / medium (σ25) | 20.96 | 35.40 | 35.53 | 22.42 | 35.31 |
| gaussian / low (σ15) | 25.14 | 37.77 | **39.13** | 26.48 | 38.75 |
| poisson_gaussian / realistic | 23.67 | **36.52** | 35.54 | 24.98 | 35.37 |

### Statistical significance (paired per-clip, n = 71)

| Axis | vs RVRT | vs FastDVDnet | Verdict |
|---|---|---|---|
| gaussian / high | **+2.25** dB, 95% CI [+1.46, +3.04], t = +5.58, wins 46/71 | **+2.40** dB [+1.75, +3.05], t = +7.20, wins 51/71 | **Significant win** |
| poisson_gaussian | **+0.98** dB [+0.55, +1.42], t = +4.41, wins 46/71 | **+1.16** dB [+0.80, +1.52], t = +6.36, wins 50/71 | **Significant win** |
| gaussian / medium | −0.13 dB [−0.85, +0.59], t = −0.35 | +0.09 dB [−0.43, +0.61], t = +0.35 | **Tie** (CI spans zero) |
| gaussian / low | **−1.36** dB [−1.90, −0.83], t = −5.00, wins 17/71 | **−0.99** dB [−1.36, −0.61], t = −5.10, wins 18/71 | **Significant loss** |

Note the win rates: even where DashMamba wins significantly it does so on ~65% of clips,
not all of them.

### SSIM — roughly tied

| Axis | DashMamba | RVRT | FastDVDnet |
|---|---|---|---|
| gaussian / high | 0.825 | **0.838** | 0.828 |
| gaussian / medium | 0.921 | 0.914 | 0.914 |
| gaussian / low | 0.945 | **0.949** | 0.949 |
| poisson_gaussian | **0.940** | 0.921 | 0.920 |

The +2.25 dB PSNR win at σ50 comes with −0.013 SSIM: more pixel-accurate, not more
structurally faithful.

### tOF temporal consistency — lower is better

| Axis | Input | DashMamba | RVRT | FastDVDnet |
|---|---|---|---|---|
| gaussian / high | 49.00 | 5.59 | 4.99 | **4.82** |
| gaussian / medium | 18.79 | 6.07 | 5.82 | **5.76** |
| gaussian / low | 11.13 | 6.24 | 6.05 | **6.02** |
| poisson_gaussian | 13.46 | 6.04 | 6.10 | **6.02** |

**DashMamba is worse than both working baselines on every axis.** It improves enormously
over the degraded input (49.0 → 5.6), but does not beat RVRT or FastDVDnet — despite
temporal consistency being the weakness the architecture was explicitly designed to fix.

### Efficiency — unambiguous win

| Model | Parameters | Relative |
|---|---|---|
| **DashMamba** | **886,840** | 1× |
| FastDVDnet | 2,483,578 | 2.8× |
| RVRT | 13,065,453 | **14.7×** |
| BasicVSR++ | 44,075,638 | 49.7× |

---

## 2. Diagnosis — both proposed signals are inactive

The architecture's stated novelty was **two decoupled control signals**. Direct
measurement of the trained Stage-2 checkpoint shows **neither is functioning.**

### Signal B (blind reliability → Δ modulation) — saturated

| True σ | Mean reliability output | Std |
|---|---|---|
| 15 | 1.0000 | 0.0000 |
| 25 | 1.0000 | 0.0000 |
| 50 | 1.0000 | 0.0000 |
| poisson (mixed) | 1.0000 | 0.0000 |

Pre-sigmoid logits average **+44** (range +5 to +56). A sigmoid saturates above roughly
+10, where its gradient is ~0 — so once training pushed the pre-activations there, the
estimator could never recover. It emits a hard constant 1.0 carrying **zero information
about noise level**.

### Signal A (motion confidence → fusion gate) — near-constant

| Frame | Gate mean | Gate std | Mean flow magnitude |
|---|---|---|---|
| all non-reference | 0.9928–0.9930 | ~0.004 | **0.002 px** |

The flow network never learned meaningful motion (magnitudes of 0.002 pixels are
effectively zero), so the forward/backward consistency check always reports "perfectly
consistent" and the gate sits at a constant ~0.99. It is not an adaptive trust signal.

### What follows from this

1. **The +2.25 dB win at severe noise is real, but is not produced by the proposed
   mechanisms.** It comes from the remaining architecture: the CNN encoder, the stacked
   bidirectional selective-scan temporal core, and the residual reconstruction head.
2. **The low-noise loss is explained.** Signal B cannot distinguish σ15 from σ50, so the
   model cannot denoise gently when it should, and over-smooths mild noise.
3. **The tOF shortfall is explained.** Signal A provides no motion-aware fusion, so no
   mechanism in the trained model actually targets temporal consistency.
4. **An ablation over these flags would show ~no effect**, because the signals are already
   inert. Running it would confirm this, not contradict it.

### Root causes and fixes (future work)

| Signal | Root cause | Proposed fix |
|---|---|---|
| B | Sigmoid saturation; dead gradient | Remove the output sigmoid (predict log-σ directly), or add normalization / clamp pre-activations; optionally supervise it against the known σ during Stage-1 training, where the ground-truth noise level *is* available |
| A | Flow network collapsed to ~zero displacement; no supervision pushes it to learn motion | Initialize from a pretrained flow network (SPyNet), or add a photometric warping loss so alignment is directly trained rather than only implicitly rewarded |

---

## 3. Honest claim to make in the paper

> A blind video restoration model with 886,840 parameters — no ground-truth noise level,
> 14.7× smaller than RVRT — achieves statistically significant gains over σ-informed
> baselines at severe noise (+2.25 dB at σ50, +0.98 dB on realistic Poisson-Gaussian
> sensor noise), at the cost of −1.36 dB at mild noise, with comparable but not superior
> temporal consistency. Direct measurement of the trained weights shows the two proposed
> adaptive signals did not activate; the observed gains therefore derive from the
> bidirectional selective-scan temporal core rather than from the adaptive mechanisms,
> and the failure modes are diagnosed rather than left unexplained.

### What must NOT be claimed

- Do not claim the decoupled signal design works — measurement says otherwise in this checkpoint.
- Do not claim improved temporal consistency — tOF is worse than both working baselines.
- Do not present the Stage-2 result as a pure architecture comparison — DashMamba was
  fine-tuned on dashcam data and the baselines were not. The Stage-1 evaluation
  (pretrained-only, matching the baselines' condition) is the fair comparison.

---

## 4. Caveats

- **n = 71 clips**, stride-2 sampled across the 5 source videos of the Track A test split;
  80 of ~300 frames per clip. Documented in `docs/Phase3_Baseline_Dataset_Scope.docx`.
- **Track A only.** Tracks B (synthetic low-light) and C (real-world night) were never
  evaluated — a deliberate scope decision under time constraints.
- **Fairness.** Stage-2 DashMamba is fine-tuned on the target domain; the three baselines
  are pretrained-only. The Stage-1 evaluation exists to remove this confound.
- **BasicVSR++'s poor numbers** reflect a checkpoint domain mismatch (trained for
  compressed-video artifact removal, not sensor noise), not an architectural verdict.

---

## 5. Reproducing

See `HANDOFF_FOR_NEW_CHAT.md`. Per-clip CSVs for every run are under `results/`; the
significance figures above are paired per-clip comparisons computed from those files.
