# Results and diagnosis — DashMamba vs. three baselines

Track A test, 71 clips × 80 frames × 4 degradation axes (284 clip-runs per model).
All numbers are means over the same 71 clips, scored by the identical protocol.

Two DashMamba checkpoints are reported:

| | Trained on | Compare against baselines? |
|---|---|---|
| **Stage-1** | DAVIS 2017 public data only | **Yes — this is the fair comparison.** Matches the baselines' condition exactly: none of the four models has seen dashcam footage. |
| **Stage-2** | Stage-1 + fine-tuned on Track A train split | No. Has a target-domain advantage the baselines lack. Reported to quantify what fine-tuning adds. |

Both are **blind**: they estimate their own noise level. RVRT and FastDVDnet are
handed the **true σ**. The asymmetry is in the baselines' favour.

---

## 1. Headline results

### PSNR (dB) — higher is better

| Axis | Input | **DashMamba S1** (fair) | DashMamba S2 (ft) | RVRT | BasicVSR++ | FastDVDnet |
|---|---|---|---|---|---|---|
| gaussian / high (σ50) | 15.50 | **30.58** | 31.04 | 28.79 | 16.54 | 28.64 |
| gaussian / medium (σ25) | 20.96 | 35.02 | 35.40 | 35.53 | 22.42 | 35.31 |
| gaussian / low (σ15) | 25.14 | 37.56 | 37.77 | **39.13** | 26.48 | 38.75 |
| poisson_gaussian / realistic | 23.67 | **36.35** | 36.52 | 35.54 | 24.98 | 35.37 |

### Statistical significance — Stage-1 vs. baselines (the fair test)

Paired per-clip, n = 71, 95% CI, t-test on per-clip differences.

| Axis | vs RVRT | vs FastDVDnet | Verdict |
|---|---|---|---|
| gaussian / high | **+1.79** dB [+1.05, +2.52], t = +4.86, wins 46/71 | **+1.94** dB [+1.34, +2.54], t = +6.50, wins 51/71 | **Significant win** |
| poisson_gaussian | **+0.81** dB [+0.38, +1.25], t = +3.70, wins 46/71 | **+0.99** dB [+0.63, +1.35], t = +5.50, wins 49/71 | **Significant win** |
| gaussian / medium | −0.51 dB [−1.18, +0.16], t = −1.51 | −0.29 dB [−0.76, +0.18], t = −1.23 | **Tie** (both CIs span zero) |
| gaussian / low | **−1.57** dB [−2.04, −1.10], t = −6.66, wins 13/71 | **−1.20** dB [−1.51, −0.88], t = −7.57, wins 10/71 | **Significant loss** |

**This is the result to lead with.** With every model trained only on public data,
a 886,840-parameter blind model beats a 13.1 M-parameter σ-informed transformer by
1.79 dB at σ50 and 0.81 dB on realistic Poisson-Gaussian sensor noise.

Note the win rates: even the significant wins are won on ~65% of clips, not all.

### What fine-tuning contributes (Stage-2 − Stage-1, paired, n = 71)

| Axis | Gain from fine-tuning | t | Wins |
|---|---|---|---|
| gaussian / high | **+0.46** dB [+0.38, +0.54] | +12.03 | 68/71 |
| gaussian / medium | **+0.38** dB [+0.31, +0.45] | +10.63 | 64/71 |
| gaussian / low | **+0.21** dB [+0.12, +0.30] | +4.64 | 57/71 |
| poisson_gaussian | **+0.17** dB [+0.14, +0.20] | +10.00 | 60/71 |

Fine-tuning helps consistently and significantly, but modestly — **+0.17 to +0.46 dB**.
It is largest where noise is worst. The architectural gain (+1.79 dB) is roughly
**four times** the domain-adaptation gain (+0.46 dB) at σ50. The win is not an
artifact of having trained on the target domain.

### Stage-2 vs. baselines (for completeness — NOT the fair test)

| Axis | vs RVRT | vs FastDVDnet |
|---|---|---|
| gaussian / high | +2.25 dB [+1.46, +3.04], t = +5.58 | +2.40 dB [+1.75, +3.05], t = +7.20 |
| poisson_gaussian | +0.98 dB [+0.55, +1.42], t = +4.41 | +1.16 dB [+0.80, +1.52], t = +6.36 |
| gaussian / medium | −0.13 dB (tie) | +0.09 dB (tie) |
| gaussian / low | −1.36 dB [−1.90, −0.83], t = −5.00 | −0.99 dB [−1.36, −0.61], t = −5.10 |

### The noise-level trend — the central failure mode

Stage-1 advantage over RVRT, plotted against how degraded the input is:

| Input PSNR | Axis | S1 − RVRT |
|---|---|---|
| 15.50 (worst) | gaussian/high | **+1.79** |
| 20.96 | gaussian/medium | −0.51 |
| 25.14 (mildest) | gaussian/low | **−1.57** |

Monotonic. The model applies one fixed denoising strength: correct at σ50,
too aggressive at σ15. §2 shows exactly why — the noise-estimation signal is dead,
so the model *cannot* modulate its strength.

### SSIM — mixed

| Axis | DashMamba S1 | RVRT | FastDVDnet | S1 − RVRT |
|---|---|---|---|---|
| gaussian / high | 0.796 | **0.838** | 0.828 | −0.041 (t = −3.7) |
| gaussian / medium | 0.911 | 0.914 | 0.914 | −0.003 (ns) |
| gaussian / low | 0.940 | **0.949** | 0.949 | −0.009 (t = −2.8) |
| poisson_gaussian | **0.937** | 0.921 | 0.920 | **+0.016** (t = +5.4) |

At σ50 the +1.79 dB PSNR win comes with −0.041 SSIM: **more pixel-accurate, less
structurally faithful** — consistent with over-smoothing. On realistic sensor noise
DashMamba wins both metrics, which is the strongest single cell in the study.

### tOF temporal consistency — lower is better

| Axis | Input | DashMamba S1 | DashMamba S2 | RVRT | FastDVDnet | S1 − RVRT |
|---|---|---|---|---|---|---|
| gaussian / high | 49.00 | 6.01 | 5.59 | 4.99 | **4.82** | +1.02 (t = +25.8) |
| gaussian / medium | 18.79 | 6.16 | 6.07 | 5.82 | **5.76** | +0.34 (t = +17.3) |
| gaussian / low | 11.13 | 6.27 | 6.24 | 6.05 | **6.02** | +0.22 (t = +19.9) |
| poisson_gaussian | 13.46 | 6.09 | 6.04 | 6.10 | **6.02** | −0.00 (t = −0.3, **tie**) |

**DashMamba does not beat the baselines on temporal consistency** — the one thing the
architecture was explicitly designed to improve. It improves enormously over the
degraded input (49.0 → 6.0) but only reaches parity with RVRT on one axis. §2 explains
why: the motion signal never activated.

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

1. **The win is real but is not produced by the proposed mechanisms.** It comes from the
   remaining architecture: the CNN encoder, the stacked bidirectional selective-scan
   temporal core, and the residual reconstruction head. That this holds in the *Stage-1*
   comparison — same training data as the baselines — makes it an architectural result.
2. **The low-noise loss is explained, and the monotonic trend above confirms it.**
   Signal B cannot distinguish σ15 from σ50, so the model applies one fixed strength.
3. **The tOF shortfall is explained.** Signal A provides no motion-aware fusion, so no
   mechanism in the trained model actually targets temporal consistency.
4. **An ablation over these flags would show ~no effect**, because the signals are already
   inert. Running it would confirm this, not contradict it.

### Root causes and fixes (future work)

| Signal | Root cause | Proposed fix |
|---|---|---|
| B | Sigmoid saturation; dead gradient | Remove the output sigmoid (predict log-σ directly), or add normalization / clamp pre-activations; optionally supervise it against the known σ during Stage-1 training, where the ground-truth noise level *is* available |
| A | Flow network collapsed to ~zero displacement; no supervision pushes it to learn motion | Initialize from a pretrained flow network (SPyNet), or add a photometric warping loss so alignment is directly trained rather than only implicitly rewarded |

A single testable prediction follows: **fixing Signal B should recover most of the
−1.57 dB low-noise deficit**, because that deficit is fully explained by inability to
modulate denoising strength. This is the highest-value next experiment.

---

## 3. Honest claim to make in the paper

> A blind video restoration model with 886,840 parameters — no ground-truth noise level,
> 14.7× smaller than RVRT — trained on the same public data as the baselines and never
> exposed to the target domain, achieves statistically significant gains over σ-informed
> baselines at severe noise (+1.79 dB over RVRT at σ50, n = 71, 95% CI [+1.05, +2.52])
> and on realistic Poisson-Gaussian sensor noise (+0.81 dB [+0.38, +1.25]), at the cost
> of −1.57 dB at mild noise, with temporal consistency at parity or slightly worse.
> Target-domain fine-tuning adds a further +0.17 to +0.46 dB. Direct measurement of the
> trained weights shows the two proposed adaptive signals did not activate; the observed
> gains therefore derive from the bidirectional selective-scan temporal core rather than
> from the adaptive mechanisms, and the failure modes are diagnosed rather than left
> unexplained.

### What must NOT be claimed

- Do not claim the decoupled signal design works — measurement says otherwise in this checkpoint.
- Do not claim improved temporal consistency — tOF is worse than both working baselines
  on three of four axes and tied on the fourth.
- Do not lead with the Stage-2 numbers against the baselines. Stage-2 was fine-tuned on
  dashcam data and the baselines were not. **Lead with Stage-1**; report Stage-2 as the
  measured value of fine-tuning.
- Do not claim a win at medium or low noise. Medium is a statistical tie; low is a
  significant loss.

---

## 4. Caveats

- **n = 71 clips**, stride-2 sampled across the 5 source videos of the Track A test split;
  80 of ~300 frames per clip. Documented in `docs/Phase3_Baseline_Dataset_Scope.docx`.
- **Track A only.** Tracks B (synthetic low-light) and C (real-world night) were never
  evaluated — a deliberate scope decision under time constraints.
- **BasicVSR++'s poor numbers** reflect a checkpoint domain mismatch (trained for
  compressed-video artifact removal, not sensor noise), not an architectural verdict.
  It is retained in the tables because silently dropping a model that performed badly
  would misrepresent the study.
- **Significance tests** use the normal-approximation t critical value (1.994, df = 70)
  and assume per-clip differences are approximately normal. Clips drawn from the same
  source video are not fully independent, so effective n is somewhat below 71.

---

## 5. Reproducing

See `HANDOFF_FOR_NEW_CHAT.md`. Per-clip CSVs for every run are under `results/`:

| Run | Directory |
|---|---|
| DashMamba Stage-1 | `results/dashmamba_stage1_track_a_test/` |
| DashMamba Stage-2 | `results/dashmamba_stage2_track_a_test/` |
| RVRT | `results/rvrt_track_a_test/` |
| BasicVSR++ | `results/bvrpp_track_a_test/` |
| FastDVDnet | `results/fastdvdnet_track_a_test/` |

Every significance figure above is a paired per-clip comparison computed from those
files, joined on `(kind, level, clip_stem)`.
