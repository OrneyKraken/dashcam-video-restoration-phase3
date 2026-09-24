# Model complexity — FLOPs, inference time, memory

Produced by [`scripts/model_complexity.py`](../../scripts/model_complexity.py) on
2026-09-24. Raw numbers: [`complexity.csv`](complexity.csv), [`complexity.json`](complexity.json).

**Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB), PyTorch 2.11.0 + CUDA 12.8,
fp32, batch 1. This is **not** the RTX 4080 SUPER the benchmark was scored on, so
absolute times will differ there; the relative ordering is what carries over.

## 1. Architecture cost (network only, same input for every model)

Input: 16 frames × 256×256 (FastDVDnet: one 5-frame window per output frame).
FLOPs scaled to one 1280×720 frame by pixel count, which is exact for these models:
re-measuring at 512×512 gave a ratio of **4.000** for all four.

| Model | Params (trainable) | Params + buffers | GFLOPs / 720p frame | GMACs / 720p frame | Time / frame @ 256² | Peak mem |
|---|---|---|---|---|---|---|
| **DashMamba** | **886,840** | **886,840** | **167** | **84** | **7.3 ms** | 0.70 GB |
| FastDVDnet | 2,479,096 | 2,483,578 | 1,173 | 586 | 15.1 ms | 0.20 GB |
| RVRT | 12,786,919 | 13,065,453 | 1,551 | 775 | 67.0 ms † | 1.85 GB |
| BasicVSR++ | 44,075,631 | 44,075,637 | 4,975 | 2,487 | 49.9 ms | 0.61 GB |

DashMamba relative to each baseline:

| vs | FLOPs | Network time | Params (trainable) |
|---|---|---|---|
| FastDVDnet | **7.0× fewer** | **2.1× faster** | 2.8× fewer |
| RVRT | **9.3× fewer** | **9.2× faster** † | 14.4× fewer |
| BasicVSR++ | **29.7× fewer** | **6.9× faster** | 49.7× fewer |

## 2. As evaluated (the exact inference path behind the reported PSNR/SSIM)

One 80-frame 1280×720 clip (`raw_video_013_clip_0009`, σ50) through each
`scripts/*_wrapper.py`, unchanged: spatial tiling, temporal chunking and CPU↔GPU
copies included. Best of 2 runs; FLOPs counted over the whole call.

| Model | Time / 720p frame | FPS | GFLOPs / frame (incl. overlap) | Peak mem | Overhead vs §1 FLOPs |
|---|---|---|---|---|---|
| FastDVDnet | **270 ms** | **3.71** | 1,173 | 3.00 GB | 1.00× — whole frame, no tiling |
| **DashMamba** | 320 ms | 3.12 | **429** | **2.13 GB** | 2.57× — 256² tiles (1.71×) + 40-frame chunks (1.5×) |
| BasicVSR++ | 1,100 ms | 0.91 | 8,399 | 3.55 GB | 1.69× — 320² tiles |
| RVRT | 2,738 ms ‡ | 0.37 | 2,029 | 2.68 GB ‡ | 1.31× — 256² tiles |

## 3. Reading these numbers

- **DashMamba is the cheapest architecture by every measure**: 7–30× fewer FLOPs
  and 2–9× faster per frame than the baselines on identical input.
- **End-to-end as evaluated, FastDVDnet is slightly faster (270 vs 320 ms/frame)**
  even though DashMamba's network is 2.1× faster. The gap is DashMamba's
  inference wrapper, not the model: overlapping 256² tiles plus overlapping
  40-frame chunks (an 80-frame clip is processed as 3 chunks = 120 frames)
  make it compute 2.57× the necessary FLOPs. FastDVDnet runs whole frames with
  no redundancy. Raising `DASHMAMBA_MAX_FRAMES` to cover the clip, or larger
  tiles, would remove most of this — but would also change the outputs slightly,
  so it would need re-scoring before being reported.
- **FLOP counting convention:** `torch.utils.flop_counter` — 2 × multiply-accumulates
  of conv / matmul / einsum; elementwise ops are not counted (same as fvcore/thop).
  This does not flatter DashMamba: its scan's elementwise updates are
  ≈ (6·C·N + 4C) × 4 scans × 180·320 locations ≈ **1.5 GFLOPs per 720p frame**,
  under 1% of the 167 counted.
- **Parameter counts:** the figures previously quoted in `RESULTS_AND_DIAGNOSIS.md`
  (RVRT 13,065,453, FastDVDnet 2,483,578, BasicVSR++ 44,075,638) are
  params **+ buffers** (BatchNorm statistics, RVRT's attention position indices).
  Counting trainable parameters only, RVRT is 12.79 M and DashMamba is
  **14.4×** smaller, not 14.7×. DashMamba has no buffers, so its count is the
  same either way. The builds here reproduce RVRT's and FastDVDnet's quoted
  totals exactly and BasicVSR++'s to within 1, which confirms the measured
  architectures are the evaluated ones.

### Caveats

† **RVRT's deformable attention** is a compiled CUDA kernel on the evaluation
machine. Here it runs as a PyTorch port (`scripts/baseline_shims.py`) that performs
the same im2col-then-matmul computation, so RVRT's **FLOPs are exact**, but its
**times may be somewhat pessimistic**.

‡ **RVRT's pipeline run used its built-in `cpu_cache` mode.** Holding a whole
80-frame tile on the GPU needs ~13 GB; on this 8 GB card Windows silently spilled
the excess into shared system memory (observed: 7.3 GB dedicated + 5.7 GB shared),
which invalidates any timing. `cpu_cache` parks features in host RAM between
recurrent steps — identical computation, much lower VRAM, but extra copies. Its
2,738 ms/frame is therefore an **upper bound**; from the network-only time
(67 ms × 18 tiles), RVRT without that overhead would be roughly **1.2 s/frame**
on this GPU — still ~4× slower than DashMamba.

- Baselines use random weights; cost does not depend on weight values.
  DashMamba's pipeline run uses the committed Stage-2 checkpoint.
- Laptop GPU: boost clocks vary with temperature, so expect a few percent of
  run-to-run variation in the times.

## Reproduce

```bash
cd scripts
python model_complexity.py --check-scaling          # everything, ~25 min on the RTX 5050
python model_complexity.py --parts params flops     # FLOPs only, ~1 min
```

Needs `external/RVRT`, `external/fastdvdnet` and `external/mmagic` (upstream
source, no builds) and `dataset/_refpool/<clip>` for the pipeline part.
