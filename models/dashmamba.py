"""DashMamba v2: Mamba-based video restoration with two DECOUPLED signals.

Design rationale lives in docs/Phase3_DashMamba_Architecture.docx (v1 design)
and its v2 revision - this file is the code side of the v2 revision. v1 fed
a single motion-magnitude signal into all three of the Mamba scan's own
parameters (dt, B, C) at once, which meant an ablation couldn't tell which
part of that signal's effect was actually doing the work. v2 fixes this by
using two INDEPENDENT signals, each routed to exactly one mechanism:

  Signal A (motion confidence): forward/backward flow-consistency check on
    stage 2's own flow estimate (a free byproduct - no new network). Low
    where the two flow directions disagree (occlusion, fast motion, an
    unreliable warp). Feeds ONLY stage 4's fusion gate - it never touches
    the SSM's own parameters. Targets the finding that temporal instability
    (tOF) is worst on high-motion daytime footage, for every baseline model.

  Signal B (blind local reliability): a small CBDNet-style FCN that
    estimates noise/exposure reliability directly from the RAW input frame -
    no ground-truth noise level is ever given to this model, unlike the
    RVRT/FastDVDnet baselines already evaluated, which are both given the
    true sigma. Feeds ONLY the scan's discretization step (dt): low
    estimated reliability -> slower state decay -> more multi-frame
    averaging; high reliability -> faster decay -> per-frame detail is kept
    instead of being blurred by unnecessary long-range averaging. Targets
    the finding that RVRT/FastDVDnet specifically lose 4-5dB PSNR when
    severe Gaussian noise combines with evening/night lighting.

Because each signal drives exactly one mechanism, the two can be ablated
independently (see the three flags on DashMambaNet, and the 5-cell ablation
table in the v2 architecture doc) - a gain from the full model alone cannot
prove which signal caused it; testing them one at a time can.

Five stages, stages 1/2/4/5 structurally unchanged from v1, only what feeds
into stage 3's dt and stage 4's fusion is new:
  1. Shallow per-frame CNN feature extractor (4x downsample).
  2. Coarse-to-fine flow-guided alignment; ALSO produces Signal A here as a
     free byproduct of computing flow in both directions.
  2b. Signal B: blind reliability estimator on the raw frame.
  3. Stack of bidirectional selective state-space (S6/Mamba) blocks. B and C
     are left as standard input-dependent Mamba parameters (untouched by
     either signal); dt is modulated by Signal B only.
  4. Feature fusion: a learned blend of the aligned spatial features and the
     Mamba stack's temporal output, gated by Signal A (motion confidence).
  5. Reconstruction head (residual RGB output).

Novelty positioning (do not overclaim beyond this - see the architecture
doc's own related-work section): the S6 recurrence itself, and applying
Mamba temporally to video, are both established (Gu & Dao; MambaOFR, EVDM,
et al.). What's specific here is the exact decoupled routing of two signals
derived from this project's own weakness analysis, plus operating fully
blind (no oracle noise level) unlike the baselines evaluated so far. This
has not been exhaustively checked against the wider literature and should
not be presented as confirmed-novel without doing that before submission.

The selective scan is a plain, correct, sequential Python loop over time -
NOT the fast parallel-scan CUDA kernel (mamba_ssm's selective_scan_cuda).
That's a follow-up performance step, not required for validation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# stage 1: spatial feature extraction (+ 4x downsample)
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class FeatureExtractor(nn.Module):
    def __init__(self, mid_channels: int, num_blocks: int = 5):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 2, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(0.1, inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(mid_channels) for _ in range(num_blocks)])

    def forward(self, x):
        # x: (N, 3, H, W) -> (N, C, H/4, W/4)
        return self.blocks(self.down(x))


# --------------------------------------------------------------------------- #
# stage 2: lightweight flow estimator + warp (+ Signal A byproduct)
# --------------------------------------------------------------------------- #
class TinyFlowNet(nn.Module):
    """Minimal learned flow estimator (single-scale, not coarse-to-fine)."""

    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 2, 3, 1, 1),
        )
        # Near-zero (not exactly zero) flow at init: a fully-zero weight
        # matrix here would make the backward Jacobian through this layer
        # zero too, starving earlier layers of gradient (found + fixed
        # during v1 smoke testing).
        self.net[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat_ref, feat_other):
        return self.net(torch.cat([feat_ref, feat_other], dim=1))  # (N, 2, H, W)


def warp(feat, flow):
    """Warp feat (N,C,H,W) toward the reference frame using flow (N,2,H,W).
    Also used, unchanged, to warp a FLOW FIELD itself (Signal A's
    forward/backward consistency check) - grid_sample doesn't care what the
    channel dimension represents."""
    n, c, h, w = feat.shape
    gy, gx = torch.meshgrid(
        torch.arange(h, device=feat.device, dtype=feat.dtype),
        torch.arange(w, device=feat.device, dtype=feat.dtype),
        indexing="ij",
    )
    grid = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(n, -1, -1, -1)  # (N,2,H,W)
    sample = grid + flow
    sample_x = 2.0 * sample[:, 0] / max(w - 1, 1) - 1.0
    sample_y = 2.0 * sample[:, 1] / max(h - 1, 1) - 1.0
    sample_grid = torch.stack([sample_x, sample_y], dim=-1)  # (N,H,W,2)
    return F.grid_sample(feat, sample_grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


class CoarseToFineFlowNet(nn.Module):
    """Two-level coarse-to-fine flow estimation (SPyNet/PWC-style residual
    refinement): half-resolution estimate first, then a full-resolution
    residual correction from the coarsely-warped pair. Handles the larger
    displacements that a single-scale estimator misses - the case that
    matters for high-motion daytime driving footage."""

    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.coarse = TinyFlowNet(in_channels, hidden)
        self.fine = TinyFlowNet(in_channels, hidden)

    def forward(self, feat_ref, feat_other):
        ref_s = F.avg_pool2d(feat_ref, 2)
        other_s = F.avg_pool2d(feat_other, 2)
        flow_coarse = self.coarse(ref_s, other_s)
        flow_up = F.interpolate(flow_coarse, scale_factor=2, mode="bilinear",
                                align_corners=True) * 2.0

        warped_other = warp(feat_other, flow_up)
        flow_residual = self.fine(feat_ref, warped_other)
        return flow_up + flow_residual


def flow_consistency_confidence(flow_to_ref, flow_from_ref):
    """Signal A: standard forward/backward consistency check.
    flow_to_ref:   warp(other, flow_to_ref)   ~= ref   (the direction actually used for alignment)
    flow_from_ref: warp(ref, flow_from_ref)   ~= other (the reverse direction, computed only to check agreement)
    If the two directions agree, warping flow_from_ref by flow_to_ref and
    adding it back to flow_to_ref should cancel to ~0. Large residual ->
    unreliable flow (occlusion, fast/complex motion) -> low confidence.
    Returns confidence in (0, 1], 1 = fully consistent."""
    warped_reverse = warp(flow_from_ref, flow_to_ref)
    fb_residual = (flow_to_ref + warped_reverse).norm(dim=1, keepdim=True)
    return torch.exp(-fb_residual)


# --------------------------------------------------------------------------- #
# stage 2b: Signal B - blind local reliability (noise/exposure) estimator
# --------------------------------------------------------------------------- #
class ReliabilityEstimator(nn.Module):
    """CBDNet-style small FCN, operating on the RAW input frame (not
    extracted features, so it sees the true sensor signal before feature
    extraction can mask it). No ground-truth noise level is ever given to
    this network - it is trained purely through the reconstruction loss,
    unlike RVRT/FastDVDnet's baselines, which are both given the true sigma
    directly (see rvrt_wrapper.py / fastdvdnet_wrapper.py)."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, frame):
        """frame: (N,3,H,W) in [0,1] -> (N,1,H,W) reliability in [0,1] (low = noisy/unreliable)."""
        return self.net(frame)


# --------------------------------------------------------------------------- #
# stage 3: selective state-space (S6) core - dt modulated by Signal B ONLY
# --------------------------------------------------------------------------- #
class SelectiveScan1D(nn.Module):
    """S6 recurrence applied along time, independently per spatial location
    (spatial dims folded into the batch dimension by the caller).

    x: (M, T, C)              M = batch * H' * W' "pixel-batch"
    reliability: (M, T, 1)    Signal B only - feeds dt, NOT B or C.
    returns: (M, T, C)
    """

    def __init__(self, channels: int, state_dim: int = 16):
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim

        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, state_dim, state_dim)).repeat(channels, 1))
        self.D = nn.Parameter(torch.ones(channels))

        self.to_dt = nn.Linear(channels + 1, channels)   # x + Signal B (reliability)
        self.to_B = nn.Linear(channels, state_dim)        # content only - untouched by either signal
        self.to_C = nn.Linear(channels, state_dim)        # content only - untouched by either signal
        nn.init.constant_(self.to_dt.bias, -2.0)  # start with small dt (slow forgetting)

    def forward(self, x, reliability):
        m, t, c = x.shape
        n = self.state_dim
        A = -torch.exp(self.A_log)  # (C, N), negative for stability

        dt = F.softplus(self.to_dt(torch.cat([x, reliability], dim=-1)))  # (M, T, C)
        Bt = self.to_B(x)                                                  # (M, T, N)
        Ct = self.to_C(x)                                                  # (M, T, N)

        h = x.new_zeros(m, c, n)
        ys = []
        for step in range(t):
            dt_s = dt[:, step]                        # (M, C)
            dA = torch.exp(dt_s.unsqueeze(-1) * A)     # (M, C, N)
            dBx = dt_s.unsqueeze(-1) * Bt[:, step].unsqueeze(1) * x[:, step].unsqueeze(-1)  # (M,C,N)
            h = dA * h + dBx
            y = torch.einsum("mcn,mn->mc", h, Ct[:, step]) + self.D * x[:, step]
            ys.append(y)
        return torch.stack(ys, dim=1)                 # (M, T, C)


class BidirectionalMambaCore(nn.Module):
    def __init__(self, channels: int, state_dim: int = 16):
        super().__init__()
        self.fwd = SelectiveScan1D(channels, state_dim)
        self.bwd = SelectiveScan1D(channels, state_dim)
        self.merge = nn.Linear(channels * 2, channels)

    def forward(self, feats, reliability):
        """feats: (B, T, C, H, W)   reliability: (B, T, 1, H, W)"""
        b, t, c, h, w = feats.shape
        x = feats.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        rel = reliability.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, 1)

        y_fwd = self.fwd(x, rel)
        y_bwd = self.bwd(x.flip(1), rel.flip(1)).flip(1)
        y = self.merge(torch.cat([y_fwd, y_bwd], dim=-1))  # (M, T, C)

        return y.reshape(b, h, w, t, c).permute(0, 3, 4, 1, 2)  # (B,T,C,H,W)


class TemporalStack(nn.Module):
    """N residually-connected BidirectionalMambaCore blocks - depth via
    stacking, matching how Mamba is normally used, not one oversized block."""

    def __init__(self, channels: int, state_dim: int, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [BidirectionalMambaCore(channels, state_dim) for _ in range(num_blocks)]
        )

    def forward(self, feats, reliability):
        x = feats
        for block in self.blocks:
            x = x + block(x, reliability)
        return x


# --------------------------------------------------------------------------- #
# stages 4-5: fusion (gated by Signal A) + reconstruction
# --------------------------------------------------------------------------- #
class ReconstructionHead(nn.Module):
    def __init__(self, mid_channels: int):
        super().__init__()
        self.fuse = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.up1 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1),
                                 nn.PixelShuffle(2), nn.LeakyReLU(0.1, inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1),
                                 nn.PixelShuffle(2), nn.LeakyReLU(0.1, inplace=True))
        self.out = nn.Conv2d(mid_channels, 3, 3, 1, 1)

    def forward(self, spatial_feat, temporal_feat, gate, frame):
        """gate: (N,1,H,W) in [0,1] - Signal A (motion confidence). gate=1 ->
        fully trust the temporal Mamba feature; gate=0 -> fully trust the
        locally-aligned spatial feature."""
        blended = gate * temporal_feat + (1.0 - gate) * spatial_feat
        x = self.fuse(blended)
        x = self.up1(x)
        x = self.up2(x)
        return frame + self.out(x)


# --------------------------------------------------------------------------- #
# full model
# --------------------------------------------------------------------------- #
class DashMambaNet(nn.Module):
    def __init__(self, mid_channels: int = 64, num_res_blocks: int = 5, state_dim: int = 16,
                num_mamba_blocks: int = 2, use_flow_align: bool = True,
                use_motion_gate: bool = True, use_reliability_delta: bool = True):
        super().__init__()
        self.use_flow_align = use_flow_align
        self.use_motion_gate = use_motion_gate and use_flow_align  # gate needs flow to exist
        self.use_reliability_delta = use_reliability_delta

        self.feat_extract = FeatureExtractor(mid_channels, num_res_blocks)
        if use_flow_align:
            self.flow = CoarseToFineFlowNet(mid_channels)
        if use_reliability_delta:
            self.reliability_net = ReliabilityEstimator()
        self.temporal_core = TemporalStack(mid_channels, state_dim, num_mamba_blocks)
        self.recon = ReconstructionHead(mid_channels)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: (B, T, 3, H, W) in [0,1]  ->  (B, T, 3, H, W)"""
        b, t, c, h, w = clip.shape
        frames = clip.reshape(b * t, c, h, w)
        feats = self.feat_extract(frames)                    # (B*T, C, H/4, W/4)
        _, cf, hf, wf = feats.shape
        feats = feats.reshape(b, t, cf, hf, wf)

        if self.use_flow_align:
            ref_idx = t // 2
            ref = feats[:, ref_idx]
            aligned, gate = [], []
            for i in range(t):
                if i == ref_idx:
                    aligned.append(ref)
                    gate.append(torch.ones(b, 1, hf, wf, device=clip.device, dtype=clip.dtype))
                    continue
                flow_to_ref = self.flow(ref, feats[:, i])
                aligned.append(warp(feats[:, i], flow_to_ref))
                if self.use_motion_gate:
                    flow_from_ref = self.flow(feats[:, i], ref)
                    gate.append(flow_consistency_confidence(flow_to_ref, flow_from_ref))
                else:
                    gate.append(torch.full((b, 1, hf, wf), 0.5, device=clip.device, dtype=clip.dtype))
            aligned = torch.stack(aligned, dim=1)             # (B,T,C,Hf,Wf)
            gate = torch.stack(gate, dim=1)                   # (B,T,1,Hf,Wf)
        else:
            # ablation: no flow alignment at all -> neutral (equal-weight) fusion gate
            aligned = feats
            gate = torch.full((b, t, 1, hf, wf), 0.5, device=clip.device, dtype=clip.dtype)

        if self.use_reliability_delta:
            reliability = self.reliability_net(frames)                    # (B*T,1,H,W)
            reliability = F.avg_pool2d(reliability, 4).reshape(b, t, 1, hf, wf)
        else:
            # ablation: content-only dt, no reliability conditioning
            reliability = torch.zeros(b, t, 1, hf, wf, device=clip.device, dtype=clip.dtype)

        temporal = self.temporal_core(aligned, reliability)   # (B,T,C,Hf,Wf)

        aligned_flat = aligned.reshape(b * t, cf, hf, wf)
        temporal_flat = temporal.reshape(b * t, cf, hf, wf)
        gate_flat = gate.reshape(b * t, 1, hf, wf)
        out = self.recon(aligned_flat, temporal_flat, gate_flat, frames)
        return out.reshape(b, t, c, h, w)


# --------------------------------------------------------------------------- #
# smoke test - the 5-cell ablation table from the v2 architecture doc
# --------------------------------------------------------------------------- #
def _run_case(name, **kwargs):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DashMambaNet(mid_channels=32, num_res_blocks=2, state_dim=8,
                         num_mamba_blocks=2, **kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    b, t, h, w = 1, 6, 64, 64
    clip = torch.rand(b, t, 3, h, w, device=device, requires_grad=True)

    out = model(clip)
    shape_ok = out.shape == clip.shape

    loss = F.mse_loss(out, torch.rand_like(out))
    loss.backward()
    used = [p for p in model.parameters() if p.requires_grad]
    n_with_grad = sum(1 for p in used if p.grad is not None and p.grad.abs().sum() > 0)

    status = "PASS" if shape_ok else "FAIL(shape)"
    print(f"[{name:22s}] params={n_params:,}  out={tuple(out.shape)}  "
         f"loss={loss.item():.4f}  grad {n_with_grad}/{len(used)}  {status}")
    return shape_ok


if __name__ == "__main__":
    results = [
        _run_case("1 baseline",       use_flow_align=False, use_motion_gate=False, use_reliability_delta=False),
        _run_case("2 +flow_align",    use_flow_align=True,  use_motion_gate=False, use_reliability_delta=False),
        _run_case("3 +motion_gate",   use_flow_align=True,  use_motion_gate=True,  use_reliability_delta=False),
        _run_case("4 +reliability",   use_flow_align=True,  use_motion_gate=False, use_reliability_delta=True),
        _run_case("5 full model",     use_flow_align=True,  use_motion_gate=True,  use_reliability_delta=True),
    ]
    print("ALL SMOKE TESTS PASSED" if all(results) else "SOME SMOKE TESTS FAILED")
