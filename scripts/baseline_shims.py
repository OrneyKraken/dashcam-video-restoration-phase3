"""Build RVRT / BasicVSR++ / FastDVDnet from their upstream source without any
compiled extension, for FLOPs and runtime measurement (model_complexity.py).

Why this exists: the evaluation machine built RVRT's deformable-attention CUDA
kernel with MSVC + CUDA 12.4 and ran BasicVSR++ through a trimmed mmagic/mmcv
checkout. Neither toolchain is needed to measure compute, so this module
substitutes the two compiled ops with equivalent PyTorch/torchvision code and
leaves every upstream file under external/ unmodified:

  RVRT       models/op/deform_attn.py JIT-compiles deform_attn_ext at import.
             Replaced by deform_attn_torch(): the same deformable im2col (here
             grid_sample, align_corners=True, zeros padding - mmcv's bilinear
             rule) followed by the same batched matmul -> softmax -> matmul
             the C++ forward performs, so the counted matmul FLOPs match.
  BasicVSR++ imports mmcv.ops.ModulatedDeformConv2d, plus mmengine/mmagic
             helpers. Replaced by stub modules; the deformable conv maps to
             torchvision.ops.deform_conv2d, which computes the same modulated
             DCNv2 with the same (group, kernel, (dy, dx)) offset layout.
  FastDVDnet pure PyTorch; only loaded under a private module name because its
             models.py collides with RVRT's `models` package.

Weights are random: FLOPs, parameter counts and runtime do not depend on them.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EXTERNAL = Path(__file__).resolve().parent.parent / "external"


def _load_file(name: str, path: Path, package: str | None = None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub(name: str, is_pkg: bool = True, **attrs):
    mod = types.ModuleType(name)
    if is_pkg:
        mod.__path__ = []
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


# --------------------------------------------------------------------------- #
# RVRT
# --------------------------------------------------------------------------- #
def deform_attn_torch(q, kv, offset, kernel_h, kernel_w, stride, padding, dilation,
                      attention_heads, deformable_groups, clip_size):
    """PyTorch port of RVRT deform_attn_cuda_forward (deform_attn_cuda_pt110.cpp).

    q:      (B, ..., P, H, W)            B = batch * clip frames, P = proj channels
    kv:     (B // clip_size, clip_size, 2P, H, W)   keys and values, concatenated
    offset: (B, clip_size * G * K * 2, H, W)
    returns a tensor shaped like q.
    """
    assert stride == 1
    b_all = q.size(0)
    p2, h, w = kv.shape[2:]
    p = p2 // 2
    heads, g, k = attention_heads, deformable_groups, kernel_h * kernel_w
    dim, cg, area = p // heads, p2 // deformable_groups, h * w

    qv = q.reshape(b_all, heads, dim, area).permute(0, 1, 3, 2) * dim ** -0.5  # (B,heads,area,dim)
    off = offset.reshape(b_all, clip_size, g, k, 2, h, w)

    dev, dt = kv.device, kv.dtype
    ys = torch.arange(h, device=dev, dtype=dt).view(h, 1)
    xs = torch.arange(w, device=dev, dtype=dt).view(1, w)
    ky = (torch.arange(kernel_h, device=dev, dtype=dt) * dilation - padding).repeat_interleave(kernel_w)
    kx = (torch.arange(kernel_w, device=dev, dtype=dt) * dilation - padding).repeat(kernel_h)
    bidx = torch.arange(b_all, device=dev)

    cols = []
    for n in range(clip_size):
        # kernel: kv[b / clip_size][(n + b) % clip_size]
        src = kv[bidx // clip_size, (n + bidx) % clip_size].reshape(b_all * g, cg, h, w)
        o = off[:, n]                                            # (B,G,K,2,H,W)
        py = ys + ky.view(k, 1, 1) + o[:, :, :, 0]               # (B,G,K,H,W)
        px = xs + kx.view(k, 1, 1) + o[:, :, :, 1]
        grid = torch.stack([2 * px / max(w - 1, 1) - 1, 2 * py / max(h - 1, 1) - 1], dim=-1)
        s = F.grid_sample(src, grid.reshape(b_all * g, k * h, w, 2), mode="bilinear",
                          padding_mode="zeros", align_corners=True)   # (B*G, cg, K*H, W)
        cols.append(s.reshape(b_all, g * cg, k, area))           # channel-major, like im2col
    cols = torch.stack(cols, dim=1)                              # (B, clip, 2P, K, area)
    cols = (cols.view(b_all, clip_size, 2, heads, dim, k, area)
                .permute(2, 0, 3, 6, 4, 1, 5)
                .reshape(2, b_all, heads, area, dim, clip_size * k))

    attn = torch.matmul(qv.unsqueeze(-2), cols[0]).softmax(-1)     # (B,heads,area,1,clip*K)
    out = torch.matmul(attn, cols[1].transpose(-1, -2))            # (B,heads,area,1,dim)
    return out.squeeze(-2).permute(0, 1, 3, 2).reshape(q.shape)


def build_rvrt():
    """RVRT exactly as scripts/rvrt_wrapper.py configures it (non-blind denoising)."""
    root = EXTERNAL / "RVRT" / "models"
    if "rvrt_models.network_rvrt" not in sys.modules:
        _stub("rvrt_models")
        _stub("rvrt_models.op")
        src = (root / "op" / "deform_attn.py").read_text(encoding="utf-8")
        start = src.index("deform_attn_ext = load(")
        end = src.index(")\n\n", start) + 2
        src = src[:start] + "deform_attn_ext = None  # replaced by baseline_shims\n" + src[end:]
        src = src.replace("from torch.utils.cpp_extension import load\n", "")
        op = _stub("rvrt_models.op.deform_attn", is_pkg=False,
                   __file__=str(root / "op" / "deform_attn.py"))
        exec(compile(src, str(root / "op" / "deform_attn.py"), "exec"), op.__dict__)
        op.deform_attn = deform_attn_torch
        _load_file("rvrt_models.network_rvrt", root / "network_rvrt.py", package="rvrt_models")
    net = sys.modules["rvrt_models.network_rvrt"].RVRT
    return net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8],
               num_blocks=[1, 2, 1], depths=[2, 2, 2], embed_dims=[192, 192, 192],
               num_heads=[6, 6, 6], inputconv_groups=[1, 3, 4, 6, 8, 4],
               deformable_groups=12, attention_heads=12, attention_window=[3, 3],
               nonblind_denoising=True, cpu_cache_length=100)


# --------------------------------------------------------------------------- #
# BasicVSR++ (mmagic)
# --------------------------------------------------------------------------- #
class _ModulatedDeformConv2d(nn.Module):
    """mmcv.ops.ModulatedDeformConv2d's constructor and parameters."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, deform_groups=1, bias=True):
        super().__init__()
        pair = nn.modules.utils._pair
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = pair(kernel_size), pair(stride)
        self.padding, self.dilation = pair(padding), pair(dilation)
        self.groups, self.deform_groups = groups, deform_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)


def _modulated_deform_conv2d(x, offset, mask, weight, bias, stride, padding, dilation,
                             groups, deform_groups):
    import torchvision
    return torchvision.ops.deform_conv2d(x, offset, weight, bias, stride=stride,
                                         padding=padding, dilation=dilation, mask=mask)


class _ConvModule(nn.Sequential):
    """mmcv.cnn.ConvModule for the only form SPyNet uses: conv (+ ReLU), no norm."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 norm_cfg=None, act_cfg=dict(type="ReLU"), **_):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if act_cfg is not None:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class _BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()


class _Registry:
    def register_module(self, *a, **k):
        return lambda cls: cls


def _constant_init(module, val, bias=0):
    if getattr(module, "weight", None) is not None:
        nn.init.constant_(module.weight, val)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


def _default_init_weights(module, scale=1):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _make_layer(block, num_blocks, **kwarg):
    return nn.Sequential(*[block(**kwarg) for _ in range(num_blocks)])


def build_basicvsrpp():
    """BasicVSR++ exactly as scripts/basicvsrpp_wrapper.py configures it."""
    mm = EXTERNAL / "mmagic" / "mmagic" / "models"
    name = "mmagic.models.editors.basicvsr_plusplus_net.basicvsr_plusplus_net"
    if name not in sys.modules:
        _stub("mmcv")
        _stub("mmcv.ops", ModulatedDeformConv2d=_ModulatedDeformConv2d,
              modulated_deform_conv2d=_modulated_deform_conv2d)
        _stub("mmcv.cnn", ConvModule=_ConvModule)
        _stub("mmengine", MMLogger=None, print_log=lambda *a, **k: None)
        _stub("mmengine.model", BaseModule=_BaseModule)
        _stub("mmengine.model.weight_init", constant_init=_constant_init)
        _stub("mmengine.runner", load_checkpoint=None)
        _stub("mmagic")
        _stub("mmagic.registry", MODELS=_Registry())
        _stub("mmagic.models")
        fw = _load_file("mmagic.models.utils.flow_warp", mm / "utils" / "flow_warp.py")
        _stub("mmagic.models.utils", flow_warp=fw.flow_warp, make_layer=_make_layer,
              default_init_weights=_default_init_weights)
        archs = _stub("mmagic.models.archs")
        srb = _load_file("mmagic.models.archs.sr_backbone", mm / "archs" / "sr_backbone.py",
                         package="mmagic.models.archs")
        ups = _load_file("mmagic.models.archs.upsample", mm / "archs" / "upsample.py",
                         package="mmagic.models.archs")
        archs.ResidualBlockNoBN, archs.PixelShufflePack = srb.ResidualBlockNoBN, ups.PixelShufflePack
        _stub("mmagic.models.editors")
        _stub("mmagic.models.editors.basicvsr")
        _stub("mmagic.models.editors.basicvsr_plusplus_net")
        _load_file("mmagic.models.editors.basicvsr.basicvsr_net",
                   mm / "editors" / "basicvsr" / "basicvsr_net.py",
                   package="mmagic.models.editors.basicvsr")
        _load_file(name, mm / "editors" / "basicvsr_plusplus_net" / "basicvsr_plusplus_net.py",
                   package="mmagic.models.editors.basicvsr_plusplus_net")
    net = sys.modules[name].BasicVSRPlusPlusNet
    return net(mid_channels=128, num_blocks=25, is_low_res_input=False,
               cpu_cache_length=100, spynet_pretrained=None)


# --------------------------------------------------------------------------- #
# FastDVDnet
# --------------------------------------------------------------------------- #
def build_fastdvdnet():
    """FastDVDnet exactly as scripts/fastdvdnet_wrapper.py configures it."""
    if "fastdvdnet_models" not in sys.modules:
        _load_file("fastdvdnet_models", EXTERNAL / "fastdvdnet" / "models.py")
    return sys.modules["fastdvdnet_models"].FastDVDnet(num_input_frames=5)
