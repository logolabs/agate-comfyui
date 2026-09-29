"""FCDM-T2-MR: T40r (models/fcdm_thinker2.py) made multi-resolution (256 and 512 px) by additive surgery.

A copy, not a subclass or an edit: T40r checkpoints keep loading into models/fcdm_thinker2.py
unchanged, and they load into THIS file too -- every pre-existing module keeps its name and shape,
and every new module is prefixed `mr_` and starts as an exact no-op. At 256 px a T40r checkpoint
gives bit-for-bit the same velocity here (tests/test_thinker2_mr.py). What is added, and why
(2026-09-26 zero-shot tests on the released weights, agate_face_tests/):

  read       Feeding the thinker a 64x64 latent (1024 perceiver keys) gives texture noise at any
             attention temperature (log-n scaling tested); feeding it the stride-2 subsample gives
             coherent 512 px images. So the thinker's grid/positions/loops are kept and only its
             READ changes: keys = in_latent(nearest subsample, the trained path, exact) +
             mr_taps (zero-init linear over the full 4x4 patch: the 48 values the subsample drops)
             + mr_read (zero-init conv branch: overlapping 3x3/7x7 convs at full resolution,
             FiLM on the timestep and the resolution, pooled onto the key grid). Nearest, not
             averaging: averaging halves the noise std (tested: worse, even with a t remap).
  resolution mr_res_q / mr_res_s: log2(scale) embedded like the timestep, minus the embedding of 0
             (exactly 0 at 256 px, forever), into the perceiver queries and the thinking stack; also
             fed to mr_read. Without it
             the thinker could still tell 256 from 512 -- from the 2x2-block artefact of the
             nearest upsample -- which would not survive other sizes.
  renderer   The core U-Net keeps running at the native 32x32 (the scale it was trained at: one
             plan cell = 2x2 cells). At 512 a thin nested outer level (Matryoshka-style) wraps
             it: mr_outer_stem + blocks at 64x64 -> zero-init stride-2 into the core's input;
             the core's last features -> 2x up -> blocks -> zero-init head, added to the core's
             velocity upsampled 2x (up2: grid-anchored, so the core's own cells stay exact). Zero-shot, a full-res core duplicated parts (two
             faces stacked): convs are not scale-equivariant. Step 0 is the core's 256 answer to
             the subsampled latent, upscaled, plus hp_prior: the closed-form linear estimate of the
             fine-scale velocity (bounded at every t). Without it 3/4 of the cells keep their noise
             at step 0 (tested: a grainy image); with it step 0 is a clean upscale and the outer level
             only learns what a linear estimate misses. The data's fine scale is not small: 30-41% of
             the latent variance per channel at 512 px (measured) -- the octave has real content.
  plan up    The 16x16 plan (all 640 dims) is upsampled ONCE per output size by mr_plan_up: bilinear
             + a teeny zero-init delta ("add what the bilinear broke": depthwise convs over the map,
             rank-k bottleneck) that also sees the raw coordinates and the latent at that size. Every
             consumer at that size (l0 steering, output modulation; at 512 the outer level's) reads
             the same sharpened plan. The 1x1 projections now run after the upsample: equal to the
             old order (both linear, bilinear weights sum to 1) up to float rounding.
  coords     Raw relative coordinates (x, y in [-1, 1] at cell centres, r = sqrt(x^2 + y^2)) reach the
             thinker through its conv read (at the TRUE latent grid, upsampled with the latent; its
             16x16 query and key grids are the same at every size, so coordinates there would be
             constants -- Fourier q_pos and RoPE already cover them) and the renderer (every level,
             the outer level, the plan upsampler) through zero-init projections. Raw, not Fourier: Fourier octaves
             alias at these sizes (octave k is clean only for 2^k <= N/4; at 32x32 octaves 4-5 are
             Nyquist/constant) and would mean different things at 256 and 512; raw coordinates mean
             the same everywhere, and their slope (2/N per cell) is itself an explicit resolution
             signal any 3x3 conv reads. fcdm2's Fourier coordinates, zero-ablated at e20 (job
             2913012): +5.5% loss, used by the encoder levels (+2.6% at 32x32) and ~0 by the decoder.
             Nothing periodic per plan cell is injected: a cell-periodic code invites a visible 16x16
             grid (texture sticking, the StyleGAN3 problem). Applied where the
             map is upsampled: l0 (16 -> 32), the outer level (16 -> 64) and the output modulations.
             Moving bilinear before the per-block 1x1 mod convs equals the old order (both linear,
             bilinear weights sum to 1) up to float rounding. The plan measured (plan_dim.py):
             >50% of its within-image energy above the lowest frequencies, which bilinear smears.

Conventions as T40r: t = 0 noise, t = 1 data, predicts the velocity x1 - x0. Supported latent
sizes: the native one (2 * grid, 32 for T40r) and exactly twice it. Aspect ratios: not yet.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcdm_planner import SpatialModBlock, fourier_coords
from .fcdm_t2i import LayerNorm2d, TimestepEmbedder


def rope_freqs(head_dim: int, device) -> torch.Tensor:
    """Angular frequencies for one axis: head_dim/4 of them (each rotates one channel pair;
    x and y get half the pairs each), log-spaced from pi/2 (one half-turn across the image)
    to 8 pi (a half-turn between neighbouring cells of a 16-cell grid)."""
    n = head_dim // 4
    return math.pi * 2.0 ** torch.linspace(-1.0, 3.0, n, device=device, dtype=torch.float32)


def grid_xy(h: int, w: int, device) -> torch.Tensor:
    """(h*w, 2) cell-centre coordinates (x, y) in [-1, 1], row-major like pixel_unshuffle."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)


def rope_angles(xy: torch.Tensor, head_dim: int, n_unrotated: int = 0) -> torch.Tensor:
    """(N, head_dim/2) rotation angles for positions xy (N, 2); `n_unrotated` extra rows of
    zeros are APPENDED (global tokens: angle 0 is the identity rotation)."""
    f = rope_freqs(head_dim, xy.device)
    ang = torch.cat([xy[:, :1] * f, xy[:, 1:] * f], 1)
    if n_unrotated:
        ang = torch.cat([ang, ang.new_zeros(n_unrotated, ang.shape[1])], 0)
    return ang


def apply_rope(x: torch.Tensor, ang: torch.Tensor | None) -> torch.Tensor:
    """Rotate channel pairs of x (B, h, N, d) by ang (N, d/2) -- in fp32 whatever the autocast
    state: a bf16 rotation scrambles positions (the Ideogram-4 rope bug, 2026-09-08)."""
    if ang is None:
        return x
    with torch.autocast(x.device.type, enabled=False):
        xf = x.float()
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        c, s = ang.cos(), ang.sin()
        out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)
    return out.to(x.dtype)


class Attn(nn.Module):
    """Multi-head attention with QK RMSNorm and optional 2D RoPE on either side."""

    def __init__(self, dim: int, kv_dim: int, head_dim: int = 64):
        super().__init__()
        self.h, self.d = dim // head_dim, head_dim
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim, 2 * dim)
        self.qn, self.kn = nn.RMSNorm(head_dim), nn.RMSNorm(head_dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, kv, pad=None, ang_q=None, ang_k=None):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.h, self.d).transpose(1, 2)
        k, v = self.kv(kv).view(B, kv.shape[1], 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.qn(q), ang_q), apply_rope(self.kn(k), ang_k)
        m = None if pad is None else ~pad[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.out(o.transpose(1, 2).reshape(B, N, -1))


class XBlock2(nn.Module):
    """Pre-LN layer: self-attention (optional, RoPE), cross-attention to a context (RoPE when
    the context has positions, i.e. the latent), MLP."""

    def __init__(self, dim: int, ctx_dim: int | None, self_attn: bool = True, mlp: int = 2):
        super().__init__()
        self.sa = Attn(dim, dim) if self_attn else None
        self.n_sa = nn.LayerNorm(dim) if self_attn else None
        self.xa = Attn(dim, ctx_dim) if ctx_dim else None
        self.n_xa = nn.LayerNorm(dim) if ctx_dim else None
        self.n_ctx = nn.LayerNorm(ctx_dim) if ctx_dim else None
        self.n_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(), nn.Linear(mlp * dim, dim))

    def forward(self, q, ctx=None, pad=None, ang_q=None, ang_ctx=None):
        if self.sa is not None:
            h = self.n_sa(q)
            q = q + self.sa(h, h, None, ang_q, ang_q)
        if self.xa is not None:
            c = self.n_ctx(ctx)
            q = q + self.xa(self.n_xa(q), c, pad, ang_q if ang_ctx is not None else None, ang_ctx)
        return q + self.mlp(self.n_mlp(q))


def zero_last(m: nn.Module) -> nn.Module:
    """Zero the weight and bias of the last Linear/Conv in m: the module starts as an exact 0."""
    last = [x for x in m.modules() if isinstance(x, (nn.Linear, nn.Conv2d))][-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)
    return m


def subsample(z: torch.Tensor, native: int) -> torch.Tensor:
    """The latent at the native size: identity there, stride 2 at twice it (the top-left cell of
    every 2x2 block -- the read that was tested zero-shot, 2026-09-26). Keeps the noise std at 1."""
    s = z.shape[-1] // native
    if z.shape[-1] == native:
        return z
    if s * native != z.shape[-1] or z.shape[-2] != z.shape[-1] or s != 2:
        raise ValueError(f"latent {tuple(z.shape[-2:])}: only {native}^2 and {2 * native}^2 are supported")
    return z[..., ::2, ::2]


def res_scale(z: torch.Tensor, native: int) -> torch.Tensor:
    """log2 of the latent size over the native one: 0 at 256 px, 1 at 512 px. (B,)"""
    return torch.full((z.shape[0],), math.log2(z.shape[-1] / native), device=z.device)


class ResEmbed(nn.Module):
    """Resolution embedding: the timestep embedder applied to log2(scale), minus its value at 0, so the
    native size gets exactly 0 however it is trained (256 px moves only through shared weights)."""

    def __init__(self, dim: int, zero_init: bool = True):
        super().__init__()
        self.emb = TimestepEmbedder(dim)
        if zero_init:
            zero_last(self.emb)

    def forward(self, r):
        return self.emb(r) - self.emb(torch.zeros_like(r))


class FiLMConvNeXt(nn.Module):
    """ConvNeXt block (7x7 depthwise -> LN -> 1x1 expand -> GELU -> 1x1) with FiLM from a
    conditioning vector (timestep + resolution) and a zero-init residual gate."""

    def __init__(self, ch: int, cond_dim: int, expansion: int = 2):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.norm = LayerNorm2d(ch, affine=False)
        self.film = nn.Linear(cond_dim, 3 * ch)
        self.pw1 = nn.Conv2d(ch, expansion * ch, 1)
        self.pw2 = nn.Conv2d(expansion * ch, ch, 1)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        shift, scale, gate = self.film(cond)[:, :, None, None].chunk(3, 1)
        h = self.norm(self.dw(x)) * (1 + scale) + shift
        return x + (1 + gate) * self.pw2(F.gelu(self.pw1(h)))


class ConvRead(nn.Module):
    """The zero-init conv branch of the thinker's read: full-resolution overlapping convs, aware of
    the timestep and the resolution, pooled onto the key grid (one vector per 2x2 native patch)."""

    def __init__(self, latent_ch: int, out_dim: int, cond_dim: int, ch: int = 96, depth: int = 2):
        super().__init__()
        self.stem = nn.Conv2d(latent_ch + N_COORD, ch, 3, padding=1)     # latent + raw coordinates
        self.blocks = nn.ModuleList(FiLMConvNeXt(ch, cond_dim) for _ in range(depth))
        self.norm = LayerNorm2d(ch)
        self.proj = zero_last(nn.Linear(ch, out_dim))

    def forward(self, z_hr, coords_hr, cond, grid_hw):
        h = self.stem(torch.cat([z_hr, coords_hr.expand(z_hr.shape[0], -1, -1, -1).to(z_hr.dtype)], 1))
        for b in self.blocks:
            h = b(h, cond)
        h = F.adaptive_avg_pool2d(self.norm(h), grid_hw)          # area pool of LEARNED features
        return self.proj(h.flatten(2).transpose(1, 2))


def raw_coords(h: int, w: int, device, dtype=torch.float32) -> torch.Tensor:
    """(1, 3, h, w): x and y of the cell centres in [-1, 1] and r = sqrt(x^2 + y^2) -- relative
    coordinates (fractions of the image, so the same place has the same value at every size; the step
    between neighbours, 2/N, tells a conv the resolution)."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy, (xx ** 2 + yy ** 2).sqrt()])[None].to(dtype)


N_COORD = 3


class CoordIn(nn.Module):
    """Zero-init 1x1 from the raw coordinates to a feature map's width (CoordConv, added). res_channel: one
    more constant input = log2(scale) (0 at 256 px, 1 at 512): an explicit 'which resolution' at every
    level, exactly 0 at the native size however it is trained."""

    def __init__(self, ch: int, res_channel: bool = False):
        super().__init__()
        self.res_channel = res_channel
        self.proj = zero_last(nn.Conv2d(N_COORD + int(res_channel), ch, 1))

    def forward(self, x, r: float = 0.0):
        c = raw_coords(x.shape[-2], x.shape[-1], x.device, x.dtype)
        if self.res_channel:
            c = torch.cat([c, torch.full_like(c[:, :1], float(r))], 1)
        return x + self.proj(c)


class PlanUp(nn.Module):
    """The plan at an output size = bilinear + a teeny zero-init delta ("add what the bilinear broke").
    The delta sees the bilinear plan through two depthwise convs (the neighbouring plan cells), the raw
    coordinates at the output size and the latent there (3x3 conv: region edges can follow the image),
    merged in a rank-k GELU bottleneck and mapped back to the plan's width. Step 0: exactly bilinear."""

    def __init__(self, dim: int, rank: int = 32, kernel: int = 5, latent_ch: int = 4):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.dw2 = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.down = nn.Conv2d(dim, rank, 1)
        self.coord = nn.Conv2d(N_COORD, rank, 1)
        self.guide = nn.Conv2d(latent_ch, rank, 3, padding=1)
        self.up = zero_last(nn.Conv2d(rank, dim, 1))

    def forward(self, m, z):
        size = z.shape[-2:]
        b = F.interpolate(m, size=size, mode="bilinear", align_corners=False)
        h = (self.down(self.dw2(F.gelu(self.dw(b)))) + self.coord(raw_coords(*size, m.device, b.dtype))
             + self.guide(z.to(b.dtype)))
        return b + self.up(F.gelu(h))


# Fine-scale variance (z minus up2 of its stride-2 subsample) of SD-VAE latents x 0.18215 at 512 px, per
# channel and per position in the 2x2 cell -- (0,1) between two columns, (1,0) between two rows, (1,1) the
# centre; (0,0) is the grid cell itself, always 0. 256 FLUX-Reason-6M images (Aesthetics-Part01 shard 7, rows
# 1000-1255, centre crop), agate_face_tests/hp_var_512_anchored.pt, 2026-09-26. Noise (N(0, 1)): exactly
# 1 + 2/4 = 1.5 between two cells and 1 + 4/16 = 1.25 in the centre (measured 1.51-1.52 / 1.265-1.267: edges).
HP_SIGMA2 = ((0.531, 0.513, 0.471), (0.491, 0.501, 0.497), (0.376, 0.397, 0.357), (0.421, 0.390, 0.376))
HP_NU2 = (1.5, 1.5, 1.25)


def up2(x: torch.Tensor) -> torch.Tensor:
    """2x upsample anchored on the grid: out[2i, 2j] = x[i, j] exactly, the cells between are the mean of
    their 2 (or 4) grid neighbours (the last row/column repeats its edge). subsample(up2(x)) == x, which
    keeps the core's own trajectory exact inside the 2x model -- bilinear (align_corners=False) does not:
    it is offset by half a cell, and the drift compounded over 50 steps (tested 2026-09-26)."""
    xr = torch.cat([x, x[..., -1:]], -1)
    row = torch.stack([x, 0.5 * (xr[..., :-1] + xr[..., 1:])], -1).flatten(-2)          # (.., H, 2W)
    rr = torch.cat([row, row[..., -1:, :]], -2)
    return torch.stack([row, 0.5 * (rr[..., :-1, :] + rr[..., 1:, :])], -2).flatten(-3, -2)


def _phase_grid(vals3, grid_val=1.0):
    """(ch, 3) values for the (0,1), (1,0), (1,1) positions -> a (1, ch, 2, 2) tile, grid cell = grid_val."""
    t = torch.full((len(vals3), 2, 2), float(grid_val))
    v = torch.tensor(vals3, dtype=torch.float32)
    t[:, 0, 1], t[:, 1, 0], t[:, 1, 1] = v[:, 0], v[:, 1], v[:, 2]
    return t[None]


COUNT_K = (1, 2, 4, 8)
COUNT_W = (1.0, 0.7, 0.5, 0.35)


def count_features(n: torch.Tensor) -> torch.Tensor:
    """(..., 10) code of an object count n (0 = no count): angle theta = pi * log2(n) / 4 over 1..16 (half a
    circle, so 16 does not wrap onto 1) at frequencies 1, 2, 4, 8 (weights 1, .7, .5, .35) + log2(n) / 4 + an
    'over 16' flag. Log spacing makes small counts far apart (1-2 is 5.7x the 15-16 gap) and the harmonics keep
    large ones separable (14 vs 16 is 2x their neighbour gap); tests/test_thinker2_mr.py checks both."""
    nf = n.float()
    th = math.pi * torch.log2(nf.clamp(1, 16)) / 4
    parts = [w * f(k * th) for k, w in zip(COUNT_K, COUNT_W) for f in (torch.cos, torch.sin)]
    parts += [torch.log2(nf.clamp(min=1)) / 4, (nf > 16).float()]
    return torch.stack(parts, -1)


class CountCode(nn.Module):
    """At the text tokens that spell an object count (data/prompt_norm.find_counts), gate-replace the
    context vector by a learned projection of count_features(n): ctx' = ctx + g * (W phi(n) - ctx), g
    zero-init per channel -- exact at step 0, free to move all the way to replacement. Why: Ettin puts the
    number words at cosine 0.994-0.999 of each other, and a count change moved the thinker's plan only
    0.57 (a Title Case change: 0.43) -- tools/count_probe.py, text_probe.py, 2026-09-26."""

    def __init__(self, ctx_dim: int):
        super().__init__()
        self.proj = nn.Linear(10, ctx_dim)
        self.gate = nn.Parameter(torch.zeros(ctx_dim))

    def forward(self, ctx, counts=None):
        if counts is None:
            return ctx
        counts = counts.to(ctx.device)
        on = (counts > 0).to(ctx.dtype)[..., None]
        code = self.proj(count_features(counts).to(ctx.dtype))
        return ctx + on * self.gate.to(ctx.dtype) * (code - ctx)


class Thinker2MR(nn.Module):
    """T40r's thinker (read -> think -> region map) with the multi-resolution read and the
    resolution embedding. Module names of the original are kept, new ones are mr_*."""

    def __init__(self, ctx_dim: int, latent_ch: int, dim: int, p_dim: int, perceiver_depth: int,
                 prelude: int, core: int, coda: int, loops: int, grid: int, n_global: int, n_freq: int = 6,
                 rope: bool = True, read_ch: int = 96, read_depth: int = 2):
        super().__init__()
        self.grid, self.n_global, self.n_freq, self.rope = grid, n_global, n_freq, rope
        self.loops = loops
        self.native = 2 * grid                                    # the latent size the thinker was trained on
        pos = 4 * n_freq
        self.queries = nn.Parameter(torch.randn(grid * grid + n_global, p_dim) * 0.02)
        self.q_pos = nn.Linear(pos, p_dim)
        self.t1 = TimestepEmbedder(p_dim)
        self.t2 = TimestepEmbedder(dim)
        self.in_latent = nn.Linear(latent_ch * 4 + pos, p_dim)
        self.perceiver = nn.ModuleList(XBlock2(p_dim, p_dim, self_attn=False) for _ in range(perceiver_depth))
        self.lift = nn.Linear(p_dim, dim)
        self.prelude = nn.ModuleList(XBlock2(dim, ctx_dim) for _ in range(prelude))
        self.core = nn.ModuleList(XBlock2(dim, ctx_dim) for _ in range(core))
        self.coda = nn.ModuleList(XBlock2(dim, ctx_dim) for _ in range(coda))
        self.adapter = nn.Linear(2 * dim, dim)
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.weight[:, :dim].copy_(torch.eye(dim))
            self.adapter.bias.zero_()
        self.norm = nn.LayerNorm(dim)
        # --- multi-resolution additions, all zero at init
        self.mr_taps = zero_last(nn.Linear(latent_ch * 16, p_dim))      # the full 4x4 patch, linear
        self.mr_read = ConvRead(latent_ch, p_dim, cond_dim=p_dim, ch=read_ch, depth=read_depth)
        self.mr_res_q = ResEmbed(p_dim)
        self.mr_res_s = ResEmbed(dim)
        self.mr_res_c = ResEmbed(p_dim, zero_init=False)               # conditioning of mr_read (feeds FiLMs)

    def read(self, z, t_emb, r):
        """Latent (native or 2x) -> the perceiver's keys, (B, (native/2)^2, p_dim)."""
        B = z.shape[0]
        zb = subsample(z, self.native)                             # the trained read: exact at 256
        hb, wb = zb.shape[-2:]
        pos = fourier_coords(hb // 2, wb // 2, self.n_freq, z.device)[None].expand(B, -1, -1)
        lat = F.pixel_unshuffle(zb, 2).flatten(2).transpose(1, 2)
        kv = self.in_latent(torch.cat([lat, pos], -1))
        # the conv read runs on the 2x grid; the coordinates are taken at the TRUE latent grid and
        # upsampled with it, so at 256 px they step every 2 cells, at 512 px every cell
        coords = raw_coords(*z.shape[-2:], z.device)
        if z.shape[-1] == 2 * self.native:
            z_hr, c_hr = z, coords
        else:
            z_hr = F.interpolate(z, scale_factor=2, mode="nearest")
            c_hr = F.interpolate(coords, scale_factor=2, mode="nearest")
        taps = F.pixel_unshuffle(z_hr, 4).flatten(2).transpose(1, 2)   # 4x4 patch of the 2x grid
        cond = t_emb + self.mr_res_c(r)
        return kv + self.mr_taps(taps) + self.mr_read(z_hr, c_hr, cond, (hb // 2, wb // 2)), (hb, wb)

    def forward(self, z, t, ctx, mask, loops: int | None = None):
        B = z.shape[0]
        g, dev = self.grid, z.device
        loops = self.loops if loops is None else loops
        r = res_scale(z, self.native)
        t1 = self.t1(t)
        kv, (hb, wb) = self.read(z, t1, r)
        q = self.queries[None].expand(B, -1, -1).clone()
        q[:, :g * g] = q[:, :g * g] + self.q_pos(fourier_coords(g, g, self.n_freq, dev)[None].expand(B, -1, -1))
        q = q + t1[:, None] + self.mr_res_q(r)[:, None]
        ang_pq = ang_pk = ang_q = None
        if self.rope:
            ang_pq = rope_angles(grid_xy(g, g, dev), 64, self.n_global)
            ang_pk = rope_angles(grid_xy(hb // 2, wb // 2, dev), 64)
            ang_q = ang_pq
        for blk in self.perceiver:
            q = blk(q, kv, None, ang_pq, ang_pk)
        q = self.lift(q) + self.t2(t)[:, None] + self.mr_res_s(r)[:, None]
        pad = ~mask.bool()
        for blk in self.prelude:
            q = blk(q, ctx, pad, ang_q)
        e, s = q, q
        for _ in range(loops):
            s = self.adapter(torch.cat([s, e], -1))
            for blk in self.core:
                s = blk(s, ctx, pad, ang_q)
        for blk in self.coda:
            s = blk(s, ctx, pad, ang_q)
        out = self.norm(s)
        return out[:, :g * g].transpose(1, 2).reshape(B, -1, g, g)


class FCDMThinker2MR(nn.Module):
    def __init__(self, latent_ch: int = 4, C: int = 224, depths: tuple = (4, 6, 2), expansion: int = 3,
                 ctx_dim: int = 512, dim: int = 512, p_dim: int = 256, perceiver_depth: int = 4,
                 prelude: int = 2, core: int = 2, coda: int = 2, loops: int = 4,
                 loop_probs: tuple = (0.1, 0.1, 0.2, 0.6), grid: int = 16, n_global: int = 16,
                 rope: bool = True, widths: tuple | None = None, steer_rank: tuple | None = None,
                 grad_checkpoint: bool = False, read_ch: int = 96, read_depth: int = 2,
                 outer_width: int = 192, outer_depth: tuple = (2, 2), up_rank: int = 32,
                 hp_sigma2: tuple = HP_SIGMA2, hp_nu2: tuple = HP_NU2, outer_mode: str = "nested",
                 hp_prior: bool = True, coord_res: bool = False):
        """T40r's arguments, plus: read_ch/read_depth (the thinker's conv read), outer_width /
        outer_depth (the nested 64x64 level: blocks before/after the core), up_rank (PlanUp),
        hp_sigma2 / hp_nu2 (fine-scale variance of data per channel and cell position / of noise per
        position, for hp_prior), outer_mode ("nested": the core at the native size inside a 2x outer level;
        "flat": the whole U-Net runs at the latent's size -- the simpler alternative, wave-2 arm B0),
        hp_prior (the analytic fine-scale velocity at 2x; False = wave-2 arm B1), coord_res (an explicit
        resolution channel in every renderer CoordIn: the renderer is told 256 vs 512 directly)."""
        if outer_mode not in ("nested", "flat"):
            raise ValueError(f"outer_mode {outer_mode!r}")
        super().__init__()
        depths = tuple(depths)
        widths = tuple(widths) if widths else (C, 2 * C, 4 * C)
        steer_rank = tuple(steer_rank) if steer_rank else (0, 0, 0)
        outer_depth = tuple(outer_depth)
        self.config = dict(latent_ch=latent_ch, C=C, depths=depths, expansion=expansion, ctx_dim=ctx_dim,
                           dim=dim, p_dim=p_dim, perceiver_depth=perceiver_depth, prelude=prelude, core=core,
                           coda=coda, loops=loops, loop_probs=tuple(loop_probs), grid=grid, n_global=n_global,
                           rope=rope, widths=widths, steer_rank=steer_rank, read_ch=read_ch,
                           read_depth=read_depth, outer_width=outer_width, outer_depth=outer_depth,
                           up_rank=up_rank, hp_sigma2=tuple(tuple(c) for c in hp_sigma2), hp_nu2=tuple(hp_nu2),
                           outer_mode=outer_mode, hp_prior=hp_prior, coord_res=coord_res)
        self.outer_mode, self.use_hp_prior = outer_mode, hp_prior
        if len(loop_probs) != loops or abs(sum(loop_probs) - 1) > 1e-6:
            raise ValueError(f"loop_probs {loop_probs} must give one probability per loop count 1..{loops}")
        self.loop_probs = tuple(loop_probs)
        self.grad_checkpoint = grad_checkpoint
        self.stateful = False
        self.native = 2 * grid
        self.thinker = Thinker2MR(ctx_dim, latent_ch, dim, p_dim, perceiver_depth, prelude, core, coda, loops,
                                  grid, n_global, rope=rope, read_ch=read_ch, read_depth=read_depth)
        self.loops = loops
        d0, d1, d2 = depths
        w0, w1, w2 = widths
        self.steer_in = nn.ModuleDict({lvl: nn.Conv2d(dim, r, 1) for lvl, r in zip(("l0", "l1", "l2"), steer_rank)
                                       if r})
        d_in = [r or dim for r in steer_rank]
        self.stem = nn.Conv2d(latent_ch, w0, 3, padding=1)
        mk = lambda w, dm, n: nn.ModuleList(SpatialModBlock(w, dm, expansion) for _ in range(n))  # noqa: E731
        self.enc0, self.enc1, self.mid = mk(w0, d_in[0], d0), mk(w1, d_in[1], d1), mk(w2, d_in[2], 2 * d2)
        self.dec1, self.dec0 = mk(w1, d_in[1], d1), mk(w0, d_in[0], d0)
        self.down0 = nn.Sequential(LayerNorm2d(w0), nn.Conv2d(w0, w1, 2, stride=2))
        self.down1 = nn.Sequential(LayerNorm2d(w1), nn.Conv2d(w1, w2, 2, stride=2))
        self.up1 = nn.ConvTranspose2d(w2, w1, 2, stride=2)
        self.up0 = nn.ConvTranspose2d(w1, w0, 2, stride=2)
        self.final_norm = LayerNorm2d(w0, affine=False)
        self.final_mod = nn.Conv2d(dim, 2 * w0, 1)
        self.final_conv = nn.Conv2d(w0, latent_ch, 3, padding=1)
        for m in (self.final_mod, self.final_conv):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        # --- multi-resolution additions
        # the plan, upsampled once per output size: native (the core's l0 and output modulation) and 2x
        self.mr_plan_up = nn.ModuleDict({"x1": PlanUp(dim, up_rank, 5, latent_ch),
                                         "x2": PlanUp(dim, up_rank, 7, latent_ch)})
        # raw coordinates into every level of the core (zero-init: T40r unchanged at step 0)
        self.mr_coord = nn.ModuleDict({k: CoordIn(w, coord_res) for k, w in
                                       (("enc0", w0), ("enc1", w1), ("mid", w2), ("dec1", w1), ("dec0", w0))})
        # the nested outer level at 2x the native size (512 px)
        wo, (eo, do) = outer_width, outer_depth
        self.mr_outer_steer = nn.Conv2d(dim, d_in[0], 1)          # init_from_t2 copies steer_in.l0 into it
        self.mr_outer_stem = nn.Conv2d(latent_ch, wo, 3, padding=1)
        self.mr_outer_coord_enc, self.mr_outer_coord_dec = CoordIn(wo, coord_res), CoordIn(wo, coord_res)
        self.mr_outer_enc = mk(wo, d_in[0], eo)
        self.mr_outer_down = zero_last(nn.Sequential(LayerNorm2d(wo), nn.Conv2d(wo, w0, 2, stride=2)))
        self.mr_outer_lift = nn.ConvTranspose2d(w0, wo, 2, stride=2)
        self.mr_outer_dec = mk(wo, d_in[0], do)
        self.mr_outer_norm = LayerNorm2d(wo, affine=False)
        self.mr_outer_mod = zero_last(nn.Conv2d(dim, 2 * wo, 1))
        self.mr_outer_head = zero_last(nn.Conv2d(wo, latent_ch, 3, padding=1))
        self.mr_count = CountCode(ctx_dim)
        self.register_buffer("mr_hp_sigma2", _phase_grid(hp_sigma2))
        self.register_buffer("mr_hp_nu2", _phase_grid([tuple(hp_nu2)] * latent_ch))

    def hp_prior(self, z, zc, t):
        """The fine-scale velocity no upsampled native answer can give, estimated linearly: z_hp = z minus
        up2 of its subsample (0 on the grid cells); with z = t x1 + (1 - t) x0 and fine-scale variances
        sigma2 (data) and nu2 (noise) at each cell position, the least-squares estimate of v_hp =
        x1_hp - x0_hp from z_hp is a(t) z_hp, a = (t sigma2 - (1-t) nu2) / (t^2 sigma2 + (1-t)^2 nu2):
        -1 at pure noise (remove it), +1 at clean data, bounded between. Without it, step 0 at 512 leaves
        3/4 of the cells' noise in. Returns (z_hp, a), a tiled to z's size."""
        z_hp = z - up2(zc)
        tt = t.float().view(-1, 1, 1, 1)
        H, W = z.shape[-2:]
        s2 = self.mr_hp_sigma2.float().repeat(1, 1, H // 2, W // 2)
        n2 = self.mr_hp_nu2.float().repeat(1, 1, H // 2, W // 2)
        a = (tt * s2 - (1 - tt) * n2) / (tt ** 2 * s2 + (1 - tt) ** 2 * n2)
        return z_hp, a.to(z.dtype)

    def draw_loops(self, step: int, seed: int = 0) -> int:
        """The loop count for one training step: seeded by the step, so every rank (and a
        resumed run) draws the same number."""
        r = random.Random(seed * 1_000_003 + step).random()
        acc = 0.0
        for n, p in enumerate(self.loop_probs, 1):
            acc += p
            if r < acc:
                return n
        return len(self.loop_probs)

    def _run(self, blocks, x, s):
        for blk in blocks:
            if self.grad_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(blk, x, s, use_reentrant=False)
            else:
                x = blk(x, s)
        return x

    def _steer(self, lvl, m, x, m_up=None):
        """The level's steering map at the level's size: the level's projection of the plan -- of the
        upsampled plan m_up when the level is finer than the plan (PlanUp), pooled when coarser."""
        if m_up is not None and m_up.shape[-2:] == x.shape[-2:]:
            m = m_up
        s = self.steer_in[lvl](m) if lvl in self.steer_in else m
        if s.shape[-1] > x.shape[-1]:
            return F.adaptive_avg_pool2d(s, x.shape[-2:])
        if s.shape[-1] < x.shape[-1]:
            return F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return s

    def core(self, zc, m, extra=None, r: float = 0.0):
        """T40r's U-Net at the native size. extra: the outer level's contribution to its input.
        -> (velocity at the native size, the last features before the output head)."""
        m_up = self.mr_plan_up["x1"](m, zc) if m.shape[-1] < zc.shape[-1] else None
        h = self.stem(zc)
        if extra is not None:
            h = h + extra
        c = self.mr_coord
        x0 = self._run(self.enc0, c["enc0"](h, r), self._steer("l0", m, h, m_up))
        x = self.down0(x0)
        x1 = self._run(self.enc1, c["enc1"](x, r), self._steer("l1", m, x, m_up))
        x = self.down1(x1)
        x = self._run(self.mid, c["mid"](x, r), self._steer("l2", m, x, m_up))
        x = self.up1(x) + x1
        x = self._run(self.dec1, c["dec1"](x, r), self._steer("l1", m, x, m_up))
        x = self.up0(x) + x0
        feat = self._run(self.dec0, c["dec0"](x, r), self._steer("l0", m, x, m_up))
        fm = self.final_mod(m_up if m_up is not None else m)
        if fm.shape[-2:] != feat.shape[-2:]:
            fm = F.interpolate(fm, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        shift, scale = fm.chunk(2, dim=1)
        return self.final_conv(self.final_norm(feat) * (1 + scale) + shift), feat

    def forward(self, z, t, ctx, mask, loops: int | None = None, counts=None):
        """z: (B, 4, H, W) with H = W = native (256 px) or 2 x native (512 px); t: (B,) in [0, 1];
        ctx: (B, L, ctx_dim); mask: (B, L), 1 = real token; counts: optional (B, L) object count at the
        tokens that spell one (data/prompt_norm.count_tensor), 0 elsewhere. -> velocity at z's size."""
        ctx = self.mr_count(ctx.float(), counts)
        m = self.thinker(z, t, ctx, mask, self.loops if loops is None else loops)
        if z.shape[-1] == self.native or self.outer_mode == "flat":
            r = math.log2(z.shape[-1] / self.native)    # 0 native, 1 at 2x: the renderer's explicit size signal
            return self.core(z, m, r=r)[0]              # flat: the whole U-Net at z's size (PlanUp x1 to it)
        zc = subsample(z, self.native)
        m2 = self.mr_plan_up["x2"](m, z)                            # the plan at 64x64, for the outer level
        so = self.mr_outer_steer(m2)
        ho = self._run(self.mr_outer_enc, self.mr_outer_coord_enc(self.mr_outer_stem(z), 1.0), so)
        v_core, feat = self.core(zc, m, extra=self.mr_outer_down(ho), r=1.0)   # the core knows it is inside 2x
        xo = self._run(self.mr_outer_dec, self.mr_outer_coord_dec(self.mr_outer_lift(feat) + ho, 1.0), so)
        shift, scale = self.mr_outer_mod(m2).chunk(2, dim=1)
        detail = self.mr_outer_head(self.mr_outer_norm(xo) * (1 + scale) + shift)
        if not self.use_hp_prior:
            return up2(v_core) + detail
        z_hp, a = self.hp_prior(z, zc, t)
        return up2(v_core) + a * z_hp + detail


MR_PREFIX = "mr_"


def is_mr(name: str) -> bool:
    """A parameter/buffer name that belongs to the multi-resolution additions."""
    return any(part.startswith(MR_PREFIX) for part in name.split("."))


def init_from_t2(model: FCDMThinker2MR, state_dict: dict) -> list[str]:
    """Load a T40r (fcdm_t2) state dict: every old key must fit, only mr_* keys may be missing.
    The outer level's steering projection starts as a copy of l0's, so it first reads the slice
    of the plan that the finest core level reads. Returns the fresh (mr_*) keys."""
    res = model.load_state_dict(state_dict, strict=False)
    bad = [k for k in res.missing_keys if not is_mr(k)] + list(res.unexpected_keys)
    if bad:
        raise RuntimeError(f"not a T40r state dict for this model: {bad[:8]}")
    if "l0" in model.steer_in:
        with torch.no_grad():
            model.mr_outer_steer.weight.copy_(model.steer_in["l0"].weight)
            model.mr_outer_steer.bias.copy_(model.steer_in["l0"].bias)
    return list(res.missing_keys)
