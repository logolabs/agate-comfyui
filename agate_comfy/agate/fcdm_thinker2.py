"""FCDM-T2: the thinker model (models/fcdm_thinker.py, "Run 1") rebuilt from its ablations.

A copy, not a subclass: Run 1's checkpoints must keep loading into the exact module they
trained with. What changed, and the measurement behind each change (2026-09-24 notes):

  renderer   Run 1's 8x8 level held 50.0M of 102.5M parameters in 8 blocks that each
             cost <= 1.5% loss when removed, while the 32x32 blocks (enc0/dec0) were the
             most critical in the model (+113-121%) and Run 1 trailed the DiT only at
             t >= 0.8 (fine detail). Depths per level are now explicit: the default
             (4, 6, 2) puts 4 blocks at 32^2, 6 at 16^2 and 2 at 8^2 on each side of the U.
  thinking   the planner is a RECURRENT core: `prelude` blocks, then `core` blocks applied
             `loops` times (each loop re-injects the prelude output through a zero-init
             adapter, as in recurrent-depth models), then `coda` blocks. Run 1's middle
             planner layers each did 2-3% of the work; sharing weights across loops spends
             that parameter budget once and lets inference choose the depth. Training draws
             the loop count per step (all ranks agree), so one run yields the loss at
             1..max loops for free.
  position   2D RoPE (fp32, applied after QK-norm) in the planner's self-attention and the
             perceiver's cross-attention: attention scores between two cells depend on their
             offset, the relation "left of" / "above" needs -- GenEval position was 0.14.
             Coordinates are fractions of the image, so the same angles serve 256 and 512.
             Global tokens are not rotated. The additive Fourier coordinates are kept.
  stability  QK RMSNorm on every attention from the first step (fcdm2 diverged from
             attention-logit growth and needed QK-norm ramped in afterwards).
  text       any encoder via ctx_dim (Ettin-68M: 512); text tokens carry no 2D position.
  widths     per-level widths (32^2, 16^2, 8^2) instead of C/2C/4C, and a LOW-RANK steering
             path (steer_rank): each level projects the region map once to r channels and its
             blocks decode shift/scale/gate from those. Run 1's effective-rank analysis
             (tools/effective_rank.py, 2026-09-24): 32^2 used 102-187 of 224 stream dims while
             16^2/8^2 MLPs used 93-96% of theirs, and the steering maps at 32^2 had participation
             ratio 2-3 (weight stable rank 2-8) -- width and steering rank belong where they are used.

No memory: the nested-chain memory run (2936742) is still being evaluated; if it earns its
place it is ported here as its own copy.

Conventions match v1: t=0 noise, t=1 data, predicts the velocity x1 - x0.
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


class Thinker2(nn.Module):
    """read (perceiver over 2x2 latent patches, RoPE) -> think (prelude, `loops` x core, coda;
    self-attention with RoPE + cross-attention to the text) -> region map."""

    def __init__(self, ctx_dim: int, latent_ch: int, dim: int, p_dim: int, perceiver_depth: int,
                 prelude: int, core: int, coda: int, loops: int, grid: int, n_global: int, n_freq: int = 6,
                 rope: bool = True):
        super().__init__()
        self.grid, self.n_global, self.n_freq, self.rope = grid, n_global, n_freq, rope
        self.loops = loops
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
        # loop input: the state and the prelude output, merged; zero-init on the prelude half
        # so loop 1 sees exactly the prelude output (state == prelude output there)
        self.adapter = nn.Linear(2 * dim, dim)
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.weight[:, :dim].copy_(torch.eye(dim))
            self.adapter.bias.zero_()
        self.norm = nn.LayerNorm(dim)

    def forward(self, z, t, ctx, mask, loops: int | None = None):
        B, _, H, W = z.shape
        g, dev = self.grid, z.device
        loops = self.loops if loops is None else loops
        pos = lambda h, w: fourier_coords(h, w, self.n_freq, dev)[None].expand(B, -1, -1)  # noqa: E731
        lat = F.pixel_unshuffle(z, 2).flatten(2).transpose(1, 2)
        kv = self.in_latent(torch.cat([lat, pos(H // 2, W // 2)], -1))
        q = self.queries[None].expand(B, -1, -1).clone()
        q[:, :g * g] = q[:, :g * g] + self.q_pos(pos(g, g))
        q = q + self.t1(t)[:, None]
        ang_pq = ang_pk = ang_q = None
        if self.rope:
            ang_pq = rope_angles(grid_xy(g, g, dev), 64, self.n_global)          # perceiver width p_dim
            ang_pk = rope_angles(grid_xy(H // 2, W // 2, dev), 64)
            ang_q = ang_pq
        for blk in self.perceiver:
            q = blk(q, kv, None, ang_pq, ang_pk)
        q = self.lift(q) + self.t2(t)[:, None]
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


class FCDMThinker2(nn.Module):
    def __init__(self, latent_ch: int = 4, C: int = 224, depths: tuple = (4, 6, 2), expansion: int = 3,
                 ctx_dim: int = 512, dim: int = 512, p_dim: int = 256, perceiver_depth: int = 4,
                 prelude: int = 2, core: int = 2, coda: int = 2, loops: int = 4,
                 loop_probs: tuple = (0.1, 0.1, 0.2, 0.6), grid: int = 16, n_global: int = 16,
                 rope: bool = True, widths: tuple | None = None, steer_rank: tuple | None = None,
                 grad_checkpoint: bool = False):
        """widths: (32^2, 16^2, 8^2) level widths, default (C, 2C, 4C). steer_rank: per-level rank
        of the steering path, 0 = full (each block decodes straight from the dim-wide map)."""
        super().__init__()
        depths = tuple(depths)
        widths = tuple(widths) if widths else (C, 2 * C, 4 * C)
        steer_rank = tuple(steer_rank) if steer_rank else (0, 0, 0)
        self.config = dict(latent_ch=latent_ch, C=C, depths=depths, expansion=expansion, ctx_dim=ctx_dim,
                           dim=dim, p_dim=p_dim, perceiver_depth=perceiver_depth, prelude=prelude, core=core,
                           coda=coda, loops=loops, loop_probs=tuple(loop_probs), grid=grid, n_global=n_global,
                           rope=rope, widths=widths, steer_rank=steer_rank)
        if len(loop_probs) != loops or abs(sum(loop_probs) - 1) > 1e-6:
            raise ValueError(f"loop_probs {loop_probs} must give one probability per loop count 1..{loops}")
        self.loop_probs = tuple(loop_probs)
        self.grad_checkpoint = grad_checkpoint
        self.stateful = False
        self.thinker = Thinker2(ctx_dim, latent_ch, dim, p_dim, perceiver_depth, prelude, core, coda, loops,
                                grid, n_global, rope=rope)
        self.loops = loops                    # used by forward; the trainer sets it per step
        d0, d1, d2 = depths
        w0, w1, w2 = widths
        # steering path per level: the region map (dim) -> r channels once, shared by the level's
        # blocks (r = 0: no projection, every block reads the full map)
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

    @staticmethod
    def _at(m, x):
        return F.adaptive_avg_pool2d(m, x.shape[-2:]) if m.shape[-1] > x.shape[-1] else m

    def _level(self, blocks, x, m, lvl):
        if lvl in self.steer_in:                      # low-rank steering: project once per level
            m = self.steer_in[lvl](m)
        m = self._at(m, x)
        for blk in blocks:
            if self.grad_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(blk, x, m, use_reentrant=False)
            else:
                x = blk(x, m)
        return x

    def forward(self, z, t, ctx, mask, loops: int | None = None):
        """z: (B, 4, H, W), H and W divisible by 4 and by the grid; t: (B,) in [0, 1];
        ctx: (B, L, ctx_dim) text states; mask: (B, L), 1 = real token; loops: thinking
        loops (default self.loops). -> velocity."""
        m = self.thinker(z, t, ctx.float(), mask, self.loops if loops is None else loops)
        x0 = self._level(self.enc0, self.stem(z), m, "l0")
        x1 = self._level(self.enc1, self.down0(x0), m, "l1")
        x = self._level(self.mid, self.down1(x1), m, "l2")
        x = self._level(self.dec1, self.up1(x) + x1, m, "l1")
        x = self._level(self.dec0, self.up0(x) + x0, m, "l0")
        fm = self.final_mod(m)
        if fm.shape[-2:] != x.shape[-2:]:
            fm = F.interpolate(fm, size=x.shape[-2:], mode="bilinear", align_corners=False)
        shift, scale = fm.chunk(2, dim=1)
        return self.final_conv(self.final_norm(x) * (1 + scale) + shift)
