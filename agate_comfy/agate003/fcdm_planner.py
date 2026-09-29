"""FCDM-P: a pure-convolution denoiser driven by a recurrent planner with memory.

Design (2026-09-24, from the fcdm2 ablations; see research_notes/daily/2026-09-24.txt):
the denoiser has NO attention and NO timestep modulation of its own. All global reasoning
and all conditioning live in a small transformer PLANNER that runs once per sampling step:

  PERCEIVER   planner queries (64 grid tokens = an 8x8 plan, 16 global, 16 memory tokens)
              read, by softmax cross-attention, the current noisy latent (2x2 patches) and
              both memory canvases -- every position carries Fourier coordinates, so the
              reads are position-aware. Cost is linear in the number of positions.
  PLANNER     self-attention over the queries + cross-attention to the text tokens, with
              the timestep added to every query. It is the only place t and the text enter.
  OUTPUTS     (1) ONE spatial MODULATION MAP (d_mod = 2048 per region of the 8x8 grid, a
              single linear layer on the planner's grid tokens -- the planner is a pure
              transformer, no MLP heads), shared by the whole denoiser: every conv block
              decodes its own shift/scale/gate from it with its own 1x1 at 8x8, bilinearly
              upsampled to the block's resolution -- the plan is smooth, the convs do detail;
              (2) the coarse canvas and the memory tokens for the next step (gated writes).

Memory carried from step to step (all start at zero, all writes gated and zero-initialised):
  coarse canvas  8x8   x mem_c    written by the planner's grid tokens (the plan, on the grid)
  mid canvas     16x16 x mem_m    written by the conv decoder at 16x16 (ConvGRU-style gate)
  memory tokens  16    x mem_tok  written by the planner (its working memory)
Both canvases are read by the convolutions (resized and added at every level) and by the
perceiver. Nothing in the loss constrains the memory: it is free to hold whatever helps
later steps (layout, bindings, counts, confidence).

Conventions: t=0 is noise, t=1 is data, the model predicts the velocity x1 - x0. One call
is one sampling step: forward(z, t, ctx, mask, state) -> (velocity, new_state); state=None
starts a chain. Classifier-free guidance keeps one state per branch.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcdm_t2i import GRN, LayerNorm2d, TimestepEmbedder

LEVELS = ("enc0", "enc1", "mid", "dec1", "dec0")


def fourier_coords(h: int, w: int, n_freq: int, device) -> torch.Tensor:
    """(h*w, 4*n_freq) sin/cos of normalised pixel-centre coordinates in [-1, 1]."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    f = (2.0 ** torch.arange(n_freq, device=device, dtype=torch.float32)) * math.pi
    feats = [fn(c.reshape(-1, 1) * f) for c in (xx, yy) for fn in (torch.sin, torch.cos)]
    return torch.cat(feats, 1)


class SpatialModBlock(nn.Module):
    """FCDM block (7x7 DWConv -> LN -> modulate -> 1x1 expand -> GELU -> GRN -> 1x1 -> gate)
    whose shift/scale/gate are per-PIXEL maps from the planner. The projection runs at the
    planner grid (8x8) and the result is upsampled; zero-initialised, so the block starts
    as the identity (gate 0)."""

    def __init__(self, dim: int, d_mod: int, expansion: int = 3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, affine=False)
        self.pw1 = nn.Conv2d(dim, expansion * dim, 1)
        self.act = nn.GELU()
        self.grn = GRN(expansion * dim)
        self.pw2 = nn.Conv2d(expansion * dim, dim, 1)
        self.mod = nn.Conv2d(d_mod, 3 * dim, 1)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        mod = self.mod(m)                                           # (B, 3C, 8, 8)
        if mod.shape[-2:] != x.shape[-2:]:
            mod = F.interpolate(mod, size=x.shape[-2:], mode="bilinear", align_corners=False)
        shift, scale, gate = mod.chunk(3, dim=1)
        h = self.norm(self.dwconv(x)) * (1 + scale) + shift
        h = self.pw2(self.grn(self.act(self.pw1(h))))
        return x + gate * h


class ConvLevel(nn.Module):
    """`depth` modulated blocks at one resolution; the resized canvases are added first
    through a zero-initialised 1x1 (the level starts blind to the memory)."""

    def __init__(self, dim: int, depth: int, d_mod: int, expansion: int, mem_in: int):
        super().__init__()
        self.blocks = nn.ModuleList(SpatialModBlock(dim, d_mod, expansion) for _ in range(depth))
        self.mem_in = nn.Conv2d(mem_in, dim, 1)
        nn.init.zeros_(self.mem_in.weight)
        nn.init.zeros_(self.mem_in.bias)

    def forward(self, x, m, mem):
        x = x + self.mem_in(F.interpolate(mem, size=x.shape[-2:], mode="bilinear", align_corners=False))
        for blk in self.blocks:
            x = blk(x, m)
        return x


class XBlock(nn.Module):
    """Pre-LN transformer layer: self-attention over the queries (optional), cross-attention
    to a context (optional), MLP. Queries get the timestep embedding added by the caller."""

    def __init__(self, dim: int, heads: int, ctx_dim: int | None, self_attn: bool = True, mlp: int = 2):
        super().__init__()
        self.sa = nn.MultiheadAttention(dim, heads, batch_first=True) if self_attn else None
        self.n_sa = nn.LayerNorm(dim) if self_attn else None
        self.xa = (nn.MultiheadAttention(dim, heads, batch_first=True, kdim=ctx_dim, vdim=ctx_dim)
                   if ctx_dim else None)
        self.n_xa = nn.LayerNorm(dim) if ctx_dim else None
        self.n_ctx = nn.LayerNorm(ctx_dim) if ctx_dim else None
        self.n_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(), nn.Linear(mlp * dim, dim))

    def forward(self, q, ctx=None, pad=None):
        if self.sa is not None:
            h = self.n_sa(q)
            q = q + self.sa(h, h, h, need_weights=False)[0]
        if self.xa is not None:
            c = self.n_ctx(ctx)
            q = q + self.xa(self.n_xa(q), c, c, key_padding_mask=pad, need_weights=False)[0]
        return q + self.mlp(self.n_mlp(q))


class Planner(nn.Module):
    """Perceiver (reads latent + canvases) + planner (reasons with the text and t)."""

    def __init__(self, dim: int, ctx_dim: int, latent_ch: int, mem_c: int, mem_m: int, mem_tok: int,
                 n_global: int, n_mem: int, perceiver_depth: int, planner_depth: int, n_freq: int = 6,
                 grid: int = 8, p_dim: int = 256):
        """Width grows through the stack: the perceiver READS at p_dim (its keys are the many
        latent/canvas positions, so narrow is where width is cheapest to save), the planner
        THINKS at dim, and the output heads (FCDMPlanner.heads) SPEAK wide."""
        super().__init__()
        self.grid, self.n_global, self.n_mem = grid, n_global, n_mem
        nq = grid * grid + n_global + n_mem
        self.queries = nn.Parameter(torch.randn(nq, p_dim) * 0.02)
        self.t_embed = TimestepEmbedder(p_dim)
        self.t_embed2 = TimestepEmbedder(dim)
        self.n_freq = n_freq
        pos = 4 * n_freq
        self.in_latent = nn.Linear(latent_ch * 4 + pos, p_dim)           # 2x2 patches
        self.in_coarse = nn.Linear(mem_c + pos, p_dim)
        self.in_mid = nn.Linear(mem_m + pos, p_dim)
        self.src = nn.Parameter(torch.zeros(3, p_dim))                    # which input a token is
        self.grid_pos = nn.Linear(pos, p_dim)
        self.mem_in = nn.Linear(mem_tok, p_dim)
        self.perceiver = nn.ModuleList(XBlock(p_dim, max(1, p_dim // 64), p_dim, self_attn=False)
                                       for _ in range(perceiver_depth))
        self.lift = nn.Linear(p_dim, dim)
        self.planner = nn.ModuleList(XBlock(dim, max(1, dim // 64), ctx_dim) for _ in range(planner_depth))
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, z, t, ctx, mask, coarse, mid, mem_tokens):
        B = z.shape[0]
        dev = z.device
        pos = lambda h, w: fourier_coords(h, w, self.n_freq, dev)[None].expand(B, -1, -1)  # noqa: E731
        H, W = z.shape[-2:]
        lat = F.pixel_unshuffle(z, 2).flatten(2).transpose(1, 2)                  # (B, HW/4, 4C)
        inputs = torch.cat([
            self.in_latent(torch.cat([lat, pos(H // 2, W // 2)], -1)) + self.src[0],
            self.in_coarse(torch.cat([coarse.flatten(2).transpose(1, 2), pos(*coarse.shape[-2:])], -1)) + self.src[1],
            self.in_mid(torch.cat([mid.flatten(2).transpose(1, 2), pos(*mid.shape[-2:])], -1)) + self.src[2]], 1)
        q = self.queries[None].expand(B, -1, -1).clone()
        g2 = self.grid * self.grid
        q[:, :g2] = q[:, :g2] + self.grid_pos(pos(self.grid, self.grid))
        q[:, -self.n_mem:] = q[:, -self.n_mem:] + self.mem_in(mem_tokens)
        q = q + self.t_embed(t)[:, None]
        for blk in self.perceiver:
            q = blk(q, inputs)
        q = self.lift(q) + self.t_embed2(t)[:, None]
        pad = ~mask.bool()
        for blk in self.planner:
            q = blk(q, ctx, pad)
        return self.norm_out(q)


class FCDMPlanner(nn.Module):
    def __init__(self, latent_ch: int = 4, C: int = 224, depths=(3, 6, 12, 6, 3), expansion: int = 3,
                 ctx_dim: int = 512, ctx_std: float = 1.0, dim: int = 640, d_mod: int = 2048,
                 p_dim: int = 256,
                 perceiver_depth: int = 4, planner_depth: int = 8, n_global: int = 16, n_mem: int = 16,
                 mem_c: int = 16, mem_m: int = 8, mem_tok: int = 256, grid: int = 8,
                 grad_checkpoint: bool = False):
        super().__init__()
        depths = tuple(int(d) for d in depths)
        self.config = dict(latent_ch=latent_ch, C=C, depths=list(depths), expansion=expansion, ctx_dim=ctx_dim,
                           ctx_std=ctx_std, dim=dim, d_mod=d_mod, p_dim=p_dim,
                           perceiver_depth=perceiver_depth,
                           planner_depth=planner_depth, n_global=n_global, n_mem=n_mem, mem_c=mem_c,
                           mem_m=mem_m, mem_tok=mem_tok, grid=grid)
        self.grad_checkpoint = grad_checkpoint
        self.grid, self.mem_c, self.mem_m, self.mem_tok, self.n_mem = grid, mem_c, mem_m, mem_tok, n_mem
        self.ctx_scale = nn.Parameter(torch.tensor(1.0 / ctx_std))           # one global scalar
        self.planner = Planner(dim, ctx_dim, latent_ch, mem_c, mem_m, mem_tok, n_global, n_mem,
                               perceiver_depth, planner_depth, grid=grid, p_dim=p_dim)
        # per-level decoding of the plan into a modulation map (+ the output head)
        # the planner's grid tokens -> one d_mod-wide instruction per region, a single linear layer
        self.to_mod = nn.Linear(dim, d_mod)
        mem_in = mem_c + mem_m
        self.stem = nn.Conv2d(latent_ch + mem_in, C, 3, padding=1)
        widths = (C, 2 * C, 4 * C, 2 * C, C)
        self.levels = nn.ModuleDict({k: ConvLevel(w, d, d_mod, expansion, mem_in)
                                     for k, w, d in zip(LEVELS, widths, depths)})
        self.down0 = nn.Sequential(LayerNorm2d(C), nn.Conv2d(C, 2 * C, 2, stride=2))
        self.down1 = nn.Sequential(LayerNorm2d(2 * C), nn.Conv2d(2 * C, 4 * C, 2, stride=2))
        self.up1 = nn.ConvTranspose2d(4 * C, 2 * C, 2, stride=2)
        self.up0 = nn.ConvTranspose2d(2 * C, C, 2, stride=2)
        self.final_norm = LayerNorm2d(C, affine=False)
        self.final_mod = nn.Conv2d(d_mod, 2 * C, 1)
        self.final_conv = nn.Conv2d(C, latent_ch, 3, padding=1)
        for m in (self.final_mod, self.final_conv):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        # memory writes (gated; candidate zero-initialised, gates biased shut -> memory starts ~0)
        self.w_coarse = nn.Linear(dim, 2 * mem_c)
        self.w_tokens = nn.Linear(dim, 2 * mem_tok)
        self.w_mid = nn.Conv2d(2 * C + mem_m, 2 * mem_m, 3, padding=1)
        for lin, n in ((self.w_coarse, mem_c), (self.w_tokens, mem_tok)):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            lin.bias.data[:n] = -2.0                                         # update gate ~0.12
        nn.init.zeros_(self.w_mid.weight)
        nn.init.zeros_(self.w_mid.bias)
        self.w_mid.bias.data[:mem_m] = -2.0

    # ---- state ------------------------------------------------------------------------------
    def init_state(self, B: int, H: int, W: int, device, dtype=torch.float32) -> dict:
        g = self.grid
        return {"coarse": torch.zeros(B, self.mem_c, g, g, device=device, dtype=dtype),
                "mid": torch.zeros(B, self.mem_m, H // 2, W // 2, device=device, dtype=dtype),
                "tokens": torch.zeros(B, self.n_mem, self.mem_tok, device=device, dtype=dtype)}

    @staticmethod
    def _gate(old, raw, n):
        z, cand = torch.sigmoid(raw[..., :n]), torch.tanh(raw[..., n:])
        return (1 - z) * old + z * cand

    # ---- one sampling step -----------------------------------------------------------------
    def forward(self, z, t, ctx, mask, state: dict | None = None):
        """z: (B, 4, H, W) noisy latent (H, W divisible by 4 and by the 8x8 grid); t: (B,)
        in [0, 1] (0 = noise); ctx: (B, L, ctx_dim) text states, mask (B, L).
        Returns (velocity (B, 4, H, W), new state)."""
        B, _, H, W = z.shape
        if state is None:
            state = self.init_state(B, H, W, z.device)
        coarse, mid, tokens = state["coarse"], state["mid"], state["tokens"]
        ctx = ctx.float() * self.ctx_scale
        q = self.planner(z, t, ctx, mask, coarse, mid, tokens)                 # (B, Nq, dim)
        g, grid_tok, mem_tok = self.grid, q[:, :self.grid ** 2], q[:, -self.n_mem:]
        mod_map = self.to_mod(grid_tok).transpose(1, 2).reshape(B, -1, g, g)       # (B, d_mod, 8, 8)
        maps = {k: mod_map for k in LEVELS + ("out",)}
        mem_full = torch.cat([F.interpolate(coarse.to(z.dtype), size=(H, W), mode="bilinear", align_corners=False),
                              F.interpolate(mid.to(z.dtype), size=(H, W), mode="bilinear", align_corners=False)], 1)

        def run(level, x):
            if self.grad_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(self.levels[level], x, maps[level], mem_full, use_reentrant=False)
            return self.levels[level](x, maps[level], mem_full)

        x0 = run("enc0", self.stem(torch.cat([z, mem_full], 1)))
        x1 = run("enc1", self.down0(x0))
        x = run("mid", self.down1(x1))
        x = run("dec1", self.up1(x) + x1)
        new_mid = self._gate(mid.float().permute(0, 2, 3, 1),
                             self.w_mid(torch.cat([x, mid.to(x.dtype)], 1)).float().permute(0, 2, 3, 1),
                             self.mem_m).permute(0, 3, 1, 2)
        x = run("dec0", self.up0(x) + x0)
        fm = F.interpolate(self.final_mod(maps["out"]), size=(H, W), mode="bilinear", align_corners=False)
        shift, scale = fm.chunk(2, dim=1)
        v = self.final_conv(self.final_norm(x) * (1 + scale) + shift)
        new_coarse = self._gate(coarse.float().permute(0, 2, 3, 1),
                                self.w_coarse(grid_tok).float().reshape(B, g, g, -1), self.mem_c).permute(0, 3, 1, 2)
        new_tokens = self._gate(tokens.float(), self.w_tokens(mem_tok).float(), self.mem_tok)
        return v, {"coarse": new_coarse, "mid": new_mid, "tokens": new_tokens}


@torch.no_grad()
def sample_chain(model, noise, ctx, mask, uncond_ctx=None, uncond_mask=None, steps: int = 32, cfg: float = 3.0):
    """Euler over t = i/steps (Supra's schedule), carrying one memory state per CFG branch."""
    z, dt = noise, 1.0 / steps
    B = noise.shape[0]
    use_cfg = cfg > 1.0 and uncond_ctx is not None
    if use_cfg:
        uc = uncond_ctx.expand(B, -1, -1) if uncond_ctx.shape[0] == 1 else uncond_ctx
        um = uncond_mask.expand(B, -1) if uncond_mask.shape[0] == 1 else uncond_mask
    s_c = s_u = None
    for i in range(steps):
        t = torch.full((B,), i / steps, device=z.device)
        v_c, s_c = model(z, t, ctx, mask, s_c)
        if use_cfg:
            v_u, s_u = model(z, t, uc, um, s_u)
            v_c = v_u + cfg * (v_c - v_u)
        z = z + dt * v_c.float()
    return z
