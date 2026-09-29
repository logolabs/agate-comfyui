"""FCDM text-to-image backbone.

The block and U-Net follow "Reviving ConvNeXt for Efficient Convolutional Diffusion
Models" (Kwon et al., arXiv 2603.09408):

  FCDM block   7x7 DWConv -> LayerNorm -> AdaLN (x*(1+gamma)+beta) -> 1x1 expand (r=3)
               -> GELU -> GRN -> 1x1 project -> gate alpha -> residual.  gamma, beta,
               alpha come from the conditioning vector c through a zero-initialised
               linear, so every block starts as the identity (DiT's adaLN-Zero).
  U-Net        3 levels (32^2, 16^2, 8^2 for a 32x32 latent), channels C, 2C, 4C,
               depths [L, 2L, 4L, 2L, L], encoder/decoder symmetric.

Two things the paper does not pin down, chosen here:
  * down = LayerNorm + 2x2 stride-2 conv (the ConvNeXt downsampler); up = 2x2 stride-2
    transposed conv; the skip is ADDED, as the paper's Figure 3c draws it.
  * text. The paper's text-to-image appendix (H) conditions on a POOLED text vector only
    and names token-level conditioning as future work. Pooled text cannot bind attributes
    to objects ("red cube, blue sphere"), and Supra2-IMG -- the model this replicates --
    cross-attends to the Flan-T5 tokens in every block. So text_mode="pooled_xattn" adds
    cross-attention after every block at the 16^2 and 8^2 levels (cheap: 256 and 64
    queries against <=128 keys); the 32^2 level stays pure convolution. text_mode="pooled"
    is the paper-faithful variant.

Conventions match Supra2-IMG's sampler: t=0 is noise, t=1 is data, the model predicts
the velocity x1 - x0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T5_DIM = 768   # Flan-T5-Base hidden size


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN on a (B, C, H, W) map with (B, C) shift/scale."""
    return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class LayerNorm2d(nn.Module):
    """LayerNorm over channels of a (B, C, H, W) map (ConvNeXt's channels_first LN).

    Routed through F.layer_norm so autocast runs it in fp32 under bf16 training."""

    def __init__(self, dim: int, affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2), channels_first. Identity at init."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x.float(), dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx.to(x.dtype)) + self.beta + x


class FCDMBlock(nn.Module):
    def __init__(self, dim: int, c_dim: int, expansion: int = 3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, affine=False)        # AdaLN supplies scale/shift
        self.pw1 = nn.Conv2d(dim, expansion * dim, 1)
        self.act = nn.GELU()
        self.grn = GRN(expansion * dim)
        self.pw2 = nn.Conv2d(expansion * dim, dim, 1)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(c_dim, 3 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.ada(c).chunk(3, dim=1)
        h = modulate(self.norm(self.dwconv(x)), shift, scale)
        h = self.pw2(self.grn(self.act(self.pw1(h))))
        return x + gate[:, :, None, None] * h


class CrossAttn2d(nn.Module):
    """Image positions attend to the (masked) text tokens. Output projection is
    zero-initialised, so the layer starts as the identity."""

    def __init__(self, dim: int, ctx_dim: int = T5_DIM, head_dim: int = 64):
        super().__init__()
        if dim % head_dim:
            raise ValueError(f"cross-attention width {dim} is not a multiple of head_dim {head_dim}")
        self.heads, self.head_dim = dim // head_dim, head_dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(ctx_dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = x.flatten(2).transpose(1, 2)                                     # (B, HW, C)
        q = self.q(self.norm(s)).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.bool()[:, None, None, :])
        s = s + self.out(a.transpose(1, 2).reshape(B, -1, C))
        return s.transpose(1, 2).reshape(B, C, H, W)


class TimestepEmbedder(nn.Module):
    """Sinusoidal t (scaled x1000, as Supra2-IMG does) -> MLP."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None] * 1000.0
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


def masked_mean(ctx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over real tokens only. Flan-T5 has no pooled output; Sentence-T5 showed the
    masked mean is a sound sentence embedding."""
    m = mask.to(ctx.dtype).unsqueeze(-1)
    return (ctx * m).sum(1) / m.sum(1).clamp(min=1.0)


class Level(nn.Module):
    """`depth` FCDM blocks at one resolution, optionally each followed by cross-attention."""

    def __init__(self, dim: int, depth: int, c_dim: int, expansion: int, xattn: bool,
                 ctx_dim: int, head_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(FCDMBlock(dim, c_dim, expansion) for _ in range(depth))
        self.xattn = (nn.ModuleList(CrossAttn2d(dim, ctx_dim, head_dim) for _ in range(depth))
                      if xattn else None)

    def forward(self, x, c, ctx, mask):
        for i, blk in enumerate(self.blocks):
            x = blk(x, c)
            if self.xattn is not None:
                x = self.xattn[i](x, ctx, mask)
        return x


# Parameter-matched to Supra2-IMG's DiT (104.09M, measured by models.count_params):
#   pooled_xattn  C=224 c_dim=464 -> 104.087M     pooled  C=256 c_dim=778 -> 104.072M
# Trainable backbone only -- like Supra's 104.1M, excluding the frozen T5 and VAE.
PARAM_MATCHED = {"pooled_xattn": dict(C=224, c_dim=464), "pooled": dict(C=256, c_dim=778)}


class FCDMT2I(nn.Module):
    TEXT_MODES = ("pooled", "pooled_xattn")

    def __init__(self, latent_ch: int = 4, C: int = 224, L: int = 2, c_dim: int = 464,
                 expansion: int = 3, text_mode: str = "pooled_xattn", ctx_dim: int = T5_DIM,
                 head_dim: int = 64, grad_checkpoint: bool = False):
        super().__init__()
        if text_mode not in self.TEXT_MODES:
            raise ValueError(f"text_mode must be one of {self.TEXT_MODES}")
        xa = text_mode == "pooled_xattn"
        self.config = dict(latent_ch=latent_ch, C=C, L=L, c_dim=c_dim, expansion=expansion,
                           text_mode=text_mode, ctx_dim=ctx_dim, head_dim=head_dim)
        self.grad_checkpoint = grad_checkpoint

        self.t_embed = TimestepEmbedder(c_dim)
        self.text_pool = nn.Sequential(nn.LayerNorm(ctx_dim), nn.Linear(ctx_dim, c_dim),
                                       nn.SiLU(), nn.Linear(c_dim, c_dim))
        self.stem = nn.Conv2d(latent_ch, C, 3, padding=1)

        mk = lambda d, n, x: Level(d, n, c_dim, expansion, x, ctx_dim, head_dim)  # noqa: E731
        self.enc0 = mk(C, L, False)
        self.down0 = nn.Sequential(LayerNorm2d(C), nn.Conv2d(C, 2 * C, 2, stride=2))
        self.enc1 = mk(2 * C, 2 * L, xa)
        self.down1 = nn.Sequential(LayerNorm2d(2 * C), nn.Conv2d(2 * C, 4 * C, 2, stride=2))
        self.mid = mk(4 * C, 4 * L, xa)
        self.up1 = nn.ConvTranspose2d(4 * C, 2 * C, 2, stride=2)
        self.dec1 = mk(2 * C, 2 * L, xa)
        self.up0 = nn.ConvTranspose2d(2 * C, C, 2, stride=2)
        self.dec0 = mk(C, L, False)

        self.final_norm = LayerNorm2d(C, affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(c_dim, 2 * C))
        self.final_conv = nn.Conv2d(C, latent_ch, 3, padding=1)
        for m in (self.final_ada[1], self.final_conv):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _run(self, level: Level, x, c, ctx, mask):
        if self.grad_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(level, x, c, ctx, mask, use_reentrant=False)
        return level(x, c, ctx, mask)

    def forward(self, z: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """z: (B, 4, H, W) latent (H, W divisible by 4); t: (B,) in [0, 1];
        ctx: (B, L, 768) Flan-T5 hidden states; mask: (B, L), 1 = real token."""
        c = self.t_embed(t) + self.text_pool(masked_mean(ctx.float(), mask))
        x0 = self._run(self.enc0, self.stem(z), c, ctx, mask)
        x1 = self._run(self.enc1, self.down0(x0), c, ctx, mask)
        x = self._run(self.mid, self.down1(x1), c, ctx, mask)
        x = self._run(self.dec1, self.up1(x) + x1, c, ctx, mask)
        x = self._run(self.dec0, self.up0(x) + x0, c, ctx, mask)
        shift, scale = self.final_ada(c).chunk(2, dim=1)
        return self.final_conv(modulate(self.final_norm(x), shift, scale))
