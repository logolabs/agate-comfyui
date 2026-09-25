"""AgatePipeline: prompt -> 256x256 image.

    from agate import AgatePipeline
    pipe = AgatePipeline.from_pretrained("Logolabs/agate-preview-001", device="cuda")
    images = pipe("a red cube on top of a blue sphere", seed=0)            # list of PIL images
    images = pipe("a portrait of an old sailor", autoguide=1.0, cfg=4.0)    # sharper faces (loads the guide)

Sampler: Euler from t=0 (noise) to t=1 (data), 50 steps, classifier-free guidance 3 against the
empty prompt; the model predicts the velocity. On CUDA each denoising step is recorded once as a
CUDA graph and replayed (the model is small enough that kernel launches, not arithmetic, dominate
an eager step); on CPU it runs eagerly.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .fcdm_thinker2 import FCDMThinker2
from .text_encoder import EttinTextEncoder

BUCKETS = (64, 128, 256, 512)


def _resolve(repo_or_dir: str) -> Path:
    p = Path(repo_or_dir)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_or_dir))


def _pad_to(ctx: torch.Tensor, mask: torch.Tensor, L: int):
    return F.pad(ctx, (0, 0, 0, L - ctx.shape[1])), F.pad(mask, (0, L - mask.shape[1]))


def _bucket(*lengths: int) -> int:
    need = max(lengths)
    return next((b for b in BUCKETS if b >= need), BUCKETS[-1])


class _Graphed:
    """model(z, t, ctx, mask) recorded as a CUDA graph per input shape and replayed."""

    def __init__(self, model):
        self.model, self.cache = model, {}

    def _run(self, *a):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(*a)

    def __call__(self, z, t, ctx, mask):
        key = (tuple(z.shape), tuple(ctx.shape))
        if key not in self.cache:
            static = [z.clone(), t.clone(), ctx.clone(), mask.clone()]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):                      # warm-up: cuDNN autotune, allocator
                    self._run(*static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._run(*static)
            self.cache[key] = (graph, static, out)
        graph, static, out = self.cache[key]
        for dst, src in zip(static, (z, t, ctx, mask)):
            dst.copy_(src)
        graph.replay()
        return out.float()


class _Eager:
    def __init__(self, model):
        self.model = model

    def __call__(self, z, t, ctx, mask):
        with torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=z.device.type == "cuda"):
            return self.model(z, t, ctx, mask).float()


class AgatePipeline:
    def __init__(self, root: Path, device: str = "cuda", fast_vae: bool = False, cuda_graphs: bool = True):
        from diffusers import AutoencoderKL, AutoencoderTiny
        self.root, self.device = Path(root), torch.device(device)
        self.cfg = json.loads((self.root / "config.json").read_text())
        on_cuda = self.device.type == "cuda"
        if on_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.enable_cudnn_sdp(False)   # the attention kernels Agate was trained with
        self._graphs = on_cuda and cuda_graphs
        self._dtype = torch.bfloat16 if on_cuda else torch.float32
        self.model = self._load_generator(self.root / "generator.safetensors")
        self.text = self._load_text(self.root / self.cfg["text_encoder"], self.cfg["text_max_len"])
        self.guide = self.guide_text = None
        vdtype = torch.float16 if on_cuda else torch.float32
        if fast_vae:
            self.vae = AutoencoderTiny.from_pretrained(self.cfg["fast_vae"], torch_dtype=vdtype)
            self.vae_div = 1.0                            # TAESD decodes the scaled latents directly
        else:
            self.vae = AutoencoderKL.from_pretrained(self.cfg["vae"], torch_dtype=vdtype)
            self.vae_div = self.cfg["vae_scale"]
        self.vae = self.vae.to(self.device).eval()

    @classmethod
    def from_pretrained(cls, repo_or_dir: str = "Logolabs/agate-preview-001", device: str = "cuda", **kw):
        return cls(_resolve(repo_or_dir), device=device, **kw)

    def _load_generator(self, path: Path):
        from safetensors.torch import load_file
        m = FCDMThinker2(**self.cfg["model_kw"])
        m.load_state_dict(load_file(str(path)))
        m = m.to(self.device, dtype=self._dtype).eval()
        if self.device.type == "cuda":
            m = m.to(memory_format=torch.channels_last)
        return _Graphed(m) if self._graphs else _Eager(m)

    def _load_text(self, path: Path, max_len: int) -> EttinTextEncoder:
        text = EttinTextEncoder(str(path), self.device, max_len=max_len)
        text.model.to(dtype=self._dtype)
        return text

    def _load_guide(self):
        g = self.cfg["guide"]
        self.guide = self._load_generator(self.root / g["generator"])
        self.guide_text = self._load_text(self.root / g["text_encoder"], g["text_max_len"])

    @torch.no_grad()
    def __call__(self, prompt: str, negative_prompt: str = "", seed: int = 0, steps: int = 50, cfg: float = 3.0,
                 num_images: int = 1, autoguide: float = 0.0, callback=None, output_type: str = "pil"):
        """autoguide > 0 also steers away from the guide model (Agate's own step-27,600 checkpoint):
        v = v_uncond + cfg (v_cond - v_uncond) + autoguide (v_cond - v_guide). Try 1.0 with cfg 4.

        comfyui-agate additions (not in the release package): `callback(step, steps)` is called
        after every denoising step, and output_type="pt" returns a float tensor (B, H, W, 3) in
        [0, 1] on the CPU instead of a list of PIL images."""
        n, dev = int(num_images), self.device
        if autoguide and self.guide is None:
            self._load_guide()
        ctx, mask = self.text([prompt] * n)
        u_ctx, u_mask = self.text([negative_prompt] * n)
        L = _bucket(ctx.shape[1], u_ctx.shape[1])
        ctx, mask = _pad_to(ctx, mask, L)
        u_ctx, u_mask = _pad_to(u_ctx, u_mask, L)
        both_ctx, both_mask = torch.cat([ctx, u_ctx]), torch.cat([mask, u_mask])
        if autoguide:
            g_ctx, g_mask = self.guide_text([prompt] * n)
            g_ctx, g_mask = _pad_to(g_ctx, g_mask, _bucket(g_ctx.shape[1]))
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        hw = self.cfg["latent_hw"]
        z = torch.randn(n, 4, hw, hw, device=dev, generator=gen)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=dev)
            vc, vu = self.model(torch.cat([z, z]), torch.cat([t, t]), both_ctx, both_mask).chunk(2)
            v = vu + cfg * (vc - vu)
            if autoguide:
                v = v + autoguide * (vc - self.guide(z, t, g_ctx, g_mask))
            z = z + dt * v
            if callback is not None:
                callback(i + 1, steps)
        x = self.vae.decode((z / self.vae_div).to(self.vae.dtype)).sample
        if output_type == "pt":
            return ((x.float().clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu()
        x = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        from PIL import Image
        return [Image.fromarray(a) for a in x]
