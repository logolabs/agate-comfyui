"""Agate inside ComfyUI: loading, memory management and the sampler.

The sampler is the release pipeline's (agate/pipeline.py, AgatePipeline.__call__) with three
generalisations, all of which reduce to it exactly for txt2img:

  * it can start part-way along the flow path from a given latent (img2img): Agate's path runs
    from noise at t = 0 to data at t = 1, x_t = (1 - t) noise + t x1, so denoise d starts at
    t0 = 1 - d from x_t0 and runs `steps` Euler steps over [t0, 1]; d = 1 is plain txt2img;
  * it returns the latent (scaled SD-VAE space, what the model samples in), and the nodes turn it
    into a ComfyUI LATENT or decode it;
  * each step reports the model's current estimate of the final latent, x1 = z + (1 - t) v, which
    the nodes turn into ComfyUI's live previews.

Weights are held by ComfyUI ModelPatchers (one for generator + text encoder, one for the VAE
decoder, one for the optional guide model), so ComfyUI loads Agate to the GPU when it samples and
can move it back to system RAM when another model needs the VRAM. Outside ComfyUI (tests) the
same code just moves modules to the device.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .agate.fcdm_thinker2 import FCDMThinker2
from .agate.pipeline import _Eager, _Graphed, _bucket, _pad_to
from .agate.text_encoder import EttinTextEncoder

log = logging.getLogger("agate-comfyui")

VAE_SCALE = 0.18215           # SD-VAE (SD 1.x) latent scale; ComfyUI's SD15 latent format uses the same


def _mm():
    try:
        import comfy.model_management as mm
        return mm
    except Exception:  # outside ComfyUI
        return None


class AgateWeights(torch.nn.Module):
    """The nn.Module a ModelPatcher manages. Moving it (ComfyUI loading or offloading it) calls
    `on_move`, which drops recorded CUDA graphs: they point at the old weight addresses."""

    def __init__(self, dtype: torch.dtype, on_move=None, **modules):
        super().__init__()
        for k, v in modules.items():
            self.add_module(k, v)
        self._agate_dtype, self._on_move = dtype, on_move
        self.device = torch.device("cpu")

    def get_dtype(self):
        return self._agate_dtype

    def _apply(self, fn, *args, **kwargs):
        if self._on_move is not None:
            self._on_move()
        return super()._apply(fn, *args, **kwargs)


class _StepFn:
    """The generator's forward, eager or replayed from CUDA graphs (the release's _Graphed/_Eager),
    re-recorded whenever the weights have moved since the graphs were captured."""

    def __init__(self, module: torch.nn.Module, graphs: bool):
        self.module, self.graphs = module, graphs
        self._impl, self._sig = None, None

    def reset(self):
        self._impl = self._sig = None

    def prepare(self, device: torch.device):
        sig = tuple(p.data_ptr() for p in self.module.parameters())
        if self._impl is None or sig != self._sig:
            use_graphs = self.graphs and device.type == "cuda"
            self._impl = _Graphed(self.module) if use_graphs else _Eager(self.module)
            self._sig = sig

    def __call__(self, z, t, ctx, mask):
        return self._impl(z, t, ctx, mask)


class _Text(EttinTextEncoder):
    """EttinTextEncoder around an already-built tokenizer + ModernBERT (from a single file)."""

    def __init__(self, tok, model, device, max_len):
        self.device, self.max_len, self.tok, self.model = torch.device(device), max_len, tok, model
        self.dim = model.config.hidden_size
        self.trainable = []


class _Managed:
    """A group of modules that ComfyUI loads and offloads as one ModelPatcher."""

    def __init__(self, name: str, bundle: AgateWeights, device: torch.device):
        self.name, self.bundle, self.device = name, bundle, device
        self.patcher = None
        mm = _mm()
        if mm is not None:
            try:
                import comfy.model_patcher
                offload = torch.device("cpu") if device.type == "cpu" else mm.unet_offload_device()
                self.patcher = comfy.model_patcher.ModelPatcher(bundle, load_device=device, offload_device=offload)
            except Exception as e:  # pragma: no cover - unexpected ComfyUI version
                log.warning("Agate: ModelPatcher unavailable (%s); managing %s memory manually", e, name)

    def size(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.bundle.state_dict().values())

    def on_device(self) -> bool:
        return all(t.device == self.device for t in self.bundle.state_dict().values())

    def evict(self):
        """Take this patcher out of ComfyUI's loaded-model list (used when the runtime is dropped)."""
        mm = _mm()
        if mm is None or self.patcher is None:
            return
        for i in range(len(mm.current_loaded_models) - 1, -1, -1):
            lm = mm.current_loaded_models[i]
            if getattr(lm, "model", None) is self.patcher:
                try:
                    mm.current_loaded_models.pop(i).model_unload()
                except Exception as e:  # pragma: no cover
                    log.debug("Agate: evicting %s failed: %s", self.name, e)


def load_to_device(groups: list, memory_required: int = 0) -> None:
    """Make every module of `groups` resident on its device, via ComfyUI when it is there.

    force_full_load: Agate is ~0.5 GB in bf16, so partial (--lowvram) loading buys little, and
    ComfyUI's partial loader only moves modules with a `weight` (Agate's GRN layers keep their
    parameters as gamma/beta). Anything ComfyUI left behind (e.g. under --novram) is moved here."""
    mm = _mm()
    patchers = [g.patcher for g in groups if g.patcher is not None]
    if mm is not None and patchers:
        mm.load_models_gpu(patchers, memory_required=memory_required, force_full_load=True)
    for g in groups:
        if not g.on_device():
            g.bundle.to(g.device)
            if g.patcher is not None:
                g.bundle.model_loaded_weight_memory = g.size()
                g.bundle.model_lowvram = False
        g.bundle.device = g.device


class AgateRuntime:
    """Generator + text encoder (+ lazily the guide model and a VAE decoder) for one device."""

    def __init__(self, cfg: dict, gen_sd: dict, text: tuple, device: torch.device, cuda_graphs: bool = True,
                 decoder: str = "sd-vae", guide_source=None, name: str = "agate"):
        self.cfg, self.device, self.name, self.decoder = cfg, torch.device(device), name, decoder
        self.on_cuda = self.device.type == "cuda"
        self.cuda_graphs = bool(cuda_graphs) and self.on_cuda
        self.dtype = torch.bfloat16 if self.on_cuda else torch.float32
        self.vae_scale = float(cfg.get("vae_scale", VAE_SCALE))
        self.latent_hw = int(cfg.get("latent_hw", 32))
        if self.on_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.enable_cudnn_sdp(False)   # the attention kernels Agate was trained with
        self.core, self.step_fn, self.text = self._build(gen_sd, text, cfg["text_max_len"], "core")
        self._guide_source = guide_source                 # callable -> (gen_sd, text, max_len)
        self.guide = self.guide_step = self.guide_text = None
        self.vae_group = self.vae = None

    # -- construction ---------------------------------------------------------------------------
    def _build(self, gen_sd: dict, text: tuple, max_len: int, name: str):
        m = FCDMThinker2(**self.cfg["model_kw"])
        m.load_state_dict(gen_sd)
        m = m.to(dtype=self.dtype).eval().requires_grad_(False)
        if self.on_cuda:
            m = m.to(memory_format=torch.channels_last)
        kind, *payload = text
        if kind == "dir":        # a release text_encoder/ folder, loaded as the release does
            enc = EttinTextEncoder(str(payload[0]), "cpu", max_len=max_len)
            enc.device = self.device
        else:                    # ("built", tokenizer, module) from a single file
            enc = _Text(payload[0], payload[1], self.device, max_len)
        enc.model.to(dtype=self.dtype).requires_grad_(False)
        step = _StepFn(m, self.cuda_graphs)
        group = _Managed(f"{self.name}:{name}", AgateWeights(self.dtype, step.reset, generator=m, text_encoder=enc.model),
                         self.device)
        return group, step, enc

    def ensure_guide(self) -> None:
        if self.guide is not None:
            return
        if self._guide_source is None:
            raise RuntimeError("this Agate checkpoint has no guide model, so autoguide is unavailable")
        with torch.inference_mode(False):
            gen_sd, text, max_len = self._guide_source()
            self.guide, self.guide_step, self.guide_text = self._build(gen_sd, text, max_len, "guide")

    def ensure_vae(self):
        if self.vae is not None:
            return
        from diffusers import AutoencoderKL, AutoencoderTiny
        vdtype = torch.float16 if self.on_cuda else torch.float32
        with torch.inference_mode(False):
            if self.decoder == "taesd":
                vae = AutoencoderTiny.from_pretrained(self.cfg["fast_vae"], torch_dtype=vdtype)
                self.vae_div = 1.0                            # TAESD decodes the scaled latents directly
            else:
                vae = AutoencoderKL.from_pretrained(self.cfg["vae"], torch_dtype=vdtype)
                self.vae_div = self.vae_scale
        self.vae = vae.eval().requires_grad_(False)
        self.vae_group = _Managed(f"{self.name}:vae", AgateWeights(vdtype, None, vae=self.vae), self.device)

    def groups(self) -> list:
        return [g for g in (self.core, self.guide, self.vae_group) if g is not None]

    def release(self) -> None:
        for g in self.groups():
            g.evict()
            g.bundle.to("cpu")
        self.step_fn.reset()
        if self.guide_step is not None:
            self.guide_step.reset()

    # -- sampling -------------------------------------------------------------------------------
    def _activation_bytes(self, n: int) -> int:
        return int(n * 2 * 160 * 2 ** 20)     # measured ~0.1 GB per CFG pair at 256 px, with margin

    @torch.no_grad()
    def sample(self, prompt: str, negative_prompt: str = "", seed: int = 0, steps: int = 50, cfg: float = 3.0,
               batch_size: int = 1, autoguide: float = 0.0, init_latent: torch.Tensor | None = None,
               denoise: float = 1.0, callback=None, interrupt=None) -> torch.Tensor:
        """-> z, (B, 4, h, w) float32 on the device, in the model's (scaled SD-VAE) latent space.

        init_latent: scaled latents (B, 4, h, w) to start from, or None. Their batch size wins over
        batch_size, and their h, w set the size (Agate was trained at 32 x 32 only).
        callback(step, steps, x1_estimate) runs after every step; interrupt() may raise."""
        denoise = float(min(max(denoise, 0.0), 1.0))
        if init_latent is not None:
            n, _, h, w = init_latent.shape
        else:
            n, h, w = int(batch_size), self.latent_hw, self.latent_hw
            if denoise < 1.0:
                raise ValueError("denoise < 1 needs a latent_image to start from (img2img); "
                                 "connect one or set denoise to 1.0")
        if h % 4 or w % 4:
            raise ValueError(f"Agate needs latent sizes divisible by 4 (images divisible by 32), got {h}x{w}")
        if autoguide:
            self.ensure_guide()
        groups = [self.core] + ([self.guide] if autoguide else [])
        load_to_device(groups, self._activation_bytes(n))
        dev = self.device
        self.step_fn.prepare(dev)
        if autoguide:
            self.guide_step.prepare(dev)

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
        z = torch.randn(n, 4, h, w, device=dev, generator=gen)
        t0 = 0.0
        if init_latent is not None and denoise < 1.0:
            t0 = 1.0 - denoise
            z = (1.0 - t0) * z + t0 * init_latent.to(dev, torch.float32)
        if steps < 1 or denoise == 0.0:
            return z if init_latent is None else init_latent.to(dev, torch.float32)
        dt = (1.0 - t0) / steps
        for i in range(steps):
            if interrupt is not None:
                interrupt()
            t = torch.full((n,), t0 + i * dt, device=dev)
            vc, vu = self.step_fn(torch.cat([z, z]), torch.cat([t, t]), both_ctx, both_mask).chunk(2)
            v = vu + cfg * (vc - vu)
            if autoguide:
                v = v + autoguide * (vc - self.guide_step(z, t, g_ctx, g_mask))
            if callback is not None:
                x1 = z + (1.0 - (t0 + i * dt)) * v
            z = z + dt * v
            if callback is not None:
                callback(i + 1, steps, x1)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latents -> ComfyUI IMAGE (B, H, W, 3) float in [0, 1] on the CPU."""
        self.ensure_vae()
        load_to_device([self.vae_group], int(z.shape[0] * 256 * 2 ** 20))
        x = self.vae.decode((z / self.vae_div).to(self.device, self.vae.dtype)).sample
        return ((x.float().clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu()


# ---------------------------------------------------------------------------------------------
# Constructors

def from_single_file(path: str | Path, device, cuda_graphs: bool = True, decoder: str = "sd-vae",
                     guide_path_fn=None) -> AgateRuntime:
    """`guide_path_fn(file_name) -> Path` finds (or downloads) the guide checkpoint on demand."""
    from .checkpoint import load_single_file
    path = Path(path)
    with torch.inference_mode(False):
        cfg, gen_sd, tok, model = load_single_file(path)
    guide = cfg.get("guide")
    guide_source = None
    if guide and guide_path_fn is not None:
        def guide_source():
            gcfg, g_sd, g_tok, g_model = load_single_file(guide_path_fn(guide["file"]))
            return g_sd, ("built", g_tok, g_model), gcfg["text_max_len"]
    return AgateRuntime(cfg, gen_sd, ("built", tok, model), device, cuda_graphs, decoder, guide_source,
                        name=path.stem)


def from_folder(root: str | Path, device, cuda_graphs: bool = True, decoder: str = "sd-vae",
                guide_root_fn=None) -> AgateRuntime:
    """A release folder (config.json, generator.safetensors, text_encoder/, guide/)."""
    from safetensors.torch import load_file
    root = Path(root)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    with torch.inference_mode(False):
        gen_sd = load_file(str(root / "generator.safetensors"))
    g = cfg.get("guide")
    guide_source = None
    if g:
        def guide_source():
            r = guide_root_fn() if guide_root_fn is not None else root
            return (load_file(str(Path(r) / g["generator"])), ("dir", Path(r) / g["text_encoder"]),
                    g["text_max_len"])
    return AgateRuntime(cfg, gen_sd, ("dir", root / cfg["text_encoder"]), device, cuda_graphs, decoder,
                        guide_source, name=root.name)
