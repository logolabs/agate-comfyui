"""ComfyUI nodes for Agate, LogoLabs' 260M text-to-image model.

Agate Loader   -> AGATE_MODEL  (generator + Ettin text encoder + VAE decoder, cached across runs)
Agate Generate -> IMAGE        (256x256, ComfyUI's (B, H, W, C) float tensor in [0, 1])

The pipeline is the vendored `agate` package (agate_comfy/agate, MIT); see agate_comfy/__init__.py
for the source version and the two small additions made to it.
"""
from __future__ import annotations

import contextlib
import gc
import logging
import os
from pathlib import Path

import torch

from .agate_comfy import SOURCE_REPO
from .agate_comfy.agate.pipeline import AgatePipeline

log = logging.getLogger("comfyui-agate")

DEFAULT_SOURCE = SOURCE_REPO
DECODERS = ("sd-vae", "taesd")
DEVICES = ("auto", "cuda", "cpu")

# What the default pipeline needs from the model repo; the guide model (another ~1 GB) is only
# fetched when load_guide is on or a Generate node asks for autoguide > 0. The model card's
# images (assets/) are never downloaded.
_BASE_FILES = ["config.json", "generator.safetensors", "text_encoder/*"]
_GUIDE_FILES = ["guide/*"]


# ---------------------------------------------------------------------------------------------
# ComfyUI integration helpers (every comfy import is guarded, so the module also imports in tests)

def _mm():
    try:
        import comfy.model_management as mm
        return mm
    except Exception:  # pragma: no cover - outside ComfyUI
        return None


def _default_device() -> torch.device:
    mm = _mm()
    if mm is not None:
        try:
            return torch.device(mm.get_torch_device())
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _free_comfy_vram(device: torch.device, nbytes: int = 2 * 1024 ** 3) -> None:
    """Ask ComfyUI to unload its own models if that is what it takes to fit Agate (~1-2 GB)."""
    mm = _mm()
    if mm is None or device.type != "cuda":
        return
    try:
        mm.free_memory(nbytes, device)
    except Exception as e:  # never let housekeeping break a load
        log.debug("free_memory failed: %s", e)


def _check_interrupt() -> None:
    mm = _mm()
    if mm is not None and hasattr(mm, "throw_exception_if_processing_interrupted"):
        mm.throw_exception_if_processing_interrupted()


def _progress_bar(total: int):
    try:
        import comfy.utils
        return comfy.utils.ProgressBar(total)
    except Exception:  # pragma: no cover - outside ComfyUI
        return None


# ---------------------------------------------------------------------------------------------
# Model resolution and the process-wide cache

def _looks_like_path(source: str) -> bool:
    """A repo id is `owner/name`; anything else that is not an existing folder is a bad path."""
    return ("\\" in source or source.count("/") != 1 or source.startswith((".", "~", "/"))
            or (len(source) > 1 and source[1] == ":"))


def resolve_source(source: str, guide: bool = False) -> Path:
    """A local folder (the unpacked model repo) or a Hugging Face repo id -> a local folder."""
    source = source.strip().strip('"')
    p = Path(source).expanduser()
    if p.is_dir():
        if not (p / "config.json").is_file():
            raise FileNotFoundError(f"{p} has no config.json; point the loader at the model folder "
                                    "that contains config.json, generator.safetensors and text_encoder/")
        return p
    if _looks_like_path(source):
        raise FileNotFoundError(f"Agate model folder not found: {source}")
    from huggingface_hub import snapshot_download
    patterns = _BASE_FILES + (_GUIDE_FILES if guide else [])
    try:
        return Path(snapshot_download(source, allow_patterns=patterns))
    except Exception as e:
        name = type(e).__name__
        if name in ("RepositoryNotFoundError", "GatedRepoError") or "401" in str(e) or "403" in str(e):
            raise RuntimeError(
                f"Could not access the Hugging Face repo '{source}' ({name}). While Agate is in private "
                "preview you need access to the repo and a login: run `huggingface-cli login` (or set "
                "HF_TOKEN) in the environment ComfyUI runs in, then retry. Alternatively download the "
                "model folder and put its local path in `source`.") from e
        raise


class AgateModel:
    """What flows along an AGATE_MODEL link: a handle that owns the pipeline.

    ComfyUI keeps node outputs in its own cache, so a handle can outlive an "Unload Models"
    click. unload() therefore drops the weights from the handle itself; the next Generate that
    uses a stale handle reloads them transparently."""

    def __init__(self, root: Path, source: str, device: torch.device, decoder: str, cuda_graphs: bool):
        self.root, self.source, self.device = Path(root), source, device
        self.decoder, self.cuda_graphs = decoder, cuda_graphs
        self._pipe: AgatePipeline | None = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    @property
    def pipe(self) -> AgatePipeline:
        if self._pipe is None:
            _free_comfy_vram(self.device)
            with torch.inference_mode(False):
                self._pipe = AgatePipeline(self.root, device=str(self.device),
                                           fast_vae=(self.decoder == "taesd"), cuda_graphs=self.cuda_graphs)
            log.info("Agate: loaded %s on %s (decoder %s, cuda_graphs %s)", self.root, self.device,
                     self.decoder, self.cuda_graphs and self.device.type == "cuda")
        return self._pipe

    def ensure_guide(self) -> None:
        pipe = self.pipe
        if pipe.guide is not None:
            return
        if not (Path(pipe.root) / pipe.cfg["guide"]["generator"]).is_file():
            # A repo id loaded without the guide: fetch it now (same snapshot folder).
            pipe.root = self.root = resolve_source(self.source, guide=True)
        with torch.inference_mode(False):
            pipe._load_guide()

    def unload(self) -> None:
        if self._pipe is None:
            return
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Agate: unloaded %s", self.source)

    def __repr__(self) -> str:  # shows up in ComfyUI's debug output
        return f"AgateModel({self.source!r}, device={self.device}, loaded={self.loaded})"


_CACHE: dict = {"key": None, "model": None}


def release_cached_model() -> None:
    """Free the cached pipeline's weights and CUDA graphs (the handle stays, and reloads on use)."""
    if _CACHE["model"] is not None:
        _CACHE["model"].unload()


def _hook_unload_all_models() -> None:
    """Make ComfyUI's "Unload Models" (and anything else calling unload_all_models) free Agate
    too. Wraps the function once; the original always runs first."""
    mm = _mm()
    if mm is None or getattr(mm.unload_all_models, "_agate_hooked", False):
        return
    original = mm.unload_all_models

    def unload_all_models(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            release_cached_model()

    unload_all_models._agate_hooked = True
    mm.unload_all_models = unload_all_models


_hook_unload_all_models()


def load_agate(source: str = DEFAULT_SOURCE, decoder: str = "sd-vae", device: str = "auto",
               cuda_graphs: bool = True, load_guide: bool = False) -> AgateModel:
    dev = _default_device() if device == "auto" else torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested but PyTorch sees no CUDA GPU")
    root = resolve_source(source, guide=load_guide)
    key = (str(root.resolve()), decoder, str(dev), bool(cuda_graphs))
    model = _CACHE["model"]
    if model is None or _CACHE["key"] != key:
        release_cached_model()                  # one Agate at a time
        model = AgateModel(root, source, dev, decoder, bool(cuda_graphs))
        _CACHE["key"], _CACHE["model"] = key, model
    model.pipe                                  # load now, so load errors surface on this node
    if load_guide:
        model.ensure_guide()
    return model

# ---------------------------------------------------------------------------------------------
# Nodes

class AgateLoader:
    CATEGORY = "LogoLabs/Agate"
    RETURN_TYPES = ("AGATE_MODEL",)
    RETURN_NAMES = ("agate",)
    FUNCTION = "load"
    DESCRIPTION = ("Loads Agate (LogoLabs, 260M text-to-image, 256x256). `source` is a Hugging Face repo id "
                   "or a local model folder. The pipeline is cached: re-running with the same settings "
                   "reuses it.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": ("STRING", {"default": DEFAULT_SOURCE, "multiline": False,
                                  "tooltip": "Hugging Face repo id or a local folder holding config.json, "
                                             "generator.safetensors and text_encoder/"}),
            "decoder": (list(DECODERS), {"default": "sd-vae",
                                         "tooltip": "sd-vae: SD-VAE ft-MSE, best quality. taesd: tiny "
                                                    "decoder, faster and lighter, slightly softer."}),
            "device": (list(DEVICES), {"default": "auto",
                                       "tooltip": "auto uses ComfyUI's device"}),
            "cuda_graphs": ("BOOLEAN", {"default": True,
                                        "tooltip": "Record each denoising step as a CUDA graph (much "
                                                   "faster; the first run per batch size is slower)"}),
            "load_guide": ("BOOLEAN", {"default": False,
                                       "tooltip": "Preload the guide model used by autoguide (+~1 GB). "
                                                  "Otherwise it loads on first use."}),
        }}

    def load(self, source, decoder, device, cuda_graphs, load_guide):
        return (load_agate(source, decoder, device, cuda_graphs, load_guide),)


class AgateGenerate:
    CATEGORY = "LogoLabs/Agate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    DESCRIPTION = ("Generates 256x256 images with Agate (Euler flow sampler, classifier-free guidance "
                   "against the negative prompt). Pair with an upscale node for larger output.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "agate": ("AGATE_MODEL",),
            "prompt": ("STRING", {"default": "a minimalist logo of a fox head, orange, flat design, "
                                             "white background", "multiline": True}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True,
                                           "tooltip": "The unconditional prompt for CFG; empty is what "
                                                      "Agate was trained with"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                             "control_after_generate": True}),
            "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1, "round": 0.01}),
            "autoguide": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1, "round": 0.01,
                                    "tooltip": "Steer away from Agate's early guide checkpoint; sharper "
                                               "faces and detail. Try 1.0 with cfg 4. Costs ~50% more time."}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
        }}

    def generate(self, agate, prompt, negative_prompt, seed, steps, cfg, autoguide, batch_size):
        if autoguide > 0:
            agate.ensure_guide()
        pipe = agate.pipe
        pbar = _progress_bar(steps)

        def on_step(i, n):
            _check_interrupt()
            if pbar is not None:
                pbar.update_absolute(i, n)

        ctx = torch.cuda.device(pipe.device) if pipe.device.type == "cuda" else contextlib.nullcontext()
        with ctx:
            images = pipe(prompt, negative_prompt=negative_prompt, seed=seed, steps=steps, cfg=cfg,
                          num_images=batch_size, autoguide=autoguide, callback=on_step, output_type="pt")
        return (images.contiguous(),)



NODE_CLASS_MAPPINGS = {
    "AgateLoader": AgateLoader,
    "AgateGenerate": AgateGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AgateLoader": "Agate Loader",
    "AgateGenerate": "Agate Generate",
}
