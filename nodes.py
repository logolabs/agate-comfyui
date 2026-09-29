"""ComfyUI nodes for Agate, LogoLabs' 260M text-to-image model.

Agate Loader   -> AGATE_MODEL  (Preview 001 / 002 / 003 single files from models/agate/, or a release folder / repo id)
Agate Sampler  -> LATENT       (SD 1.x latent: decode with the stock VAE Decode; optional img2img input)
Agate Generate -> IMAGE        (all in one: sample + Agate's own SD-VAE / TAESD decode; AI-output marking)
Agate Plan Viewer -> IMAGEs    (the thinker's 16 x 16 plan at every step, as frames; agate_comfy/plan.py)
Agate Watermark   -> IMAGE     (adds the Agate AI-output watermark, e.g. after Agate Sampler + VAE Decode)
Agate Save Image               (PNG with the provenance text chunks the Python packages write)

The model code is the vendored release packages (agate_comfy/agate for 001/002, agate_comfy/agate003 for
003, MIT); agate_comfy/runtime.py holds the ComfyUI side: memory management, img2img, previews;
agate_comfy/marking.py the AI-output marking (EU AI Act Art. 50(2)).
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import torch

from .agate_comfy import marking as mk
from .agate_comfy import runtime as rt
from .agate_comfy.checkpoint import DEFAULT_NAME, GUIDE_REPO, MAIN_NAME, OFFICIAL

log = logging.getLogger("agate-comfyui")

HF_REPO = OFFICIAL[MAIN_NAME]            # 001's repo (also hosts the guide model)
HF_SUBFOLDER = "comfyui"
MODEL_FOLDER = "agate"                  # ComfyUI/models/agate/
DECODERS = ("sd-vae", "taesd")
DEVICES = ("auto", "cuda", "cpu")

# Release-folder loading (the optional `model_folder` input): what the pipeline needs from the repo.
_BASE_FILES = ["config.json", "generator.safetensors", "text_encoder/*"]
_GUIDE_FILES = ["guide/*"]


# ---------------------------------------------------------------------------------------------
# ComfyUI integration helpers (every comfy import is guarded, so the module also imports in tests)

def _mm():
    return rt._mm()


def _default_device() -> torch.device:
    mm = _mm()
    if mm is not None:
        try:
            return torch.device(mm.get_torch_device())
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _check_interrupt() -> None:
    mm = _mm()
    if mm is not None and hasattr(mm, "throw_exception_if_processing_interrupted"):
        mm.throw_exception_if_processing_interrupted()


def _intermediate_device() -> torch.device:
    mm = _mm()
    try:
        return torch.device(mm.intermediate_device()) if mm is not None else torch.device("cpu")
    except Exception:
        return torch.device("cpu")


def _progress(steps: int, device: torch.device, previews: bool = True, preview_mode: str = "side_by_side"):
    """-> callback(step, steps, x1, plan=None) that drives ComfyUI's progress bar and live preview."""
    try:
        import comfy.utils
        pbar = comfy.utils.ProgressBar(steps)
    except Exception:  # outside ComfyUI
        return None
    if not previews or preview_mode == "none":
        return lambda step, total, *args: pbar.update_absolute(step, total, None)
    previewer = None
    if preview_mode in ("image", "side_by_side"):
        try:
            import comfy.latent_formats
            import latent_preview
            previewer = latent_preview.get_previewer(device, comfy.latent_formats.SD15())
        except Exception as e:
            log.debug("Agate: no latent previewer (%s)", e)

    basis = {"vt": None, "lo": None, "hi": None}

    def callback(step, total, x1, plan=None):
        preview = None
        if previewer is not None or (plan is not None and preview_mode in ("plan", "side_by_side")):
            try:
                from PIL import Image
                img_im = None
                if previewer is not None and preview_mode in ("image", "side_by_side"):
                    ret = previewer.decode_latent_to_preview_image("JPEG", x1)
                    if isinstance(ret, tuple) and len(ret) >= 2 and hasattr(ret[1], "size"):
                        img_im = ret[1]
                    else:
                        preview = ret

                plan_im = None
                if plan is not None and preview_mode in ("plan", "side_by_side"):
                    P = plan[:1].float()
                    X = P.permute(0, 2, 3, 1).reshape(-1, 640).double()
                    X = X - X.mean(0, keepdim=True)
                    if basis["vt"] is None:
                        _, vecs = torch.linalg.eigh(X.T @ X)
                        vt = vecs.flip(1)[:, :3].T
                        pivot = vt.gather(1, vt.abs().argmax(1, keepdim=True))
                        vt = vt * torch.where(pivot < 0, -1.0, 1.0).to(vt)
                        basis["vt"] = vt
                        Y = (X @ vt.T).float()
                        basis["lo"] = torch.quantile(Y, 0.02, dim=0)
                        basis["hi"] = torch.quantile(Y, 0.98, dim=0)
                    else:
                        vt = basis["vt"]
                        Y = (X @ vt.T).float()
                    lo, hi = basis["lo"], basis["hi"]
                    rgb = (((Y - lo) / (hi - lo + 1e-8)).clamp(0, 1) * 255).round().to(torch.uint8).reshape(16, 16, 3).cpu().numpy()
                    target_sz = img_im.size if img_im is not None else (256, 256)
                    plan_im = Image.fromarray(rgb).resize(target_sz, Image.NEAREST)

                if preview_mode == "side_by_side":
                    if img_im is not None and plan_im is not None:
                        w, h = img_im.size
                        comp = Image.new("RGB", (w + plan_im.width, max(h, plan_im.height)))
                        comp.paste(img_im, (0, 0))
                        comp.paste(plan_im, (w, 0))
                        preview = ("JPEG", comp, max(comp.size))
                    elif img_im is not None and preview is None:
                        preview = ("JPEG", img_im, max(img_im.size))
                    elif plan_im is not None and preview is None:
                        preview = ("JPEG", plan_im, max(plan_im.size))
                elif preview_mode == "plan" and plan_im is not None:
                    preview = ("JPEG", plan_im, max(plan_im.size))
                elif preview_mode == "image" and img_im is not None and preview is None:
                    preview = ("JPEG", img_im, max(img_im.size))
            except Exception as e:  # a preview must never break sampling
                log.debug("Agate: preview failed: %s", e)
        pbar.update_absolute(step, total, preview)
    return callback


def _folder_paths():
    try:
        import folder_paths
        return folder_paths
    except Exception:
        return None


def _register_model_folder() -> Path | None:
    """ComfyUI/models/agate/ (plus any `agate:` entries in extra_model_paths.yaml)."""
    fp = _folder_paths()
    if fp is None:
        return None
    path = Path(fp.models_dir) / MODEL_FOLDER
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    if str(path) not in fp.folder_names_and_paths.get(MODEL_FOLDER, ([], set()))[0]:
        fp.add_model_folder_path(MODEL_FOLDER, str(path))
    return path


_DEFAULT_DIR = _register_model_folder()


def checkpoint_choices() -> list[str]:
    names = []
    fp = _folder_paths()
    if fp is not None:
        try:
            names = [n for n in fp.get_filename_list(MODEL_FOLDER)
                     if n.endswith(".safetensors") and "agate-guide-" not in Path(n).name]
        except Exception:
            names = []
    return list(OFFICIAL) + sorted(n for n in names if n not in OFFICIAL)


def _find_in_model_folder(name: str) -> Path | None:
    fp = _folder_paths()
    if fp is not None:
        try:
            p = fp.get_full_path(MODEL_FOLDER, name)
            if p and os.path.isfile(p):
                return Path(p)
        except Exception:
            pass
    return None


def _repo_for(name: str) -> str:
    """The Hugging Face model repo of an official single file (the guide lives in 001's repo)."""
    return OFFICIAL.get(name, GUIDE_REPO)


def _count_load(repo: str = HF_REPO) -> None:
    """Read the model repo's root config.json in the background. The Hub counts a model download only on a
    request for that file, so this makes every load of an official checkpoint count once, as a Python
    from_pretrained() does -- on the repo of the release that is loaded. Best effort: offline or blocked just
    skips it."""
    def ping():
        try:
            from huggingface_hub import get_session, hf_hub_url
            get_session().get(hf_hub_url(repo, "config.json"), timeout=5)
        except Exception:
            pass
    import threading
    threading.Thread(target=ping, name="agate-count", daemon=True).start()


def _is_official(name: str) -> bool:
    return name in OFFICIAL or name.startswith("agate-guide-")


def _download(name: str, dest_dir: Path | None) -> Path:
    """Fetch comfyui/<name> from its Hugging Face model repo into models/agate/."""
    from huggingface_hub import hf_hub_download
    repo = _repo_for(name)
    if dest_dir is None:                                     # outside ComfyUI: the HF cache
        return Path(hf_hub_download(repo, f"{HF_SUBFOLDER}/{name}"))
    log.info("Agate: downloading %s/%s/%s into %s", repo, HF_SUBFOLDER, name, dest_dir)
    staging = dest_dir / ".download"
    try:
        got = Path(hf_hub_download(repo, f"{HF_SUBFOLDER}/{name}", local_dir=str(staging)))
    except Exception as e:
        raise RuntimeError(
            f"Could not download {name} from https://huggingface.co/{repo} ({type(e).__name__}: {e}). "
            f"Download it by hand from https://huggingface.co/{repo}/tree/main/{HF_SUBFOLDER} and put it "
            f"in {dest_dir}") from e
    target = dest_dir / name
    os.replace(got, target)
    shutil.rmtree(staging, ignore_errors=True)
    return target


def resolve_checkpoint(name: str) -> Path:
    """A file name from the dropdown (or an absolute path) -> a local file, downloading the
    official checkpoints from Hugging Face when they are missing."""
    p = Path(name)
    if _is_official(p.name):
        _count_load(_repo_for(p.name))
    if p.is_absolute() and p.is_file():
        return p
    found = _find_in_model_folder(name)
    if found is not None:
        return found
    if _is_official(p.name):
        return _download(p.name, _DEFAULT_DIR)
    raise FileNotFoundError(f"Agate checkpoint {name!r} not found in models/{MODEL_FOLDER}/")


def _guide_path(name: str, next_to: Path) -> Path:
    for cand in (next_to.parent / name, _find_in_model_folder(name)):
        if cand is not None and Path(cand).is_file():
            return Path(cand)
    return _download(name, _DEFAULT_DIR)


def _looks_like_path(source: str) -> bool:
    """A repo id is `owner/name`; anything else that is not an existing folder is a bad path."""
    return ("\\" in source or source.count("/") != 1 or source.startswith((".", "~", "/"))
            or (len(source) > 1 and source[1] == ":"))


def resolve_source(source: str, guide: bool = False) -> Path:
    """A local release folder or a Hugging Face repo id -> a local folder."""
    source = source.strip().strip('"')
    p = Path(source).expanduser()
    if p.is_dir():
        if not (p / "config.json").is_file():
            raise FileNotFoundError(f"{p} has no config.json; point model_folder at the folder that contains "
                                    "config.json, generator.safetensors and text_encoder/")
        return p
    if _looks_like_path(source):
        raise FileNotFoundError(f"Agate model folder not found: {source}")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(source, allow_patterns=_BASE_FILES + (_GUIDE_FILES if guide else [])))


# ---------------------------------------------------------------------------------------------
# The AGATE_MODEL handle and the process-wide cache

class AgateModel:
    """What flows along an AGATE_MODEL link. Owns an AgateRuntime; ComfyUI's memory manager moves
    its weights between VRAM and RAM, and unload() drops them entirely (a later use reloads)."""

    def __init__(self, kind: str, target: str, device: torch.device, decoder: str, cuda_graphs: bool):
        self.kind, self.target, self.device = kind, target, device
        self.decoder, self.cuda_graphs = decoder, cuda_graphs
        self._rt: rt.AgateRuntime | None = None

    @property
    def loaded(self) -> bool:
        return self._rt is not None

    @property
    def runtime(self) -> rt.AgateRuntime:
        if self._rt is None:
            if self.kind == "file":
                path = resolve_checkpoint(self.target)
                self._rt = rt.from_single_file(path, self.device, self.cuda_graphs, self.decoder,
                                               guide_path_fn=lambda n: _guide_path(n, path))
            else:
                root = resolve_source(self.target)
                self._rt = rt.from_folder(root, self.device, self.cuda_graphs, self.decoder,
                                          guide_root_fn=lambda: resolve_source(self.target, guide=True))
            log.info("Agate: loaded %s (%s) for %s, cuda_graphs %s", self.target, self.kind, self.device,
                     self._rt.cuda_graphs)
        return self._rt

    def unload(self) -> None:
        if self._rt is None:
            return
        self._rt.release()
        self._rt = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Agate: released %s", self.target)

    def __repr__(self) -> str:
        return f"AgateModel({self.target!r}, device={self.device}, loaded={self.loaded})"


_CACHE: dict = {"key": None, "model": None}


def release_cached_model() -> None:
    if _CACHE["model"] is not None:
        _CACHE["model"].unload()


def load_agate(checkpoint: str = DEFAULT_NAME, decoder: str = "sd-vae", device: str = "auto",
               cuda_graphs: bool = True, load_guide: bool = False, model_folder: str = "") -> AgateModel:
    dev = _default_device() if device == "auto" else torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested but PyTorch sees no CUDA GPU")
    folder = (model_folder or "").strip()
    kind, target = ("folder", folder) if folder else ("file", checkpoint)
    key = (kind, target, decoder, str(dev), bool(cuda_graphs))
    model = _CACHE["model"]
    if model is None or _CACHE["key"] != key:
        release_cached_model()                  # one Agate at a time
        model = AgateModel(kind, target, dev, decoder, bool(cuda_graphs))
        _CACHE["key"], _CACHE["model"] = key, model
    r = model.runtime                           # load now, so load errors surface on this node
    if load_guide:
        r.ensure_guide()
    return model


# ---------------------------------------------------------------------------------------------
# LATENT conversion. ComfyUI's LATENT holds SD 1.x latents unscaled (what VAE Encode returns and
# VAE Decode takes; a model's latent_format multiplies by 0.18215 on the way in). Agate samples
# in the scaled space.

def latent_to_model(samples: torch.Tensor) -> torch.Tensor:
    return samples.float() * rt.VAE_SCALE


def model_to_latent(z: torch.Tensor) -> torch.Tensor:
    return z.float() / rt.VAE_SCALE


# ---------------------------------------------------------------------------------------------
# Nodes

_PROMPT = "a minimalist logo of a fox head, orange, flat design, white background"
RESOLUTIONS = ("auto", "512", "256")


def _res(resolution) -> int | None:
    return None if resolution in (None, "", "auto") else int(resolution)


def _optional_res():
    return (list(RESOLUTIONS), {"default": "auto",
            "tooltip": "Preview 003: auto = 512 px (native); 256 px is ~4x faster. 001 / 002 are 256 px only."})


def _live_preview():
    return (["side_by_side", "image", "plan", "none"], {"default": "side_by_side",
            "tooltip": "Live preview on this node during sampling: side_by_side shows "
                       "both the developing image and the thinker's 16x16 plan."})


def _optional_marking():
    return {
        "watermark": ("BOOLEAN", {"default": True,
                      "tooltip": "Invisible AI-output watermark in every image (invisible-watermark dwtDctSvd, payload "
                                 "AGATE + release; EU AI Act Art. 50). Read it with agate.detect_watermark()."}),
        "metadata": ("BOOLEAN", {"default": True,
                     "tooltip": "Add ai_generated / agate provenance entries to the workflow metadata, so the stock "
                                "Save Image writes them into the PNG (Agate Save Image writes the exact keys)."}),
    }


def _mark(images, r, watermark: bool):
    return mk.mark_image_tensor(images, r.release_tag) if watermark else images


def _add_provenance(extra_pnginfo, r) -> None:
    """ComfyUI's Save Image writes every entry of the prompt's extra_pnginfo (json-encoded) as a PNG text chunk.
    `ai_generated` (json true -> the text "true", the packages' value) and an `agate` object with the other
    provenance fields are added there. No prompt is added (the workflow chunk ComfyUI writes anyway holds it)."""
    if not isinstance(extra_pnginfo, dict):
        return
    info = mk.provenance(r.release_tag)
    extra_pnginfo["ai_generated"] = True
    extra_pnginfo["agate"] = {k: v for k, v in info.items() if k != "ai_generated"}


def _common_inputs():
    return {
        "agate": ("AGATE_MODEL",),
        "prompt": ("STRING", {"default": _PROMPT, "multiline": True}),
        "negative_prompt": ("STRING", {"default": "", "multiline": True,
                                       "tooltip": "The unconditional prompt for CFG; empty is what Agate "
                                                  "was trained with"}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
        "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
        "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1, "round": 0.01}),
        "autoguide": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1, "round": 0.01,
                                "tooltip": "Also steer away from Agate's early (step 27,600) checkpoint: "
                                           "sharper faces and detail. Try 1.0 with cfg 4. ~50% slower; "
                                           "loads the guide model (+0.5 GB) on first use."}),
        "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
    }


class AgateLoader:
    CATEGORY = "LogoLabs/Agate"
    RETURN_TYPES = ("AGATE_MODEL",)
    RETURN_NAMES = ("agate",)
    FUNCTION = "load"
    DESCRIPTION = ("Loads Agate (LogoLabs, 260M text-to-image) from models/agate/: Preview 003 (512 px, or 256; "
                   "prompt pipeline built in), Preview 002 or Preview 001 (256 px). The official checkpoints "
                   "download from Hugging Face on first use. Cached: re-running reuses it.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "checkpoint": (checkpoint_choices(), {"default": DEFAULT_NAME,
                           "tooltip": f"The Agate version: agate-preview-003 (newest, 512 px), -002 or -001 "
                                      f"(256 px), or another single file in models/{MODEL_FOLDER}/. The official "
                                      f"files are downloaded from huggingface.co/Logolabs/agate-preview-00X if "
                                      f"they are missing."}),
            "decoder": (list(DECODERS), {"default": "sd-vae",
                        "tooltip": "Only used by Agate Generate. sd-vae: SD-VAE ft-MSE, best quality. "
                                   "taesd: tiny decoder, faster and lighter, slightly softer."}),
            "device": (list(DEVICES), {"default": "auto", "tooltip": "auto uses ComfyUI's device"}),
            "cuda_graphs": ("BOOLEAN", {"default": True,
                            "tooltip": "Record each denoising step as a CUDA graph (much faster; the "
                                       "first run per batch size and prompt length is slower)"}),
            "load_guide": ("BOOLEAN", {"default": False,
                           "tooltip": "Preload the guide model used by autoguide. Otherwise it loads on "
                                      "first use."}),
        }, "optional": {
            "model_folder": ("STRING", {"default": "", "multiline": False,
                             "tooltip": "Advanced: instead of the checkpoint, a Hugging Face repo id or a "
                                        "local folder in the release layout (config.json, "
                                        "generator.safetensors, text_encoder/)"}),
        }}

    def load(self, checkpoint, decoder, device, cuda_graphs, load_guide, model_folder=""):
        return (load_agate(checkpoint, decoder, device, cuda_graphs, load_guide, model_folder),)


class AgateSampler:
    CATEGORY = "LogoLabs/Agate"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    DESCRIPTION = ("Samples Agate into an SD 1.x LATENT (4 x 32 x 32 for 256 px). Decode it with the stock "
                   "VAE Decode and an SD 1.5 VAE (vae-ft-mse-840000). With a latent_image and denoise < 1 "
                   "it does img2img along Agate's flow path.")

    @classmethod
    def INPUT_TYPES(cls):
        d = _common_inputs()
        d["batch_size"] = ("INT", {"default": 1, "min": 1, "max": 64,
                                   "tooltip": "Ignored when latent_image is connected (its batch is used)"})
        d["denoise"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "1.0: ignore latent_image's content (txt2img). Lower keeps more "
                                             "of latent_image: sampling starts at t = 1 - denoise."})
        # optional (it used to be required): API-format workflows from before 0.3.1 lack it. Being the first
        # optional widget, it keeps its place in saved UI workflows' widgets_values.
        return {"required": d, "optional": {"live_preview": _live_preview(), "latent_image": ("LATENT",),
                                            "resolution": _optional_res()}}

    def sample(self, agate, prompt, negative_prompt, seed, steps, cfg, autoguide, batch_size, denoise,
               live_preview="side_by_side", latent_image=None, resolution="auto"):
        r = agate.runtime
        init = None
        if latent_image is not None:
            s = latent_image["samples"]
            if s.ndim != 4 or s.shape[1] != 4:
                raise ValueError(f"Agate needs an SD 1.x latent (B, 4, H, W); got {tuple(s.shape)}")
            if s.shape[-1] not in [v["latent_hw"] for v in r.resolutions.values()] or s.shape[-1] != s.shape[-2]:
                log.warning("Agate: latent_image is %dx%d px; this Agate was trained at %s only",
                            s.shape[-1] * 8, s.shape[-2] * 8, r.sizes_text())
            if denoise <= 0.0:                          # nothing to do: pass the latent through
                return ({"samples": s.clone()},)
            init = latent_to_model(s)
        z = r.sample(prompt, negative_prompt, seed, steps, cfg, batch_size, autoguide, init_latent=init,
                     denoise=denoise, callback=_progress(steps, r.device, preview_mode=live_preview),
                     interrupt=_check_interrupt, resolution=_res(resolution))
        _log_prepared(r, prompt)
        return ({"samples": model_to_latent(z).to(_intermediate_device())},)


class AgateGenerate:
    CATEGORY = "LogoLabs/Agate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    DESCRIPTION = ("All in one: samples Agate and decodes with the loader's decoder (SD-VAE or TAESD): 512x512 "
                   "(or 256) with Preview 003, 256x256 with 001 / 002. Every image carries Agate's invisible "
                   "AI-output watermark and the workflow metadata gets the provenance entries (both options, ON "
                   "by default).")

    @classmethod
    def INPUT_TYPES(cls):
        d = _common_inputs()
        return {"required": d, "optional": {"live_preview": _live_preview(), "resolution": _optional_res(),
                                            **_optional_marking()},
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"}}

    def generate(self, agate, prompt, negative_prompt, seed, steps, cfg, autoguide, batch_size,
                 live_preview="side_by_side", resolution="auto", watermark=True, metadata=True, extra_pnginfo=None):
        r = agate.runtime
        z = r.sample(prompt, negative_prompt, seed, steps, cfg, batch_size, autoguide,
                     callback=_progress(steps, r.device, preview_mode=live_preview),
                     interrupt=_check_interrupt, resolution=_res(resolution))
        _log_prepared(r, prompt)
        if metadata:
            _add_provenance(extra_pnginfo, r)
        return (_mark(r.decode(z), r, watermark).contiguous(),)


def _log_prepared(r, prompt: str) -> None:
    if r.mr and r.last_prepared and (r.last_prepared[0] != prompt or r.last_prepared[1]):
        log.info("Agate 003 prompt pipeline: %r -> %r (negative %r)", prompt, *r.last_prepared)


def _to_image(a) -> torch.Tensor:
    """uint8 (B, H, W, 3) numpy -> ComfyUI IMAGE (float 0-1) on the intermediate device."""
    import numpy as np
    return torch.from_numpy(np.array(a, dtype=np.uint8, copy=True)).float().div_(255).to(_intermediate_device())


class AgatePlanViewer:
    CATEGORY = "LogoLabs/Agate"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "LATENT")
    RETURN_NAMES = ("image", "plan_frames", "region_frames", "prediction_frames", "change_frames", "panel",
                    "change_curve", "latent")
    OUTPUT_TOOLTIPS = (
        "The final image, decoded like Agate Generate (the loader's decoder)",
        "The thinker's plan at every step (one frame per step): its 640 channels as RGB (PCA fitted over "
        "all steps, so a colour means the same thing throughout)",
        "The plan's regions at every step: k-means over the plan cells of all steps, one colour per region",
        "What the model expects the final image to be at every step (x1 = z + (1 - t) v, decoded with TAESD)",
        "How much the plan changed since the previous step, per cell (dark blue: not at all, red: most)",
        "One frame per step: plan | regions | change | prediction with a label. Feed it to Save Animated "
        "WEBP / Video Combine",
        "The mean plan change per step, as a plot",
        "The final SD 1.x LATENT, as Agate Sampler outputs it",
    )
    FUNCTION = "view"
    DESCRIPTION = ("Samples one image and shows how Agate plans it. Agate's thinker lays the picture out on a "
                   "16 x 16 grid (the plan) and the renderer paints the image from it; this node records the "
                   "plan at every denoising step and returns it as frames: PCA colours, regions, change, and "
                   "the model's running prediction of the final image. The image is the same as Agate "
                   "Sampler's for the same seed.")

    @classmethod
    def INPUT_TYPES(cls):
        d = _common_inputs()
        del d["autoguide"], d["batch_size"]
        d["regions"] = ("INT", {"default": 6, "min": 2, "max": 10,
                                "tooltip": "Number of plan regions (k-means clusters) in region_frames"})
        d["frame_size"] = ("INT", {"default": 256, "min": 64, "max": 1024, "step": 16,
                                   "tooltip": "Side of each frame in pixels (the 16 x 16 plan is upscaled "
                                              "with nearest neighbour)"})
        d["denoise"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "With latent_image: how much to change it (as in Agate Sampler)"})
        return {"required": d, "optional": {"latent_image": ("LATENT",), "resolution": _optional_res(),
                                            **_optional_marking()},
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"}}

    def view(self, agate, prompt, negative_prompt, seed, steps, cfg, regions, frame_size, denoise=1.0,
             latent_image=None, resolution="auto", watermark=True, metadata=True, extra_pnginfo=None):
        import numpy as np
        from .agate_comfy import plan as pv
        r = agate.runtime
        init = None
        if latent_image is not None:
            s = latent_image["samples"]
            if s.ndim != 4 or s.shape[1] != 4:
                raise ValueError(f"Agate needs an SD 1.x latent (B, 4, H, W); got {tuple(s.shape)}")
            if s.shape[0] > 1:
                log.warning("Agate Plan Viewer: latent_image has a batch of %d; using the first", s.shape[0])
            s = s[:1]
            if denoise > 0.0:
                init = latent_to_model(s)
        if init is None and denoise < 1.0:
            if latent_image is None:
                raise ValueError("denoise < 1 needs a latent_image to start from (img2img); "
                                 "connect one or set denoise to 1.0")
            raise ValueError("denoise = 0 leaves the latent unchanged: there are no steps to view")

        plans, preds, ts = [], [], []

        def on_step(i, t, x1, plan):
            if plan is None:
                raise RuntimeError("Agate Plan Viewer: this model has no thinker output to capture")
            plans.append(plan[:1].detach().clone())
            preds.append(x1[:1].detach().clone())
            ts.append(float(t))

        progress = _progress(steps, r.device)
        z = r.sample(prompt, negative_prompt, seed, steps, cfg, 1, 0.0, init_latent=init, denoise=denoise,
                     callback=progress, interrupt=_check_interrupt, on_step=on_step, resolution=_res(resolution))
        if metadata:
            _add_provenance(extra_pnginfo, r)
        image = _mark(r.decode(z), r, watermark).contiguous()   # the final image; the analysis frames are not marked
        pred = (r.decode_fast(torch.cat(preds)).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        A = pv.analyse(torch.cat(plans).float(), int(regions))       # (S, C, gh, gw), on the device
        F = int(frame_size)
        plan_f = pv.upscale_nearest(A["pca"], F)
        reg_f = pv.upscale_nearest(pv.regions_rgb(A["labels"]), F)
        chg_f = pv.upscale_nearest(pv.heat_rgb(A["change"]), F)
        pred_f = pv.resize_smooth(pred, F)
        panel = pv.panels(plan_f, reg_f, chg_f, pred_f, A["curve"], np.array(ts), int(regions))
        curve = pv.curve_image(A["curve"])[None]
        return (image, _to_image(plan_f), _to_image(reg_f), _to_image(pred_f), _to_image(chg_f),
                _to_image(panel), _to_image(curve), {"samples": model_to_latent(z).to(_intermediate_device())})


RELEASES = ("003", "002", "001")


class AgateWatermark:
    CATEGORY = "LogoLabs/Agate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "mark"
    DESCRIPTION = ("Adds Agate's invisible AI-output watermark (invisible-watermark dwtDctSvd, payload AGATE + "
                   "release, the same as the Python packages and the WebGPU Space) to images, e.g. after Agate "
                   "Sampler + VAE Decode, which output a LATENT and cannot mark it. Connect the loader's agate "
                   "output to take the release from the model. Images under 256 x 256 px are passed unchanged.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "release": (list(RELEASES), {"default": "003",
                        "tooltip": "The Agate release that made the images (ignored when agate is connected)"}),
        }, "optional": {"agate": ("AGATE_MODEL",)}, "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"}}

    def mark(self, images, release, agate=None, extra_pnginfo=None):
        rel = agate.runtime.release_tag if agate is not None else release
        if isinstance(extra_pnginfo, dict):
            extra_pnginfo["ai_generated"] = True
            extra_pnginfo["agate"] = {k: v for k, v in mk.provenance(rel).items() if k != "ai_generated"}
        return (mk.mark_image_tensor(images, rel).contiguous(),)


class AgateSaveImage:
    CATEGORY = "LogoLabs/Agate"
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    DESCRIPTION = ("Saves PNGs with the provenance text chunks the Agate Python packages write (ai_generated, "
                   "generator, model, watermark), plus ComfyUI's usual prompt/workflow chunks unless "
                   "include_workflow is off. ComfyUI's stock Save Image cannot write these exact keys (it JSON-"
                   "encodes its entries). release auto reads the watermark to find the release; "
                   "ensure_watermark adds the mark to images that lack it.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "Agate"}),
            "release": (["auto"] + list(RELEASES), {"default": "auto",
                        "tooltip": "auto: the release read from the image's watermark"}),
            "ensure_watermark": ("BOOLEAN", {"default": True,
                                 "tooltip": "Add the watermark to images in which it is not found"}),
            "include_workflow": ("BOOLEAN", {"default": True,
                                 "tooltip": "Also write ComfyUI's prompt/workflow chunks (they contain the prompt)"}),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    def save(self, images, filename_prefix="Agate", release="auto", ensure_watermark=True, include_workflow=True,
             prompt=None, extra_pnginfo=None):
        import json
        import numpy as np
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        fp = _folder_paths()
        out_dir = Path(fp.get_output_directory()) if fp is not None else Path("output")
        h, w = images.shape[1], images.shape[2]
        if fp is not None:
            full, fname, counter, sub, _ = fp.get_save_image_path(filename_prefix, str(out_dir), w, h)
        else:
            out_dir.mkdir(exist_ok=True)
            full, fname, counter, sub = str(out_dir), filename_prefix, 1, ""
        try:
            from comfy.cli_args import args
            no_meta = bool(getattr(args, "disable_metadata", False))
        except Exception:
            no_meta = False
        results = []
        for b, img in enumerate(images):
            a = (img.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            rel = release
            det = mk.detect_watermark(a) if min(a.shape[:2]) >= 64 else {"detected": False, "release": None}
            if rel == "auto":
                rel = det["release"]
                if rel is None:
                    raise ValueError("Agate Save Image: no Agate watermark found in the image, so its release is "
                                     "unknown; pick the release instead of auto")
            if ensure_watermark and not (det["detected"] and det["release"] == rel):
                a = mk.mark_rgb_uint8(a, rel)
            meta = PngInfo()
            for k, v in mk.provenance(rel).items():
                meta.add_text(k, v)
            if include_workflow and not no_meta:
                if prompt is not None:
                    meta.add_text("prompt", json.dumps(prompt))
                for k, v in (extra_pnginfo or {}).items():
                    if k not in ("ai_generated", "agate"):
                        meta.add_text(k, json.dumps(v))
            file = f"{fname}_{counter:05}_.png"
            Image.fromarray(np.ascontiguousarray(a)).save(str(Path(full) / file), pnginfo=meta, compress_level=4)
            results.append({"filename": file, "subfolder": sub, "type": "output"})
            counter += 1
        return {"ui": {"images": results}}


def _hook_unload_all_models() -> None:
    """ComfyUI's "Unload Models" offloads Agate like any other model (its weights are registered
    through ModelPatchers). The hook additionally frees the CUDA graphs' memory pools."""
    mm = _mm()
    if mm is None or getattr(mm.unload_all_models, "_agate_hooked", False):
        return
    original = mm.unload_all_models

    def unload_all_models(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            m = _CACHE["model"]
            if m is not None and m._rt is not None:
                m._rt.step_fn.reset()
                if m._rt.guide_step is not None:
                    m._rt.guide_step.reset()

    unload_all_models._agate_hooked = True
    mm.unload_all_models = unload_all_models


_hook_unload_all_models()


NODE_CLASS_MAPPINGS = {
    "AgateLoader": AgateLoader,
    "AgateSampler": AgateSampler,
    "AgateGenerate": AgateGenerate,
    "AgatePlanViewer": AgatePlanViewer,
    "AgateWatermark": AgateWatermark,
    "AgateSaveImage": AgateSaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AgateLoader": "Agate Loader",
    "AgateSampler": "Agate Sampler",
    "AgateGenerate": "Agate Generate",
    "AgatePlanViewer": "Agate Plan Viewer",
    "AgateWatermark": "Agate Watermark",
    "AgateSaveImage": "Agate Save Image",
}
