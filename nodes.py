"""ComfyUI nodes for Agate, LogoLabs' 260M text-to-image model.

Agate Loader   -> AGATE_MODEL  (single-file checkpoint from models/agate/, or a release folder / repo id)
Agate Sampler  -> LATENT       (SD 1.x latent: decode with the stock VAE Decode; optional img2img input)
Agate Generate -> IMAGE        (all in one: sample + Agate's own SD-VAE / TAESD decode)
Agate Plan Viewer -> IMAGEs    (the thinker's 16 x 16 plan at every step, as frames; agate_comfy/plan.py)

The model code is the vendored release package (agate_comfy/agate, MIT); agate_comfy/runtime.py
holds the ComfyUI side: memory management, img2img, previews.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import torch

from .agate_comfy import SOURCE_REPO
from .agate_comfy import runtime as rt
from .agate_comfy.checkpoint import MAIN_NAME

log = logging.getLogger("agate-comfyui")

HF_REPO = SOURCE_REPO
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
    return [MAIN_NAME] + sorted(n for n in names if n != MAIN_NAME)


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


def _count_load() -> None:
    """Read the model repo's root config.json in the background. The Hub counts a model download only on a
    request for that file, so this makes every load of an official checkpoint count once, as a Python
    from_pretrained() does. Best effort: offline or blocked just skips it."""
    def ping():
        try:
            from huggingface_hub import get_session, hf_hub_url
            get_session().get(hf_hub_url(HF_REPO, "config.json"), timeout=5)
        except Exception:
            pass
    import threading
    threading.Thread(target=ping, name="agate-count", daemon=True).start()


def _download(name: str, dest_dir: Path | None) -> Path:
    """Fetch comfyui/<name> from the Hugging Face model repo into models/agate/."""
    from huggingface_hub import hf_hub_download
    if dest_dir is None:                                     # outside ComfyUI: the HF cache
        return Path(hf_hub_download(HF_REPO, f"{HF_SUBFOLDER}/{name}"))
    log.info("Agate: downloading %s/%s/%s into %s", HF_REPO, HF_SUBFOLDER, name, dest_dir)
    staging = dest_dir / ".download"
    try:
        got = Path(hf_hub_download(HF_REPO, f"{HF_SUBFOLDER}/{name}", local_dir=str(staging)))
    except Exception as e:
        raise RuntimeError(
            f"Could not download {name} from https://huggingface.co/{HF_REPO} ({type(e).__name__}: {e}). "
            f"Download it by hand from https://huggingface.co/{HF_REPO}/tree/main/{HF_SUBFOLDER} and put it "
            f"in {dest_dir}") from e
    target = dest_dir / name
    os.replace(got, target)
    shutil.rmtree(staging, ignore_errors=True)
    return target


def resolve_checkpoint(name: str) -> Path:
    """A file name from the dropdown (or an absolute path) -> a local file, downloading the
    official checkpoints from Hugging Face when they are missing."""
    p = Path(name)
    if p.name == MAIN_NAME or p.name.startswith("agate-guide-"):
        _count_load()
    if p.is_absolute() and p.is_file():
        return p
    found = _find_in_model_folder(name)
    if found is not None:
        return found
    if p.name == MAIN_NAME or p.name.startswith("agate-guide-"):
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


def load_agate(checkpoint: str = MAIN_NAME, decoder: str = "sd-vae", device: str = "auto",
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
    DESCRIPTION = ("Loads Agate (LogoLabs, 260M text-to-image, 256x256) from models/agate/. The official "
                   "checkpoint downloads from Hugging Face on first use. Cached: re-running reuses it.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "checkpoint": (checkpoint_choices(), {"default": MAIN_NAME,
                           "tooltip": f"Single-file checkpoints in models/{MODEL_FOLDER}/. {MAIN_NAME} is "
                                      f"downloaded from huggingface.co/{HF_REPO} if it is missing."}),
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
        d["live_preview"] = (["side_by_side", "image", "plan", "none"], {"default": "side_by_side",
                                  "tooltip": "Live preview on this node during sampling: side_by_side shows "
                                             "both the developing image and the thinker's 16x16 plan."})
        return {"required": d, "optional": {"latent_image": ("LATENT",)}}

    def sample(self, agate, prompt, negative_prompt, seed, steps, cfg, autoguide, batch_size, denoise,
               live_preview="side_by_side", latent_image=None):
        r = agate.runtime
        init = None
        if latent_image is not None:
            s = latent_image["samples"]
            if s.ndim != 4 or s.shape[1] != 4:
                raise ValueError(f"Agate needs an SD 1.x latent (B, 4, H, W); got {tuple(s.shape)}")
            if tuple(s.shape[-2:]) != (r.latent_hw, r.latent_hw):
                log.warning("Agate: latent_image is %dx%d px; Agate was trained at %d px only",
                            s.shape[-1] * 8, s.shape[-2] * 8, r.latent_hw * 8)
            if denoise <= 0.0:                          # nothing to do: pass the latent through
                return ({"samples": s.clone()},)
            init = latent_to_model(s)
        z = r.sample(prompt, negative_prompt, seed, steps, cfg, batch_size, autoguide, init_latent=init,
                     denoise=denoise, callback=_progress(steps, r.device, preview_mode=live_preview),
                     interrupt=_check_interrupt)
        return ({"samples": model_to_latent(z).to(_intermediate_device())},)


class AgateGenerate:
    CATEGORY = "LogoLabs/Agate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    DESCRIPTION = ("All in one: samples Agate and decodes with the loader's decoder (SD-VAE or TAESD) to "
                   "256x256 images. Pair with an upscale node for larger output.")

    @classmethod
    def INPUT_TYPES(cls):
        d = _common_inputs()
        d["live_preview"] = (["side_by_side", "image", "plan", "none"], {"default": "side_by_side",
                                  "tooltip": "Live preview on this node during sampling: side_by_side shows "
                                             "both the developing image and the thinker's 16x16 plan."})
        return {"required": d}

    def generate(self, agate, prompt, negative_prompt, seed, steps, cfg, autoguide, batch_size,
                 live_preview="side_by_side"):
        r = agate.runtime
        z = r.sample(prompt, negative_prompt, seed, steps, cfg, batch_size, autoguide,
                     callback=_progress(steps, r.device, preview_mode=live_preview),
                     interrupt=_check_interrupt)
        return (r.decode(z).contiguous(),)


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
        return {"required": d, "optional": {"latent_image": ("LATENT",)}}

    def view(self, agate, prompt, negative_prompt, seed, steps, cfg, regions, frame_size, denoise=1.0,
             latent_image=None):
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
                     callback=progress, interrupt=_check_interrupt, on_step=on_step)
        image = r.decode(z).contiguous()
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AgateLoader": "Agate Loader",
    "AgateSampler": "Agate Sampler",
    "AgateGenerate": "Agate Generate",
    "AgatePlanViewer": "Agate Plan Viewer",
}
