"""Tests of the Agate nodes outside ComfyUI, with `comfy.*` stubbed.

Needs the model. AGATE_CKPT is the single-file checkpoint (default: downloaded from
Logolabs/agate-preview-001, comfyui/agate-preview-001.safetensors); AGATE_SOURCE is the release
folder or repo id, used for the parity test against the release pipeline. Tests that need a
model skip when it cannot be found.

    AGATE_CKPT=models/agate/agate-preview-001.safetensors AGATE_SOURCE=/path/to/agate-preview-001 \
        python -m pytest -q tests
    AGATE_EXAMPLE=docs/example.png python -m pytest -q tests -k example   # re-render docs/example.png
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get("AGATE_SOURCE", "Logolabs/agate-preview-001")
PROMPT = "a red cube on top of a blue sphere"


class _ProgressBar:
    instances: list = []

    def __init__(self, total):
        self.total, self.calls, self.previews = total, [], []
        _ProgressBar.instances.append(self)

    def update_absolute(self, value, total=None, preview=None):
        self.calls.append(value)
        self.previews.append(preview)


class _Previewer:
    seen: list = []

    def decode_latent_to_preview_image(self, fmt, x0):
        _Previewer.seen.append(tuple(x0.shape))
        return (fmt, "img", 512)


def _install_comfy_stubs():
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mm.intermediate_device = lambda: torch.device("cpu")
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.unloaded = 0

    def unload_all_models():
        mm.unloaded += 1
    mm.unload_all_models = unload_all_models
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = _ProgressBar
    formats = types.ModuleType("comfy.latent_formats")

    class SD15:
        scale_factor = 0.18215
    formats.SD15 = SD15
    lp = types.ModuleType("latent_preview")
    lp.get_previewer = lambda device, fmt: _Previewer()
    comfy.model_management, comfy.utils, comfy.latent_formats = mm, utils, formats
    sys.modules.update({"comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils,
                        "comfy.latent_formats": formats, "latent_preview": lp})
    return mm


@pytest.fixture(scope="module")
def pack():
    mm = _install_comfy_stubs()
    spec = importlib.util.spec_from_file_location("comfyui_agate", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_agate"] = mod
    spec.loader.exec_module(mod)
    return mod, sys.modules["comfyui_agate.nodes"], mm


@pytest.fixture(scope="module")
def ckpt(pack):
    _, nodes, _ = pack
    p = os.environ.get("AGATE_CKPT")
    if p:
        return str(Path(p).resolve())
    try:
        return str(nodes.resolve_checkpoint(nodes.MAIN_NAME))
    except Exception as e:  # offline and not given
        pytest.skip(f"Agate checkpoint not available: {e}")


@pytest.fixture(scope="module")
def agate(pack, ckpt):
    mod, _, _ = pack
    (a,) = mod.NODE_CLASS_MAPPINGS["AgateLoader"]().load(ckpt, "sd-vae", "auto", True, False)
    return a


def _sampler_args(seed=0, steps=4, batch=1, denoise=1.0):
    return dict(prompt=PROMPT, negative_prompt="", seed=seed, steps=steps, cfg=3.0, autoguide=0.0,
                batch_size=batch, denoise=denoise)


# -- no model needed ----------------------------------------------------------------------------

def test_mappings(pack):
    mod, _, _ = pack
    assert set(mod.NODE_CLASS_MAPPINGS) == {"AgateLoader", "AgateSampler", "AgateGenerate", "AgatePlanViewer"}
    assert set(mod.NODE_DISPLAY_NAME_MAPPINGS) == set(mod.NODE_CLASS_MAPPINGS)
    for cls in mod.NODE_CLASS_MAPPINGS.values():
        assert "required" in cls.INPUT_TYPES()
        assert callable(getattr(cls, cls.FUNCTION))
    assert mod.NODE_CLASS_MAPPINGS["AgateSampler"].RETURN_TYPES == ("LATENT",)
    assert "latent_image" in mod.NODE_CLASS_MAPPINGS["AgateSampler"].INPUT_TYPES()["optional"]


def test_latent_format(pack):
    """ComfyUI LATENTs are unscaled SD 1.x latents; Agate samples in the scaled space."""
    _, nodes, _ = pack
    import comfy.latent_formats
    assert nodes.rt.VAE_SCALE == comfy.latent_formats.SD15.scale_factor == 0.18215
    x = torch.randn(2, 4, 32, 32)
    assert torch.allclose(nodes.latent_to_model(x), x * 0.18215)
    assert torch.allclose(nodes.model_to_latent(nodes.latent_to_model(x)), x, atol=1e-6)


def test_bad_path(pack):
    _, nodes, _ = pack
    with pytest.raises(FileNotFoundError):
        nodes.resolve_source(r"C:\definitely\not\here" if os.name == "nt" else "/definitely/not/here")


def test_single_file_metadata(pack, ckpt):
    from comfyui_agate.agate_comfy.checkpoint import read_metadata
    meta = read_metadata(ckpt)
    cfg = json.loads(meta["agate.config"])
    assert cfg["vae_scale"] == 0.18215 and cfg["latent_hw"] == 32 and cfg["text_max_len"] == 512
    assert cfg["sampler"]["steps"] == 50 and cfg["sampler"]["cfg"] == 3.0
    assert cfg["guide"]["file"] == "agate-guide-27600.safetensors"


def test_kmeans_stable(pack):
    """Deterministic, recovers well-separated clusters, orders them by size (largest first)."""
    from comfyui_agate.agate_comfy import plan as pv
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(3, 16)) * 10
    sizes = [300, 120, 40]
    X = np.concatenate([c + rng.normal(size=(n, 16)) * 0.1 for c, n in zip(centers, sizes)])
    perm = rng.permutation(len(X))
    a = pv.kmeans(X[perm], 3).numpy()
    b = pv.kmeans(X[perm], 3).numpy()
    assert np.array_equal(a, b), "same input, same result"
    for j, c in enumerate(centers.astype(np.float32)):     # cluster j is true cluster j (sizes descend)
        assert np.abs(a[j] - c).max() < 0.1
    c = pv.kmeans(X[rng.permutation(len(X))], 3).numpy()           # the order of the points does not matter
    assert np.allclose(a, c, atol=1e-4)


def test_plan_analysis_synthetic(pack):
    """A plan with two fixed halves: regions are constant over steps, change is ~0."""
    from comfyui_agate.agate_comfy import plan as pv
    rng = np.random.default_rng(0)
    S, C, H, W = 5, 32, 16, 16
    left, right = rng.normal(size=C), rng.normal(size=C)
    P = np.empty((S, C, H, W), np.float32)
    P[:, :, :, :8] = left[None, :, None, None]
    P[:, :, :, 8:] = right[None, :, None, None]
    P += rng.normal(size=P.shape).astype(np.float32) * 0.01
    A = pv.analyse(P, 2)
    assert A["pca"].shape == (S, H, W, 3) and A["pca"].dtype == np.uint8
    lab = A["labels"]
    assert lab.shape == (S, H, W)
    assert (lab[:, :, :8] == lab[0, 0, 0]).all() and (lab[:, :, 8:] == lab[0, 0, 8]).all()
    assert lab[0, 0, 0] != lab[0, 0, 8]
    assert (lab == pv.region_labels(P, 2)).all(), "labels are reproducible"
    assert A["change"].shape == (S, H, W) and A["change"][0].max() == 0 and A["curve"].max() < 1e-3
    img = pv.curve_image(A["curve"])
    assert img.shape == (260, 640, 3) and img.dtype == np.uint8


# -- with the model -----------------------------------------------------------------------------

def test_sampler_latent_and_previews(pack, agate):
    mod, nodes, _ = pack
    _ProgressBar.instances.clear()
    _Previewer.seen.clear()
    (lat,) = mod.NODE_CLASS_MAPPINGS["AgateSampler"]().sample(agate, **_sampler_args())
    s = lat["samples"]
    assert s.shape == (1, 4, 32, 32) and s.dtype == torch.float32 and s.device.type == "cpu"
    # Unscaled SD-VAE latents have a std of roughly 1 / 0.18215 ~ 5.5, scaled ones ~1.
    assert 2.0 < s.std().item() < 12.0
    assert _ProgressBar.instances[-1].calls == [1, 2, 3, 4]
    assert all(p is not None for p in _ProgressBar.instances[-1].previews)
    assert _Previewer.seen == [(1, 4, 32, 32)] * 4


def test_sampler_matches_generate(pack, agate):
    """Sampler -> decode of the LATENT is exactly the all-in-one Generate."""
    mod, nodes, _ = pack
    (lat,) = mod.NODE_CLASS_MAPPINGS["AgateSampler"]().sample(agate, **_sampler_args(seed=3))
    (img,) = mod.NODE_CLASS_MAPPINGS["AgateGenerate"]().generate(agate, PROMPT, "", 3, 4, 3.0, 0.0, 1)
    assert img.shape == (1, 256, 256, 3) and 0.0 <= img.min().item() and img.max().item() <= 1.0
    again = agate.runtime.decode(nodes.latent_to_model(lat["samples"]).to(agate.device))
    assert (again - img).abs().max().item() <= 1 / 255


def test_img2img_denoise_one_is_txt2img(pack, agate):
    mod, _, _ = pack
    sampler = mod.NODE_CLASS_MAPPINGS["AgateSampler"]()
    (ref,) = sampler.sample(agate, **_sampler_args(seed=5, batch=2))
    init = {"samples": torch.randn(2, 4, 32, 32) * 5}
    (got,) = sampler.sample(agate, **_sampler_args(seed=5, batch=1), latent_image=init)
    assert got["samples"].shape == (2, 4, 32, 32), "the latent's batch size wins"
    assert torch.equal(got["samples"], ref["samples"]), "denoise=1 must ignore the latent's content"


def test_img2img_partial_denoise(pack, agate):
    mod, _, _ = pack
    sampler = mod.NODE_CLASS_MAPPINGS["AgateSampler"]()
    (src,) = sampler.sample(agate, **_sampler_args(seed=1, steps=8))
    (same,) = sampler.sample(agate, **_sampler_args(seed=9, steps=8, denoise=0.0), latent_image=src)
    assert torch.equal(same["samples"], src["samples"]), "denoise=0 returns the input"
    (light,) = sampler.sample(agate, **_sampler_args(seed=9, steps=8, denoise=0.3), latent_image=src)
    (fresh,) = sampler.sample(agate, **_sampler_args(seed=9, steps=8))
    d_light = (light["samples"] - src["samples"]).pow(2).mean().item()
    d_fresh = (fresh["samples"] - src["samples"]).pow(2).mean().item()
    assert d_light < 0.5 * d_fresh, (d_light, d_fresh)
    with pytest.raises(ValueError):
        sampler.sample(agate, **_sampler_args(denoise=0.5))           # img2img without a latent


def test_generate_determinism_batch_and_reload(pack, ckpt, agate):
    mod, nodes, mm = pack
    (again,) = mod.NODE_CLASS_MAPPINGS["AgateLoader"]().load(ckpt, "sd-vae", "auto", True, False)
    assert again is agate, "same settings must reuse the cached pipeline"
    gen = mod.NODE_CLASS_MAPPINGS["AgateGenerate"]()
    (img,) = gen.generate(agate, PROMPT, "", 0, 4, 3.0, 0.0, 1)
    (img2,) = gen.generate(agate, PROMPT, "", 0, 4, 3.0, 0.0, 1)
    assert torch.equal(img, img2), "same seed must give the same image"
    (batch,) = gen.generate(agate, "a cat", "blurry", 7, 3, 3.0, 0.0, 2)
    assert batch.shape == (2, 256, 256, 3)
    mm.unload_all_models()                          # the hook drops CUDA graphs; images unchanged
    (img3,) = gen.generate(agate, PROMPT, "", 0, 4, 3.0, 0.0, 1)
    assert torch.equal(img, img3)
    agate.unload()
    assert not agate.loaded
    (img4,) = gen.generate(agate, PROMPT, "", 0, 4, 3.0, 0.0, 1)
    assert agate.loaded and torch.equal(img, img4), "a released handle reloads on use"


def test_single_file_matches_release_pipeline(pack, ckpt):
    """The single bf16 file reproduces the release pipeline (fp32 files, bf16 compute on CUDA)."""
    _, nodes, _ = pack
    from comfyui_agate.agate_comfy.agate.pipeline import AgatePipeline
    try:
        root = nodes.resolve_source(SOURCE)
    except Exception as e:
        pytest.skip(f"release folder not available from {SOURCE!r}: {e}")
    dev = nodes._default_device()
    if dev.type != "cuda":
        pytest.skip("bit-exact parity holds on CUDA (the release runs fp32 weights on the CPU)")
    nodes.release_cached_model()
    pipe = AgatePipeline(root, device=str(dev))
    ref = pipe(PROMPT, seed=0, steps=8, output_type="pt")
    del pipe
    r = nodes.rt.from_single_file(ckpt, dev, True)
    img = r.decode(r.sample(PROMPT, seed=0, steps=8))
    r.release()
    assert (img - ref).abs().max().item() * 255 <= 1.0


@pytest.mark.skipif(not os.environ.get("AGATE_EXAMPLE"), reason="set AGATE_EXAMPLE=<png path> to render")
def test_render_example(pack, agate):
    mod, _, _ = pack
    from PIL import Image
    gen = mod.NODE_CLASS_MAPPINGS["AgateGenerate"]()
    prompt = "a minimalist logo of a fox head, orange, flat design, white background"
    gen.generate(agate, prompt, "", 0, 50, 3.0, 0.0, 1)          # warm-up: records the CUDA graph
    t = time.perf_counter()
    (img,) = gen.generate(agate, prompt, "", 0, 50, 3.0, 0.0, 1)
    print(f"\n50 steps: {time.perf_counter() - t:.2f} s")
    Image.fromarray((img[0].numpy() * 255).round().astype("uint8")).save(os.environ["AGATE_EXAMPLE"])


def _viewer_args(seed=0, steps=4, frame_size=128, regions=4):
    return dict(prompt=PROMPT, negative_prompt="", seed=seed, steps=steps, cfg=3.0, regions=regions,
                frame_size=frame_size, denoise=1.0)


def test_plan_viewer_outputs(pack, agate):
    mod, _, _ = pack
    _ProgressBar.instances.clear()
    out = mod.NODE_CLASS_MAPPINGS["AgatePlanViewer"]().view(agate, **_viewer_args())
    assert len(out) == len(mod.NODE_CLASS_MAPPINGS["AgatePlanViewer"].RETURN_TYPES) == 8
    image, plan, regions, pred, change, panel, curve, latent = out
    assert image.shape == (1, 256, 256, 3)
    for frames in (plan, regions, pred, change):
        assert frames.shape == (4, 128, 128, 3)
    assert panel.shape[0] == 4 and panel.shape[2] == 4 * 128 + 3 * 4 and panel.shape[1] > 128
    assert curve.shape == (1, 260, 640, 3)
    for x in out[:7]:
        assert x.dtype == torch.float32 and 0.0 <= x.min().item() and x.max().item() <= 1.0
    assert latent["samples"].shape == (1, 4, 32, 32)
    assert _ProgressBar.instances[-1].calls == [1, 2, 3, 4]
    # 16 x 16 plan, nearest-upscaled: every 8 x 8 block of a 128 px frame is one colour
    blocks = plan[:, ::8, ::8]
    assert torch.equal(plan, blocks.repeat_interleave(8, 1).repeat_interleave(8, 2))
    # the predictions converge towards the final image
    import torch.nn.functional as F
    p256 = F.interpolate(pred.permute(0, 3, 1, 2), size=256, mode="bilinear").permute(0, 2, 3, 1)
    err = [(p256[i] - image[0]).abs().mean().item() for i in range(4)]
    assert err[-1] < err[0], err


@pytest.mark.parametrize("graphs", [True, False])
def test_plan_viewer_matches_sampler(pack, ckpt, graphs):
    """Capturing the plan does not change the image: the LATENT equals Agate Sampler output for the
    same seed, with and without CUDA graphs."""
    mod, nodes, _ = pack
    (a,) = mod.NODE_CLASS_MAPPINGS["AgateLoader"]().load(ckpt, "sd-vae", "auto", graphs, False)
    (ref,) = mod.NODE_CLASS_MAPPINGS["AgateSampler"]().sample(a, **_sampler_args(seed=11, steps=6))
    out = mod.NODE_CLASS_MAPPINGS["AgatePlanViewer"]().view(a, **_viewer_args(seed=11, steps=6))
    assert torch.equal(out[7]["samples"], ref["samples"])
    img = a.runtime.decode(nodes.latent_to_model(ref["samples"]).to(a.device))
    assert (out[0] - img).abs().max().item() <= 1 / 255
    (again,) = mod.NODE_CLASS_MAPPINGS["AgateSampler"]().sample(a, **_sampler_args(seed=11, steps=6))
    assert torch.equal(again["samples"], ref["samples"]), "the viewer leaves the sampler unchanged"
    grabbed = []
    a.runtime.sample(PROMPT, "", 11, 3, 3.0, 1, on_step=lambda i, t, x1, plan: grabbed.append(plan.clone()))
    assert len(grabbed) == 3 and grabbed[0].shape[-2:] == (16, 16) and grabbed[0].shape[0] == 2
    assert not torch.equal(grabbed[0], grabbed[-1]), "a fresh plan per step, not a stale buffer"


def test_plan_capture_graph_equals_eager(pack, ckpt):
    """The plan read from the CUDA graph buffer is the real thinker output (as captured eagerly)."""
    mod, nodes, _ = pack
    if nodes._default_device().type != "cuda":
        pytest.skip("CUDA graphs need CUDA")
    runs = {}
    for graphs in (True, False):
        (a,) = mod.NODE_CLASS_MAPPINGS["AgateLoader"]().load(ckpt, "sd-vae", "auto", graphs, False)
        grabbed = []
        a.runtime.sample(PROMPT, "", 2, 3, 3.0, 1,
                         on_step=lambda i, t, x1, plan: grabbed.append(plan.float().clone()))
        runs[graphs] = torch.stack(grabbed)
    cos = torch.nn.functional.cosine_similarity(runs[True].flatten(1), runs[False].flatten(1))
    assert cos.min().item() > 0.999, cos


@pytest.mark.skipif(not os.environ.get("AGATE_PLAN_DOCS"), reason="set AGATE_PLAN_DOCS=<dir> to render")
def test_render_plan_docs(pack, ckpt):
    """Renders the README animated plan examples (docs/plan_*.webp) and their change curves."""
    mod, _, _ = pack
    from PIL import Image
    (agate,) = mod.NODE_CLASS_MAPPINGS["AgateLoader"]().load(ckpt, "sd-vae", "auto", True, False)
    out_dir = Path(os.environ["AGATE_PLAN_DOCS"])
    viewer = mod.NODE_CLASS_MAPPINGS["AgatePlanViewer"]()
    for name, prompt in (("plan_dog_cat", "a dog sitting to the left of a cat"),
                         ("plan_fox_logo", "a minimalist logo of a fox head, orange, flat design, white background")):
        t = time.perf_counter()
        out = viewer.view(agate, prompt, "", 0, 50, 3.0, 6, 160, 1.0)
        print(f"\n{name}: {time.perf_counter() - t:.2f} s")
        frames = [Image.fromarray((f.numpy() * 255).round().astype("uint8")) for f in out[5]]
        durations = [100] * (len(frames) - 1) + [2000]                # hold the last frame
        frames[0].save(out_dir / f"{name}.webp", save_all=True, append_images=frames[1:], duration=durations,
                       loop=0, quality=80, method=6)
        Image.fromarray((out[6][0].numpy() * 255).round().astype("uint8")).save(out_dir / f"{name}_change.png")
