"""End-to-end test of Agate Loader + Agate Generate outside ComfyUI, with `comfy.*` stubbed.

Needs the model: set AGATE_SOURCE to a local model folder or a repo id you can access
(default: Logolabs/agate-preview-001). Skips when the model cannot be resolved.

    AGATE_SOURCE=/path/to/agate-preview-001 python -m pytest -q tests
    AGATE_EXAMPLE=docs/example.png python -m pytest -q tests -k example   # re-render docs/example.png
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get("AGATE_SOURCE", "Logolabs/agate-preview-001")


class _ProgressBar:
    instances: list = []

    def __init__(self, total):
        self.total, self.calls = total, []
        _ProgressBar.instances.append(self)

    def update_absolute(self, value, total=None, preview=None):
        self.calls.append(value)


def _install_comfy_stubs():
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mm.free_memory = lambda *a, **k: None
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.unloaded = 0

    def unload_all_models():
        mm.unloaded += 1
    mm.unload_all_models = unload_all_models
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = _ProgressBar
    comfy.model_management, comfy.utils = mm, utils
    sys.modules.update({"comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils})
    return mm


@pytest.fixture(scope="module")
def pack():
    mm = _install_comfy_stubs()
    spec = importlib.util.spec_from_file_location("comfyui_agate", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_agate"] = mod
    spec.loader.exec_module(mod)
    nodes = sys.modules["comfyui_agate.nodes"]
    try:
        nodes.resolve_source(SOURCE)
    except Exception as e:  # no local folder and no HF access
        pytest.skip(f"Agate model not available from {SOURCE!r}: {e}")
    return mod, nodes, mm


def test_mappings(pack):
    mod, _, _ = pack
    assert set(mod.NODE_CLASS_MAPPINGS) == {"AgateLoader", "AgateGenerate"}
    assert set(mod.NODE_DISPLAY_NAME_MAPPINGS) == set(mod.NODE_CLASS_MAPPINGS)
    for cls in mod.NODE_CLASS_MAPPINGS.values():
        assert "required" in cls.INPUT_TYPES()
        assert callable(getattr(cls, cls.FUNCTION))


def test_bad_path(pack):
    _, nodes, _ = pack
    with pytest.raises(FileNotFoundError):
        nodes.resolve_source(r"C:\definitely\not\here" if os.name == "nt" else "/definitely/not/here")


def test_loader_generate(pack):
    mod, nodes, mm = pack
    (agate,) = mod.NODE_CLASS_MAPPINGS["AgateLoader"]().load(SOURCE, "sd-vae", "auto", True, False)
    (again,) = mod.NODE_CLASS_MAPPINGS["AgateLoader"]().load(SOURCE, "sd-vae", "auto", True, False)
    assert again is agate, "same settings must reuse the cached pipeline"
    gen = mod.NODE_CLASS_MAPPINGS["AgateGenerate"]()
    _ProgressBar.instances.clear()
    (img,) = gen.generate(agate, "a red cube on top of a blue sphere", "", 0, 4, 3.0, 0.0, 1)
    assert img.shape == (1, 256, 256, 3) and img.dtype == torch.float32
    assert 0.0 <= img.min().item() and img.max().item() <= 1.0 and img.std().item() > 0.01
    assert _ProgressBar.instances[-1].calls == [1, 2, 3, 4]
    (img2,) = gen.generate(agate, "a red cube on top of a blue sphere", "", 0, 4, 3.0, 0.0, 1)
    assert torch.allclose(img, img2, atol=2 / 255), "same seed must give the same image"
    (batch,) = gen.generate(agate, "a cat", "blurry", 7, 3, 3.0, 0.0, 2)
    assert batch.shape == (2, 256, 256, 3)
    mm.unload_all_models()                      # ComfyUI's "Unload Models" frees Agate too
    assert mm.unloaded == 1 and not agate.loaded
    (img3,) = gen.generate(agate, "a red cube on top of a blue sphere", "", 0, 4, 3.0, 0.0, 1)
    assert agate.loaded and torch.allclose(img, img3, atol=2 / 255), "a stale handle reloads on use"


@pytest.mark.skipif(not os.environ.get("AGATE_EXAMPLE"), reason="set AGATE_EXAMPLE=<png path> to render")
def test_render_example(pack):
    mod, _, _ = pack
    from PIL import Image
    (agate,) = mod.NODE_CLASS_MAPPINGS["AgateLoader"]().load(SOURCE, "sd-vae", "auto", True, False)
    gen = mod.NODE_CLASS_MAPPINGS["AgateGenerate"]()
    prompt = "a minimalist logo of a fox head, orange, flat design, white background"
    gen.generate(agate, prompt, "", 0, 50, 3.0, 0.0, 1)          # warm-up: records the CUDA graph
    t = time.perf_counter()
    (img,) = gen.generate(agate, prompt, "", 0, 50, 3.0, 0.0, 1)
    print(f"\n50 steps: {time.perf_counter() - t:.2f} s")
    Image.fromarray((img[0].numpy() * 255).round().astype("uint8")).save(os.environ["AGATE_EXAMPLE"])
