"""Agate as single-file ComfyUI checkpoints.

One `.safetensors` file per model holds everything the sampler needs:

    generator.*      the FCDM-Thinker2 flow model (bf16)
    text_encoder.*   the fine-tuned Ettin-68M text encoder, a ModernBERT (bf16)
    metadata         agate.config (model_kw, text_max_len, vae_scale, sampler defaults, guide file),
                     the Ettin config.json and its tokenizer files, and modelspec.* fields

bf16 is the precision the release pipeline runs in on CUDA (weights cast to bf16, bf16 autocast),
so on a GPU the single file gives the release pipeline's images bit for bit. On the CPU the release
runs in fp32; there the bf16-rounded weights differ from the fp32 originals by <= 2^-9 relative.

Build the files from the release folder (the Hugging Face repo layout) with

    python -m agate_comfy.checkpoint /path/to/agate-preview-001 out_dir/

which writes out_dir/agate-preview-001.safetensors and out_dir/agate-guide-27600.safetensors. The same
command builds agate-preview-002.safetensors (same architecture; its guide is 001's) and
agate-preview-003.safetensors (arch fcdm_t2mr, 512 px; no guide). The release name is the folder name.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

FORMAT = "agate-comfyui/1"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
MAIN_NAME = "agate-preview-001.safetensors"          # the first release; old workflows name it
STORAGE_DTYPE = torch.bfloat16
# The official single files: file name -> the Hugging Face model repo that hosts it under comfyui/.
OFFICIAL = {
    "agate-preview-003.safetensors": "Logolabs/agate-preview-003",
    "agate-preview-002.safetensors": "Logolabs/agate-preview-002",
    MAIN_NAME: "Logolabs/agate-preview-001",
}
GUIDE_REPO = "Logolabs/agate-preview-001"                 # agate-guide-27600 (used by 001 and 002)
DEFAULT_NAME = "agate-preview-003.safetensors"


def guide_file_name(step: int) -> str:
    return f"agate-guide-{int(step)}.safetensors"


def _state(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


def _pack(generator_sd: dict, text_dir: Path, config: dict, title: str) -> tuple[dict, dict]:
    mr = config.get("arch") == "fcdm_t2mr"
    res = int(config.get("resolution", 256))
    tensors = {}
    for prefix, sd in (("generator.", generator_sd), ("text_encoder.", _state(text_dir / "model.safetensors"))):
        for k, v in sd.items():
            tensors[prefix + k] = v.to(STORAGE_DTYPE).contiguous() if v.is_floating_point() else v.contiguous()
    meta = {
        "agate.format": FORMAT,
        "agate.config": json.dumps(config, separators=(",", ":")),
        "agate.text_encoder.config": (text_dir / "config.json").read_text(encoding="utf-8"),
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture": "agate/fcdm-thinker2-mr" if mr else "agate/fcdm-thinker2",
        "modelspec.implementation": "https://github.com/logolabs/agate-comfyui",
        "modelspec.title": title,
        "modelspec.author": "LogoLabs",
        "modelspec.license": "MIT",
        "modelspec.resolution": f"{res}x{res}",
    }
    for f in TOKENIZER_FILES:
        meta[f"agate.tokenizer.{f}"] = (text_dir / f).read_text(encoding="utf-8")
    return tensors, meta


def convert_release(release: str | Path, out_dir: str | Path) -> list[Path]:
    """Release folder (config.json, generator.safetensors, text_encoder/, guide/) -> single files."""
    from safetensors.torch import save_file
    release, out_dir = Path(release), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((release / "config.json").read_text(encoding="utf-8"))
    g = cfg.get("guide")
    base = {k: cfg[k] for k in ("model_type", "arch", "model_kw", "params_generator", "params_generator_active",
                                "text_max_len", "vae", "vae_scale", "fast_vae", "resolution", "latent_hw",
                                "resolutions", "prompt_pipeline", "sampler", "training")
            if k in cfg}
    name = release.name if release.name.startswith("agate-preview-") else "agate-preview-001"
    num = name.rsplit("-", 1)[-1]
    base["release"] = name
    if g:
        base["guide"] = {"file": guide_file_name(g["step"]), "step": g["step"], "text_max_len": g["text_max_len"]}
    written = []
    tensors, meta = _pack(_state(release / "generator.safetensors"), release / cfg["text_encoder"], base,
                          f"Agate preview {num}")
    save_file(tensors, str(out_dir / f"{name}.safetensors"), metadata=meta)
    written.append(out_dir / f"{name}.safetensors")
    if g:
        gcfg = dict(base, role="guide", text_max_len=g["text_max_len"], training={"step": g["step"]})
        gcfg.pop("guide")
        tensors, meta = _pack(_state(release / g["generator"]), release / g["text_encoder"], gcfg,
                              f"Agate preview 001 guide (step {g['step']})")
        save_file(tensors, str(out_dir / guide_file_name(g["step"])), metadata=meta)
        written.append(out_dir / guide_file_name(g["step"]))
    return written


# ---------------------------------------------------------------------------------------------
# Loading

def read_metadata(path: str | Path) -> dict:
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    if meta.get("agate.format") != FORMAT:
        raise ValueError(f"{path} is not an Agate ComfyUI checkpoint (metadata agate.format="
                         f"{meta.get('agate.format')!r}, expected {FORMAT!r})")
    return meta


def load_single_file(path: str | Path):
    """-> (config dict, generator state dict, text encoder (tokenizer, ModernBERT module on CPU, fp32))."""
    from safetensors.torch import load_file
    meta = read_metadata(path)
    cfg = json.loads(meta["agate.config"])
    sd = load_file(str(path))
    gen = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    txt = {k[len("text_encoder."):]: v for k, v in sd.items() if k.startswith("text_encoder.")}
    if not gen or not txt:
        raise ValueError(f"{path}: expected generator.* and text_encoder.* tensors")
    tok, model = _build_text_model(meta, txt)
    return cfg, gen, tok, model


def _build_text_model(meta: dict, state: dict):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    with tempfile.TemporaryDirectory(prefix="agate_tok_") as d:
        d = Path(d)
        (d / "config.json").write_text(meta["agate.text_encoder.config"], encoding="utf-8")
        for f in TOKENIZER_FILES:
            (d / f).write_text(meta[f"agate.tokenizer.{f}"], encoding="utf-8")
        tok = AutoTokenizer.from_pretrained(str(d))
        config = AutoConfig.from_pretrained(str(d))
    model = AutoModel.from_config(config).float()
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    if missing:
        raise KeyError(f"text encoder tensors missing from the checkpoint: {missing[:5]}")
    if unexpected:
        raise KeyError(f"unexpected text encoder tensors in the checkpoint: {unexpected[:5]}")
    return tok, model.eval().requires_grad_(False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release", help="the unpacked Logolabs/agate-preview-00X folder (named agate-preview-00X)")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    for p in convert_release(a.release, a.out_dir):
        print(f"{p}  {p.stat().st_size / 2**20:.1f} MiB")
