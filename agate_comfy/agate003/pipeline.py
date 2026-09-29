"""AgatePipeline (Preview 003): prompt -> 512x512 image (or 256x256).

    from agate import AgatePipeline
    pipe = AgatePipeline.from_pretrained("Logolabs/agate-preview-003", device="cuda")
    images = pipe("a red cube on top of a blue sphere", seed=0)                   # 512 x 512
    images = pipe('a shop sign that says "OPEN"', seed=0, resolution=256)        # 256 x 256

Preview 003 is the multi-resolution model (arch fcdm_t2mr). It was trained with a prompt pipeline, and
sampling uses the same one (prompt_norm.py):
  * normaliser: whitespace, SHOUTED or Title Cased prompts lower-cased outside quotes, number words and
    small digits made canonical ("THREE", "3" -> "three");
  * negatives: "without X" / "no X" are cut from the prompt and become the negative prompt (CFG pushes
    away from it; the model itself barely reacts to negation);
  * spelling: text inside double quotes is also spelled out letter by letter after " || spell: ", so the
    text encoder sees one token per letter;
  * count code: the value of every object count ("three cats") is passed to the model at those tokens.
normalize=False / spell=False switch the steps off (not how the model was trained).

Sampler: Euler from t=0 (noise) to t=1 (data), 50 steps, classifier-free guidance 3 against the empty
prompt (or the negatives), velocity prediction. 512 px uses the SD3 resolution shift 2 on the time grid,
as trained; 256 px uses no shift. On CUDA each denoising step is recorded once as a CUDA graph and
replayed; on CPU it runs eagerly.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from . import prompt_norm as PN
from .fcdm_thinker2_mr import FCDMThinker2MR
from .text_encoder import EttinTextEncoder
from .marking import add_watermark, provenance

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


def shift_t(t: float, shift: float) -> float:
    """SD3's resolution shift in Agate's convention (t = 0 noise): with the noise level s = 1 - t,
    s' = shift * s / (1 + (shift - 1) * s). shift = 1 is the identity."""
    if shift == 1.0:
        return t
    s = 1 - t
    return 1 - shift * s / (1 + (shift - 1) * s)


class _Graphed:
    """model(z, t, ctx, mask, counts) recorded as a CUDA graph per input shape and replayed."""

    def __init__(self, model):
        self.model, self.cache = model, {}

    def _run(self, z, t, ctx, mask, counts):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(z, t, ctx, mask, counts=counts)

    def __call__(self, z, t, ctx, mask, counts):
        key = (tuple(z.shape), tuple(ctx.shape))
        if key not in self.cache:
            static = [z.clone(), t.clone(), ctx.clone(), mask.clone(), counts.clone()]
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
        for dst, src in zip(static, (z, t, ctx, mask, counts)):
            dst.copy_(src)
        graph.replay()
        return out.float()


class _Eager:
    def __init__(self, model):
        self.model = model

    def __call__(self, z, t, ctx, mask, counts):
        with torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=z.device.type == "cuda"):
            return self.model(z, t, ctx, mask, counts=counts).float()


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
        vdtype = torch.float16 if on_cuda else torch.float32
        if fast_vae:
            self.vae = AutoencoderTiny.from_pretrained(self.cfg["fast_vae"], torch_dtype=vdtype)
            self.vae_div = 1.0                            # TAESD decodes the scaled latents directly
        else:
            self.vae = AutoencoderKL.from_pretrained(self.cfg["vae"], torch_dtype=vdtype)
            self.vae_div = self.cfg["vae_scale"]
        self.vae = self.vae.to(self.device).eval()

    @classmethod
    def from_pretrained(cls, repo_or_dir: str = "Logolabs/agate-preview-003", device: str = "cuda", **kw):
        return cls(_resolve(repo_or_dir), device=device, **kw)

    def _load_generator(self, path: Path):
        from safetensors.torch import load_file
        m = FCDMThinker2MR(**self.cfg["model_kw"])
        m.load_state_dict(load_file(str(path)))
        m = m.to(self.device, dtype=self._dtype).eval()
        if self.device.type == "cuda":
            m = m.to(memory_format=torch.channels_last)
        return _Graphed(m) if self._graphs else _Eager(m)

    def _load_text(self, path: Path, max_len: int) -> EttinTextEncoder:
        text = EttinTextEncoder(str(path), self.device, max_len=max_len)
        text.model.to(dtype=self._dtype)
        return text

    def prepare(self, prompt: str, negative_prompt: str = "", normalize: bool = True, spell: bool = True):
        """The prompt pipeline alone: -> (prompt as the text encoder sees it, negative prompt)."""
        negs = []
        if normalize:
            prompt, negs = PN.split_negatives(PN.normalize(prompt))
        if spell:
            prompt = PN.add_spelling(prompt)
        negative = ", ".join(n for n in [negative_prompt.strip(), *negs] if n)
        return prompt, negative

    @torch.no_grad()
    def __call__(self, prompt: str, negative_prompt: str = "", seed: int = 0, steps: int = 50, cfg: float = 3.0,
                 num_images: int = 1, resolution: int | None = None, normalize: bool = True, spell: bool = True,
                 watermark: bool = True, metadata: bool = True, record_prompt: bool = False):
        """resolution: 512 (default) or 256. negative_prompt is combined with the negatives the normaliser
        cut from the prompt ("without X", "no X"); when both are empty the unconditional prompt is "".
        watermark (default True): an invisible dwtDct watermark (marking.PAYLOAD) in every image; metadata (default
        True): provenance entries in img.info ("ai_generated", "generator", "model"); record_prompt (default False)
        also stores the prompt and seed there. Save with agate.save(img, path) to keep the entries in a PNG."""
        n, dev = int(num_images), self.device
        res = self.cfg["resolutions"][str(int(resolution or self.cfg["resolution"]))]
        hw, shift = res["latent_hw"], float(res["shift"])
        text, negative = self.prepare(prompt, negative_prompt, normalize, spell)
        ids, am = self.text.tokenize([text] * n)
        ctx, mask = self.text.encode(ids, am)
        counts = PN.count_tensor(self.text, [text] * n, ids.shape[1]) if normalize else torch.zeros(n, ids.shape[1])
        u_ctx, u_mask = self.text([negative] * n)
        L = _bucket(ctx.shape[1], u_ctx.shape[1])
        ctx, mask = _pad_to(ctx, mask, L)
        u_ctx, u_mask = _pad_to(u_ctx, u_mask, L)
        counts = F.pad(counts.to(dev).float(), (0, L - counts.shape[1]))
        both_ctx, both_mask = torch.cat([ctx, u_ctx]), torch.cat([mask, u_mask])
        both_counts = torch.cat([counts, torch.zeros_like(counts)])        # the unconditional half gets none
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        z = torch.randn(n, 4, hw, hw, device=dev, generator=gen)
        grid = [shift_t(i / steps, shift) for i in range(steps + 1)]
        for i in range(steps):
            t = torch.full((n,), grid[i], device=dev)
            vc, vu = self.model(torch.cat([z, z]), torch.cat([t, t]), both_ctx, both_mask, both_counts).chunk(2)
            z = z + (grid[i + 1] - grid[i]) * (vu + cfg * (vc - vu))
        x = self.vae.decode((z / self.vae_div).to(self.vae.dtype)).sample
        x = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        from PIL import Image
        imgs = [Image.fromarray(a) for a in x]
        if watermark:
            imgs = [add_watermark(im) for im in imgs]
        if metadata:
            for k, im in enumerate(imgs):
                im.info.update(provenance(prompt if record_prompt else None, int(seed) if record_prompt else None,
                                          {"image_index": str(k)} if n > 1 else None))
        return imgs
