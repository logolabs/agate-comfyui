# Agate for ComfyUI

ComfyUI nodes for **[Agate](https://huggingface.co/Logolabs/agate-preview-001)**, the 260M-parameter
text-to-image model from [LogoLabs](https://logolabs.org). Agate was trained from scratch in 145 GPU-hours and
scores 0.550 on GenEval with the official scorer. It renders 256 × 256 images in about a second on a
consumer GPU.

<p align="center"><img src="docs/example.png" alt="Agate output: an orange flat-design fox head logo on a white background" width="256"></p>
<p align="center"><sub><i>"a minimalist logo of a fox head, orange, flat design, white background"</i>, seed 0, 50 steps, cfg 3</sub></p>

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/logolabs/comfyui-agate
pip install -r comfyui-agate/requirements.txt     # use ComfyUI's Python (the portable build: python_embeded\python.exe -m pip ...)
```

Restart ComfyUI. The nodes appear under **LogoLabs → Agate**. Drag
[`example_workflows/agate_basic.json`](example_workflows/agate_basic.json) onto the canvas to get
Loader → Generate → Save Image.

On first use the loader downloads the model from Hugging Face (about 1 GB, plus 1 GB for the optional
guide model) into the normal Hugging Face cache, and the SD-VAE or TAESD decoder from their own repos.

> **Private preview.** `Logolabs/agate-preview-001` is not public yet. Until it is, you need access to the
> repo and a login in the environment ComfyUI runs in: `huggingface-cli login` (or set `HF_TOKEN`).
> You can also put the path of a downloaded model folder in the loader's `source` field.

Requirements: ComfyUI's own PyTorch (CUDA, CPU or other devices), plus `diffusers>=0.30` and
`transformers>=4.48` (ModernBERT, which the Ettin text encoder uses; older ComfyUI builds ship an
older transformers). Nothing in `requirements.txt` pins or replaces torch.

## Nodes

### Agate Loader → `AGATE_MODEL`

| Input | Default | |
|---|---|---|
| `source` | `Logolabs/agate-preview-001` | A Hugging Face repo id, or a local folder with `config.json`, `generator.safetensors` and `text_encoder/` |
| `decoder` | `sd-vae` | `sd-vae` (SD-VAE ft-MSE, best quality) or `taesd` (tiny decoder: faster and lighter, a little softer) |
| `device` | `auto` | `auto` uses ComfyUI's device; or force `cuda` / `cpu` |
| `cuda_graphs` | on | Records each denoising step as a CUDA graph and replays it. Agate is small enough that kernel launches, not arithmetic, dominate an eager step. Ignored off CUDA |
| `load_guide` | off | Preloads the guide model that `autoguide` uses (about +1 GB). Otherwise it loads the first time a Generate node asks for it |

The loaded pipeline is cached for the whole ComfyUI session. Re-running a workflow, or another
workflow with the same loader settings, reuses it. Changing a setting replaces it, so only one Agate
is in memory at a time. ComfyUI's **Unload Models** button frees it too, and the next generation
reloads it.

### Agate Generate → `IMAGE`

| Input | Default | |
|---|---|---|
| `agate` | | From Agate Loader |
| `prompt` | | English prompt; up to 512 tokens are read |
| `negative_prompt` | empty | The unconditional side of classifier-free guidance. Agate was trained with the empty prompt there, so leave it empty unless you want to push away from something |
| `seed` | 0 | With the usual control_after_generate (fixed / increment / randomize) |
| `steps` | 50 | 1–100 Euler steps. 20 is a usable draft, 8 a rough preview |
| `cfg` | 3.0 | Classifier-free guidance scale |
| `autoguide` | 0.0 | 0–3. Also steers away from Agate's own early (step 27,600) checkpoint, which sharpens faces and fine detail. Try 1.0 with cfg 4. Costs about 50% more time |
| `batch_size` | 1 | 1–16 images from one seed |

The output is a standard ComfyUI image batch (B × 256 × 256 × 3, floats in 0–1), so it connects to
Save Image, Preview Image, or any image node. The progress bar tracks the denoising steps, and Cancel
stops the run at the next step.

**Larger output.** Agate works natively at 256 × 256 only; it was never trained above that. For bigger
images, add an upscaler after Generate: *Load Upscale Model* + *Upscale Image (using Model)* with a
4× ESRGAN-type model gives 1024 × 1024, and logos also trace cleanly to SVG with
[Inkvec](https://github.com/logolabs/inkvec-comfyui).

## Speed

Measured in ComfyUI on an RTX 4060 (8 GB), SD-VAE decoder, CUDA graphs on, 50 steps:

| | Time |
|---|---|
| One image | 1.1 s to sample and decode; 1.4 s per queued prompt including the PNG save |
| Batch of 4 | 5.0 s (1.25 s per image) |
| One image with `autoguide` 1.0 | 2.0 s |
| First run after loading | 20–75 s: weights load, cuDNN picks its kernels, and the CUDA graph is recorded |

The model card quotes 1.9 s per image (2.9 s with autoguide) for the plain pipeline. The slow first run
happens once per session, and again briefly for each new batch size or prompt-length bucket
(64 / 128 / 256 / 512 tokens), because each input shape gets its own CUDA graph. If you change
`batch_size` a lot, turn `cuda_graphs` off: every step then pays the kernel-launch overhead again,
but no shape is ever slow. Agate needs about 1–2 GB of VRAM and also runs on the CPU, slowly.

Note: on CUDA, loading Agate sets `torch.backends.cudnn.benchmark = True` and disables PyTorch's cuDNN
attention backend (`enable_cudnn_sdp(False)`) for the ComfyUI process, because those are the kernels Agate
was trained with. Other models keep working, but they then run with those settings too.

## What Agate is bad at

Exact text, counting above three, negation ("a bowl with
no fruit" comes back full of fruit), and anything above 256 px. It is a research preview and not
yet converged. The [model card](https://huggingface.co/Logolabs/agate-preview-001) has the full
evaluation.

## How it works

The pack vendors Agate's MIT-licensed inference code (`agate_comfy/agate/`, from the
`agate-preview-001` release). The generator is a 191M thinker-steered convolutional flow model,
the text encoder is a fine-tuned Ettin-68M (ModernBERT), and the image comes from the SD-VAE decoder.
The only changes to the vendored code are a per-step callback and a tensor output mode in
`AgatePipeline.__call__`, which the nodes use for the progress bar and ComfyUI's image format. See
`agate_comfy/__init__.py`.

## Tests

```bash
AGATE_SOURCE=/path/to/agate-preview-001 python -m pytest -q tests
```

This runs Loader and Generate end to end with `comfy.*` stubbed out, covering output shape and range,
seed determinism, batching, the progress bar, and unload/reload. Set `AGATE_EXAMPLE=docs/example.png`
to re-render the example image.

## License

MIT; see [LICENSE](LICENSE). The Agate weights are MIT-licensed as well; see the model card.

## Acknowledgement

We acknowledge EuroHPC JU for awarding the project ID EHPC-AIF-2026PG01-907 access to resources on
Arrhenius GPU at NAISS, Sweden.
