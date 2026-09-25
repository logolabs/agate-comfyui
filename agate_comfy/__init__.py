"""Vendored copy of the `agate` inference package (MIT, Copyright (c) 2026 LogoLabs).

Source: the `agate/` folder of the Hugging Face model repo Logolabs/agate-preview-001
(release "agate-preview-001", generator step 99,250), copied on 2026-09-25.

Every file is byte-identical to the release except `agate/pipeline.py`, whose
`AgatePipeline.__call__` gained two keyword arguments for ComfyUI:
  * callback(step, steps)  -- called after every denoising step (progress bar, interrupt);
  * output_type="pt"       -- return a float (B, H, W, 3) tensor in [0, 1] instead of PIL images.
With the defaults (callback=None, output_type="pil") it behaves exactly like the release.
"""

SOURCE_REPO = "Logolabs/agate-preview-001"
SOURCE_VERSION = "agate-preview-001"
