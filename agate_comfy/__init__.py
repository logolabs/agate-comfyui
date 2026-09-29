"""Vendored copy of the `agate` inference package (MIT, Copyright (c) 2026 LogoLabs).

Source: the `agate/` folder of the Hugging Face model repo Logolabs/agate-preview-001
(release "agate-preview-001", generator step 99,250), copied on 2026-09-25.

Every file is byte-identical to the release except `agate/pipeline.py`, whose
`AgatePipeline.__call__` gained two keyword arguments for ComfyUI:
  * callback(step, steps)  -- called after every denoising step (progress bar, interrupt);
  * output_type="pt"       -- return a float (B, H, W, 3) tensor in [0, 1] instead of PIL images.
With the defaults (callback=None, output_type="pil") it behaves exactly like the release.

`agate003/` is an unmodified copy of the `agate/` folder of Logolabs/agate-preview-003 (2026-09-29): the
multi-resolution generator (fcdm_thinker2_mr.py), its prompt pipeline (prompt_norm.py) and pipeline pieces
(shift_t, _Eager/_Graphed). Its marking.py is not used by the nodes (agate_comfy/marking.py is the numpy port).

The ComfyUI nodes do not call AgatePipeline itself (the tests use it as the parity reference):
  * runtime.py    -- the release sampler generalised to img2img, ComfyUI memory management, previews;
  * checkpoint.py -- the single-file checkpoints in ComfyUI/models/agate/ (build and load).
"""

SOURCE_REPO = "Logolabs/agate-preview-001"
SOURCE_VERSION = "agate-preview-001"
