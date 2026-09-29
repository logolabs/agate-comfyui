"""AI-generated content marking for Agate outputs (EU AI Act Art. 50(2): machine-readable, detectable marking).

Two layers, both ON by default in AgatePipeline:
  * an invisible watermark in the pixels: invisible-watermark (imwatermark, MIT), method 'dwtDctSvd', a fixed 64-bit
    payload b"AGATE" + the release id (e.g. b"AGATE002"). detect_watermark(img) reads it back.
  * provenance metadata: text entries in img.info ("ai_generated", "generator", "model", optionally "prompt" /
    "seed"). PIL only writes them to disk through a PngInfo: use save(img, path), which does that for PNG.

Neither layer is tamper-proof: metadata disappears on most re-encodes and screenshots, and the watermark can be
removed by heavy edits or deliberately. Deployers still have to label AI-generated output themselves (Art. 50).
C2PA content credentials are future work.
"""
from __future__ import annotations

RELEASE = "003"
PAYLOAD = b"AGATE" + RELEASE.encode()                  # 8 bytes = 64 bits, fixed
GENERATOR = f"Agate Preview {RELEASE} (LogoLabs)"
MODEL = f"Logolabs/agate-preview-{RELEASE}"
METHOD = "dwtDctSvd"   # dwtDct (the library default) read back only 36-61% of the bits on Agate images, 2026-09-29


def _lib():
    try:
        from imwatermark import WatermarkDecoder, WatermarkEncoder
    except ImportError as e:                           # the pipeline asks for it only when watermark=True
        raise ImportError("Agate's output watermark needs the invisible-watermark package: "
                          "pip install invisible-watermark  (or pass watermark=False)") from e
    return WatermarkEncoder, WatermarkDecoder


def add_watermark(img, payload: bytes = PAYLOAD):
    """PIL RGB image -> a new PIL image carrying `payload` invisibly (dwtDct)."""
    import numpy as np
    from PIL import Image
    Enc, _ = _lib()
    enc = Enc()
    enc.set_watermark("bytes", payload)
    bgr = np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])
    out = enc.encode(bgr, METHOD)
    return Image.fromarray(np.ascontiguousarray(out[:, :, ::-1]))


def read_watermark(img, n_bytes: int = len(PAYLOAD), size: int | None = None) -> bytes:
    """The payload bytes read from img, optionally after resizing it to size x size (the library needs at least
    256 x 256; the mark is read best at the size it was embedded at)."""
    import numpy as np
    from PIL import Image
    img = img.convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    _, Dec = _lib()
    dec = Dec("bytes", 8 * n_bytes)
    return bytes(dec.decode(np.ascontiguousarray(np.asarray(img)[:, :, ::-1]), METHOD))


def detect_watermark(img, payload: bytes = PAYLOAD, min_bit_accuracy: float = 0.9) -> dict:
    """-> {"detected": bool, "bit_accuracy": float, "payload": bytes read, "release": "001"/"002"/... or None}.
    "detected" means "an Agate output": the release payloads AGATE001/002/003 differ in only 1-2 of the 64 bits, so an
    image of another release also scores >= 0.98 here. "release" is set only when the read payload is exactly
    AGATE + three digits, which is what tells the releases apart. Detected when at least
    `min_bit_accuracy` of the 64 payload bits match (a random image matches ~50%; up to three reads are tried, so a little more).
    Images are read as given and resized to 256 and 512 px squares, which recovers marks after a resize."""
    sizes = ([None] if min(img.size) >= 256 else []) + [s for s in (256, 512) if img.size != (s, s)]
    best = None
    for size in sizes:                                  # as given, then at Agate's two output sizes
        got = read_watermark(img, len(payload), size)
        acc = sum(8 - bin(a ^ b).count("1") for a, b in zip(got, payload)) / (8 * len(payload))
        if best is None or acc > best[0]:
            best = (acc, got, size)
    got = best[1]
    exact = got[:5] == b"AGATE" and got[5:].isdigit() and len(got) == 8
    return {"detected": best[0] >= min_bit_accuracy, "bit_accuracy": best[0], "payload": got, "read_at": best[2],
            "release": got[5:].decode() if exact else None}


def provenance(prompt: str | None = None, seed: int | None = None, extra: dict | None = None) -> dict:
    info = {"ai_generated": "true", "generator": GENERATOR, "model": MODEL,
            "watermark": f"invisible-watermark {METHOD}, payload {PAYLOAD.decode()}"}
    if prompt is not None:
        info["prompt"] = prompt
    if seed is not None:
        info["seed"] = str(seed)
    info.update(extra or {})
    return info


def save(img, path, **kw) -> None:
    """Save keeping the provenance entries: PNG text chunks via PngInfo (other formats: PIL's own handling, which
    for JPEG drops them -- the pixel watermark is then the remaining mark)."""
    from pathlib import Path
    from PIL.PngImagePlugin import PngInfo
    if Path(path).suffix.lower() == ".png":
        meta = PngInfo()
        for k, v in img.info.items():
            if isinstance(v, str):
                meta.add_text(k, v)
        img.save(path, pnginfo=meta, **kw)
    else:
        img.save(path, **kw)
