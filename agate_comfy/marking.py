"""AI-generated content marking for the images the Agate nodes produce (EU AI Act Art. 50(2)).

The same two marks as the Python packages (agate/marking.py of Logolabs/agate-preview-002 / -003) and the
Agate WebGPU Space, so one detector reads all of them:

  * an invisible watermark: invisible-watermark's (MIT, github.com/ShieldMnt/invisible-watermark) 'dwtDctSvd'
    method with the fixed 64-bit payload b"AGATE" + release (b"AGATE001" / b"AGATE002" / b"AGATE003"). It is
    re-implemented below in plain numpy, bit-identical to invisible-watermark 0.2.0 (tests/test_marking.py), so
    the pack needs neither OpenCV nor PyWavelets. If `imwatermark` is installed it is not used either: the output
    is the same.
  * provenance text entries: "ai_generated" = "true", "generator" = "Agate Preview 00X (LogoLabs)",
    "model" = "Logolabs/agate-preview-00X", "watermark" = "invisible-watermark dwtDctSvd, payload AGATE00X".
    The prompt is not included.

detect_watermark() follows the packages' rule: "detected" (>= 58 of 64 bits against the payload) means "an
Agate output" -- the release payloads differ in only 1-2 bits -- and "release" is set only when the payload read
is exactly AGATE + three digits. Neither mark is tamper-proof.
"""
from __future__ import annotations

import re

import numpy as np

METHOD = "dwtDctSvd"
SCALE = 36.0
BLOCK = 4
UNKNOWN_RELEASE = "000"          # a checkpoint that is not an official release (e.g. a fine-tune)


def release_id(cfg: dict | None, name: str = "") -> str:
    """'003' from a checkpoint config ("release": "agate-preview-003") or a file/folder/repo name."""
    for s in ((cfg or {}).get("release", ""), name):
        m = re.search(r"agate-preview-(\d{3})", str(s))
        if m:
            return m.group(1)
    return UNKNOWN_RELEASE


def payload(release: str) -> bytes:
    return b"AGATE" + release.encode()


def provenance(release: str) -> dict:
    p = payload(release).decode()
    if release == UNKNOWN_RELEASE:
        gen, model = "Agate-based model (LogoLabs Agate architecture)", "unknown"
    else:
        gen, model = f"Agate Preview {release} (LogoLabs)", f"Logolabs/agate-preview-{release}"
    return {"ai_generated": "true", "generator": gen, "model": model, "watermark": f"invisible-watermark {METHOD}, payload {p}"}


# ---- dwtDctSvd, numpy -------------------------------------------------------------------------
def _desc(x):
    return (x + (1 << 13)) >> 14


def bgr2yuv(bgr: np.ndarray) -> np.ndarray:
    b, g, r = (bgr[..., i].astype(np.int64) for i in range(3))
    y = _desc(b * 1868 + g * 9617 + r * 4899)
    u = np.clip(_desc((b - y) * 8061 + (128 << 14)), 0, 255)
    v = np.clip(_desc((r - y) * 14369 + (128 << 14)), 0, 255)
    return np.stack([y, u, v], -1).astype(np.uint8)


def yuv2bgr(yuv: np.ndarray) -> np.ndarray:
    y, u, v = (yuv[..., i].astype(np.int64) - (0, 128, 128)[i] for i in range(3))
    b = np.clip(y + _desc(u * 33292), 0, 255)
    g = np.clip(y + _desc(u * -6472 + v * -9519), 0, 255)
    r = np.clip(y + _desc(v * 18678), 0, 255)
    return np.stack([b, g, r], -1).astype(np.uint8)


_n = np.arange(BLOCK)
_C = np.sqrt(np.where(_n[:, None] == 0, 1.0 / BLOCK, 2.0 / BLOCK)) * np.cos(np.pi * (2 * _n[None, :] + 1) * _n[:, None] / (2 * BLOCK))


def _blocks(ll):
    """(h, w) -> (h//4 * w//4, 4, 4) row-major blocks (the library's block order), and the cropped shape."""
    h, w = (ll.shape[0] // BLOCK) * BLOCK, (ll.shape[1] // BLOCK) * BLOCK
    x = ll[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).transpose(0, 2, 1, 3).reshape(-1, BLOCK, BLOCK)
    return x, h, w


_S = 0.7071067811865476            # PyWavelets' Haar filter tap, 1/sqrt(2)


def _haar(u):
    """pywt.dwt2(u, 'haar') with PyWavelets' exact float order: pairs along axis 0 first, then axis 1.
    -> (cA, cH, cV, cD)."""
    lo, hi = u[0::2] * _S + u[1::2] * _S, u[0::2] * _S - u[1::2] * _S
    ca, cv = lo[:, 0::2] * _S + lo[:, 1::2] * _S, lo[:, 0::2] * _S - lo[:, 1::2] * _S
    ch, cd = hi[:, 0::2] * _S + hi[:, 1::2] * _S, hi[:, 0::2] * _S - hi[:, 1::2] * _S
    return ca, ch, cv, cd


def _ihaar_swapped(ca, ch, cv, cd):
    """pywt.idwt2((ca, (cv, ch, cd)), 'haar') -- the library passes the two details swapped -- with PyWavelets'
    float order (axis 1 first, then axis 0). Bit-identical, which matters: the result is truncated to uint8."""
    h_in, v_in = cv, ch                                  # the swap
    def up(lo, hi, axis):
        out = np.empty(tuple(n * 2 if k == axis else n for k, n in enumerate(lo.shape)))
        ev, od = [slice(None)] * 2, [slice(None)] * 2
        ev[axis], od[axis] = slice(0, None, 2), slice(1, None, 2)
        out[tuple(ev)], out[tuple(od)] = lo * _S + hi * _S, lo * _S - hi * _S
        return out
    return up(up(ca, v_in, 1), up(h_in, cd, 1), 0)


def _to_uint8_unsafe(x):
    return (np.trunc(x).astype(np.int64) % 256).astype(np.uint8)


def payload_bits(payload: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(payload, np.uint8))


def embed(bgr: np.ndarray, payload: bytes) -> np.ndarray:
    """uint8 BGR (H, W, 3), H * W >= 256 * 256 -> watermarked uint8 BGR."""
    rows, cols = bgr.shape[:2]
    if rows * cols < 256 * 256:
        raise ValueError("image too small, should be larger than 256x256")
    bits = payload_bits(payload)
    yuv = bgr2yuv(bgr)
    R, Cc = rows // 4 * 4, cols // 4 * 4
    u = yuv[:R, :Cc, 1].astype(np.float64)
    ca, ch, cv, cd = _haar(u)
    blk, h, w = _blocks(ca)
    wm = bits[np.arange(len(blk)) % len(bits)].astype(np.float64)
    dct = _C @ blk @ _C.T
    U, s, Vt = np.linalg.svd(dct)
    s[:, 0] = (s[:, 0] // SCALE + 0.25 + 0.5 * wm) * SCALE
    rec = _C.T @ (U @ (s[:, :, None] * Vt)) @ _C
    ca = ca.copy()
    ca[:h, :w] = rec.reshape(h // BLOCK, w // BLOCK, BLOCK, BLOCK).transpose(0, 2, 1, 3).reshape(h, w)
    yuv[:R, :Cc, 1] = _to_uint8_unsafe(_ihaar_swapped(ca, ch, cv, cd))
    return yuv2bgr(yuv)


def decode_bits(bgr: np.ndarray, n_bits: int = 64) -> np.ndarray:
    rows, cols = bgr.shape[:2]
    if rows * cols < 256 * 256:
        raise ValueError("image too small, should be larger than 256x256")
    yuv = bgr2yuv(bgr)
    u = yuv[:rows // 4 * 4, :cols // 4 * 4, 1].astype(np.float64)
    blk, _, _ = _blocks(_haar(u)[0])
    s0 = np.linalg.svd(_C @ blk @ _C.T, compute_uv=False)[:, 0]
    score = ((s0 % SCALE) > SCALE * 0.5).astype(np.float64)
    idx = np.arange(len(blk)) % n_bits
    avg = np.array([score[idx == k].mean() for k in range(n_bits)])
    return (avg * 255 > 127).astype(np.uint8)


def decode(bgr: np.ndarray, n_bytes: int = 8) -> bytes:
    return np.packbits(decode_bits(bgr, 8 * n_bytes)).tobytes()


# ---- images ------------------------------------------------------------------------------------

def mark_rgb_uint8(rgb: np.ndarray, release: str) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> watermarked copy. Images under 256 x 256 pixels are returned unchanged (the
    method needs at least that many; Agate's outputs are 256 or 512)."""
    h, w = rgb.shape[:2]
    if h * w < 256 * 256:
        return rgb
    return np.ascontiguousarray(embed(np.ascontiguousarray(rgb[:, :, ::-1]), payload(release))[:, :, ::-1])


def mark_image_tensor(images, release: str):
    """ComfyUI IMAGE (B, H, W, 3) float [0, 1] -> watermarked IMAGE (same device/dtype). The pixels are
    quantised to 8 bit first (what any image file stores)."""
    import torch
    x = (images.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    out = np.stack([mark_rgb_uint8(a, release) for a in x])
    return to_image_tensor(out, images.device, images.dtype)


def to_image_tensor(u8: np.ndarray, device="cpu", dtype=None):
    """uint8 (B, H, W, 3) -> IMAGE float in [0, 1] that gives back exactly these bytes however it is quantised:
    (v + 0.25) / 255, capped at 1. ComfyUI's Save Image TRUNCATES 255 * x (np.clip(...).astype(uint8)), and
    v / 255 * 255 lands just below v for many v in float32, which shifts those pixels by -1 and was enough to
    break the mark (0.89 of the bits in a test); + 0.25 survives both truncation and rounding."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(u8)).to(device).float().add_(0.25).div_(255).clamp_(max=1.0)
    return t if dtype is None else t.to(dtype)


def _read(rgb: np.ndarray, n_bytes: int) -> bytes:
    return decode(np.ascontiguousarray(rgb[:, :, ::-1]), n_bytes)


def detect_watermark(img, release: str = "003", min_bit_accuracy: float = 0.9) -> dict:
    """PIL image or (H, W, 3) uint8 RGB -> {"detected", "bit_accuracy", "payload", "release"}; reads as given
    (if >= 256 px on the short side) and resized to 256 and 512 px squares, keeping the best (the packages'
    detect_watermark)."""
    from PIL import Image
    im = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img, np.uint8))
    im = im.convert("RGB")
    want = payload(release)
    sizes = ([None] if min(im.size) >= 256 else []) + [s for s in (256, 512) if im.size != (s, s)]
    best = None
    for s in sizes:
        x = im if s is None else im.resize((s, s), Image.LANCZOS)
        got = _read(np.asarray(x), len(want))
        acc = sum(8 - bin(a ^ b).count("1") for a, b in zip(got, want)) / (8 * len(want))
        if best is None or acc > best[0]:
            best = (acc, got)
    got = best[1]
    exact = got[:5] == b"AGATE" and got[5:].isdigit() and len(got) == 8
    return {"detected": best[0] >= min_bit_accuracy, "bit_accuracy": best[0], "payload": got,
            "release": got[5:].decode() if exact else None}
