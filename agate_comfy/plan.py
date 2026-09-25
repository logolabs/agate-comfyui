"""Views of Agate's plan: what the Agate Plan Viewer node shows.

Agate's thinker writes a plan, m (640 channels on a 16 x 16 grid for a 256 px image), and the
renderer paints the image from m alone. For a run of S steps the node records m_k for every step
(conditional branch) and turns the stack (S, C, H, W) into pictures:

  * PCA: the C channels projected to RGB, fitted once over all steps and cells, so a colour
    means the same thing at every step (2-98 percentile scaling per component);
  * regions: per step the mean over cells is removed (a shared, timestep-driven offset) and each
    cell vector is L2-normalised, then ONE k-means is fitted over all steps pooled and every step
    is assigned to those clusters, so region colours persist over time;
  * change: 1 - cosine similarity between consecutive plans per cell (heat), and its mean per
    step (the change curve).

The analysis runs in torch on the plans' device (the GPU in ComfyUI: a few ms); the pictures use
numpy and Pillow. The k-means is a small k-means++ / Lloyd implementation (scikit-learn's
algorithm and tolerance), so there is no scikit-learn dependency.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Region colours: red, blue, green, amber, pink, violet, dark green, orange, grey, black.
PALETTE = np.array([[244, 41, 31], [42, 120, 214], [27, 175, 122], [237, 161, 0], [232, 123, 164], [74, 58, 167],
                    [0, 131, 0], [235, 104, 52], [120, 120, 120], [11, 10, 8]], np.uint8)
MAX_FIT_CELLS = 6000


def _t(x) -> torch.Tensor:
    return (torch.from_numpy(np.asarray(x)) if not torch.is_tensor(x) else x).float()


# -- analysis ------------------------------------------------------------------------------------

def pca_rgb(plans) -> np.ndarray:
    """(S, C, H, W) -> (S, H, W, 3) uint8: the top three principal components as RGB."""
    P = _t(plans)
    S, C, H, W = P.shape
    X = P.permute(0, 2, 3, 1).reshape(-1, C).double()
    X = X - X.mean(0, keepdim=True)
    _, vecs = torch.linalg.eigh(X.T @ X)                      # covariance eigenvectors, ascending
    vt = vecs.flip(1)[:, :3].T
    # Eigenvector signs are arbitrary: fix them so the colours do not flip between runs or devices.
    pivot = vt.gather(1, vt.abs().argmax(1, keepdim=True))
    vt = vt * torch.where(pivot < 0, -1.0, 1.0).to(vt)
    Y = (X @ vt.T).float()
    if Y.shape[1] < 3:
        Y = torch.nn.functional.pad(Y, (0, 3 - Y.shape[1]))
    lo, hi = torch.quantile(Y, 0.02, dim=0), torch.quantile(Y, 0.98, dim=0)
    rgb = ((Y - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return (rgb * 255).round().to(torch.uint8).reshape(S, H, W, 3).cpu().numpy()


def _sqdist(X: torch.Tensor, x2: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return (x2[:, None] - 2 * X @ c.T + (c * c).sum(1)[None]).clamp_min(0)


def kmeans(X, k: int, n_init: int = 4, seed: int = 0, iters: int = 300, tol: float = 1e-4) -> torch.Tensor:
    """k-means++ seeding + Lloyd iterations, best of n_init by inertia (scikit-learn's KMeans
    algorithm and tol). -> centers (k, D) on X's device, ordered by cluster size (largest first).
    Deterministic for a given seed."""
    X = _t(X)
    n, dev = len(X), X.device
    k = int(max(1, min(k, n)))
    x2 = (X * X).sum(1)
    stop = tol * float(X.var(0, unbiased=False).mean())
    gen = torch.Generator().manual_seed(int(seed))              # CPU draws: identical on every device
    best, best_inertia = None, float("inf")
    for _ in range(n_init):
        idx = [int(torch.randint(n, (1,), generator=gen))]
        d2 = _sqdist(X, x2, X[idx[-1]][None])[:, 0]
        for _ in range(1, k):
            p = d2.double().cpu()
            nxt = int(torch.multinomial(p, 1, generator=gen)) if p.sum() > 0 else int(torch.randint(n, (1,), generator=gen))
            idx.append(nxt)
            d2 = torch.minimum(d2, _sqdist(X, x2, X[nxt][None])[:, 0])
        c = X[idx].clone()
        for _ in range(iters):
            d = _sqdist(X, x2, c)
            dmin, lab = d.min(1)
            counts = torch.bincount(lab, minlength=k)
            onehot = (lab[None, :] == torch.arange(k, device=dev)[:, None]).float()
            new = (onehot @ X) / counts.clamp_min(1)[:, None].float()
            for j in torch.nonzero(counts == 0).flatten().tolist():   # empty: take the worst-fit point
                far = int(dmin.argmax())
                new[j], dmin[far] = X[far], 0.0
            shift = float(((new - c) ** 2).sum())
            c = new
            if shift <= stop:
                break
        inertia = float(_sqdist(X, x2, c).min(1).values.double().sum())
        if inertia < best_inertia - 1e-9:
            best, best_inertia = c, inertia
    lab = _sqdist(X, x2, best).argmin(1)
    order = torch.argsort(-torch.bincount(lab, minlength=k).cpu(), stable=True).to(dev)
    return best[order]


def region_labels(plans, k: int = 6, max_fit: int = MAX_FIT_CELLS, seed: int = 0) -> np.ndarray:
    """(S, C, H, W) -> (S, H, W) int region index, one clustering shared by all steps."""
    P = _t(plans)
    S, C, H, W = P.shape
    Z = P.permute(0, 2, 3, 1).reshape(S, H * W, C)
    Z = Z - Z.mean(1, keepdim=True)
    Z = Z / (Z.norm(dim=2, keepdim=True) + 1e-8)
    Z = Z.reshape(-1, C)
    gen = torch.Generator().manual_seed(int(seed))
    pick = torch.randperm(len(Z), generator=gen)[:min(len(Z), max_fit)].sort().values.to(Z.device)
    centers = kmeans(Z[pick], k, n_init=4, seed=seed)
    return _sqdist(Z, (Z * Z).sum(1), centers).argmin(1).reshape(S, H, W).cpu().numpy()


def plan_change(plans) -> np.ndarray:
    """(S, C, H, W) -> (S, H, W): 1 - cos(m_k, m_{k-1}) per cell; 0 at the first step."""
    P = _t(plans)
    S, C, H, W = P.shape
    a = P.reshape(S, C, -1).double()
    an = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    cos = (an[1:] * an[:-1]).sum(1)
    return torch.cat([torch.zeros(1, H * W, dtype=cos.dtype, device=cos.device), 1 - cos]).reshape(S, H, W).cpu().numpy()


def analyse(plans, k: int = 6) -> dict:
    """plans (S, C, H, W), numpy or torch on any device -> numpy views."""
    change = plan_change(plans)
    return {"pca": pca_rgb(plans), "labels": region_labels(plans, k), "change": change,
            "curve": change.reshape(len(change), -1).mean(1)}


# -- pictures ------------------------------------------------------------------------------------

def regions_rgb(labels: np.ndarray) -> np.ndarray:
    return PALETTE[labels % len(PALETTE)]


def heat_rgb(change: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Change -> dark blue (none) .. red/orange (most). vmax: the 99th percentile over steps >= 1."""
    if vmax is None:
        vmax = float(np.percentile(change[1:], 99)) if len(change) > 1 else 1.0
    x = np.clip(change / (vmax + 1e-8), 0, 1)
    return np.stack([255 * x, 255 * (x ** 2) * 0.35, 30 + 60 * (1 - x)], -1).round().astype(np.uint8)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)          # Pillow >= 10.1: scalable
    except TypeError:
        return ImageFont.load_default()


def _text(d: ImageDraw.ImageDraw, xy, text: str, fill, f, anchor: str = "la") -> None:
    try:
        d.text(xy, text, fill=fill, font=f, anchor=anchor)
    except ValueError:                                   # bitmap fonts (old Pillow) take no anchor
        d.text(xy, text, fill=fill, font=f)


def label_strip(width: int, height: int, cells: list[tuple[int, str]], bg=(11, 10, 8), fg=(245, 243, 238)):
    """A dark strip with text at the given x offsets."""
    im = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(im)
    f = font(max(9, int(height * 0.6)))
    for x, text in cells:
        _text(d, (x + 6, height / 2), text, fg, f, "lm")
    return np.asarray(im)


def curve_image(curve: np.ndarray, width: int = 640, height: int = 260, t=None) -> np.ndarray:
    """Mean plan change vs step, drawn with PIL. -> (height, width, 3) uint8."""
    im = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f, ft = font(12), font(14)
    left, right, top, bottom = 62, 16, 34, 40
    pw, ph = width - left - right, height - top - bottom
    S = len(curve)
    y = np.asarray(curve[1:] if S > 1 else curve, np.float64)
    xs = np.arange(2, S + 1) if S > 1 else np.arange(1, S + 1)
    ymax = float(y.max()) if len(y) and y.max() > 0 else 1.0
    ymax *= 1.05
    x0, x1 = (xs[0], xs[-1]) if len(xs) > 1 else (xs[0] - 1, xs[0] + 1)

    def px(xv, yv):
        return (left + (xv - x0) / (x1 - x0) * pw, top + ph - yv / ymax * ph)

    _text(d, (left, 10), "mean plan change vs step  (1 - cos between consecutive plans)", (11, 10, 8), ft)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):                       # horizontal grid + y ticks
        yy = top + ph - frac * ph
        d.line([(left, yy), (left + pw, yy)], fill=(230, 228, 222))
        _text(d, (left - 6, yy), f"{frac * ymax:.2g}", (90, 90, 90), f, "rm")
    ticks = sorted({int(round(v)) for v in np.linspace(x0, x1, 6)})
    for tx in ticks:
        xx = px(tx, 0)[0]
        d.line([(xx, top + ph), (xx, top + ph + 4)], fill=(11, 10, 8))
        _text(d, (xx, top + ph + 6), str(tx), (90, 90, 90), f, "mt")
    d.line([(left, top), (left, top + ph), (left + pw, top + ph)], fill=(11, 10, 8))
    _text(d, (left + pw / 2, height - 6), "step", (11, 10, 8), f, "md")
    pts = [px(a, b) for a, b in zip(xs, y)]
    if len(pts) > 1:
        d.line(pts, fill=(244, 41, 31), width=2, joint="curve")
    for p in pts if len(pts) <= 60 else []:
        d.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=(244, 41, 31))
    return np.asarray(im)


def upscale_nearest(a: np.ndarray, size: int) -> np.ndarray:
    """(S, h, w, 3) uint8 -> (S, size, size, 3), nearest neighbour (keeps the grid visible)."""
    return np.stack([np.asarray(Image.fromarray(x).resize((size, size), Image.NEAREST)) for x in a])


def resize_smooth(a: np.ndarray, size: int) -> np.ndarray:
    if a.shape[1] == size and a.shape[2] == size:
        return a
    return np.stack([np.asarray(Image.fromarray(x).resize((size, size), Image.LANCZOS)) for x in a])


def panels(plan: np.ndarray, regions: np.ndarray, change: np.ndarray, pred: np.ndarray, curve: np.ndarray,
           t: np.ndarray, k: int, gap: int = 4) -> np.ndarray:
    """Per step [plan | regions | change | prediction] with a label strip. Inputs (S, F, F, 3) uint8."""
    S, F = plan.shape[0], plan.shape[1]
    width = 4 * F + 3 * gap
    strip_h = max(16, F // 11)
    out = np.empty((S, strip_h + F, width, 3), np.uint8)
    xs = [i * (F + gap) for i in range(4)]
    for s in range(S):
        out[s] = 11                                                   # the gaps: near-black
        out[s, :strip_h] = label_strip(width, strip_h, [
            (xs[0], f"plan  step {s + 1}/{S}"), (xs[1], f"regions (k={k})"),
            (xs[2], f"change {curve[s]:.3f}" if s else "change"), (xs[3], f"prediction  t={t[s]:.2f}")])
        for x, tile in zip(xs, (plan[s], regions[s], change[s], pred[s])):
            out[s, strip_h:, x:x + F] = tile
    return out
