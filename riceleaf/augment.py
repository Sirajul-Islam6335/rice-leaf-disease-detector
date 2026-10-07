"""On-the-fly data augmentation written in NumPy.

Each epoch sees a freshly perturbed copy of every training image, which acts as
a regulariser: the network must learn features that are invariant to the
perturbations (position, scale, orientation, lighting) rather than memorising
pixels.
"""
from __future__ import annotations

import numpy as np

from .config import AugmentConfig

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def bilinear_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of an (H, W, C) image (half-pixel-centre convention)."""
    h, w = img.shape[:2]
    if (h, w) == (out_h, out_w):
        return img
    ys = (np.arange(out_h) + 0.5) * h / out_h - 0.5
    xs = (np.arange(out_w) + 0.5) * w / out_w - 0.5
    ys, xs = np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(img.dtype)


class Augmenter:
    def __init__(self, cfg: AugmentConfig, rng: np.random.Generator) -> None:
        self.cfg, self.rng = cfg, rng

    def __call__(self, img: np.ndarray) -> np.ndarray:
        c, r = self.cfg, self.rng
        h, w = img.shape[:2]
        # 1. random crop (independent scale per axis) then resize back
        if c.min_crop_scale < 1.0:
            ch = int(round(h * r.uniform(c.min_crop_scale, 1.0)))
            cw = int(round(w * r.uniform(c.min_crop_scale, 1.0)))
            y, x = r.integers(0, h - ch + 1), r.integers(0, w - cw + 1)
            img = bilinear_resize(img[y:y + ch, x:x + cw], h, w)
        # 2. flips (a leaf has no canonical up/down or left/right)
        if r.random() < c.hflip:
            img = img[:, ::-1]
        if r.random() < c.vflip:
            img = img[::-1]
        # 3. photometric jitter
        if c.brightness:
            img = img * r.uniform(1 - c.brightness, 1 + c.brightness)
        if c.contrast:
            mean = img.mean()
            img = (img - mean) * r.uniform(1 - c.contrast, 1 + c.contrast) + mean
        if c.saturation:
            gray = (img @ _LUMA)[..., None]
            img = (img - gray) * r.uniform(1 - c.saturation, 1 + c.saturation) + gray
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def batch(self, images: np.ndarray) -> np.ndarray:
        return np.stack([self(im) for im in images])
