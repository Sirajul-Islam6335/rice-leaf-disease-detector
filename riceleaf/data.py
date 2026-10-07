"""Dataset discovery, image loading, stratified splitting and normalisation."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
try:
    _RESAMPLE = Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9.1
    _RESAMPLE = Image.BICUBIC


@dataclass
class Dataset:
    paths: list[str]
    labels: np.ndarray            # (N,) int64
    class_names: list[str]
    images: np.ndarray | None = None  # (N, H, W, 3) float32 in [0, 1]

    def __len__(self) -> int:
        return len(self.paths)

    def subset(self, idx) -> "Dataset":
        idx = np.asarray(idx)
        return Dataset([self.paths[i] for i in idx], self.labels[idx], self.class_names,
                       None if self.images is None else self.images[idx])


def scan_directory(root: str | Path) -> Dataset:
    """Expect ``root/<class_name>/<image files>``; class indices follow sorted folder names."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if len(class_dirs) < 2:
        raise ValueError(f"Need at least two class sub-folders in {root}")
    paths, labels = [], []
    for idx, d in enumerate(class_dirs):
        files = sorted(f for f in d.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        paths += [str(f) for f in files]
        labels += [idx] * len(files)
    return Dataset(paths, np.array(labels, dtype=np.int64), [d.name for d in class_dirs])


def load_image(source, size: tuple[int, int]) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1] with shape (H, W, 3).

    Portrait images are rotated to landscape because every leaf in the dataset is
    photographed lying horizontally; this keeps the model's input geometry stable.
    """
    img = source if isinstance(source, Image.Image) else Image.open(source)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if img.height > img.width:
        img = img.rotate(90, expand=True)
    h, w = size
    img = img.resize((w, h), _RESAMPLE)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_images(ds: Dataset, size: tuple[int, int]) -> Dataset:
    log.info("Loading %d images at %dx%d ...", len(ds), size[0], size[1])
    ds.images = np.stack([load_image(p, size) for p in ds.paths]).astype(np.float32)
    return ds


def stratified_split(labels: np.ndarray, fractions: list[float], seed: int) -> list[np.ndarray]:
    """Split indices into len(fractions) groups, preserving class proportions.

    The last group receives the remainder, so ``fractions`` need not sum to 1.
    """
    rng = np.random.default_rng(seed)
    groups: list[list[int]] = [[] for _ in fractions]
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        start = 0
        for g, frac in enumerate(fractions[:-1]):
            n = int(round(frac * len(idx)))
            groups[g] += idx[start:start + n].tolist()
            start += n
        groups[-1] += idx[start:].tolist()
    return [np.array(sorted(g), dtype=np.int64) for g in groups]


def stratified_kfold(labels: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Return k disjoint index folds with (near) equal class balance."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        for i, chunk in enumerate(np.array_split(idx, k)):
            folds[i] += chunk.tolist()
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


def channel_stats(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std, computed on TRAINING images only (no leakage)."""
    mean = images.mean(axis=(0, 1, 2))
    std = images.std(axis=(0, 1, 2)) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def to_model_input(images: np.ndarray, mean, std) -> np.ndarray:
    """(N, H, W, 3) in [0, 1] -> normalised, contiguous (N, 3, H, W)."""
    x = (images - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32)


def write_split_manifest(path: str | Path, splits: dict[str, Dataset]) -> None:
    """Record exactly which file landed in which split (reproducibility / auditing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "path", "label"])
        for name, ds in splits.items():
            for p, y in zip(ds.paths, ds.labels):
                w.writerow([name, p, ds.class_names[y]])
