"""A small, dependency-free (NumPy only) deep-learning engine."""

from __future__ import annotations

from .layers import (BatchNorm2D, Conv2D, Dense, Dropout, Flatten, GlobalAvgPool2D, Layer, MaxPool2D, ReLU)
from .losses import softmax_cross_entropy
from .model import Sequential, build_rice_leaf_net, load_checkpoint, save_checkpoint
from .optim import SGD, AdamW, clip_grad_norm, warmup_cosine

__all__ = [
    "Layer", "Conv2D", "BatchNorm2D", "ReLU", "MaxPool2D", "GlobalAvgPool2D", "Flatten", "Dropout", "Dense",
    "softmax_cross_entropy", "Sequential", "build_rice_leaf_net", "save_checkpoint", "load_checkpoint",
    "SGD", "AdamW", "clip_grad_norm", "warmup_cosine",
]
