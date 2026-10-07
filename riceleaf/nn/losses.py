"""Loss functions."""
from __future__ import annotations

import numpy as np

from .functional import log_softmax


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray, label_smoothing: float = 0.0,
                          class_weights: np.ndarray | None = None):
    """Softmax + (label-smoothed, optionally class-weighted) cross-entropy.

    With smoothed targets  t_k = (1 - eps) * 1[k = y] + eps / K  the loss is

        L = (1 / sum_i w_i) * sum_i w_i * ( - sum_k t_ik * log softmax(z_i)_k )

    and its gradient w.r.t. the logits has the famously simple form

        dL/dz_i = w_i * (softmax(z_i) - t_i) / sum_i w_i.

    Returns
    -------
    loss : float
    dlogits : array with the same shape as ``logits``
    """
    n, k = logits.shape
    logp = log_softmax(logits.astype(np.float64))
    target = np.full((n, k), label_smoothing / k)
    target[np.arange(n), y] += 1.0 - label_smoothing
    w = np.ones(n) if class_weights is None else np.asarray(class_weights, dtype=np.float64)[y]
    w_sum = w.sum()
    per_sample = -(target * logp).sum(axis=1)
    loss = float((w * per_sample).sum() / w_sum)
    dlogits = (np.exp(logp) - target) * (w / w_sum)[:, None]
    return loss, dlogits.astype(logits.dtype)
