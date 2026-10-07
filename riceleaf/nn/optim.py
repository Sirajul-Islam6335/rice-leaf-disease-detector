"""Optimisers, learning-rate schedules and gradient clipping."""
from __future__ import annotations

import math

import numpy as np


class Optimizer:
    def __init__(self, model, lr: float) -> None:
        self.model = model
        self.lr = lr
        self.refs = list(model.parameters())  # [(layer, name, decay: bool)]

    def step(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class SGD(Optimizer):
    """SGD with (Nesterov) momentum and L2 weight decay."""

    def __init__(self, model, lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0):
        super().__init__(model, lr)
        self.mu, self.nesterov, self.wd = momentum, nesterov, weight_decay
        self.v = [np.zeros_like(layer.params[name]) for layer, name, _ in self.refs]

    def step(self):
        for (layer, name, decay), v in zip(self.refs, self.v):
            p, g = layer.params[name], layer.grads[name]
            if decay and self.wd:
                g = g + self.wd * p
            v *= self.mu
            v += g
            update = g + self.mu * v if self.nesterov else v
            p -= self.lr * update


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

        m_t = b1 m_{t-1} + (1-b1) g_t
        v_t = b2 v_{t-1} + (1-b2) g_t^2
        m^  = m_t / (1 - b1^t),  v^ = v_t / (1 - b2^t)
        theta <- theta - lr * ( m^ / (sqrt(v^) + eps) + wd * theta )
    """

    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        super().__init__(model, lr)
        self.b1, self.b2 = betas
        self.eps, self.wd = eps, weight_decay
        self.t = 0
        self.m = [np.zeros_like(layer.params[name]) for layer, name, _ in self.refs]
        self.v = [np.zeros_like(layer.params[name]) for layer, name, _ in self.refs]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for (layer, name, decay), m, v in zip(self.refs, self.m, self.v):
            p, g = layer.params[name], layer.grads[name]
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            if decay and self.wd:
                p -= self.lr * self.wd * p          # decoupled decay
            p -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)


def warmup_cosine(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 0.0) -> float:
    """Linear warm-up followed by cosine annealing to ``min_lr``."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def clip_grad_norm(model, max_norm: float) -> float:
    """Rescale all gradients so their global L2 norm is at most ``max_norm``.

    Returns the norm *before* clipping (useful for monitoring)."""
    grads = [layer.grads[name] for layer, name, _ in model.parameters()]
    total = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads)))
    if max_norm and total > max_norm:
        scale = max_norm / (total + 1e-12)
        for g in grads:
            g *= scale
    return total
