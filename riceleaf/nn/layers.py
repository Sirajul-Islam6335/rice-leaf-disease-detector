"""Neural-network layers implemented from scratch with NumPy.

Every layer implements

* ``forward(x)``   – computes the output and caches what backward needs,
* ``backward(dy)`` – receives dL/d(output), stores dL/d(param) in ``self.grads``
  and returns dL/d(input).

Tensors use the ``(N, C, H, W)`` layout (batch, channels, height, width).
"""
from __future__ import annotations

import numpy as np

from .functional import col2im, conv_output_size, im2col


class Layer:
    """Base class. Parameters live in ``params``; matching gradients in ``grads``."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.training: bool = True
        # When True the layer may skip computing dL/dx (used for the first layer).
        self.skip_input_grad: bool = False
        self._cache = None

    # --- interface -------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def output_shape(self, in_shape: tuple) -> tuple:
        return in_shape

    # --- bookkeeping -----------------------------------------------------
    def buffers(self) -> dict[str, np.ndarray]:
        """Non-trainable state that must be saved (e.g. BatchNorm running stats)."""
        return {}

    def no_decay(self) -> set[str]:
        """Parameter names excluded from weight decay (biases, norm scales)."""
        return set()

    def num_params(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Conv2D(Layer):
    """2-D convolution (cross-correlation) computed as im2col + GEMM.

    y[n, f, i, j] = b[f] + sum_{c, u, v} W[f, c, u, v] * x_pad[n, c, i*s + u, j*s + v]
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int | str = "same", bias: bool = True,
                 rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.k, self.stride = kernel_size, stride
        self.pad = kernel_size // 2 if padding == "same" else int(padding)
        fan_in = in_channels * kernel_size * kernel_size
        # He (Kaiming) normal initialisation: Var[W] = 2 / fan_in, suited to ReLU.
        std = np.sqrt(2.0 / fan_in)
        self.params["W"] = (rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)) * std).astype(np.float32)
        self.use_bias = bias
        if bias:
            self.params["b"] = np.zeros(out_channels, dtype=np.float32)

    def no_decay(self) -> set[str]:
        return {"b"}

    def output_shape(self, in_shape):
        c, h, w = in_shape
        return (self.out_channels, conv_output_size(h, self.k, self.stride, self.pad),
                conv_output_size(w, self.k, self.stride, self.pad))

    def forward(self, x):
        n = x.shape[0]
        cols, ho, wo = im2col(x, self.k, self.k, self.stride, self.pad)
        w_mat = self.params["W"].reshape(self.out_channels, -1)          # (F, C*k*k)
        out = cols @ w_mat.T                                               # (N*Ho*Wo, F)
        if self.use_bias:
            out += self.params["b"]
        if self.training:
            self._cache = (x.shape, cols)
        return out.reshape(n, ho, wo, self.out_channels).transpose(0, 3, 1, 2)

    def backward(self, dout):
        x_shape, cols = self._cache
        f = self.out_channels
        d = dout.transpose(0, 2, 3, 1).reshape(-1, f)                     # (N*Ho*Wo, F)
        self.grads["W"] = (d.T @ cols).reshape(self.params["W"].shape)    # dL/dW = dY^T X_col
        if self.use_bias:
            self.grads["b"] = d.sum(axis=0)                                # dL/db = sum dY
        self._cache = None
        if self.skip_input_grad:
            return None
        dcols = d @ self.params["W"].reshape(f, -1)                       # dL/dX_col = dY W
        return col2im(dcols, x_shape, self.k, self.k, self.stride, self.pad)

    def __repr__(self):
        return f"Conv2D({self.in_channels}->{self.out_channels}, k={self.k}, s={self.stride}, p={self.pad})"


class BatchNorm2D(Layer):
    """Batch normalisation over (N, H, W) for every channel."""

    def __init__(self, num_channels: int, momentum: float = 0.9, eps: float = 1e-5) -> None:
        super().__init__()
        self.c, self.momentum, self.eps = num_channels, momentum, eps
        self.params["gamma"] = np.ones(num_channels, dtype=np.float32)
        self.params["beta"] = np.zeros(num_channels, dtype=np.float32)
        self.running_mean = np.zeros(num_channels, dtype=np.float32)
        self.running_var = np.ones(num_channels, dtype=np.float32)

    def buffers(self):
        return {"running_mean": self.running_mean, "running_var": self.running_var}

    def no_decay(self):
        return {"gamma", "beta"}

    @staticmethod
    def _bc(v):
        return v[None, :, None, None]

    def forward(self, x):
        if self.training:
            mu = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))
            m = x.shape[0] * x.shape[2] * x.shape[3]
            unbiased = var * m / max(m - 1, 1)
            self.running_mean[...] = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var[...] = self.momentum * self.running_var + (1 - self.momentum) * unbiased
        else:
            mu, var = self.running_mean, self.running_var
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - self._bc(mu)) * self._bc(inv_std)
        if self.training:
            self._cache = (x_hat, inv_std)
        return self._bc(self.params["gamma"]) * x_hat + self._bc(self.params["beta"])

    def backward(self, dout):
        x_hat, inv_std = self._cache
        axes = (0, 2, 3)
        m = dout.shape[0] * dout.shape[2] * dout.shape[3]
        self.grads["beta"] = dout.sum(axis=axes)
        self.grads["gamma"] = (dout * x_hat).sum(axis=axes)
        dx_hat = dout * self._bc(self.params["gamma"])
        # dx = (1/m) * inv_std * (m*dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
        dx = self._bc(inv_std / m) * (
            m * dx_hat
            - self._bc(dx_hat.sum(axis=axes))
            - x_hat * self._bc((dx_hat * x_hat).sum(axis=axes))
        )
        self._cache = None
        return dx

    def __repr__(self):
        return f"BatchNorm2D({self.c})"


class ReLU(Layer):
    def forward(self, x):
        mask = x > 0
        if self.training:
            self._cache = mask
        return x * mask

    def backward(self, dout):
        mask, self._cache = self._cache, None
        return dout * mask


class MaxPool2D(Layer):
    """Non-overlapping max pooling (window = stride = ``size``).

    Odd trailing rows/columns are dropped (floor mode); their gradient is zero.
    """

    def __init__(self, size: int = 2) -> None:
        super().__init__()
        self.p = size

    def output_shape(self, in_shape):
        c, h, w = in_shape
        return (c, h // self.p, w // self.p)

    def forward(self, x):
        n, c, h, w = x.shape
        p = self.p
        ho, wo = h // p, w // p
        xc = x[:, :, : ho * p, : wo * p]
        win = xc.reshape(n, c, ho, p, wo, p).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, ho, wo, p * p)
        idx = win.argmax(axis=-1)
        out = np.take_along_axis(win, idx[..., None], axis=-1)[..., 0]
        if self.training:
            self._cache = (x.shape, idx)
        return out

    def backward(self, dout):
        (n, c, h, w), idx = self._cache
        p = self.p
        ho, wo = h // p, w // p
        dwin = np.zeros((n, c, ho, wo, p * p), dtype=dout.dtype)
        # The gradient flows only to the arg-max of every window.
        np.put_along_axis(dwin, idx[..., None], dout[..., None], axis=-1)
        dxc = dwin.reshape(n, c, ho, wo, p, p).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, ho * p, wo * p)
        dx = np.zeros((n, c, h, w), dtype=dout.dtype)
        dx[:, :, : ho * p, : wo * p] = dxc
        self._cache = None
        return dx

    def __repr__(self):
        return f"MaxPool2D({self.p})"


class GlobalAvgPool2D(Layer):
    """Averages each feature map to a single number: (N, C, H, W) -> (N, C)."""

    def output_shape(self, in_shape):
        return (in_shape[0],)

    def forward(self, x):
        if self.training:
            self._cache = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, dout):
        n, c, h, w = self._cache
        self._cache = None
        return np.broadcast_to((dout / (h * w))[:, :, None, None], (n, c, h, w)).copy()


class Flatten(Layer):
    def output_shape(self, in_shape):
        return (int(np.prod(in_shape)),)

    def forward(self, x):
        if self.training:
            self._cache = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        shape, self._cache = self._cache, None
        return dout.reshape(shape)


class Dropout(Layer):
    """Inverted dropout: activations are scaled by 1/(1-p) at train time so
    inference needs no rescaling."""

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p
        self.rng = rng or np.random.default_rng()

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = (self.rng.random(x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        self._cache = mask
        return x * mask

    def backward(self, dout):
        if self._cache is None:
            return dout
        mask, self._cache = self._cache, None
        return dout * mask

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Dense(Layer):
    """Fully connected layer: y = x W + b."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        rng = rng or np.random.default_rng()
        self.in_features, self.out_features = in_features, out_features
        # Glorot/Xavier uniform: keeps logit variance moderate for the softmax head.
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.params["W"] = rng.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        self.params["b"] = np.zeros(out_features, dtype=np.float32)

    def no_decay(self):
        return {"b"}

    def output_shape(self, in_shape):
        return (self.out_features,)

    def forward(self, x):
        if self.training:
            self._cache = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        x, self._cache = self._cache, None
        self.grads["W"] = x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        if self.skip_input_grad:
            return None
        return dout @ self.params["W"].T

    def __repr__(self):
        return f"Dense({self.in_features}->{self.out_features})"
