"""Low-level array operations used by the layers.

The key trick is *im2col*: every receptive field of the input is unrolled into
one row of a matrix, so a convolution becomes a single matrix multiplication
(GEMM) that NumPy hands to an optimised BLAS routine.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided


def conv_output_size(n: int, k: int, stride: int, pad: int) -> int:
    """Spatial output size of a convolution / pooling window: floor((n + 2p - k) / s) + 1."""
    return (n + 2 * pad - k) // stride + 1


def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1, pad: int = 0):
    """Unroll sliding windows of ``x`` into rows.

    Parameters
    ----------
    x : (N, C, H, W) array
    kh, kw : kernel height / width
    stride, pad : convolution stride and symmetric zero padding

    Returns
    -------
    cols : (N * Ho * Wo, C * kh * kw) array. Row ``(n, i, j)`` holds the
        receptive field that produces output pixel ``(i, j)`` of sample ``n``.
    Ho, Wo : output spatial size
    """
    n, c, h, w = x.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho = conv_output_size(h, kh, stride, pad)
    wo = conv_output_size(w, kw, stride, pad)
    s_n, s_c, s_h, s_w = x.strides
    # A zero-copy 6-D view: (sample, out_row, out_col, channel, k_row, k_col)
    windows = as_strided(
        x,
        shape=(n, ho, wo, c, kh, kw),
        strides=(s_n, s_h * stride, s_w * stride, s_c, s_h, s_w),
        writeable=False,
    )
    return windows.reshape(n * ho * wo, c * kh * kw), ho, wo


def col2im(cols: np.ndarray, x_shape, kh: int, kw: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Adjoint of :func:`im2col`: scatter-add rows back to image positions.

    Overlapping windows *add* their contributions, which is exactly what the
    chain rule requires because each input pixel feeds several outputs.
    """
    n, c, h, w = x_shape
    ho = conv_output_size(h, kh, stride, pad)
    wo = conv_output_size(w, kw, stride, pad)
    cols6 = cols.reshape(n, ho, wo, c, kh, kw)
    dx = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        i_end = i + stride * ho
        for j in range(kw):
            j_end = j + stride * wo
            dx[:, :, i:i_end:stride, j:j_end:stride] += cols6[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    if pad:
        dx = dx[:, :, pad:-pad, pad:-pad]
    return dx


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the last axis."""
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))
