"""Sequential container plus checkpoint (de)serialisation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .layers import (BatchNorm2D, Conv2D, Dense, Dropout, GlobalAvgPool2D, Layer, MaxPool2D, ReLU)


class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers
        if layers:
            layers[0].skip_input_grad = True  # nobody needs dL/d(image)
        self.training = True

    # --- mode switching --------------------------------------------------
    def train(self) -> "Sequential":
        self.training = True
        for layer in self.layers:
            layer.training = True
        return self

    def eval(self) -> "Sequential":
        self.training = False
        for layer in self.layers:
            layer.training = False
        return self

    # --- propagation -----------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    __call__ = forward

    def backward(self, dout: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            if dout is None:
                break

    # --- parameters ------------------------------------------------------
    def parameters(self):
        """Yield ``(layer, param_name, apply_weight_decay)`` triples."""
        for layer in self.layers:
            exempt = layer.no_decay()
            for name in layer.params:
                yield layer, name, name not in exempt

    def num_params(self) -> int:
        return sum(layer.num_params() for layer in self.layers)

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {}
        for i, layer in enumerate(self.layers):
            for name, arr in {**layer.params, **layer.buffers()}.items():
                state[f"{i}.{name}"] = arr.copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for i, layer in enumerate(self.layers):
            for name, arr in layer.params.items():
                arr[...] = state[f"{i}.{name}"]
            for name, arr in layer.buffers().items():
                arr[...] = state[f"{i}.{name}"]

    def summary(self, input_shape: tuple) -> str:
        rows, shape = [], tuple(input_shape)
        for i, layer in enumerate(self.layers):
            shape = layer.output_shape(shape)
            rows.append((f"{i:>2}  {layer!r}", str(shape), layer.num_params()))
        w1 = max(len(r[0]) for r in rows) + 2
        w2 = max(len(r[1]) for r in rows) + 2
        line = "-" * (w1 + w2 + 10)
        out = [line, f"{'Layer':<{w1}}{'Output (C,H,W)':<{w2}}{'Params':>10}", line]
        out += [f"{a:<{w1}}{b:<{w2}}{c:>10,}" for a, b, c in rows]
        out += [line, f"Total trainable parameters: {self.num_params():,}", line]
        return "\n".join(out)


def build_rice_leaf_net(num_classes: int, channels=(16, 32, 64, 128), kernel_size: int = 3,
                        dropout: float = 0.3, in_channels: int = 3,
                        rng: np.random.Generator | None = None) -> Sequential:
    """[Conv -> BatchNorm -> ReLU -> MaxPool] x len(channels) -> GAP -> Dropout -> Dense."""
    rng = rng or np.random.default_rng()
    layers: list[Layer] = []
    c_in = in_channels
    for c_out in channels:
        # Bias is redundant before BatchNorm (beta plays that role).
        layers += [Conv2D(c_in, c_out, kernel_size, bias=False, rng=rng), BatchNorm2D(c_out), ReLU(), MaxPool2D(2)]
        c_in = c_out
    layers += [GlobalAvgPool2D(), Dropout(dropout, rng=rng), Dense(c_in, num_classes, rng=rng)]
    return Sequential(layers)


# --- checkpoints ---------------------------------------------------------
META_KEY = "__meta__"


def save_checkpoint(path: str | Path, model: Sequential, meta: dict) -> Path:
    """Save weights + JSON metadata to a single compressed ``.npz`` (no pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **model.state_dict(), **{META_KEY: blob})
    return path


def load_checkpoint(path: str | Path) -> tuple[Sequential, dict]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data[META_KEY].tobytes().decode("utf-8"))
        state = {k: data[k] for k in data.files if k != META_KEY}
    arch = meta["architecture"]
    model = build_rice_leaf_net(num_classes=len(meta["class_names"]), channels=tuple(arch["channels"]),
                                kernel_size=arch["kernel_size"], dropout=arch["dropout"])
    model.load_state_dict(state)
    return model.eval(), meta
