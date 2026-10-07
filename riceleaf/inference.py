"""Production inference API."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import load_image, to_model_input
from .nn import load_checkpoint
from .trainer import predict_proba


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    uncertain: bool
    source: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    """Load once, predict many.

    >>> p = Predictor("artifacts/model.npz")
    >>> p.predict("leaf.jpg").label
    'Brown spot'

    ``uncertain`` is True when the top probability is below ``threshold``. The model
    only knows three diseases, so a healthy leaf or an unrelated photo will still be
    forced into one of them - the flag is a cue to ask for human review.
    """

    def __init__(self, checkpoint: str | Path, threshold: float = 0.6, tta: bool = True) -> None:
        self.model, self.meta = load_checkpoint(checkpoint)
        self.class_names: list[str] = self.meta["class_names"]
        self.size = tuple(self.meta["image_size"])
        self.mean, self.std = np.array(self.meta["mean"]), np.array(self.meta["std"])
        self.threshold, self.tta = threshold, tta

    def predict_batch(self, sources) -> list[Prediction]:
        images = np.stack([load_image(s, self.size) for s in sources])
        probs = predict_proba(self.model, to_model_input(images, self.mean, self.std), tta=self.tta)
        out = []
        for src, p in zip(sources, probs):
            i = int(p.argmax())
            out.append(Prediction(
                label=self.class_names[i], confidence=float(p[i]),
                probabilities={n: float(v) for n, v in zip(self.class_names, p)},
                uncertain=bool(p[i] < self.threshold),
                source=str(src) if isinstance(src, (str, Path)) else None,
            ))
        return out

    def predict(self, source) -> Prediction:
        return self.predict_batch([source])[0]
