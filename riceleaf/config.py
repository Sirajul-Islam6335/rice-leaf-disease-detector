"""Typed, JSON-serialisable experiment configuration."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class AugmentConfig:
    enabled: bool = True
    min_crop_scale: float = 0.80   # random crop keeps 80-100 % of each side
    hflip: float = 0.5             # probability of a horizontal flip
    vflip: float = 0.5             # probability of a vertical flip
    brightness: float = 0.15       # multiply by U(1-b, 1+b)
    contrast: float = 0.15
    saturation: float = 0.15       # hue is NOT jittered: lesion colour is diagnostic


@dataclass
class ModelConfig:
    channels: tuple = (16, 32, 64, 128)
    kernel_size: int = 3
    dropout: float = 0.3


@dataclass
class TrainConfig:
    data_dir: str = "data/rice_leaf_diseases"
    output_dir: str = "artifacts"
    # Leaves are long horizontal strips (~3.4:1), so the input is not square.
    image_height: int = 80
    image_width: int = 256
    epochs: int = 80
    batch_size: int = 16
    lr: float = 2e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    grad_clip: float = 5.0
    patience: int = 20             # early stopping on validation loss
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    tta: bool = True               # flip test-time augmentation at evaluation
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_height, self.image_width

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model"]["channels"] = list(self.model.channels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        model = ModelConfig(**{k: v for k, v in d.pop("model", {}).items()})
        model.channels = tuple(model.channels)
        augment = AugmentConfig(**d.pop("augment", {}))
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d, model=model, augment=augment)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
