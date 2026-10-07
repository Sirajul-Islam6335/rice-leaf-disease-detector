"""Mini-batch training loop with warm-up/cosine LR, AdamW, early stopping and TTA."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .augment import Augmenter
from .config import TrainConfig
from .data import Dataset, channel_stats, to_model_input
from .metrics import evaluate_predictions
from .nn import AdamW, Sequential, build_rice_leaf_net, clip_grad_norm, softmax_cross_entropy, warmup_cosine
from .nn.functional import softmax

log = logging.getLogger(__name__)


def predict_proba(model: Sequential, x: np.ndarray, batch_size: int = 32, tta: bool = False) -> np.ndarray:
    """Class probabilities for normalised NCHW input. With ``tta`` the softmax
    outputs of the original, horizontally- and vertically-flipped views are averaged."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        p = softmax(model(xb).astype(np.float64))
        if tta:
            p += softmax(model(np.ascontiguousarray(xb[..., ::-1])).astype(np.float64))
            p += softmax(model(np.ascontiguousarray(xb[:, :, ::-1, :])).astype(np.float64))
            p /= 3.0
        out.append(p)
    return np.concatenate(out)


@dataclass
class FitResult:
    model: Sequential
    mean: np.ndarray
    std: np.ndarray
    best_epoch: int
    history: list[dict] = field(default_factory=list)


class Trainer:
    def __init__(self, cfg: TrainConfig, class_names: list[str]) -> None:
        self.cfg = cfg
        self.class_names = class_names

    def build_model(self, rng: np.random.Generator) -> Sequential:
        m = self.cfg.model
        return build_rice_leaf_net(len(self.class_names), channels=m.channels, kernel_size=m.kernel_size,
                                   dropout=m.dropout, rng=rng)

    def fit(self, train: Dataset, val: Dataset) -> FitResult:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        model = self.build_model(rng)
        augmenter = Augmenter(cfg.augment, rng) if cfg.augment.enabled else None

        mean, std = channel_stats(train.images)          # training statistics only
        x_val = to_model_input(val.images, mean, std)

        opt = AdamW(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
        n = len(train)
        bs = min(cfg.batch_size, n)
        steps_per_epoch = max(1, n // bs)                 # drop_last: avoid tiny BatchNorm batches
        total_steps = cfg.epochs * steps_per_epoch
        warmup_steps = cfg.warmup_epochs * steps_per_epoch

        best = {"val_loss": np.inf, "epoch": 0, "state": model.state_dict()}
        history, step, bad_epochs = [], 0, 0
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            perm = rng.permutation(n)
            losses, correct, seen, gnorms = [], 0, 0, []
            for b in range(steps_per_epoch):
                idx = perm[b * bs:(b + 1) * bs]
                imgs = augmenter.batch(train.images[idx]) if augmenter else train.images[idx]
                x, y = to_model_input(imgs, mean, std), train.labels[idx]

                opt.lr = warmup_cosine(step, total_steps, warmup_steps, cfg.lr, cfg.min_lr)
                logits = model(x)
                loss, dlogits = softmax_cross_entropy(logits, y, cfg.label_smoothing)
                model.backward(dlogits)
                gnorms.append(clip_grad_norm(model, cfg.grad_clip))
                opt.step()

                losses.append(loss)
                correct += int((logits.argmax(1) == y).sum())
                seen += len(y)
                step += 1

            probs = predict_proba(model, x_val)
            vm = evaluate_predictions(val.labels, probs, self.class_names)
            rec = {
                "epoch": epoch, "lr": opt.lr, "train_loss": float(np.mean(losses)), "train_acc": correct / seen,
                "val_loss": vm["log_loss"], "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"],
                "grad_norm": float(np.mean(gnorms)), "seconds": time.perf_counter() - t0,
            }
            history.append(rec)
            improved = rec["val_loss"] < best["val_loss"] - 1e-4
            if improved:
                best = {"val_loss": rec["val_loss"], "epoch": epoch, "state": model.state_dict()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            log.info("epoch %3d/%d | lr %.2e | train loss %.4f acc %.3f | val loss %.4f acc %.3f f1 %.3f | %.1fs%s",
                     epoch, cfg.epochs, rec["lr"], rec["train_loss"], rec["train_acc"], rec["val_loss"],
                     rec["val_acc"], rec["val_macro_f1"], rec["seconds"], "  *" if improved else "")
            if bad_epochs >= cfg.patience:
                log.info("Early stopping: no val-loss improvement for %d epochs.", cfg.patience)
                break

        model.load_state_dict(best["state"])
        model.eval()
        log.info("Restored best weights from epoch %d (val loss %.4f).", best["epoch"], best["val_loss"])
        return FitResult(model, mean, std, best["epoch"], history)
