"""High-level workflows: hold-out training, k-fold cross-validation, evaluation."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .config import TrainConfig
from .data import (Dataset, load_images, scan_directory, stratified_kfold, stratified_split, to_model_input,
                   write_split_manifest)
from .metrics import classification_report, evaluate_predictions
from .nn import load_checkpoint, save_checkpoint
from .trainer import FitResult, Trainer, predict_proba

log = logging.getLogger(__name__)


def _load(cfg: TrainConfig) -> Dataset:
    ds = scan_directory(cfg.data_dir)
    counts = {n: int((ds.labels == i).sum()) for i, n in enumerate(ds.class_names)}
    log.info("Found %d images in %d classes: %s", len(ds), len(ds.class_names), counts)
    return load_images(ds, cfg.image_size)


def _evaluate(fit: FitResult, ds: Dataset, tta: bool) -> tuple[dict, np.ndarray]:
    probs = predict_proba(fit.model, to_model_input(ds.images, fit.mean, fit.std), tta=tta)
    return evaluate_predictions(ds.labels, probs, ds.class_names), probs


def _plots(out: Path, fit: FitResult, cm, names, title):
    try:
        from .plots import plot_confusion, plot_history
        plot_history(fit.history, out / "training_curves.png", fit.best_epoch)
        plot_confusion(cm, names, out / "confusion_matrix.png", title)
    except ImportError:
        log.warning("matplotlib not installed - skipping plots.")


def build_meta(cfg: TrainConfig, fit: FitResult, class_names: list[str], metrics: dict) -> dict:
    return {
        "format_version": 1,
        "package_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "class_names": class_names,
        "image_size": list(cfg.image_size),
        "mean": fit.mean.tolist(),
        "std": fit.std.tolist(),
        "architecture": {"name": "RiceLeafNet", "channels": list(cfg.model.channels),
                         "kernel_size": cfg.model.kernel_size, "dropout": cfg.model.dropout},
        "best_epoch": fit.best_epoch,
        "metrics": metrics,
        "config": cfg.to_dict(),
    }


def train_holdout(cfg: TrainConfig) -> dict:
    """Stratified train / val / test split -> train -> pick best epoch on val -> report on test."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    ds = _load(cfg)
    train_frac = 1.0 - cfg.val_fraction - cfg.test_fraction
    tr_idx, va_idx, te_idx = stratified_split(ds.labels, [train_frac, cfg.val_fraction, cfg.test_fraction], cfg.seed)
    train, val, test = ds.subset(tr_idx), ds.subset(va_idx), ds.subset(te_idx)
    write_split_manifest(out / "split.csv", {"train": train, "val": val, "test": test})
    log.info("Split sizes  train=%d  val=%d  test=%d", len(train), len(val), len(test))

    trainer = Trainer(cfg, ds.class_names)
    log.info("\n%s", trainer.build_model(np.random.default_rng(0)).summary((3, *cfg.image_size)))
    fit = trainer.fit(train, val)

    val_m, _ = _evaluate(fit, val, cfg.tta)
    test_m, _ = _evaluate(fit, test, cfg.tta)
    metrics = {"val": val_m, "test": test_m}
    save_checkpoint(out / "model.npz", fit.model, build_meta(cfg, fit, ds.class_names, metrics))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(fit.history, indent=2))
    report = classification_report(test_m, ds.class_names)
    (out / "test_report.txt").write_text(report + "\n")
    _plots(out, fit, test_m["confusion_matrix"], ds.class_names, f"Hold-out test set (n={len(test)})")
    log.info("TEST SET (never seen during training or model selection)\n%s", report)
    return metrics


def cross_validate(cfg: TrainConfig, k: int = 5) -> dict:
    """Stratified k-fold CV. Each fold is a test set once; a stratified slice of the
    remaining data is held out as validation for early stopping. Reports mean +/- std
    and the pooled out-of-fold confusion matrix over all images."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    ds = _load(cfg)
    folds = stratified_kfold(ds.labels, k, cfg.seed)
    oof = np.zeros((len(ds), len(ds.class_names)))
    fold_metrics = []
    val_frac = min(0.5, cfg.val_fraction / (1.0 - 1.0 / k))
    for i, te_idx in enumerate(folds):
        rest = np.setdiff1d(np.arange(len(ds)), te_idx)
        tr_rel, va_rel = stratified_split(ds.labels[rest], [1 - val_frac, val_frac], cfg.seed + i)
        train, val, test = ds.subset(rest[tr_rel]), ds.subset(rest[va_rel]), ds.subset(te_idx)
        log.info("Fold %d/%d  train=%d  val=%d  test=%d", i + 1, k, len(train), len(val), len(test))
        fold_cfg = TrainConfig.from_dict({**cfg.to_dict(), "seed": cfg.seed + i})
        fit = Trainer(fold_cfg, ds.class_names).fit(train, val)
        m, probs = _evaluate(fit, test, cfg.tta)
        oof[te_idx] = probs
        fold_metrics.append({"fold": i + 1, "best_epoch": fit.best_epoch, **{key: m[key] for key in
                             ("accuracy", "macro_f1", "log_loss")}})
        log.info("Fold %d test  acc %.3f  macro-F1 %.3f", i + 1, m["accuracy"], m["macro_f1"])

    pooled = evaluate_predictions(ds.labels, oof, ds.class_names)
    summary = {
        "k": k,
        "folds": fold_metrics,
        **{f"{key}_mean": float(np.mean([f[key] for f in fold_metrics])) for key in ("accuracy", "macro_f1", "log_loss")},
        **{f"{key}_std": float(np.std([f[key] for f in fold_metrics], ddof=1)) for key in ("accuracy", "macro_f1", "log_loss")},
        "pooled_out_of_fold": pooled,
    }
    (out / "cv_results.json").write_text(json.dumps(summary, indent=2))
    report = classification_report(pooled, ds.class_names)
    (out / "cv_report.txt").write_text(report + "\n")
    try:
        from .plots import plot_confusion
        plot_confusion(pooled["confusion_matrix"], ds.class_names, out / "cv_confusion_matrix.png",
                       f"{k}-fold CV, out-of-fold (n={len(ds)})")
    except ImportError:
        pass
    log.info("%d-fold CV  accuracy %.3f +/- %.3f   macro-F1 %.3f +/- %.3f\n%s", k,
             summary["accuracy_mean"], summary["accuracy_std"], summary["macro_f1_mean"], summary["macro_f1_std"], report)
    return summary


def evaluate_checkpoint(checkpoint: str | Path, data_dir: str | Path, tta: bool = True) -> dict:
    """Evaluate a saved model on every image under ``data_dir/<class>/``."""
    model, meta = load_checkpoint(checkpoint)
    ds = scan_directory(data_dir)
    if ds.class_names != meta["class_names"]:
        raise ValueError(f"Class folders {ds.class_names} do not match model classes {meta['class_names']}")
    load_images(ds, tuple(meta["image_size"]))
    probs = predict_proba(model, to_model_input(ds.images, meta["mean"], meta["std"]), tta=tta)
    m = evaluate_predictions(ds.labels, probs, ds.class_names)
    log.info("\n%s", classification_report(m, ds.class_names))
    return m
