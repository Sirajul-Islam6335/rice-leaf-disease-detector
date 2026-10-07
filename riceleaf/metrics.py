"""Classification metrics implemented from first principles."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """cm[i, j] = number of samples of true class i predicted as class j."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_metrics(cm: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": cm.sum(axis=1)}


def evaluate_predictions(y_true, probs: np.ndarray, class_names: list[str]) -> dict:
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(axis=1)
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, k)
    pc = per_class_metrics(cm)
    eps = 1e-12
    nll = float(-np.log(probs[np.arange(len(y_true)), y_true] + eps).mean())
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_precision": float(pc["precision"].mean()),
        "macro_recall": float(pc["recall"].mean()),
        "macro_f1": float(pc["f1"].mean()),
        "log_loss": nll,
        "n_samples": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
        "per_class": {name: {**{m: float(pc[m][i]) for m in ("precision", "recall", "f1")}, "support": int(pc["support"][i])}
                      for i, name in enumerate(class_names)},
    }


def classification_report(metrics: dict, class_names: list[str]) -> str:
    w = max(len(n) for n in class_names + ["macro avg"]) + 2
    lines = [f"{'':<{w}}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}", ""]
    for n in class_names:
        m = metrics["per_class"][n]
        lines.append(f"{n:<{w}}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>10d}")
    lines += ["", f"{'accuracy':<{w}}{'':>20}{metrics['accuracy']:>10.3f}{metrics['n_samples']:>10d}",
              f"{'macro avg':<{w}}{metrics['macro_precision']:>10.3f}{metrics['macro_recall']:>10.3f}"
              f"{metrics['macro_f1']:>10.3f}{metrics['n_samples']:>10d}"]
    cm = np.array(metrics["confusion_matrix"])
    lines += ["", "Confusion matrix (rows = true, cols = predicted):"]
    lines += ["  " + " ".join(f"{v:>4d}" for v in row) + f"   {class_names[i]}" for i, row in enumerate(cm)]
    return "\n".join(lines)
