"""Diagnostic figures (optional: requires matplotlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

TRAIN_COLOR = "#2a78d6"   # categorical slot 1
VAL_COLOR = "#eb6834"     # categorical slot 2
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
SEQ_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False,
    })
    return plt


def plot_history(history: list[dict], path: str | Path, best_epoch: int | None = None) -> Path:
    plt = _plt()
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    panels = [("Loss (cross-entropy)", "train_loss", "val_loss"), ("Accuracy", "train_acc", "val_acc")]
    for ax, (title, tk, vk) in zip(axes, panels):
        ax.plot(ep, [h[tk] for h in history], color=TRAIN_COLOR, lw=2, label="train (augmented)")
        ax.plot(ep, [h[vk] for h in history], color=VAL_COLOR, lw=2, label="validation")
        if best_epoch:
            ax.axvline(best_epoch, color=MUTED, lw=1, ls="--")
            ax.text(best_epoch, 1.0, f" best epoch ({best_epoch})", transform=ax.get_xaxis_transform(),
                    color=MUTED, fontsize=8, ha="left", va="bottom")
        ax.set_title(title, loc="left", color=INK, pad=14)
        ax.set_xlabel("epoch")
    axes[1].set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_confusion(cm, class_names: list[str], path: str | Path, title: str = "Confusion matrix") -> Path:
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cm = np.asarray(cm)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.grid(False)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(1, cm.max()))
    k = len(class_names)
    ax.set_xticks(range(k), class_names, rotation=20, ha="right")
    ax.set_yticks(range(k), class_names)
    for i in range(k):
        for j in range(k):
            dark = cm[i, j] > 0.55 * cm.max()
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                    color="#ffffff" if dark else INK, fontweight="bold" if i == j else "normal")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left", color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)
