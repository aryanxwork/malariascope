"""Static report figures (PNG) written next to each run — for the README / report / slides."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e"


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d9d8d4")
    ax.tick_params(colors=MUTED)
    ax.grid(True, color="#ecebe8", linewidth=1)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_matrix_plot(conf: dict, path: Path, title: str = "Confusion matrix (test)"):
    m = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(m, cmap="Blues")
    labels = ["Uninfected", "Parasitized"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for i in range(2):
        for j in range(2):
            pct = m[i, j] / max(m[i].sum(), 1)
            color = "white" if m[i, j] > m.max() * 0.5 else INK
            ax.text(j, i, f"{m[i, j]:,}\n{pct:.1%}", ha="center", va="center", color=color, fontsize=11)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("Actual", color=MUTED)
    _save(fig, path)


def roc_plot(roc: list[dict], auc: float, op: tuple[float, float] | None, path: Path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1, label="Chance")
    ax.plot([d["fpr"] for d in roc], [d["tpr"] for d in roc], color=BLUE, linewidth=2, label=f"Model (AUC {auc:.3f})")
    if op:
        ax.plot(*op, "o", color=ORANGE, markersize=8, markeredgecolor="white", markeredgewidth=2, label="Operating point")
    _style(ax, "ROC curve (test)", "False positive rate", "True positive rate (sensitivity)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)


def pr_plot(pr: list[dict], ap: float, path: Path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([d["recall"] for d in pr], [d["precision"] for d in pr], color=BLUE, linewidth=2)
    _style(ax, f"Precision–recall (test, AP {ap:.3f})", "Recall (sensitivity)", "Precision")
    ax.set_ylim(0, 1.02)
    _save(fig, path)


def reliability_plot(before: dict, after: dict, path: Path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1, label="Perfect calibration")
    for rel, color, name in ((before, ORANGE, "Raw"), (after, BLUE, "Temperature-scaled")):
        xs = [b["confidence"] for b in rel["bins"]]
        ys = [b["frequency"] for b in rel["bins"]]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=6, markeredgecolor="white",
                label=f"{name} (ECE {rel['ece']:.3f})")
    _style(ax, "Reliability diagram (test)", "Predicted probability", "Observed frequency")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def history_plot(history: dict, path: Path, finetune_start: int | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title in ((axes[0], "loss", "Loss"), (axes[1], "auc", "ROC-AUC")):
        if key in history:
            ax.plot(history[key], color=BLUE, linewidth=2, label="Train")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], color=ORANGE, linewidth=2, label="Validation")
        if finetune_start:
            ax.axvline(finetune_start - 0.5, color=GRAY, linewidth=1)
            ax.text(finetune_start - 0.4, ax.get_ylim()[1], " fine-tune", va="top", color=MUTED, fontsize=9)
        _style(ax, title, "Epoch", "")
        ax.legend(frameon=False)
    _save(fig, path)


def gradcam_gallery(rows: list[dict], path: Path):
    """rows: dicts with 'image' (uint8), 'overlay' (uint8), 'title'."""
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.2))
    axes = np.array(axes).reshape(2, n)
    for j, r in enumerate(rows):
        axes[0, j].imshow(r["image"])
        axes[1, j].imshow(r["overlay"])
        axes[0, j].set_title(r["title"], fontsize=8, color=INK)
        for i in range(2):
            axes[i, j].axis("off")
    fig.suptitle("Grad-CAM: evidence for 'Parasitized' (top: input, bottom: heatmap)", fontsize=10, x=0.01, ha="left")
    _save(fig, path)
