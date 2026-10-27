"""Plotting helpers with one consistent theme. Every function returns a Figure.

Confusion-matrix, ROC and feature-importance plots adapted from Nelson, Fridhi
& Wagner, LMU CMSI 3300 HW4 (2022), reused with instructor approval.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

PALETTE = {"<=50K": "#8DA0CB", ">50K": "#E07A5F"}
ACCENT = "#3D405B"
SEQ_CMAP = sns.light_palette("#E07A5F", as_cmap=True)


def set_theme():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05,
                  rc={"axes.titleweight": "bold", "axes.titlesize": 13,
                      "figure.dpi": 110, "savefig.dpi": 160,
                      "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out_dir, name):
    if out_dir is not None:
        p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / f"{name}.png", bbox_inches="tight")
    return fig


def class_balance(df, cfg):
    counts = df[cfg["target"]].value_counts().reindex([cfg["negative_label"], cfg["positive_label"]])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PALETTE,
                legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set(title="Class balance", xlabel=cfg["target"], ylabel="People")
    ax.set_ylim(0, counts.max() * 1.2)
    return fig


def numeric_by_class(df, cfg):
    cols = cfg["numeric_cols"]
    ncol = min(3, len(cols)); nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        data = df[[c, cfg["target"]]]
        skewed = (data[c] == 0).mean() > 0.5  # capital gain/loss: mostly zeros
        if skewed:
            data = data[data[c] > 0]
        sns.histplot(data=data, x=c, hue=cfg["target"], palette=PALETTE, bins=40,
                     stat="density", common_norm=False, element="step", ax=ax,
                     log_scale=True if skewed else None)
        ax.set_title(c + (" (non-zero only)" if skewed else ""))
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Numeric features by income class (each class normalized)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def positive_rate(df, cfg, col, top=15, order=None):
    pos = df[cfg["target"]] == cfg["positive_label"]
    g = pos.groupby(df[col]).agg(["mean", "size"])
    g = g.reindex([o for o in order if o in g.index]) if order else \
        g.sort_values("mean", ascending=False)
    g = g.head(top) if not order else g
    base = pos.mean()
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(g) + 1.4))
    sns.barplot(x=g["mean"], y=g.index.astype(str), color=PALETTE[">50K"], ax=ax)
    ax.axvline(base, color=ACCENT, ls="--", lw=1.3, label=f"Overall {base:.1%}")
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(m, i, f" {m:.0%}  (n={n:,})", va="center", fontsize=9)
    ax.set(title=f"Share earning >50K by {col}", xlabel="Positive rate", ylabel="")
    ax.set_xlim(0, min(1, g["mean"].max() * 1.35))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower right")
    return fig


def corr_heatmap(mat, title, fmt=".2f", cmap="vlag", center=0):
    fig, ax = plt.subplots(figsize=(0.75 * len(mat) + 3, 0.6 * len(mat) + 2.5))
    sns.heatmap(mat, annot=True, fmt=fmt, cmap=cmap, center=center, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title(title)
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion matrix"):
    m = confusion_matrix(y_true, y_pred)
    pct = m / m.sum(axis=1, keepdims=True)
    labels = np.array([[f"{v:,}\n{p:.1%} of row" for v, p in zip(r, pr)]
                       for r, pr in zip(m, pct)])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(pct, annot=labels, fmt="", cmap=SEQ_CMAP, cbar=False, vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, linewidths=1, ax=ax)
    ax.set(title=title, ylabel="Actual", xlabel="Predicted")
    return fig


def plot_roc_curve(curves, title="ROC curve"):
    """curves: dict name -> (y_true, y_score)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = sns.color_palette("Set2", len(curves))
    for (name, (yt, ys)), c in zip(curves.items(), colors):
        fpr, tpr, _ = roc_curve(yt, ys)
        ax.plot(fpr, tpr, lw=2, color=c, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
           ylabel="True positive rate", title=title)
    ax.legend(loc="lower right", fontsize=9)
    return fig


def plot_feature_importance(importances, feature_names, top=15,
                            title="Top feature importances"):
    s = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.5))
    sns.barplot(x=s.values, y=s.index, color=ACCENT, ax=ax)
    ax.set(title=title, xlabel="Relative importance", ylabel="")
    return fig


def model_comparison(res):
    long = res.melt(id_vars="Model", value_vars=["Accuracy", "F1", "ROC-AUC"],
                    var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sns.barplot(data=long, x="Score", y="Model", hue="Metric", palette="Set2", ax=ax)
    ax.set(xlim=(0.4, 1), title="Model comparison (test set)", ylabel="")
    ax.legend(loc="lower right", fontsize=9)
    return fig


def fairness_bars(tab, attr):
    long = tab.reset_index().melt(id_vars="g", value_vars=["Base rate", "Predicted positive rate", "TPR"],
                                  var_name="Metric", value_name="Rate")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(tab) * 1.6 + 1.5))
    sns.barplot(data=long, x="Rate", y="g", hue="Metric",
                palette=["#BBBBBB", PALETTE[">50K"], ACCENT], ax=ax)
    ax.set(title=f"Fairness peek by {attr} (best model, test set)", ylabel="", xlim=(0, 1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower right", fontsize=9)
    return fig
