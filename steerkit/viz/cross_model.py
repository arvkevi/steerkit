from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..probe import Probe


def _short_label(label: str) -> str:
    """Compact model label for legends."""
    return label.split("/", 1)[-1]


def _metric_label(metric: str) -> str:
    if metric == "cohens_d_logistic":
        return "concept separation (Cohen's d)"
    if metric == "auc_test_logistic":
        return "held-out AUC"
    return metric


def plot_cross_model_overlay(
    probes_per_model: dict[str, dict[int, Probe]],
    *,
    by: str = "auc_test_logistic",
    title: str | None = None,
    mark_best: bool = True,
) -> Figure:
    """Overlay layer-selection curves from multiple models on a normalized-depth x-axis.

    Useful for comparing where the same concept is most cleanly *classified* across
    models — a methodology-comparison plot, not a steering-vector-transfer claim.
    Each entry of `probes_per_model` is `model_label -> dict[int, Probe]` (e.g. the
    per-layer fits returned by `Probe.fit_all`). Curves are aligned via
    `Probe.normalized_depth` so models with different layer counts can be compared
    visually.

    Args:
        probes_per_model: Mapping from model label to per-layer `Probe.fit_all` results.
        by: Metric to plot, for example `auc_test_logistic` or `cohens_d_logistic`.
        title: Optional figure title.
        mark_best: If True, mark each model's best layer with a larger hollow dot.
    """
    if not probes_per_model:
        raise ValueError("no models to plot")

    series: list[tuple[str, list[float], list[float]]] = []
    for label, probes in probes_per_model.items():
        depths: list[float] = []
        ys: list[float] = []
        for layer in sorted(probes.keys()):
            metric = probes[layer].metrics.get(by)
            if metric is None:
                continue
            depths.append(probes[layer].normalized_depth)
            ys.append(metric)
        if ys:
            series.append((label, depths, ys))
    if not series:
        raise ValueError(f"no model has metric {by!r} to plot")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    cmap = plt.get_cmap("tab10")
    for i, (label, depths, ys) in enumerate(series):
        color = cmap(i % 10)
        ax.plot(
            depths,
            ys,
            marker="o",
            markersize=3.2,
            markeredgewidth=0,
            color=color,
            linewidth=2.1,
            alpha=0.92,
            label=_short_label(label),
        )
        if mark_best:
            best_i = max(range(len(ys)), key=ys.__getitem__)
            ax.scatter(
                [depths[best_i]],
                [ys[best_i]],
                s=72,
                marker="o",
                facecolor="white",
                edgecolor=color,
                linewidth=2.0,
                zorder=4,
            )

    ax.set_xlabel("normalized model depth")
    ax.set_ylabel(_metric_label(by))
    ax.set_xlim(-0.015, 1.015)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["embed", "0.2", "0.4", "0.6", "0.8", "final"])
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="model")
    ax.grid(True, axis="y", color="#bbbbbb", alpha=0.28, linewidth=0.8)
    ax.grid(True, axis="x", color="#dddddd", alpha=0.18, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", colors="#333333")
    if title:
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.text(
        0,
        -0.18,
        "Hollow markers show each model's best layer. The same concept is refit per model.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout()
    return fig
