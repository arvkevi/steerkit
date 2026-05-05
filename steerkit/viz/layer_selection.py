from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..probe import Probe


def plot_layer_selection(
    probes: dict[int, Probe],
    *,
    by_classifier: str = "auc_test_logistic",
    by_steering: str = "steering_effect",
    x_axis: Literal["layer", "normalized_depth"] = "layer",
    title: str | None = None,
) -> Figure:
    """Dual-curve plot of probe-classifier metric and (if available) LLM-judge steering effect
    as a function of layer depth.

    The classifier curve is always drawn (left y-axis); the steering curve is drawn on the
    right y-axis only if at least one probe has the `by_steering` metric attached.
    """
    layers = sorted(probes.keys())
    if not layers:
        raise ValueError("no probes to plot")

    xs: list[float]
    if x_axis == "layer":
        xs = [float(layers[i]) for i in range(len(layers))]
        xlabel = "layer index"
    elif x_axis == "normalized_depth":
        xs = [probes[layer].normalized_depth for layer in layers]
        xlabel = "normalized depth"
    else:
        raise ValueError(f"x_axis must be 'layer' or 'normalized_depth', got {x_axis!r}")

    classifier_ys: list[float] = []
    classifier_xs: list[float] = []
    for x, layer in zip(xs, layers, strict=True):
        m = probes[layer].metrics.get(by_classifier)
        if m is not None:
            classifier_ys.append(m)
            classifier_xs.append(x)

    steering_xs: list[float] = []
    steering_ys: list[float] = []
    for x, layer in zip(xs, layers, strict=True):
        m = probes[layer].metrics.get(by_steering)
        if m is not None and m == m:  # filter out NaN
            steering_xs.append(x)
            steering_ys.append(m)

    fig, ax_classifier = plt.subplots(figsize=(8, 4))
    ax_classifier.plot(
        classifier_xs,
        classifier_ys,
        marker="o",
        color="tab:blue",
        label=by_classifier,
    )
    ax_classifier.set_xlabel(xlabel)
    ax_classifier.set_ylabel(by_classifier, color="tab:blue")
    ax_classifier.tick_params(axis="y", labelcolor="tab:blue")
    ax_classifier.grid(True, alpha=0.3)

    if steering_ys:
        ax_steering = ax_classifier.twinx()
        ax_steering.plot(
            steering_xs,
            steering_ys,
            marker="s",
            color="tab:orange",
            label=by_steering,
        )
        ax_steering.set_ylabel(by_steering, color="tab:orange")
        ax_steering.tick_params(axis="y", labelcolor="tab:orange")

    if title:
        ax_classifier.set_title(title)
    fig.tight_layout()
    return fig
