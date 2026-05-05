from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_alpha_curve(
    ratios: dict[float, float],
    *,
    ratio_max: float = 1.5,
    chosen_alpha: float | None = None,
    title: str | None = None,
) -> Figure:
    """Plot α vs perplexity ratio from `calibrate_alpha`'s output.

    A horizontal line at `ratio_max` shows the coherence ceiling; the chosen α
    (if provided) is annotated with a vertical marker. Intent is to make the
    auto-α decision transparent: which α values stayed under the ceiling, and
    which one was picked.
    """
    if not ratios:
        raise ValueError("no α/ratio entries to plot")

    alphas = sorted(ratios.keys())
    ys = [ratios[a] for a in alphas]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, ys, marker="o", color="tab:blue", label="perplexity ratio")
    ax.axhline(ratio_max, color="tab:red", linestyle="--", alpha=0.7, label=f"ceiling = {ratio_max}")
    if chosen_alpha is not None:
        ax.axvline(chosen_alpha, color="tab:green", linestyle=":", alpha=0.7, label=f"chosen α = {chosen_alpha}")
    ax.set_xlabel("α (steering strength)")
    ax.set_ylabel("steered ppl / unsteered ppl")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
