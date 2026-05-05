"""Per-token probe-score visualization.

Renders the output of `Probe.score_tokens(...)` as a horizontal bar chart with
one bar per token. Bars to the right of zero are positions where the probe
direction is most active (positive class); bars to the left are negative.

The deliberate stylistic choice is "looks like `plot_logit_lens` but for
sequence positions instead of vocabulary positions" — both are sanity checks
of where a learned direction puts its weight.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..probe import TokenScores


def plot_token_scores(
    scores: TokenScores,
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    color_pos: str = "tab:red",
    color_neg: str = "tab:blue",
    mark_response_start: bool = True,
) -> Figure:
    """Render per-token probe scores as a horizontal bar chart.

    Args:
        scores: a `TokenScores` from `Probe.score_tokens(...)`.
        title: optional figure title; defaults to one mentioning the layer + method.
        figsize: matplotlib figure size; defaults scale with the number of tokens.
        color_pos: bar color for positive scores (concept-active positions).
        color_neg: bar color for negative scores.
        mark_response_start: when True and `scores.response_start > 0`, draws a
            horizontal divider between the prompt and response tokens.

    Returns the `Figure` (no `plt.show()` / `plt.close()` — caller decides).
    """
    tokens = scores.tokens
    vals = scores.scores.float().cpu().numpy()
    n = len(tokens)

    if figsize is None:
        figsize = (8.0, max(2.5, 0.22 * n))
    fig, ax = plt.subplots(figsize=figsize)

    # Plot top-to-bottom (token 0 at the top).
    y_pos = list(range(n))
    colors = [color_pos if v >= 0 else color_neg for v in vals]
    ax.barh(y_pos, vals, color=colors)
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels([repr(t) for t in tokens], fontsize=8)
    ax.axvline(0, color="black", lw=0.5)

    if mark_response_start and scores.response_start > 0:
        # Draw a horizontal line just above the first response token.
        ax.axhline(scores.response_start - 0.5, color="gray", lw=0.6, ls="--")
        ax.text(
            ax.get_xlim()[1],
            scores.response_start - 0.5,
            " response →",
            va="center",
            ha="right",
            fontsize=8,
            color="gray",
        )

    ax.set_xlabel(f"probe score ({scores.method} direction)")
    if title is None:
        title = f"Per-token probe scores (layer {scores.layer}, method={scores.method})"
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
