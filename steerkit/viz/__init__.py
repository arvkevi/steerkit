"""Visualization plots.

All plot functions return a `matplotlib.figure.Figure`; they do not call `.show()`
or `plt.close()`. The caller decides whether to display, save, or further compose.
Default colormaps: viridis for ordinal, tab10 for categorical, override-friendly.

Public surface:
  plot_layer_selection      — dual-curve AUC + steering effect across layers
  plot_activation_projection — PCA of [n_pairs, 2, d_model] activations colored by class
  plot_alpha_curve          — α vs perplexity ratio from calibrate_alpha output
  plot_logit_lens           — top-K vocab tokens that the steering direction promotes
  plot_similarity_heatmap   — cosine similarity between probe directions in a group
  plot_cross_model_overlay  — normalized-depth layer-curve overlay across models (the hero plot)
  plot_token_scores         — per-token probe-score bars over a single sequence
"""

from .alpha_curve import plot_alpha_curve
from .cross_model import plot_cross_model_overlay
from .layer_selection import plot_layer_selection
from .logit_lens import plot_logit_lens
from .projection import plot_activation_projection
from .similarity import plot_similarity_heatmap
from .token_scores import plot_token_scores

__all__ = [
    "plot_activation_projection",
    "plot_alpha_curve",
    "plot_cross_model_overlay",
    "plot_layer_selection",
    "plot_logit_lens",
    "plot_similarity_heatmap",
    "plot_token_scores",
]
