from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from sklearn.decomposition import PCA


def plot_activation_projection(
    activations: torch.Tensor,
    *,
    method: Literal["pca"] = "pca",
    title: str | None = None,
    pos_label: str = "concept",
    neg_label: str = "neutral",
) -> Figure:
    """2D projection of a [n_pairs, 2, d_model] activations tensor, colored by class.

    The second axis is the contrast pair: index 0 is the positive (concept-bearing)
    response and index 1 is the negative (neutral) response. PCA only for now;
    UMAP can be added as an optional extra later.
    """
    if activations.ndim != 3 or activations.shape[1] != 2:
        raise ValueError(
            f"expected activations shape [n_pairs, 2, d_model]; got {tuple(activations.shape)}"
        )
    if method != "pca":
        raise ValueError(f"only method='pca' is supported in v1, got {method!r}")

    arr = activations.numpy()
    n_pairs = arr.shape[0]
    flat = arr.reshape(2 * n_pairs, -1)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat)
    pos_coords = coords[:n_pairs]
    neg_coords = coords[n_pairs:]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pos_coords[:, 0], pos_coords[:, 1], c="tab:orange", s=40, alpha=0.7, label=pos_label)
    ax.scatter(neg_coords[:, 0], neg_coords[:, 1], c="tab:blue", s=40, alpha=0.7, label=neg_label)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
