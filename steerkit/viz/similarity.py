from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..probe import MultinomialProbe, Probe


def plot_similarity_heatmap(
    source: MultinomialProbe | dict[str, Probe],
    *,
    method: str | None = None,
    title: str | None = None,
    cmap: str = "RdBu_r",
) -> Figure:
    """Cosine-similarity heatmap between class direction vectors.

    Accepts either:
      - a `MultinomialProbe` whose `weights` rows are per-class directions
      - a `dict[name -> Probe]` whose entries each carry a binary direction
        (typically `GroupFit.best`).

    A diagonal of 1.0 is expected; off-diagonals at ~0 indicate orthogonal
    concepts; off-diagonals near ±1 indicate redundancy (e.g., joy ≈ −sadness).
    """
    if isinstance(source, MultinomialProbe):
        names = list(source.class_names)
        sim = source.similarity_matrix().cpu().numpy()
    elif isinstance(source, dict):
        names = list(source.keys())
        if not names:
            raise ValueError("empty probes dict")
        vectors = []
        for name in names:
            v = source[name].get_direction(method)
            vectors.append(v.cpu().numpy())
        mat = np.stack(vectors, axis=0)
        norm = mat / (np.linalg.norm(mat, axis=-1, keepdims=True) + 1e-12)
        sim = norm @ norm.T
    else:
        raise TypeError(
            "source must be MultinomialProbe or dict[str, Probe], "
            f"got {type(source).__name__}"
        )

    fig, ax = plt.subplots(figsize=(max(4.5, len(names) * 0.6), max(4.5, len(names) * 0.6)))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{sim[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if abs(sim[i, j]) < 0.5 else "white",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, label="cosine similarity")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
