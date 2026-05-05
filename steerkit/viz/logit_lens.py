from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from ..models import ModelHandle
from ..probe import Probe


@torch.no_grad()
def plot_logit_lens(
    probe: Probe,
    model: ModelHandle,
    *,
    top_k: int = 20,
    method: str | None = None,
    title: str | None = None,
) -> Figure:
    """Push the steering direction through the model's unembedding to get vocab logits,
    and render the top-K tokens as a horizontal bar chart.

    A high-quality steering direction for "joy" should produce top tokens like
    "happy", "joyful", "delighted"; if the top tokens look unrelated, the probe
    is likely broken — this plot is the cheapest interpretability sanity check.
    """
    direction = probe.get_direction(method).to(model.device).to(model.hooked.cfg.dtype)
    # TL HookedTransformer exposes the unembed weight as model.hooked.W_U
    # of shape [d_model, vocab].
    W_U = model.hooked.W_U
    logits = direction @ W_U  # [vocab]
    top_vals, top_indices = logits.topk(top_k)
    tokenizer = model.tokenizer
    assert tokenizer is not None
    tokens = [str(tokenizer.decode([int(i.item())])) for i in top_indices]
    values = top_vals.float().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, max(3.5, top_k * 0.25)))
    y_pos = list(range(top_k))
    ax.barh(y_pos, values[::-1], color="tab:purple")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([repr(t) for t in tokens[::-1]])
    ax.set_xlabel("direction · W_U (logit contribution)")
    if title is None:
        title = (
            f"Logit-lens for probe at layer {probe.layer} "
            f"(method={method or probe.default_method})"
        )
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
