"""Tests for steerkit.viz — verify each plot returns a Figure with sensible structure.

We use the Agg backend so tests run headlessly. The assertions focus on shape
(Figure type, axes count, line count) rather than pixel-perfect rendering.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from steerkit import (  # noqa: E402
    PROBE_METHODS,
    MultinomialProbe,
    Probe,
    plot_activation_projection,
    plot_alpha_curve,
    plot_cross_model_overlay,
    plot_layer_selection,
    plot_logit_lens,
    plot_similarity_heatmap,
)


def _make_probe(layer: int, *, auc: float = 0.9, steering: float | None = None) -> Probe:
    rng = torch.Generator().manual_seed(layer + 13)
    directions = {}
    for method in PROBE_METHODS:
        v = torch.randn(8, generator=rng)
        directions[method] = v / v.norm()
    metrics = {"auc_test_logistic": auc}
    if steering is not None:
        metrics["steering_effect"] = steering
    return Probe(
        directions=directions,
        bias={m: 0.0 for m in PROBE_METHODS},
        layer=layer,
        metrics=metrics,
        model_id="fake/test",
        hook_site="resid_post",
        hook_name=f"blocks.{layer}.hook_resid_post",
        n_total_layers=12,
    )


def _close(fig: Figure) -> None:
    plt.close(fig)


def test_plot_layer_selection_classifier_only():
    probes = {i: _make_probe(i, auc=0.5 + i * 0.05) for i in range(8)}
    fig = plot_layer_selection(probes)
    assert isinstance(fig, Figure)
    # Single y-axis when no steering metric is present.
    assert len(fig.axes) == 1
    line_ys = fig.axes[0].lines[0].get_ydata()
    assert len(line_ys) == 8
    _close(fig)


def test_plot_layer_selection_dual_when_steering_present():
    probes = {i: _make_probe(i, auc=0.5 + i * 0.05, steering=3.0 + i * 0.2) for i in range(6)}
    fig = plot_layer_selection(probes)
    # twinx() introduces a second axis.
    assert len(fig.axes) == 2
    _close(fig)


def test_plot_layer_selection_normalized_depth():
    probes = {i: _make_probe(i, auc=0.7) for i in range(4)}
    fig = plot_layer_selection(probes, x_axis="normalized_depth")
    xs = fig.axes[0].lines[0].get_xdata()
    # Normalized depths are in [0, 1].
    assert all(0 <= x <= 1 for x in xs)
    _close(fig)


def test_plot_layer_selection_rejects_empty():
    with pytest.raises(ValueError, match="no probes"):
        plot_layer_selection({})


def test_plot_activation_projection_returns_figure():
    rng = np.random.default_rng(0)
    pos = rng.normal(loc=2.0, size=(20, 16))
    neg = rng.normal(loc=-2.0, size=(20, 16))
    activations = torch.tensor(np.stack([pos, neg], axis=1), dtype=torch.float32)
    fig = plot_activation_projection(activations)
    ax = fig.axes[0]
    # Two scatter collections (positives + negatives).
    assert len(ax.collections) == 2
    _close(fig)


def test_plot_activation_projection_rejects_bad_shape():
    bad = torch.zeros(10, 3, 16)  # second axis != 2
    with pytest.raises(ValueError, match="\\[n_pairs, 2, d_model\\]"):
        plot_activation_projection(bad)


def test_plot_alpha_curve():
    ratios = {0.5: 1.02, 1.0: 1.05, 2.0: 1.20, 4.0: 1.30, 8.0: 1.65}
    fig = plot_alpha_curve(ratios, ratio_max=1.5, chosen_alpha=4.0)
    ax = fig.axes[0]
    # main curve + ceiling line + chosen-α line = 3 lines
    assert len(ax.lines) == 3
    _close(fig)


def test_plot_alpha_curve_no_chosen():
    fig = plot_alpha_curve({1.0: 1.1, 2.0: 1.4}, ratio_max=1.5)
    # Without chosen_alpha there are only 2 lines: curve + ceiling
    assert len(fig.axes[0].lines) == 2
    _close(fig)


def test_plot_alpha_curve_rejects_empty():
    with pytest.raises(ValueError, match="no α/ratio"):
        plot_alpha_curve({})


def test_plot_logit_lens_with_stub_model():
    """logit_lens needs `model.hooked.W_U` and a tokenizer; we stub both."""
    probe = _make_probe(layer=3)
    model = MagicMock()
    model.device = "cpu"
    model.hooked = MagicMock()
    model.hooked.cfg.dtype = torch.float32
    # W_U: [d_model, vocab]
    rng = torch.Generator().manual_seed(0)
    model.hooked.W_U = torch.randn(8, 100, generator=rng)
    model.tokenizer = MagicMock()
    model.tokenizer.decode = MagicMock(side_effect=lambda ids: f"tok{ids[0]}")
    fig = plot_logit_lens(probe, model, top_k=10)
    ax = fig.axes[0]
    # 10 horizontal bars
    bars = [p for p in ax.patches if p.get_height() != 0 or p.get_width() != 0]
    assert len(bars) == 10
    _close(fig)


def test_plot_similarity_heatmap_from_multinomial():
    weights = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],  # parallel to row 0 with opposite sign
        ]
    )
    mn = MultinomialProbe(
        weights=weights,
        biases=torch.zeros(3),
        class_names=["a", "b", "c"],
        layer=4,
        hook_name="blocks.4.hook_resid_post",
        model_id="fake/test",
        hook_site="resid_post",
        n_total_layers=12,
    )
    fig = plot_similarity_heatmap(mn)
    ax = fig.axes[0]
    assert ax.get_xticklabels()[0].get_text() == "a"
    _close(fig)


def test_plot_similarity_heatmap_from_probes_dict():
    probes = {"joy": _make_probe(3), "sadness": _make_probe(5)}
    fig = plot_similarity_heatmap(probes)
    ax = fig.axes[0]
    assert {t.get_text() for t in ax.get_xticklabels()} == {"joy", "sadness"}
    _close(fig)


def test_plot_similarity_rejects_unknown_type():
    with pytest.raises(TypeError, match="MultinomialProbe or dict"):
        plot_similarity_heatmap("not a valid source")


def test_plot_cross_model_overlay_two_models():
    probes_a = {i: _make_probe(i, auc=0.5 + i * 0.05) for i in range(6)}
    probes_b = {i: _make_probe(i, auc=0.4 + i * 0.07) for i in range(8)}
    fig = plot_cross_model_overlay(
        {"model-a": probes_a, "model-b": probes_b}, by="auc_test_logistic"
    )
    ax = fig.axes[0]
    # One line per model
    assert len(ax.lines) == 2
    # Legend has both labels
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "model-a" in legend_texts and "model-b" in legend_texts
    _close(fig)


def test_plot_cross_model_overlay_rejects_empty():
    with pytest.raises(ValueError, match="no models"):
        plot_cross_model_overlay({})


def test_probe_plot_logit_lens_method():
    """Probe.plot_logit_lens() should delegate to the free function."""
    probe = _make_probe(layer=3)
    model = MagicMock()
    model.device = "cpu"
    model.hooked = MagicMock()
    model.hooked.cfg.dtype = torch.float32
    model.hooked.W_U = torch.randn(8, 50)
    model.tokenizer = MagicMock()
    model.tokenizer.decode = MagicMock(side_effect=lambda ids: f"t{ids[0]}")
    fig = probe.plot_logit_lens(model, top_k=5)
    assert isinstance(fig, Figure)
    _close(fig)


def test_multinomial_plot_similarity_method():
    """MultinomialProbe.plot_similarity() should delegate to the free function."""
    mn = MultinomialProbe(
        weights=torch.eye(3, 8),
        biases=torch.zeros(3),
        class_names=["a", "b", "c"],
        layer=4,
        hook_name="blocks.4.hook_resid_post",
        model_id="fake/test",
        hook_site="resid_post",
        n_total_layers=12,
    )
    fig = mn.plot_similarity()
    assert isinstance(fig, Figure)
    _close(fig)


def test_groupfit_plot_layer_selection_requires_per_concept():
    from steerkit import Concept, ConceptGroup, GroupFit

    group = ConceptGroup(
        name="g",
        relationship="multi_label",
        neutral_reference="...",
        concepts=[Concept("c", "...")],
    )
    fit = GroupFit(group=group, best={"c": _make_probe(3)}, per_concept=None)
    with pytest.raises(RuntimeError, match="per_concept is None"):
        fit.plot_layer_selection("c")
