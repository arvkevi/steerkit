"""Tests for the HTML one-pager report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from steerkit import (  # noqa: E402
    PROBE_METHODS,
    Concept,
    ConceptGroup,
    GroupFit,
    MultinomialProbe,
    Probe,
)
from steerkit.report import (  # noqa: E402
    render_group_report,
    render_probe_report,
    write_report,
)


def _probe(layer: int = 3) -> Probe:
    rng = torch.Generator().manual_seed(layer + 1)
    directions = {}
    for method in PROBE_METHODS:
        v = torch.randn(8, generator=rng)
        directions[method] = v / v.norm()
    return Probe(
        directions=directions,
        bias={m: 0.0 for m in PROBE_METHODS},
        layer=layer,
        metrics={"auc_test_logistic": 0.92, "cohens_d_logistic": 1.5},
        model_id="fake/test",
        hook_site="resid_post",
        hook_name=f"blocks.{layer}.hook_resid_post",
        n_total_layers=12,
    )


def _per_layer() -> dict[int, Probe]:
    return {i: _probe(i) for i in range(5)}


def test_render_probe_report_summary_only():
    """With no extras, the report still renders summary + metrics."""
    html = render_probe_report(_probe())
    assert "<h1>" in html
    assert "fake/test" in html
    assert "auc_test_logistic" in html
    # No images when nothing else passed.
    assert "<img" not in html


def test_render_probe_report_with_per_layer_includes_layer_plot():
    html = render_probe_report(_probe(), per_layer=_per_layer())
    assert "Layer selection" in html
    assert "data:image/png;base64," in html


def test_render_probe_report_with_activations_includes_pca():
    rng = np.random.default_rng(0)
    activations = torch.tensor(
        np.stack([rng.normal(2, 1, (10, 8)), rng.normal(-2, 1, (10, 8))], axis=1),
        dtype=torch.float32,
    )
    html = render_probe_report(_probe(), activations=activations)
    assert "Activation projection" in html
    assert "data:image/png;base64," in html


def test_write_report_creates_file(tmp_path: Path):
    html = render_probe_report(_probe())
    out = tmp_path / "report.html"
    p = write_report(html, out)
    assert p.exists()
    assert p.read_text().startswith("<!doctype html>")


def test_probe_report_method_writes_file(tmp_path: Path):
    out = tmp_path / "report.html"
    result = _probe().report(per_layer=_per_layer(), out=out)
    assert result == str(out)
    assert out.exists()


def test_probe_report_method_returns_html_when_no_out():
    html = _probe().report(per_layer=_per_layer())
    assert html.startswith("<!doctype html>")


def _group_fit_with_full_per_concept() -> GroupFit:
    group = ConceptGroup(
        name="emotion",
        relationship="mutually_exclusive",
        neutral_reference="Plain.",
        concepts=[Concept("joy", "upbeat"), Concept("sadness", "downcast")],
    )
    return GroupFit(
        group=group,
        best={"joy": _probe(2), "sadness": _probe(4)},
        per_concept={"joy": _per_layer(), "sadness": _per_layer()},
        multinomial=MultinomialProbe(
            weights=torch.eye(2, 8),
            biases=torch.zeros(2),
            class_names=["joy", "sadness"],
            layer=3,
            hook_name="blocks.3.hook_resid_post",
            model_id="fake/test",
            hook_site="resid_post",
            n_total_layers=12,
        ),
    )


def test_render_group_report_includes_per_concept_curves_and_similarity():
    html = render_group_report(_group_fit_with_full_per_concept())
    assert "emotion" in html
    assert "joy" in html and "sadness" in html
    assert "Cross-concept similarity" in html
    # We expect at least 3 embedded images: 2 per-concept + 1 similarity heatmap.
    assert html.count("data:image/png;base64,") >= 3


def test_render_group_report_no_per_concept_skips_layer_plots():
    fit = _group_fit_with_full_per_concept()
    fit.per_concept = None  # simulate loaded-from-disk
    html = render_group_report(fit)
    # Similarity heatmap still appears.
    assert "Cross-concept similarity" in html
    # No "Layer selection per concept" section without per_concept.
    assert "Layer selection per concept" not in html


def test_groupfit_report_method_writes_file(tmp_path: Path):
    out = tmp_path / "fit.html"
    fit = _group_fit_with_full_per_concept()
    result = fit.report(out=out)
    assert result == str(out)
    assert out.exists()
    body = out.read_text()
    assert "<!doctype html>" in body and "emotion" in body
