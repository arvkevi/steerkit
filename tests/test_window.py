"""Tests for window-of-(2k+1) multi-layer steering."""

from __future__ import annotations

import pytest
import torch

from steerkit import (
    PROBE_METHODS,
    Concept,
    ConceptGroup,
    GroupFit,
    Probe,
    window,
)


def _probe_at(layer: int) -> Probe:
    rng = torch.Generator().manual_seed(layer + 1)
    directions = {}
    for method in PROBE_METHODS:
        v = torch.randn(8, generator=rng)
        directions[method] = v / v.norm()
    return Probe(
        directions=directions,
        bias={m: 0.0 for m in PROBE_METHODS},
        layer=layer,
        metrics={"auc_test_logistic": 0.9},
        model_id="fake/test",
        hook_site="resid_post",
        hook_name=f"blocks.{layer}.hook_resid_post",
        n_total_layers=12,
    )


def _full_layer_dict() -> dict[int, Probe]:
    # 6 consecutive layers: 0..5
    return {i: _probe_at(i) for i in range(6)}


def test_window_default_k1_picks_three_layers():
    probes = _full_layer_dict()
    composite = window(probes, center_layer=3, k=1)
    assert len(composite.probes) == 3
    assert [p.layer for p in composite.probes] == [2, 3, 4]
    # Equal weights summing to ~1.0
    assert composite.weights == [pytest.approx(1.0 / 3)] * 3


def test_window_k0_collapses_to_single_layer():
    probes = _full_layer_dict()
    composite = window(probes, center_layer=2, k=0)
    assert len(composite.probes) == 1
    assert composite.probes[0].layer == 2
    assert composite.weights == [1.0]


def test_window_clips_at_left_edge():
    probes = _full_layer_dict()
    composite = window(probes, center_layer=0, k=1)
    # Center=0 has no left neighbor; window is {0, 1}.
    assert [p.layer for p in composite.probes] == [0, 1]
    assert composite.weights == [pytest.approx(0.5), pytest.approx(0.5)]


def test_window_clips_at_right_edge():
    probes = _full_layer_dict()
    composite = window(probes, center_layer=5, k=2)
    # Center=5 with k=2 wants {3, 4, 5, 6, 7}; only {3, 4, 5} exist.
    assert [p.layer for p in composite.probes] == [3, 4, 5]


def test_window_with_boundary_layers():
    """Boundary-layer keys (-1 and n_layers) should be ordered as expected."""
    probes = {-1: _probe_at(-1), 0: _probe_at(0), 1: _probe_at(1), 2: _probe_at(2), 12: _probe_at(12)}
    composite = window(probes, center_layer=0, k=1)
    # sorted keys are [-1, 0, 1, 2, 12]; center=0 → [{-1, 0, 1}].
    assert [p.layer for p in composite.probes] == [-1, 0, 1]


def test_window_rejects_missing_center():
    with pytest.raises(KeyError, match="not in probes"):
        window(_full_layer_dict(), center_layer=99)


def test_window_rejects_negative_k():
    with pytest.raises(ValueError, match="k must be"):
        window(_full_layer_dict(), center_layer=2, k=-1)


def test_window_rejects_empty_dict():
    with pytest.raises(ValueError, match="no probes"):
        window({}, center_layer=0)


def _fake_group() -> ConceptGroup:
    return ConceptGroup(
        name="g",
        relationship="multi_label",
        neutral_reference="...",
        concepts=[Concept("c", "...")],
    )


def test_groupfit_window_uses_best_layer_as_center():
    per_concept = {"c": _full_layer_dict()}
    fit = GroupFit(
        group=_fake_group(),
        best={"c": _probe_at(3)},
        per_concept=per_concept,
    )
    composite = fit.window("c", k=1)
    assert [p.layer for p in composite.probes] == [2, 3, 4]


def test_groupfit_window_requires_per_concept():
    fit = GroupFit(
        group=_fake_group(),
        best={"c": _probe_at(3)},
        per_concept=None,  # loaded-from-disk state
    )
    with pytest.raises(RuntimeError, match="per_concept is None"):
        fit.window("c")


def test_groupfit_window_unknown_concept():
    per_concept = {"c": _full_layer_dict()}
    fit = GroupFit(
        group=_fake_group(),
        best={"c": _probe_at(3)},
        per_concept=per_concept,
    )
    with pytest.raises(KeyError, match="no concept named"):
        fit.window("nonexistent")
