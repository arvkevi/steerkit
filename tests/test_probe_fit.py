"""Unit tests for Probe.fit_all using synthetic activations.

We construct two well-separated Gaussian clusters in d_model space and check that:
  - All three candidate directions produce unit-normalized directions.
  - Held-out AUC is high for all three when classes are linearly separable.
  - Cohen's d on the logistic direction is well above 0 for separable data.
  - Best layer selection works with the default and fallback metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from steerkit import PROBE_METHODS, Probe


@dataclass
class _FakeModel:
    """Minimal duck-type for ModelHandle, used in the synthetic test."""

    n_layers: int
    model_id: str = "fake/test"


def _synth_activations(
    n_pairs: int,
    n_layers: int,
    d_model: int,
    *,
    separation: float,
    seed: int = 0,
) -> dict[int, torch.Tensor]:
    """Generate a [n_pairs, 2, d_model] tensor per layer where positives and negatives
    are drawn from N(±separation*e1, I). All layers share the same separation here.
    """
    rng = np.random.default_rng(seed)
    direction = np.zeros(d_model)
    direction[0] = 1.0
    out: dict[int, torch.Tensor] = {}
    for layer in range(n_layers):
        pos = rng.normal(loc=separation * direction, scale=1.0, size=(n_pairs, d_model))
        neg = rng.normal(loc=-separation * direction, scale=1.0, size=(n_pairs, d_model))
        stacked = np.stack([pos, neg], axis=1)  # [n_pairs, 2, d_model]
        out[layer] = torch.tensor(stacked, dtype=torch.float32)
    return out


def test_fit_all_produces_three_unit_directions():
    activations = _synth_activations(n_pairs=40, n_layers=3, d_model=32, separation=2.0)
    model = _FakeModel(n_layers=3)
    probes = Probe.fit_all(activations, model)  # type: ignore[arg-type]
    assert set(probes.keys()) == {0, 1, 2}
    for probe in probes.values():
        assert set(probe.directions.keys()) == set(PROBE_METHODS)
        for v in probe.directions.values():
            assert v.shape == (32,)
            assert abs(v.norm().item() - 1.0) < 1e-5


def test_fit_all_well_separated_data_gets_high_auc():
    activations = _synth_activations(n_pairs=40, n_layers=2, d_model=32, separation=3.0)
    model = _FakeModel(n_layers=2)
    probes = Probe.fit_all(activations, model, test_fraction=0.25)
    for probe in probes.values():
        for method in PROBE_METHODS:
            key = f"auc_test_{method}"
            assert key in probe.metrics
            assert probe.metrics[key] > 0.85, f"{key} too low: {probe.metrics[key]}"
        assert probe.metrics["cohens_d_logistic"] > 1.0


def test_fit_all_best_layer_with_explicit_metric():
    # Layer 0 has weak separation, layer 1 has strong separation. Best should be layer 1.
    weak = _synth_activations(n_pairs=40, n_layers=1, d_model=32, separation=0.3, seed=0)
    strong = _synth_activations(n_pairs=40, n_layers=1, d_model=32, separation=3.0, seed=1)
    activations = {0: weak[0], 1: strong[0]}
    model = _FakeModel(n_layers=2)
    probes = Probe.fit_all(activations, model, test_fraction=0.25)
    best = Probe.best_layer(probes, by="auc_test_logistic")
    assert best.layer == 1


def test_fit_all_best_layer_falls_back_when_no_test_split():
    activations = _synth_activations(n_pairs=20, n_layers=2, d_model=32, separation=2.0)
    model = _FakeModel(n_layers=2)
    probes = Probe.fit_all(activations, model, test_fraction=0.0)
    # No test split → no auc_test_* metrics. best_layer should fall back gracefully.
    sample = next(iter(probes.values()))
    assert "auc_test_logistic" not in sample.metrics
    best = Probe.best_layer(probes)  # request default; should fall back to train metric
    assert best.layer in (0, 1)


def test_fit_all_invalid_default_method_raises():
    activations = _synth_activations(n_pairs=20, n_layers=1, d_model=16, separation=1.0)
    model = _FakeModel(n_layers=1)
    with pytest.raises(ValueError, match="default_method"):
        Probe.fit_all(activations, model, default_method="nonsense")


def test_split_keeps_pair_pos_and_neg_together():
    """Sanity-check that the train/test split is at the pair level, not at the sample level."""
    from steerkit.probe import _split_pair_indices

    train, test = _split_pair_indices(n_pairs=10, test_fraction=0.2, seed=0)
    assert len(set(train.tolist()) & set(test.tolist())) == 0
    assert len(train) + len(test) == 10
