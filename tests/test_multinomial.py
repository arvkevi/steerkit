"""Tests for MultinomialProbe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from steerkit import MultinomialProbe


@dataclass
class _FakeModel:
    n_layers: int
    model_id: str = "fake/multi"


def _synth_concept_acts(
    n_pairs: int, n_layers: int, d_model: int, *, mean: np.ndarray, seed: int
) -> dict[int, torch.Tensor]:
    rng = np.random.default_rng(seed)
    out = {}
    for layer in range(n_layers):
        # positives: cluster around `mean`; negatives: around -mean (we ignore negatives for multinomial)
        pos = rng.normal(loc=mean, scale=0.6, size=(n_pairs, d_model))
        neg = rng.normal(loc=-mean, scale=0.6, size=(n_pairs, d_model))
        out[layer] = torch.tensor(np.stack([pos, neg], axis=1), dtype=torch.float32)
    return out


def _three_class_activations(d_model: int = 32, n_pairs: int = 30, n_layers: int = 2):
    e1 = np.zeros(d_model)
    e1[0] = 2.5
    e2 = np.zeros(d_model)
    e2[1] = 2.5
    e3 = np.zeros(d_model)
    e3[2] = 2.5
    return {
        "joy": _synth_concept_acts(n_pairs, n_layers, d_model, mean=e1, seed=10),
        "sadness": _synth_concept_acts(n_pairs, n_layers, d_model, mean=e2, seed=20),
        "fear": _synth_concept_acts(n_pairs, n_layers, d_model, mean=e3, seed=30),
    }


def test_fit_at_layer_perfect_separation():
    acts = _three_class_activations()
    model = _FakeModel(n_layers=2)
    probe = MultinomialProbe.fit_at_layer(acts, layer=0, model=model)  # type: ignore[arg-type]
    assert probe.class_names == ["joy", "sadness", "fear"]
    assert probe.weights.shape == (3, 32)
    assert probe.biases.shape == (3,)
    assert probe.metrics["accuracy_test"] >= 0.9


def test_fit_best_layer_picks_a_layer():
    acts = _three_class_activations()
    model = _FakeModel(n_layers=2)
    probe = MultinomialProbe.fit_best_layer(acts, model=model)  # type: ignore[arg-type]
    assert probe.layer in (0, 1)
    assert probe.metrics["accuracy_test"] >= 0.9


def test_similarity_matrix_orthogonal_for_orthogonal_means():
    acts = _three_class_activations()
    model = _FakeModel(n_layers=1)
    probe = MultinomialProbe.fit_at_layer(acts, layer=0, model=model)  # type: ignore[arg-type]
    sim = probe.similarity_matrix()
    # Diagonal must be 1.0; off-diagonals should be smallish for orthogonal class means.
    assert torch.allclose(torch.diagonal(sim), torch.ones(3), atol=1e-5)
    off = sim - torch.eye(3)
    assert off.abs().max().item() < 0.7


def test_save_load_roundtrip(tmp_path: Path):
    acts = _three_class_activations()
    model = _FakeModel(n_layers=2)
    probe = MultinomialProbe.fit_at_layer(acts, layer=0, model=model)  # type: ignore[arg-type]
    path = tmp_path / "mn.probe.safetensors"
    probe.save(path)
    loaded = MultinomialProbe.load(path)
    assert loaded.class_names == probe.class_names
    assert torch.allclose(loaded.weights, probe.weights)
    assert torch.allclose(loaded.biases, probe.biases)
    assert loaded.layer == probe.layer
    assert loaded.hook_name == probe.hook_name
    assert loaded.metrics == probe.metrics


def test_fit_rejects_single_class():
    acts = {"only": _synth_concept_acts(20, 1, 16, mean=np.zeros(16), seed=0)}
    model = _FakeModel(n_layers=1)
    with pytest.raises(ValueError, match="at least 2 classes"):
        MultinomialProbe.fit_at_layer(acts, layer=0, model=model)  # type: ignore[arg-type]
