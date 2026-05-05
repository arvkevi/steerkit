"""Unit tests for the Zarr activation cache."""

from __future__ import annotations

from pathlib import Path

import torch

from steerkit import ContrastPair
from steerkit.cache import (
    cache_path,
    cache_signature,
    hash_pairs,
    load_activations_zarr,
    save_activations_zarr,
)


def _sample_pairs() -> list[ContrastPair]:
    return [
        ContrastPair(prompt="hi", positive_response="yes!", negative_response="no."),
        ContrastPair(prompt="bye", positive_response="see ya!", negative_response="ok."),
    ]


def _sample_activations(n_pairs: int = 2, n_sites: int = 5, d_model: int = 16) -> dict[int, torch.Tensor]:
    layers = [-1, 0, 1, 2, 3]  # embed + 3 blocks + final_ln slot at 3 means n=3 here
    return {
        layer: torch.randn(n_pairs, 2, d_model, generator=torch.Generator().manual_seed(layer + 100))
        for layer in layers
    }


def test_hash_pairs_deterministic():
    pairs = _sample_pairs()
    assert hash_pairs(pairs) == hash_pairs(pairs)


def test_hash_pairs_changes_with_content():
    pairs = _sample_pairs()
    h1 = hash_pairs(pairs)
    pairs[0].positive_response = "different"
    h2 = hash_pairs(pairs)
    assert h1 != h2


def test_cache_signature_includes_inputs():
    sig_a = cache_signature(
        model_id="org/model-a", hook_site="resid_post", include_boundaries=True, pairs_hash="abc"
    )
    sig_b = cache_signature(
        model_id="org/model-b", hook_site="resid_post", include_boundaries=True, pairs_hash="abc"
    )
    sig_c = cache_signature(
        model_id="org/model-a", hook_site="resid_pre", include_boundaries=True, pairs_hash="abc"
    )
    sig_d = cache_signature(
        model_id="org/model-a", hook_site="resid_post", include_boundaries=False, pairs_hash="abc"
    )
    sig_e = cache_signature(
        model_id="org/model-a", hook_site="resid_post", include_boundaries=True, pairs_hash="xyz"
    )
    assert len({sig_a, sig_b, sig_c, sig_d, sig_e}) == 5
    # Filesystem-safe: no slashes after replacement.
    assert "/" not in sig_a
    assert sig_a == "org--model-a__resid_post__boundaries__abc"


def test_save_and_load_activations_roundtrip(tmp_path: Path):
    activations = _sample_activations()
    sig = "test-roundtrip"
    target = cache_path(tmp_path, sig)
    save_activations_zarr(
        activations,
        target,
        metadata={"model_id": "fake/test", "hook_site": "resid_post"},
    )
    assert target.exists()

    loaded, meta = load_activations_zarr(target)
    assert set(loaded.keys()) == set(activations.keys())
    for layer, tensor in activations.items():
        assert torch.allclose(loaded[layer], tensor, atol=1e-6)
    assert meta["model_id"] == "fake/test"
    assert meta["hook_site"] == "resid_post"
    assert meta["cache_schema_version"] == 1


def test_save_overwrites_existing_cache(tmp_path: Path):
    sig = "test-overwrite"
    target = cache_path(tmp_path, sig)
    a = _sample_activations()
    save_activations_zarr(a, target, metadata={"model_id": "fake/a", "hook_site": "x"})

    b = _sample_activations(n_pairs=2, n_sites=5, d_model=32)  # different shape
    save_activations_zarr(b, target, metadata={"model_id": "fake/b", "hook_site": "y"})

    loaded, meta = load_activations_zarr(target)
    assert meta["model_id"] == "fake/b"
    sample = next(iter(loaded.values()))
    assert sample.shape[-1] == 32
