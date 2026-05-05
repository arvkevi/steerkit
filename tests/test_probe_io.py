from pathlib import Path

import torch

from steerkit import PROBE_METHODS, Probe


def _make_probe(d_model: int = 64, layer: int = 7, n_layers: int = 12) -> Probe:
    rng = torch.Generator().manual_seed(0)
    directions = {}
    for method in PROBE_METHODS:
        v = torch.randn(d_model, generator=rng)
        directions[method] = v / v.norm()
    return Probe(
        directions=directions,
        bias={"logistic": -0.123, "diff_of_means": 0.0, "mass_mean": 0.05},
        layer=layer,
        metrics={
            "auc_train_logistic": 0.99,
            "auc_test_logistic": 0.92,
            "auc_test_diff_of_means": 0.88,
            "auc_test_mass_mean": 0.91,
            "cohens_d_logistic": 1.7,
            "n_train_pairs": 19.0,
            "n_test_pairs": 5.0,
        },
        model_id="fake/Tiny-Test-Model",
        hook_site="resid_post",
        hook_name=f"blocks.{layer}.hook_resid_post",
        n_total_layers=n_layers,
        default_method="logistic",
        auto_alpha=2.0,
    )


def test_probe_save_load_roundtrip(tmp_path: Path):
    probe = _make_probe()
    path = tmp_path / "probe.safetensors"
    probe.save(path)

    loaded = Probe.load(path)
    for method in PROBE_METHODS:
        assert torch.allclose(loaded.directions[method], probe.directions[method])
    assert loaded.bias == probe.bias
    assert loaded.layer == probe.layer
    assert loaded.metrics == probe.metrics
    assert loaded.model_id == probe.model_id
    assert loaded.hook_site == probe.hook_site
    assert loaded.n_total_layers == probe.n_total_layers
    assert loaded.default_method == probe.default_method
    assert loaded.auto_alpha == probe.auto_alpha
    assert loaded.hook_name == probe.hook_name
    # New normalized_depth: (layer + 1) / (n_total_layers + 1) = 8/13.
    assert abs(loaded.normalized_depth - 8 / 13) < 1e-6


def test_probe_auto_alpha_none_roundtrips(tmp_path: Path):
    probe = _make_probe()
    probe.auto_alpha = None
    path = tmp_path / "probe.safetensors"
    probe.save(path)
    loaded = Probe.load(path)
    assert loaded.auto_alpha is None


def test_probe_default_direction_property():
    probe = _make_probe()
    assert torch.allclose(probe.direction, probe.directions["logistic"])
    probe.default_method = "diff_of_means"
    assert torch.allclose(probe.direction, probe.directions["diff_of_means"])


def test_probe_get_direction_unknown_method_raises():
    import pytest

    probe = _make_probe()
    with pytest.raises(KeyError, match="no direction for method"):
        probe.get_direction("nonexistent")


