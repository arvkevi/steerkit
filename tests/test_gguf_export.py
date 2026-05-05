"""Tests for the GGUF control-vector export.

The `gguf` package is in the [llamacpp] optional extra. We probe-import it at
test collection and skip the whole module if unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from steerkit import (
    PROBE_METHODS,
    Probe,
    compose,
    export_composite_to_gguf,
    export_probe_to_gguf,
)

pytest.importorskip("gguf")
import gguf  # noqa: E402  # safe after importorskip


def _probe(layer: int = 5, n_layers: int = 12) -> Probe:
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
        model_id="fake/test-model",
        hook_site="resid_post",
        hook_name=f"blocks.{layer}.hook_resid_post",
        n_total_layers=n_layers,
    )


def _read_gguf(path: Path):
    """Read a GGUF file back and return (kv_dict, tensors_dict)."""
    reader = gguf.GGUFReader(str(path))
    kv = {f.name: f.parts[f.data[0]].tolist() if f.data else None for f in reader.fields.values()}
    tensors = {t.name: torch.tensor(t.data) for t in reader.tensors}
    return kv, tensors


def test_export_probe_writes_one_tensor(tmp_path: Path):
    probe = _probe(layer=5)
    out = tmp_path / "probe.gguf"
    written = export_probe_to_gguf(probe, out)
    assert written == out
    assert out.exists()
    _, tensors = _read_gguf(out)
    assert "direction.5" in tensors
    assert tensors["direction.5"].shape == (8,)


def test_export_probe_with_method_override(tmp_path: Path):
    probe = _probe(layer=3)
    out = tmp_path / "p.gguf"
    export_probe_to_gguf(probe, out, method="diff_of_means")
    _, tensors = _read_gguf(out)
    expected = probe.directions["diff_of_means"]
    actual = tensors["direction.3"]
    assert torch.allclose(actual, expected, atol=1e-5)


def test_export_probe_scale_baked_in(tmp_path: Path):
    probe = _probe(layer=2)
    out = tmp_path / "scaled.gguf"
    export_probe_to_gguf(probe, out, scale=3.0)
    _, tensors = _read_gguf(out)
    expected = probe.direction * 3.0
    actual = tensors["direction.2"]
    assert torch.allclose(actual, expected, atol=1e-5)


def test_export_composite_writes_one_tensor_per_layer(tmp_path: Path):
    probes = [_probe(layer=3), _probe(layer=5), _probe(layer=7)]
    composite = compose(probes, weights=[0.5, 1.0, 0.5])
    out = tmp_path / "composite.gguf"
    export_composite_to_gguf(composite, out)
    _, tensors = _read_gguf(out)
    for layer in (3, 5, 7):
        assert f"direction.{layer}" in tensors
        assert tensors[f"direction.{layer}"].shape == (8,)


def test_export_composite_folds_same_layer(tmp_path: Path):
    """Two probes at the same layer should be summed (with their weights) into one entry."""
    p1 = _probe(layer=4)
    p2 = _probe(layer=4)
    composite = compose([p1, p2], weights=[0.5, 0.5])
    out = tmp_path / "fold.gguf"
    export_composite_to_gguf(composite, out)
    _, tensors = _read_gguf(out)
    # Only one direction.4 entry; value = 0.5*p1 + 0.5*p2.
    assert list(tensors.keys()) == ["direction.4"]
    expected = 0.5 * p1.direction + 0.5 * p2.direction
    assert torch.allclose(tensors["direction.4"], expected, atol=1e-5)


def test_export_composite_rejects_empty(tmp_path: Path):
    composite = compose([_probe(0)])
    composite.probes = []  # force empty for the test
    composite.weights = []
    with pytest.raises(ValueError, match="no probes"):
        export_composite_to_gguf(composite, tmp_path / "x.gguf")


def test_probe_export_gguf_method(tmp_path: Path):
    """Probe.export_gguf(...) should delegate to the free function."""
    probe = _probe(layer=5)
    out = tmp_path / "method.gguf"
    result = probe.export_gguf(out)
    assert result == out
    assert out.exists()
