"""Tests for CompositeProbe and compose() — cross-group steering composition."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from steerkit import PROBE_METHODS, CompositeProbe, Probe, compose


def _make_probe(
    *, layer: int, hook_name: str | None = None, n_layers: int = 12, auto_alpha: float | None = 2.0
) -> Probe:
    rng = torch.Generator().manual_seed(layer + 7)
    directions = {}
    for method in PROBE_METHODS:
        v = torch.randn(8, generator=rng)
        directions[method] = v / v.norm()
    return Probe(
        directions=directions,
        bias={m: 0.0 for m in PROBE_METHODS},
        layer=layer,
        metrics={},
        model_id="fake/test",
        hook_site="resid_post",
        hook_name=hook_name or f"blocks.{layer}.hook_resid_post",
        n_total_layers=n_layers,
        auto_alpha=auto_alpha,
    )


def test_compose_default_weights():
    probes = [_make_probe(layer=3), _make_probe(layer=5)]
    c = compose(probes)
    assert c.weights == [1.0, 1.0]


def test_compose_explicit_weights():
    probes = [_make_probe(layer=3), _make_probe(layer=5)]
    c = compose(probes, weights=[0.7, 0.3])
    assert c.weights == [0.7, 0.3]


def test_compose_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="must match"):
        CompositeProbe(probes=[_make_probe(layer=3)], weights=[1.0, 2.0])


def _stub_model() -> MagicMock:
    """A duck-type ModelHandle whose hooked.generate returns prompt + dummy tokens."""
    model = MagicMock()
    model.device = "cpu"
    model.hooked = MagicMock()
    model.hooked.cfg.dtype = torch.float32
    model.format_chat = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    # generate echoes back prompt + 5 fake tokens
    model.hooked.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 10, 11, 12, 13, 14]]))
    model.tokenizer = MagicMock()
    model.tokenizer.decode = MagicMock(return_value="dummy completion")
    # hooks() must be a context manager that captures fwd_hooks for assertion.
    captured = {"hooks": None}

    class _HookCtx:
        def __enter__(self_ctx):
            return self_ctx

        def __exit__(self_ctx, *a):
            return False

    def _hooks_call(fwd_hooks=None):
        captured["hooks"] = fwd_hooks
        return _HookCtx()

    model.hooked.hooks = MagicMock(side_effect=_hooks_call)
    return model, captured


def test_steer_installs_one_hook_per_distinct_layer():
    """Two probes at different layers should produce two separate hook entries."""
    probes = [_make_probe(layer=3), _make_probe(layer=5)]
    c = compose(probes, weights=[0.5, 0.5])
    model, captured = _stub_model()
    out = c.steer(model, "hi")
    assert out == "dummy completion"
    fwd_hooks = captured["hooks"]
    assert fwd_hooks is not None
    hook_names = [name for name, _ in fwd_hooks]
    assert "blocks.3.hook_resid_post" in hook_names
    assert "blocks.5.hook_resid_post" in hook_names
    assert len(fwd_hooks) == 2


def test_steer_folds_same_hook_into_one():
    """Two probes at the SAME layer should be combined into a single hook."""
    probes = [_make_probe(layer=4), _make_probe(layer=4)]
    c = compose(probes)
    model, captured = _stub_model()
    c.steer(model, "hi")
    fwd_hooks = captured["hooks"]
    assert len(fwd_hooks) == 1
    assert fwd_hooks[0][0] == "blocks.4.hook_resid_post"


def test_steer_uses_auto_alpha_by_default():
    """If `alphas` is not passed, each probe's auto_alpha should be used."""
    p1 = _make_probe(layer=3, auto_alpha=4.0)
    p2 = _make_probe(layer=5, auto_alpha=None)  # → falls back to 2.0
    c = compose([p1, p2])
    model, captured = _stub_model()
    # Patch out the actual hook execution; we just want to verify the construction completes.
    c.steer(model, "hi")
    # Two distinct hooks, one per probe.
    assert len(captured["hooks"]) == 2


def test_steer_explicit_alphas_override_auto():
    p = _make_probe(layer=3, auto_alpha=4.0)
    c = compose([p])
    model, _ = _stub_model()
    # Just verify it doesn't error when overriding alpha.
    c.steer(model, "hi", alphas=[1.5])


def test_steer_rejects_mismatched_alphas():
    p = _make_probe(layer=3)
    c = compose([p])
    model, _ = _stub_model()
    with pytest.raises(ValueError, match="alphas and methods"):
        c.steer(model, "hi", alphas=[1.0, 2.0])
