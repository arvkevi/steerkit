"""Tests for the four intervention operations.

Pure-tensor unit tests at the math level + a sanity test that Probe.steer
dispatches to the right op via a stubbed model.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from steerkit import (
    PROBE_METHODS,
    Probe,
    apply_addition,
    apply_clamp,
    apply_multiplicative,
    apply_projection,
)
from steerkit.intervention import make_hook


def _direction(d_model: int = 4) -> torch.Tensor:
    """Unit-norm direction along the first axis."""
    v = torch.zeros(d_model)
    v[0] = 1.0
    return v


def _activation(batch: int = 1, seq: int = 3, d_model: int = 4) -> torch.Tensor:
    """A simple activation tensor with known projection along e0."""
    rng = torch.Generator().manual_seed(0)
    return torch.randn(batch, seq, d_model, generator=rng)


def test_addition_adds_alpha_along_direction():
    act = _activation()
    direction = _direction()
    alpha = 3.0
    out = apply_addition(act, direction, alpha)
    delta = out - act
    # Only the first component should change; change should equal alpha at every position.
    assert torch.allclose(delta[..., 0], torch.full_like(delta[..., 0], alpha))
    assert torch.allclose(delta[..., 1:], torch.zeros_like(delta[..., 1:]))


def test_projection_zeros_component_along_direction():
    act = _activation()
    direction = _direction()
    out = apply_projection(act, direction)
    # Projection onto direction should be (approximately) zero everywhere.
    proj = (out * direction).sum(dim=-1)
    assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-6)


def test_clamp_forces_target_projection():
    act = _activation()
    direction = _direction()
    target = 2.5
    out = apply_clamp(act, direction, target)
    proj = (out * direction).sum(dim=-1)
    assert torch.allclose(proj, torch.full_like(proj, target), atol=1e-6)


def test_clamp_preserves_orthogonal_components():
    act = _activation()
    direction = _direction()
    out = apply_clamp(act, direction, target=1.0)
    # Components orthogonal to `direction` should be unchanged.
    assert torch.allclose(out[..., 1:], act[..., 1:])


def test_multiplicative_scales_existing_component():
    act = _activation()
    direction = _direction()
    gamma = 3.0
    out = apply_multiplicative(act, direction, gamma)
    in_proj = (act * direction).sum(dim=-1)
    out_proj = (out * direction).sum(dim=-1)
    assert torch.allclose(out_proj, gamma * in_proj, atol=1e-6)


def test_multiplicative_gamma_zero_equals_projection():
    """γ=0 should produce the same result as op='projection'."""
    act = _activation()
    direction = _direction()
    out_mult = apply_multiplicative(act, direction, gamma=0.0)
    out_proj = apply_projection(act, direction)
    assert torch.allclose(out_mult, out_proj, atol=1e-6)


def test_multiplicative_gamma_one_is_noop():
    act = _activation()
    direction = _direction()
    out = apply_multiplicative(act, direction, gamma=1.0)
    assert torch.allclose(out, act)


def test_make_hook_addition_returns_correct_function():
    direction = _direction()
    hook = make_hook("addition", direction, alpha=2.0)
    act = _activation()
    out = hook(act, None)  # second arg is the TL `hook` object, ignored
    assert torch.allclose(out, apply_addition(act, direction, 2.0))


def test_make_hook_requires_op_specific_param():
    direction = _direction()
    with pytest.raises(ValueError, match="alpha"):
        make_hook("addition", direction)
    with pytest.raises(ValueError, match="target"):
        make_hook("clamp", direction)
    with pytest.raises(ValueError, match="gamma"):
        make_hook("multiplicative", direction)


def test_make_hook_rejects_unknown_op():
    direction = _direction()
    with pytest.raises(ValueError, match="unknown op"):
        make_hook("nonsense", direction)


def _make_probe() -> Probe:
    rng = torch.Generator().manual_seed(0)
    directions = {}
    for method in PROBE_METHODS:
        v = torch.randn(8, generator=rng)
        directions[method] = v / v.norm()
    return Probe(
        directions=directions,
        bias={m: 0.0 for m in PROBE_METHODS},
        layer=3,
        metrics={},
        model_id="fake/test",
        hook_site="resid_post",
        hook_name="blocks.3.hook_resid_post",
        n_total_layers=12,
        auto_alpha=2.0,
    )


def _stub_model() -> tuple[MagicMock, dict]:
    model = MagicMock()
    model.model_id = "fake/test"
    model.device = "cpu"
    model.hooked = MagicMock()
    model.hooked.cfg.dtype = torch.float32
    model.format_chat = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    model.hooked.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 99]]))
    model.tokenizer = MagicMock()
    model.tokenizer.decode = MagicMock(return_value="ok")
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


def test_steer_addition_default():
    probe = _make_probe()
    model, captured = _stub_model()
    probe.steer(model, "hi")
    assert captured["hooks"] is not None
    assert len(captured["hooks"]) == 1
    assert captured["hooks"][0][0] == "blocks.3.hook_resid_post"


def test_steer_projection_via_op():
    probe = _make_probe()
    model, _ = _stub_model()
    probe.steer(model, "hi", op="projection")  # should not error


def test_steer_clamp_requires_target():
    probe = _make_probe()
    model, _ = _stub_model()
    with pytest.raises(ValueError, match="target"):
        probe.steer(model, "hi", op="clamp")


def test_steer_multiplicative_requires_gamma():
    probe = _make_probe()
    model, _ = _stub_model()
    with pytest.raises(ValueError, match="gamma"):
        probe.steer(model, "hi", op="multiplicative")


def test_steer_rejects_unknown_op():
    probe = _make_probe()
    model, _ = _stub_model()
    with pytest.raises(ValueError, match="unknown op"):
        probe.steer(model, "hi", op="nonsense")


def test_convenience_methods_dispatch():
    probe = _make_probe()
    model, _ = _stub_model()
    # Each convenience method should call into steer with the right op.
    probe.ablate(model, "hi")
    probe.clamp(model, "hi", target=1.5)
    probe.amplify(model, "hi", gamma=2.0)
