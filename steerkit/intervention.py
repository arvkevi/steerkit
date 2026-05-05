"""Four steering operations on residual-stream activations.

These are pure tensor functions; the steering hook installed by `Probe.steer`
is a thin closure over one of them. Each op interprets a probe's unit-direction
vector differently:

  addition       — push toward / away from the concept; α controls magnitude.
  projection     — ablate the concept-component entirely (α has no effect).
  clamp          — force the projection onto the direction to a target value.
  multiplicative — scale whatever signal is already there; γ=2 doubles it,
                   γ=0 ablates (equivalent to projection), γ=−1 reverses sign.

All four assume `direction` is unit-normalized (steerkit's fitters guarantee
this) and shape `[d_model]`. `activation` is `[batch, seq, d_model]`. They all
return a tensor of the same shape as the input activation.
"""

from __future__ import annotations

from typing import Literal

import torch

Operation = Literal["addition", "projection", "clamp", "multiplicative"]
OPERATIONS: tuple[str, ...] = ("addition", "projection", "clamp", "multiplicative")


def apply_addition(activation: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    """`act ← act + α·v`. The standard CAA / repeng-style steering."""
    return activation + alpha * direction


def apply_projection(activation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """`act ← act − (act·v̂)v̂`. Removes any component along `direction`.

    Useful when you want to *neutralize* the concept rather than push toward it
    (e.g., "make the model produce neither overtly refusing nor overtly compliant
    responses, just whatever else is in there").
    """
    proj_scalar = (activation * direction).sum(dim=-1, keepdim=True)
    return activation - proj_scalar * direction


def apply_clamp(activation: torch.Tensor, direction: torch.Tensor, target: float) -> torch.Tensor:
    """`act ← act + (target − act·v̂)v̂`. Forces the projection onto `direction` to
    equal `target` exactly, while leaving the orthogonal component untouched.

    More predictable than addition because it's invariant to whatever signal was
    already there: a positive `target` produces a fixed concept strength rather
    than additive perturbation.
    """
    current = (activation * direction).sum(dim=-1, keepdim=True)
    return activation + (target - current) * direction


def apply_multiplicative(
    activation: torch.Tensor, direction: torch.Tensor, gamma: float
) -> torch.Tensor:
    """`act ← act + (γ−1)(act·v̂)v̂`. Scales the existing component along `direction`
    by `γ` (γ=1 is no-op, γ=0 is equivalent to projection, γ>1 amplifies, γ<0 reverses).
    """
    current = (activation * direction).sum(dim=-1, keepdim=True)
    return activation + (gamma - 1.0) * current * direction


def make_hook(
    op: Operation,
    direction: torch.Tensor,
    *,
    alpha: float | None = None,
    target: float | None = None,
    gamma: float | None = None,
):
    """Return a TL-compatible forward hook closure that applies the given op.

    Required parameter per op:
      addition       → alpha
      projection     → (none)
      clamp          → target
      multiplicative → gamma
    """
    if op == "addition":
        if alpha is None:
            raise ValueError("op='addition' requires alpha")
        a = float(alpha)

        def addition_hook(activation, hook):  # noqa: ARG001
            return apply_addition(activation, direction, a)

        return addition_hook
    if op == "projection":

        def projection_hook(activation, hook):  # noqa: ARG001
            return apply_projection(activation, direction)

        return projection_hook
    if op == "clamp":
        if target is None:
            raise ValueError("op='clamp' requires target")
        t = float(target)

        def clamp_hook(activation, hook):  # noqa: ARG001
            return apply_clamp(activation, direction, t)

        return clamp_hook
    if op == "multiplicative":
        if gamma is None:
            raise ValueError("op='multiplicative' requires gamma")
        g = float(gamma)

        def multiplicative_hook(activation, hook):  # noqa: ARG001
            return apply_multiplicative(activation, direction, g)

        return multiplicative_hook
    raise ValueError(f"unknown op {op!r}; choose one of {OPERATIONS}")
