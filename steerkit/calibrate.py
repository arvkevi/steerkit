"""Auto-α calibration for steering strength.

Sweep α candidates on a small calibration set; pick the largest α at which the
*steered* output's perplexity (under the unsteered model — "how surprised does
the model itself look at what it just produced?") stays within a ratio of the
*unsteered* baseline. The intent is a coherence ceiling: bigger α gives more
concept push, but past some threshold the model goes out of distribution and
generates whitespace, repetition, or gibberish.

Cost: 2*K*M generations (K = candidate count, M = calibration prompts) plus
2*K*M short forward passes for perplexity. Default K=5, M=4 → ~40 generations
of 30 tokens — seconds on a small model.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .models import ModelHandle
from .probe import Probe

DEFAULT_ALPHA_CANDIDATES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_CALIBRATION_PROMPTS: tuple[str, ...] = (
    "Tell me about your day.",
    "What's a good way to spend a Sunday afternoon?",
    "Describe your ideal weekend.",
    "What's the best book you've read recently?",
)


@torch.no_grad()
def _response_perplexity(
    model: ModelHandle, prompt_ids: torch.Tensor, response_ids: torch.Tensor
) -> float:
    """Average per-token perplexity of `response_ids` conditioned on `prompt_ids`,
    computed under the unsteered model. Skips no-op cases (response of length 0).
    """
    response_ids = response_ids.to(prompt_ids.device)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    if response_ids.shape[-1] == 0:
        return math.inf
    full = torch.cat([prompt_ids, response_ids], dim=-1)
    logits = model.hooked(full)  # [1, T, vocab]
    P = prompt_ids.shape[-1]
    R = response_ids.shape[-1]
    target_logits = logits[0, P - 1 : P - 1 + R, :]  # [R, vocab]
    log_probs = F.log_softmax(target_logits, dim=-1)
    response_log_probs = log_probs.gather(-1, response_ids[0].view(-1, 1)).squeeze(-1)
    avg_nll = -response_log_probs.mean().item()
    return float(math.exp(min(avg_nll, 50)))  # clamp to avoid overflow


@torch.no_grad()
def _generate(
    model: ModelHandle,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    fwd_hooks: list | None = None,
) -> torch.Tensor:
    """Greedy generation, optionally with steering hooks installed. Returns response-only token ids."""
    if fwd_hooks is None:
        fwd_hooks = []
    with model.hooked.hooks(fwd_hooks=fwd_hooks):
        output = model.hooked.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            verbose=False,
        )
    assert isinstance(output, torch.Tensor)
    return output[0, prompt_ids.shape[-1]:]


def calibrate_alpha(
    probe: Probe,
    model: ModelHandle,
    *,
    prompts: list[str] | None = None,
    candidates: list[float] | tuple[float, ...] = DEFAULT_ALPHA_CANDIDATES,
    perplexity_ratio_max: float = 1.5,
    max_new_tokens: int = 30,
    method: str | None = None,
    attach: bool = True,
) -> tuple[float, dict[float, float]]:
    """Pick the largest α whose steered-output perplexity is within ratio_max of unsteered.

    Args:
        probe: a fitted Probe to calibrate.
        model: the model to steer.
        prompts: small calibration set (default: 4 generic prompts).
        candidates: α values to sweep, in increasing order is fine but we sort.
        perplexity_ratio_max: ceiling on steered_ppl / baseline_ppl.
        max_new_tokens: response length per generation.
        method: which probe direction to use (default: probe.default_method).
        attach: if True, set probe.auto_alpha = chosen value before returning.

    Returns:
        (best_alpha, ratios): best alpha and the full {alpha: perplexity_ratio} mapping.
        best_alpha is 0.0 if no candidate satisfies the constraint (steering destroys coherence
        for every α tried).
    """
    if prompts is None:
        prompts = list(DEFAULT_CALIBRATION_PROMPTS)
    candidates = sorted(set(candidates))

    # Pre-LayerNorm transformers accumulate residual-stream norm through depth, so a
    # candidate set sized for a normalized layer (~25 norm) is invisible at an
    # unnormalized middle layer (~5000 norm). Scale α candidates by the layer's
    # mean activation norm if it's recorded on the probe (Probe.fit_all stores it
    # as `metrics["activation_norm_mean"]`). Falls back to the literal candidates
    # for back-compat if the probe predates this metric.
    norm_scale = probe.metrics.get("activation_norm_mean", 1.0)
    candidates = [a * norm_scale / 25.0 for a in candidates]
    candidates = sorted(set(candidates))

    direction = probe.get_direction(method).to(model.device).to(model.hooked.cfg.dtype)
    target_hook = probe.hook_name

    # Baseline: unsteered perplexity of unsteered generation, averaged over prompts.
    baseline_ppls: list[float] = []
    for prompt in prompts:
        prompt_ids = model.format_chat(prompt)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        response_ids = _generate(model, prompt_ids, max_new_tokens, fwd_hooks=[])
        baseline_ppls.append(_response_perplexity(model, prompt_ids, response_ids))
    baseline = float(np.mean(baseline_ppls))
    if not math.isfinite(baseline) or baseline <= 0:
        if attach:
            probe.auto_alpha = 0.0
        return 0.0, {a: math.inf for a in candidates}

    ratios: dict[float, float] = {}
    best_alpha = 0.0
    for alpha in candidates:

        def steering_hook(activation, hook, _alpha=alpha):  # noqa: ARG001 - hook signature
            return activation + _alpha * direction

        steered_ppls: list[float] = []
        for prompt in prompts:
            prompt_ids = model.format_chat(prompt)
            if prompt_ids.dim() == 1:
                prompt_ids = prompt_ids.unsqueeze(0)
            response_ids = _generate(
                model, prompt_ids, max_new_tokens, fwd_hooks=[(target_hook, steering_hook)]
            )
            steered_ppls.append(_response_perplexity(model, prompt_ids, response_ids))
        steered = float(np.mean(steered_ppls))
        ratio = steered / baseline if baseline > 0 else math.inf
        ratios[alpha] = ratio
        if ratio <= perplexity_ratio_max:
            best_alpha = alpha  # keep updating; we want the LARGEST α that satisfies

    if attach:
        probe.auto_alpha = best_alpha
    return best_alpha, ratios
