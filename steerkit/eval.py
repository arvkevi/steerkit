"""Steering-effect evaluation: a small library of probe-quality metrics.

The cheap-tier (held-out AUC + Cohen's d) lives on `Probe.metrics`. This module
adds higher-cost metrics that involve actual generation or model-side scoring,
and an aggregator `evaluate_probe(...)` that runs whichever of them you choose.

Public surface:

  evaluate_steering_effect(probes, model, teacher, ...)
      LLM-judge expensive tier — narrow to top-K layers and ask the teacher to
      rate steered completions on a 1-7 scale. Attaches `steering_effect`.

  logit_lens_vocab_score(probe, model, target_vocab, top_k=20)
      Cheapest interpretability check. Score = fraction of the top-K tokens
      promoted by the direction that match a user-supplied vocabulary set.

  perplexity_ratio(probe, model, prompts, alpha)
      Ratio of steered-output perplexity to unsteered baseline, under the
      unsteered model. Same metric `calibrate_alpha` uses for its ceiling
      check, exposed standalone.

  external_classifier_score(probe, model, prompts, classifier, alpha)
      Generates unsteered + steered completions on each prompt, hands each
      to a user-supplied classifier function, returns mean shift.

  evaluate_probe(probe, model, ..., target_vocab=, classifier=)
      One-shot aggregator that runs the cheap-tier metrics from
      `probe.metrics` plus any of the above the caller has supplied inputs
      for. Returns an EvalReport.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .models import ModelHandle
from .probe import Probe
from .teacher import TeacherModel

DEFAULT_EVAL_PROMPTS: tuple[str, ...] = (
    "Tell me about your morning routine.",
    "What's a good way to spend a Sunday afternoon?",
    "Recommend a book for a long flight.",
    "Describe a memorable meal.",
    "What's a hobby you'd suggest to someone new in town?",
    "Tell me about a movie you enjoyed recently.",
    "What's the best advice you've received?",
    "Describe your ideal weekend.",
)

JUDGE_SYSTEM_TEMPLATE = """You are an evaluator. Rate how strongly the assistant response exhibits the given concept.

Scale:
1 = not at all
2 = barely
3 = slightly
4 = moderately
5 = clearly
6 = strongly
7 = overwhelmingly

Output ONLY a single integer 1-7. No reasoning, no commentary, no formatting."""

JUDGE_USER_TEMPLATE = """Concept: {description}

Response: {response}

Rating (1-7):"""

_RATING_RE = re.compile(r"\b([1-7])\b")


def _parse_rating(text: str) -> float | None:
    """Extract a 1-7 integer rating from the teacher's output. Returns None on failure."""
    match = _RATING_RE.search(text)
    if match is None:
        return None
    return float(match.group(1))


@torch.no_grad()
def _generate_steered(
    probe: Probe,
    model: ModelHandle,
    prompt: str,
    *,
    alpha: float,
    method: str | None,
    max_new_tokens: int,
) -> str:
    direction = probe.get_direction(method).to(model.device).to(model.hooked.cfg.dtype)

    def steering_hook(activation, hook):  # noqa: ARG001 — TL hook signature
        return activation + alpha * direction

    tokens = model.format_chat(prompt)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    with model.hooked.hooks(fwd_hooks=[(probe.hook_name, steering_hook)]):
        output_ids = model.hooked.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            verbose=False,
        )
    assert isinstance(output_ids, torch.Tensor)
    new_tokens = output_ids[0, tokens.shape[-1]:]
    tokenizer = model.tokenizer
    assert tokenizer is not None
    return str(tokenizer.decode(new_tokens, skip_special_tokens=True))


def evaluate_steering_effect(
    probes: dict[int, Probe],
    model: ModelHandle,
    teacher: TeacherModel,
    *,
    concept_description: str,
    eval_prompts: list[str] | None = None,
    top_k: int = 5,
    by: str = "auc_test_logistic",
    alpha: float = 4.0,
    max_new_tokens: int = 60,
    method: str | None = None,
    on_failure: Callable[[int, str, str], None] | None = None,
    attach: bool = True,
) -> dict[int, float]:
    """Score steering effect size for the top-K probes by a cheap-tier metric.

    Args:
        probes: a dict of layer -> Probe (e.g. from Probe.fit_all).
        model: the model to steer.
        teacher: the TeacherModel to use as judge (often the same as the generator).
        concept_description: short description of the concept (e.g. "refusal", "verbose, expansive language").
        eval_prompts: list of evaluation prompts; defaults to a small bundled set.
        top_k: how many top layers (by `by`) to evaluate. Use a number >= len(probes) to evaluate all.
        by: cheap-tier metric for narrowing. Falls back through train metrics if missing.
        alpha: steering strength during evaluation.
        max_new_tokens: response length per generation.
        method: which probe direction to use (defaults to each probe's default_method).
        on_failure: optional callback (layer, response, raw_judge_text) when rating parsing fails.
        attach: if True, write `metrics["steering_effect"]` and `metrics["steering_effect_n"]`
            on each evaluated probe in place.

    Returns a dict mapping layer index -> mean steering-effect score.
    Layers not in the top-K are not evaluated and not present in the return.
    """
    if eval_prompts is None:
        eval_prompts = list(DEFAULT_EVAL_PROMPTS)

    sample = next(iter(probes.values()))
    metric_key = by
    if metric_key not in sample.metrics:
        for fallback in ("auc_train_logistic", "auc_train_diff_of_means", "auc_train_mass_mean"):
            if fallback in sample.metrics:
                metric_key = fallback
                break
        else:
            raise KeyError(f"no usable cheap-tier metric on probes; tried {by!r} and train fallbacks")

    ranked = sorted(probes.items(), key=lambda kv: -kv[1].metrics[metric_key])
    chosen = ranked[: max(1, top_k)]

    judge_system = JUDGE_SYSTEM_TEMPLATE
    results: dict[int, float] = {}
    for layer, probe in chosen:
        ratings: list[float] = []
        for prompt in eval_prompts:
            try:
                response = _generate_steered(
                    probe, model, prompt, alpha=alpha, method=method, max_new_tokens=max_new_tokens
                )
            except Exception as e:  # noqa: BLE001
                if on_failure is not None:
                    on_failure(layer, prompt, f"<generate error: {type(e).__name__}: {e}>")
                continue

            user = JUDGE_USER_TEMPLATE.format(
                description=concept_description, response=response.strip() or "<empty>"
            )
            try:
                judge_text = teacher.complete(judge_system, user, max_tokens=8, temperature=0.0)
            except Exception as e:  # noqa: BLE001
                if on_failure is not None:
                    on_failure(layer, response, f"<judge error: {type(e).__name__}: {e}>")
                continue
            rating = _parse_rating(judge_text)
            if rating is None:
                if on_failure is not None:
                    on_failure(layer, response, judge_text)
                continue
            ratings.append(rating)

        score = math.nan if not ratings else statistics.fmean(ratings)
        results[layer] = score
        if attach:
            probe.metrics["steering_effect"] = score
            probe.metrics["steering_effect_n"] = float(len(ratings))

    return results


# ---- Logit-lens vocab match ---------------------------------------------------


@torch.no_grad()
def logit_lens_vocab_score(
    probe: Probe,
    model: ModelHandle,
    target_vocab: set[str] | list[str],
    *,
    top_k: int = 20,
    method: str | None = None,
    case_insensitive: bool = True,
    strip: bool = True,
) -> dict:
    """Score a probe direction by how many of its top-K vocabulary tokens are in
    `target_vocab`. The cheapest interpretability sanity check.

    A "joy" direction whose top tokens are happy/joyful/exuberant scores ~1.0;
    a broken probe whose top tokens are random subwords scores ~0.0.

    Returns a dict with `fraction` (0..1), `matches` (list of matching tokens),
    and `top_k_tokens` (the full top-K list, useful for debugging).
    """
    direction = probe.get_direction(method).to(model.device).to(model.hooked.cfg.dtype)
    W_U = model.hooked.W_U  # [d_model, vocab]
    logits = direction @ W_U
    _vals, top_idx = logits.topk(top_k)
    tokenizer = model.tokenizer
    assert tokenizer is not None
    raw_tokens = [str(tokenizer.decode([int(i.item())])) for i in top_idx]

    def _norm(t: str) -> str:
        return (t.strip() if strip else t).lower() if case_insensitive else (t.strip() if strip else t)

    target_set = {_norm(t) for t in target_vocab}
    normalized = [_norm(t) for t in raw_tokens]
    matches = [n for n in normalized if n in target_set]
    return {
        "fraction": len(matches) / max(top_k, 1),
        "matches": matches,
        "top_k_tokens": raw_tokens,
    }


# ---- Perplexity ratio (standalone, same metric used by calibrate_alpha) -------


@torch.no_grad()
def perplexity_ratio(
    probe: Probe,
    model: ModelHandle,
    prompts: list[str],
    *,
    alpha: float | None = None,
    method: str | None = None,
    max_new_tokens: int = 30,
) -> dict:
    """Compute the ratio `mean(steered ppl) / mean(unsteered ppl)` averaged over
    the given prompts, with the steered output evaluated under the unsteered
    forward pass.

    Same metric `calibrate_alpha` uses for its ceiling check, exposed standalone
    so you can ask "at this alpha, how much coherence does steering cost?"
    without rerunning the calibration sweep. `alpha=None` uses `probe.auto_alpha`.
    """
    from .calibrate import _generate, _response_perplexity

    if alpha is None:
        alpha = probe.auto_alpha if probe.auto_alpha is not None else 2.0

    direction = probe.get_direction(method).to(model.device).to(model.hooked.cfg.dtype)

    def steering_hook(activation, hook):  # noqa: ARG001
        return activation + alpha * direction

    baseline_ppls: list[float] = []
    steered_ppls: list[float] = []
    for prompt in prompts:
        prompt_ids = model.format_chat(prompt)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        unsteered_response = _generate(model, prompt_ids, max_new_tokens, fwd_hooks=[])
        baseline_ppls.append(_response_perplexity(model, prompt_ids, unsteered_response))
        steered_response = _generate(
            model, prompt_ids, max_new_tokens, fwd_hooks=[(probe.hook_name, steering_hook)]
        )
        steered_ppls.append(_response_perplexity(model, prompt_ids, steered_response))

    baseline = statistics.fmean(baseline_ppls)
    steered = statistics.fmean(steered_ppls)
    ratio = steered / baseline if baseline > 0 else math.inf
    return {
        "alpha": float(alpha),
        "ratio": ratio,
        "baseline_perplexity": baseline,
        "steered_perplexity": steered,
        "n_prompts": len(prompts),
    }


# ---- External classifier (steer-then-classify) --------------------------------


@torch.no_grad()
def external_classifier_score(
    probe: Probe,
    model: ModelHandle,
    prompts: list[str],
    classifier: Callable[[str], float],
    *,
    alpha: float | None = None,
    method: str | None = None,
    max_new_tokens: int = 60,
) -> dict:
    """Generate unsteered + steered completions on each prompt; pass each to a
    user-supplied `classifier` (any callable returning a float — sentiment score,
    refusal probability, length, anything) and report the mean shift.

    Returns `{shift, baseline_mean, steered_mean, baseline_scores, steered_scores}`.
    A positive shift means steering moves outputs in the classifier's positive
    direction; sign and magnitude depend on the classifier's convention.
    """
    from .calibrate import _generate

    if alpha is None:
        alpha = probe.auto_alpha if probe.auto_alpha is not None else 2.0

    direction = probe.get_direction(method).to(model.device).to(model.hooked.cfg.dtype)

    def steering_hook(activation, hook):  # noqa: ARG001
        return activation + alpha * direction

    tokenizer = model.tokenizer
    assert tokenizer is not None
    baseline_scores: list[float] = []
    steered_scores: list[float] = []
    for prompt in prompts:
        prompt_ids = model.format_chat(prompt)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        u = _generate(model, prompt_ids, max_new_tokens, fwd_hooks=[])
        s = _generate(model, prompt_ids, max_new_tokens, fwd_hooks=[(probe.hook_name, steering_hook)])
        baseline_scores.append(float(classifier(str(tokenizer.decode(u, skip_special_tokens=True)))))
        steered_scores.append(float(classifier(str(tokenizer.decode(s, skip_special_tokens=True)))))

    bm = statistics.fmean(baseline_scores)
    sm = statistics.fmean(steered_scores)
    return {
        "shift": sm - bm,
        "baseline_mean": bm,
        "steered_mean": sm,
        "baseline_scores": baseline_scores,
        "steered_scores": steered_scores,
        "alpha": float(alpha),
    }


# ---- One-shot aggregator ------------------------------------------------------


@dataclass
class EvalReport:
    """Aggregated result of `evaluate_probe`. Each field is None when the
    corresponding metric wasn't requested or its inputs weren't supplied."""

    cheap: dict[str, float] = field(default_factory=dict)
    logit_lens: dict | None = None
    perplexity: dict | None = None
    classifier: dict | None = None

    def summary(self) -> str:
        """Human-readable single-line summary of the most informative metrics."""
        parts: list[str] = []
        if self.cheap:
            for k in ("auc_test_logistic", "cohens_d_logistic"):
                if k in self.cheap:
                    parts.append(f"{k}={self.cheap[k]:.3f}")
        if self.logit_lens is not None:
            parts.append(f"vocab_match={self.logit_lens['fraction']:.2f}")
        if self.perplexity is not None:
            parts.append(f"ppl_ratio={self.perplexity['ratio']:.2f}")
        if self.classifier is not None:
            parts.append(f"clf_shift={self.classifier['shift']:+.3f}")
        return " | ".join(parts) or "(no metrics)"


def evaluate_probe(
    probe: Probe,
    model: ModelHandle | None = None,
    *,
    target_vocab: set[str] | list[str] | None = None,
    perplexity_prompts: list[str] | None = None,
    classifier_prompts: list[str] | None = None,
    classifier: Callable[[str], float] | None = None,
    alpha: float | None = None,
    method: str | None = None,
    top_k_logit_lens: int = 20,
    max_new_tokens: int = 30,
) -> EvalReport:
    """Run all probe-quality metrics for which inputs are supplied.

    Always pulls cheap-tier metrics from `probe.metrics`. The other three
    metrics each require their own inputs:

      * `target_vocab` triggers `logit_lens_vocab_score` (needs `model`).
      * `perplexity_prompts` triggers `perplexity_ratio` (needs `model`).
      * `classifier_prompts` + `classifier` triggers `external_classifier_score`
        (needs `model`).

    Returns an EvalReport. None of the metrics raise on partial input — they
    just don't run. Use `report.summary()` for a one-line readout.
    """
    report = EvalReport(cheap=dict(probe.metrics))

    if model is None:
        return report

    if target_vocab is not None:
        report.logit_lens = logit_lens_vocab_score(
            probe, model, target_vocab, top_k=top_k_logit_lens, method=method
        )
    if perplexity_prompts is not None:
        report.perplexity = perplexity_ratio(
            probe, model, perplexity_prompts, alpha=alpha, method=method, max_new_tokens=max_new_tokens
        )
    if classifier_prompts is not None and classifier is not None:
        report.classifier = external_classifier_score(
            probe,
            model,
            classifier_prompts,
            classifier,
            alpha=alpha,
            method=method,
            max_new_tokens=max_new_tokens,
        )

    return report
