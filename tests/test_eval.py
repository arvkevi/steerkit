"""Tests for the LLM-judge expensive tier (evaluate_steering_effect).

These tests use a mock teacher and a stub generation path, so no real model is
needed. The focus is correctness of the orchestration: top-K narrowing, score
attachment, and graceful handling of bad judge outputs.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from steerkit import PROBE_METHODS, Probe, TeacherModel, evaluate_steering_effect
from steerkit.eval import _parse_rating


class FakeRatingTeacher(TeacherModel):
    """Maps each generated response to a rating drawn deterministically from a queue."""

    def __init__(self, ratings: list[int | str]):
        self._ratings = list(ratings)
        self.call_count = 0

    @property
    def identifier(self) -> str:
        return "fake-rating:test"

    def complete(self, system, user, *, max_tokens=8, temperature=0.0):
        self.call_count += 1
        if not self._ratings:
            return "1"
        return str(self._ratings.pop(0))


def _make_probe(layer: int, *, auc: float = 0.9, d_model: int = 16) -> Probe:
    rng = torch.Generator().manual_seed(layer + 1)
    directions = {}
    for method in PROBE_METHODS:
        v = torch.randn(d_model, generator=rng)
        directions[method] = v / v.norm()
    return Probe(
        directions=directions,
        bias={m: 0.0 for m in PROBE_METHODS},
        layer=layer,
        metrics={
            "auc_test_logistic": auc,
            "auc_test_diff_of_means": auc,
            "auc_test_mass_mean": auc,
            "cohens_d_logistic": 1.0,
        },
        model_id="fake/test",
        hook_site="resid_post",
        hook_name=f"blocks.{layer}.hook_resid_post",
        n_total_layers=12,
    )


def test_parse_rating_basic():
    assert _parse_rating("5") == 5.0
    assert _parse_rating("Rating: 7") == 7.0
    assert _parse_rating("I'd rate this 3, not higher.") == 3.0
    assert _parse_rating("garbage") is None
    # Out-of-range integers are not matched.
    assert _parse_rating("9") is None
    assert _parse_rating("0") is None


def test_evaluate_top_k_narrowing():
    """Only the top_k probes by `by` should be evaluated; others get no steering_effect entry."""
    probes = {
        0: _make_probe(0, auc=0.6),
        1: _make_probe(1, auc=0.95),  # best
        2: _make_probe(2, auc=0.8),
        3: _make_probe(3, auc=0.7),
    }
    # 2 prompts × top 2 layers = 4 ratings
    teacher = FakeRatingTeacher([6, 5, 3, 4])
    # Stub the actual model generation so we don't need a real ModelHandle.
    with patch("steerkit.eval._generate_steered", return_value="dummy steered response"):
        results = evaluate_steering_effect(
            probes,
            model=None,  # type: ignore[arg-type] — patched out
            teacher=teacher,
            concept_description="formality",
            eval_prompts=["p1", "p2"],
            top_k=2,
        )
    assert set(results.keys()) == {1, 2}, "should evaluate the 2 highest-AUC layers only"
    assert results[1] == 5.5  # mean(6, 5)
    assert results[2] == 3.5  # mean(3, 4)
    assert teacher.call_count == 4
    assert "steering_effect" in probes[1].metrics
    assert probes[1].metrics["steering_effect"] == 5.5
    assert probes[1].metrics["steering_effect_n"] == 2.0
    assert "steering_effect" not in probes[0].metrics
    assert "steering_effect" not in probes[3].metrics


def test_evaluate_handles_bad_judge_outputs_gracefully():
    """If the judge can't return a parseable rating, the probe should still get a score
    (averaged over successful ratings only) and the failure callback should fire."""
    probes = {0: _make_probe(0, auc=0.9)}
    # 3 prompts → 3 calls; first is unparseable, second is 4, third is 6.
    teacher = FakeRatingTeacher(["lol no idea", 4, 6])
    failures: list[tuple[int, str, str]] = []
    with patch("steerkit.eval._generate_steered", return_value="dummy"):
        results = evaluate_steering_effect(
            probes,
            model=None,  # type: ignore[arg-type]
            teacher=teacher,
            concept_description="x",
            eval_prompts=["p1", "p2", "p3"],
            top_k=1,
            on_failure=lambda layer, r, t: failures.append((layer, r, t)),
        )
    assert results[0] == 5.0  # mean(4, 6)
    assert probes[0].metrics["steering_effect_n"] == 2.0
    assert len(failures) == 1
    assert failures[0][0] == 0


def test_evaluate_no_attach():
    probes = {0: _make_probe(0, auc=0.9)}
    teacher = FakeRatingTeacher([5])
    with patch("steerkit.eval._generate_steered", return_value="dummy"):
        results = evaluate_steering_effect(
            probes,
            model=None,  # type: ignore[arg-type]
            teacher=teacher,
            concept_description="x",
            eval_prompts=["p1"],
            top_k=1,
            attach=False,
        )
    assert results[0] == 5.0
    assert "steering_effect" not in probes[0].metrics
