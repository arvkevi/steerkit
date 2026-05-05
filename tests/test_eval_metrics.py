"""Tests for the new eval metrics: logit-lens vocab match, perplexity ratio,
external classifier, and the evaluate_probe aggregator. All use stub models /
mocks — no real model loads.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from steerkit import (
    PROBE_METHODS,
    EvalReport,
    Probe,
    evaluate_probe,
    external_classifier_score,
    logit_lens_vocab_score,
    perplexity_ratio,
)


def _make_probe(layer: int = 3, d_model: int = 16, auto_alpha: float | None = 2.0) -> Probe:
    rng = torch.Generator().manual_seed(layer + 1)
    directions = {}
    for method in PROBE_METHODS:
        v = torch.randn(d_model, generator=rng)
        directions[method] = v / v.norm()
    return Probe(
        directions=directions,
        bias={m: 0.0 for m in PROBE_METHODS},
        layer=layer,
        metrics={"auc_test_logistic": 0.93, "cohens_d_logistic": 1.4},
        model_id="fake/test",
        hook_site="resid_post",
        hook_name=f"blocks.{layer}.hook_resid_post",
        n_total_layers=12,
        auto_alpha=auto_alpha,
    )


def _stub_model(d_model: int = 16, vocab_size: int = 50) -> MagicMock:
    """Minimal duck-type for ModelHandle covering the bits eval calls."""
    model = MagicMock()
    model.device = "cpu"
    model.hooked = MagicMock()
    model.hooked.cfg.dtype = torch.float32
    # W_U for logit-lens: we'll set this so direction @ W_U gives a controllable top-K
    model.hooked.W_U = torch.zeros(d_model, vocab_size)
    return model


def test_logit_lens_vocab_score_perfect_match():
    """Set up the unembed so the direction's top-1 is a known token."""
    probe = _make_probe(d_model=16)
    model = _stub_model(d_model=16, vocab_size=10)
    # Make logit at vocab index 0 exactly equal to the unit-direction's first component.
    # i.e. direction @ W_U gives logit[i] = direction · W_U[:, i]; setting W_U[:, 0] = direction
    # ensures index 0 has the highest logit.
    direction = probe.get_direction()
    model.hooked.W_U = torch.zeros(16, 10)
    model.hooked.W_U[:, 0] = direction
    model.hooked.W_U[:, 1] = direction * 0.9
    model.hooked.W_U[:, 2] = direction * 0.8

    tokenizer = MagicMock()
    tokenizer.decode = MagicMock(side_effect=lambda ids: f"tok_{int(ids[0])}")
    model.tokenizer = tokenizer

    result = logit_lens_vocab_score(probe, model, target_vocab={"tok_0", "tok_1"}, top_k=5)
    assert "tok_0" in result["matches"]
    assert "tok_1" in result["matches"]
    assert result["fraction"] == 2 / 5


def test_logit_lens_vocab_score_no_match():
    probe = _make_probe(d_model=16)
    model = _stub_model(d_model=16, vocab_size=10)
    tokenizer = MagicMock()
    tokenizer.decode = MagicMock(side_effect=lambda ids: f"tok_{int(ids[0])}")
    model.tokenizer = tokenizer

    result = logit_lens_vocab_score(
        probe, model, target_vocab={"banana", "elephant"}, top_k=5
    )
    assert result["matches"] == []
    assert result["fraction"] == 0.0
    assert len(result["top_k_tokens"]) == 5


def test_logit_lens_vocab_score_case_insensitive():
    probe = _make_probe(d_model=16)
    model = _stub_model(d_model=16, vocab_size=10)
    direction = probe.get_direction()
    model.hooked.W_U = torch.zeros(16, 10)
    model.hooked.W_U[:, 0] = direction
    tokenizer = MagicMock()
    tokenizer.decode = MagicMock(side_effect=lambda ids: "  Joy  " if int(ids[0]) == 0 else "x")
    model.tokenizer = tokenizer
    result = logit_lens_vocab_score(probe, model, target_vocab={"joy"}, top_k=3)
    assert "joy" in result["matches"]


def test_perplexity_ratio_uses_auto_alpha_by_default():
    """Patch the calibrate-side helpers so we can exercise the API without a real model."""
    from unittest.mock import patch

    probe = _make_probe(auto_alpha=4.0)
    model = _stub_model()
    model.format_chat = MagicMock(return_value=torch.tensor([[1, 2, 3]]))

    fake_response = torch.tensor([10, 11, 12])
    with patch("steerkit.calibrate._generate", return_value=fake_response), patch(
        "steerkit.calibrate._response_perplexity",
        side_effect=[10.0, 12.0, 11.0, 14.0],  # baseline x2, steered x2 across 2 prompts
    ):
        result = perplexity_ratio(probe, model, prompts=["a", "b"])
    assert result["alpha"] == 4.0
    assert result["baseline_perplexity"] == 10.5  # mean(10, 11)
    assert result["steered_perplexity"] == 13.0  # mean(12, 14)
    assert abs(result["ratio"] - 13.0 / 10.5) < 1e-6
    assert result["n_prompts"] == 2


def test_external_classifier_score_records_shift():
    from unittest.mock import patch

    probe = _make_probe(auto_alpha=2.0)
    model = _stub_model()
    model.format_chat = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    model.tokenizer = MagicMock()
    model.tokenizer.decode = MagicMock(side_effect=lambda t, **k: "decoded")

    # Classifier returns +1 for steered, 0 for unsteered (we'll alternate via _generate calls).
    call_count = {"i": 0}

    def fake_generate(*a, fwd_hooks=None, **k):
        call_count["i"] += 1
        return torch.tensor([10, 11])

    def classifier(text: str) -> float:
        # Alternates baseline=0, steered=1, baseline=0, steered=1
        return float(call_count["i"] % 2)

    with patch("steerkit.calibrate._generate", side_effect=fake_generate):
        result = external_classifier_score(
            probe, model, prompts=["a", "b"], classifier=classifier
        )
    assert result["alpha"] == 2.0
    assert "shift" in result
    # baseline calls happen on odd i, steered on even i (per the alternating logic above)
    assert len(result["baseline_scores"]) == 2
    assert len(result["steered_scores"]) == 2


def test_evaluate_probe_returns_cheap_only_without_model():
    probe = _make_probe()
    report = evaluate_probe(probe)
    assert isinstance(report, EvalReport)
    assert report.cheap == probe.metrics
    assert report.logit_lens is None
    assert report.perplexity is None
    assert report.classifier is None


def test_evaluate_probe_runs_logit_lens_when_target_vocab_supplied():
    probe = _make_probe()
    model = _stub_model()
    direction = probe.get_direction()
    model.hooked.W_U = torch.zeros(16, 10)
    model.hooked.W_U[:, 0] = direction
    tokenizer = MagicMock()
    tokenizer.decode = MagicMock(side_effect=lambda ids: f"tok_{int(ids[0])}")
    model.tokenizer = tokenizer

    report = evaluate_probe(probe, model, target_vocab={"tok_0"}, top_k_logit_lens=3)
    assert report.logit_lens is not None
    assert "tok_0" in report.logit_lens["matches"]
    # Cheap metrics still passed through.
    assert "auc_test_logistic" in report.cheap


def test_eval_report_summary_includes_supplied_metrics():
    report = EvalReport(
        cheap={"auc_test_logistic": 0.93, "cohens_d_logistic": 1.4},
        logit_lens={"fraction": 0.4, "matches": [], "top_k_tokens": []},
        perplexity={"alpha": 2.0, "ratio": 1.2, "baseline_perplexity": 10, "steered_perplexity": 12, "n_prompts": 4},
        classifier={"shift": 0.5, "baseline_mean": 0.0, "steered_mean": 0.5, "baseline_scores": [], "steered_scores": [], "alpha": 2.0},
    )
    s = report.summary()
    assert "auc_test_logistic" in s
    assert "vocab_match" in s
    assert "ppl_ratio" in s
    assert "clf_shift" in s
