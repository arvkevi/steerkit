"""Tests for `Probe.score_tokens` and `viz.plot_token_scores`.

Uses a stub model (MagicMock) so tests are fast and don't require a real
HuggingFace download. The goal is to verify the wiring — tokenization,
shape alignment, prompt-vs-response slicing, method dispatch — not the
quality of any particular probe.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from steerkit import PROBE_METHODS, Probe, TokenScores, plot_token_scores

D_MODEL = 8
N_LAYERS = 12
PROBE_LAYER = 3


def _make_probe(d_model: int = D_MODEL, layer: int = PROBE_LAYER) -> Probe:
    rng = torch.Generator().manual_seed(0)
    directions = {}
    for method in PROBE_METHODS:
        v = torch.randn(d_model, generator=rng)
        directions[method] = v / v.norm()
    return Probe(
        directions=directions,
        bias={m: 0.0 for m in PROBE_METHODS},
        layer=layer,
        metrics={},
        model_id="fake/test",
        hook_site="resid_post",
        hook_name=f"blocks.{layer}.hook_resid_post",
        n_total_layers=N_LAYERS,
    )


def _stub_model(
    *,
    full_seq_len: int = 12,
    prompt_len: int = 5,
    d_model: int = D_MODEL,
    hook_name: str = f"blocks.{PROBE_LAYER}.hook_resid_post",
) -> MagicMock:
    """Build a MagicMock model that returns deterministic activations.

    `format_chat(prompt, response=None)` returns:
      - prompt-only sequence of `prompt_len` tokens when response is None
      - full prompt+response sequence of `full_seq_len` tokens otherwise
    """
    model = MagicMock()
    model.model_id = "fake/test"
    model.device = "cpu"
    model.hooked = MagicMock()
    model.hooked.cfg.dtype = torch.float32

    def _format_chat(prompt, response=None):
        seq_len = full_seq_len if response is not None else prompt_len
        return torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    model.format_chat = MagicMock(side_effect=_format_chat)

    # Activations: deterministic per position so different methods produce
    # different token-score patterns.
    g = torch.Generator().manual_seed(42)
    fake_act = torch.randn(1, full_seq_len, d_model, generator=g)

    def _run_with_cache(tokens, names_filter=None):
        used_len = tokens.shape[-1]
        return None, {hook_name: fake_act[:, :used_len, :]}

    model.hooked.run_with_cache = MagicMock(side_effect=_run_with_cache)

    tokenizer = MagicMock()
    tokenizer.decode = MagicMock(side_effect=lambda ids: f"<tok{ids[0]}>")
    model.tokenizer = tokenizer

    return model


# --------------------------------------------------------------------------
# Probe.score_tokens
# --------------------------------------------------------------------------


def test_score_tokens_returns_token_scores():
    probe = _make_probe()
    model = _stub_model()
    out = probe.score_tokens(model, "hi", "there friend")
    assert isinstance(out, TokenScores)


def test_score_tokens_response_only_slices_off_prompt():
    probe = _make_probe()
    model = _stub_model(full_seq_len=12, prompt_len=5)
    out = probe.score_tokens(model, "hi", "there")  # default include_prompt=False
    # 12 total tokens minus 5 prompt tokens = 7 response tokens
    assert len(out.tokens) == 7
    assert out.scores.shape == (7,)
    assert out.response_start == 0


def test_score_tokens_include_prompt_returns_full_sequence():
    probe = _make_probe()
    model = _stub_model(full_seq_len=12, prompt_len=5)
    out = probe.score_tokens(model, "hi", "there", include_prompt=True)
    assert len(out.tokens) == 12
    assert out.scores.shape == (12,)
    assert out.response_start == 5


def test_score_tokens_no_response_scores_prompt_only():
    probe = _make_probe()
    model = _stub_model(full_seq_len=12, prompt_len=5)
    out = probe.score_tokens(model, "hi")
    # No response → score the prompt-only formatting (5 tokens).
    assert len(out.tokens) == 5
    assert out.scores.shape == (5,)
    assert out.response_start == 0


def test_score_tokens_method_dispatch_changes_scores():
    probe = _make_probe()
    model = _stub_model()
    a = probe.score_tokens(model, "hi", "there", method="logistic")
    b = probe.score_tokens(model, "hi", "there", method="diff_of_means")
    # Same shape, same tokens, but the directions differ so the projections should too.
    assert a.scores.shape == b.scores.shape
    assert a.tokens == b.tokens
    assert not torch.allclose(a.scores, b.scores)
    assert a.method == "logistic"
    assert b.method == "diff_of_means"


def test_score_tokens_uses_default_method_when_unspecified():
    probe = _make_probe()
    model = _stub_model()
    out = probe.score_tokens(model, "hi", "there")
    assert out.method == probe.default_method == "logistic"


def test_score_tokens_records_layer():
    probe = _make_probe(layer=7)
    model = _stub_model(hook_name="blocks.7.hook_resid_post")
    # Need to also rewrite the hook_name attribute on probe.
    probe.hook_name = "blocks.7.hook_resid_post"
    out = probe.score_tokens(model, "hi", "there")
    assert out.layer == 7


# --------------------------------------------------------------------------
# TokenScores dataclass
# --------------------------------------------------------------------------


def test_token_scores_validates_shape():
    with pytest.raises(ValueError, match="1-D"):
        TokenScores(tokens=["a"], scores=torch.zeros(1, 1), layer=0, method="logistic")


def test_token_scores_validates_length_match():
    with pytest.raises(ValueError, match="same length"):
        TokenScores(tokens=["a", "b"], scores=torch.tensor([0.1]), layer=0, method="logistic")


def test_token_scores_plot_convenience_returns_figure():
    ts = TokenScores(
        tokens=["a", "b", "c"],
        scores=torch.tensor([0.1, -0.2, 0.5]),
        layer=4,
        method="logistic",
    )
    fig = ts.plot()
    assert fig is not None
    # Should have rendered len(tokens) bars
    ax = fig.axes[0]
    assert len(ax.get_yticklabels()) == 3


# --------------------------------------------------------------------------
# plot_token_scores
# --------------------------------------------------------------------------


def test_plot_token_scores_returns_figure_with_correct_shape():
    ts = TokenScores(
        tokens=["foo", "bar", "baz", "qux"],
        scores=torch.tensor([1.0, -0.5, 0.0, 2.0]),
        layer=5,
        method="diff_of_means",
    )
    fig = plot_token_scores(ts)
    ax = fig.axes[0]
    # As many y-ticks as tokens
    assert len(ax.get_yticklabels()) == 4
    # Title mentions the layer + method
    assert "layer 5" in ax.get_title()
    assert "diff_of_means" in ax.get_title()


def test_plot_token_scores_marks_response_start():
    ts = TokenScores(
        tokens=["sys", "user", "asst", "hi", "there"],
        scores=torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0]),
        layer=2,
        method="logistic",
        response_start=3,
    )
    fig = plot_token_scores(ts, mark_response_start=True)
    ax = fig.axes[0]
    # axhline + axvline both produce Line2D artifacts; we expect ≥2.
    lines = ax.get_lines()
    assert len(lines) >= 2


def test_plot_token_scores_mark_response_start_false_skips_divider():
    ts = TokenScores(
        tokens=["a", "b", "c"],
        scores=torch.tensor([0.1, 0.2, 0.3]),
        layer=1,
        method="logistic",
        response_start=1,
    )
    fig_with = plot_token_scores(ts, mark_response_start=True)
    fig_without = plot_token_scores(ts, mark_response_start=False)
    # The version without the divider should have one fewer line.
    n_with = len(fig_with.axes[0].get_lines())
    n_without = len(fig_without.axes[0].get_lines())
    assert n_with == n_without + 1
