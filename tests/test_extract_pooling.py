"""Fast unit tests for pooling strategies in `extract_activations`.

Uses a stub model (MagicMock) with deterministic activations so we can verify:

* pooling="last"  — current behavior, takes the final real position
* pooling="mean"  — averages across real positions only (pads excluded)
* pooling="max"   — element-wise max across real positions
* the cache signature varies with pooling so different strategies don't collide
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from steerkit import ContrastPair, extract_activations
from steerkit.cache import cache_signature
from steerkit.extract import _pool

D_MODEL = 4
N_LAYERS = 3
HOOK = "resid_post"


def _hook_name_for(layer: int) -> str:
    if layer == -1:
        return "hook_embed"
    if layer == N_LAYERS:
        return "ln_final.hook_normalized"
    return f"blocks.{layer}.hook_{HOOK}"


def _stub_model(seq_per_item: int = 5) -> MagicMock:
    """Build a stub model that returns deterministic activations for any input."""
    model = MagicMock()
    model.model_id = "fake/test"
    model.device = "cpu"
    model.n_layers = N_LAYERS
    model.d_model = D_MODEL

    # Deterministic per-position activations: position i → vector [i, i, i, i].
    # This makes pooling math trivial to verify by hand.
    def _format_chat(prompt, response=None):
        return torch.arange(seq_per_item, dtype=torch.long).unsqueeze(0)

    model.format_chat = MagicMock(side_effect=_format_chat)

    def _run_with_cache(tokens, names_filter=None):
        seq_len = tokens.shape[-1]
        batch = tokens.shape[0] if tokens.dim() > 1 else 1
        # cache[hook_name][b, t, d] = t  (broadcast over batch + d_model)
        cache = {}
        for layer in (-1, 0, 1, 2, 3):
            arr = torch.arange(seq_len, dtype=torch.float32).view(1, -1, 1).expand(batch, seq_len, D_MODEL).contiguous()
            cache[_hook_name_for(layer)] = arr
        return None, cache

    model.hooked = MagicMock()
    model.hooked.run_with_cache = MagicMock(side_effect=_run_with_cache)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0
    model.tokenizer = tokenizer
    return model


def _pairs(n: int = 2) -> list[ContrastPair]:
    return [
        ContrastPair(prompt=f"p{i}", positive_response=f"+{i}", negative_response=f"-{i}")
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# _pool primitive
# --------------------------------------------------------------------------


def test_pool_last_returns_final_row():
    seq = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = _pool(seq, "last")
    assert torch.equal(out, torch.tensor([5.0, 6.0]))


def test_pool_mean_averages_rows():
    seq = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = _pool(seq, "mean")
    assert torch.allclose(out, torch.tensor([3.0, 4.0]))


def test_pool_max_takes_elementwise_max():
    seq = torch.tensor([[1.0, 9.0], [3.0, 4.0], [5.0, 6.0]])
    out = _pool(seq, "max")
    assert torch.equal(out, torch.tensor([5.0, 9.0]))


def test_pool_unknown_raises():
    with pytest.raises(ValueError, match="unknown pooling"):
        _pool(torch.zeros(2, 3), "median")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# extract_activations dispatches pooling correctly
# --------------------------------------------------------------------------


def test_extract_default_is_last_position():
    """seq_per_item=5; default pooling=last picks position 4 → vector [4,4,4,4]."""
    model = _stub_model(seq_per_item=5)
    out = extract_activations(_pairs(2), model, hook_site=HOOK, include_boundaries=False, batch_size=8)
    # 3 block layers, [n_pairs=2, 2, d_model=4]
    assert set(out.keys()) == {0, 1, 2}
    for layer_acts in out.values():
        assert layer_acts.shape == (2, 2, D_MODEL)
        # Every (pair, response) row → final position (4) → [4, 4, 4, 4]
        assert torch.allclose(layer_acts, torch.full((2, 2, D_MODEL), 4.0))


def test_extract_mean_pools_over_real_positions():
    """seq_per_item=5; pooling=mean averages positions 0..4 → mean is 2.0."""
    model = _stub_model(seq_per_item=5)
    out = extract_activations(
        _pairs(2), model, hook_site=HOOK, include_boundaries=False, batch_size=8, pooling="mean"
    )
    for layer_acts in out.values():
        assert torch.allclose(layer_acts, torch.full((2, 2, D_MODEL), 2.0))


def test_extract_max_picks_largest_real_position():
    """seq_per_item=5; pooling=max gives the final position value (4) since
    activations are monotonically increasing with position in the stub."""
    model = _stub_model(seq_per_item=5)
    out = extract_activations(
        _pairs(2), model, hook_site=HOOK, include_boundaries=False, batch_size=8, pooling="max"
    )
    for layer_acts in out.values():
        assert torch.allclose(layer_acts, torch.full((2, 2, D_MODEL), 4.0))


def test_extract_sequential_path_matches_batched():
    """batch_size=1 takes the sequential path; should agree with batched path."""
    model = _stub_model(seq_per_item=5)
    seq_out = extract_activations(
        _pairs(2), model, hook_site=HOOK, include_boundaries=False, batch_size=1, pooling="mean"
    )
    model = _stub_model(seq_per_item=5)
    batched_out = extract_activations(
        _pairs(2), model, hook_site=HOOK, include_boundaries=False, batch_size=8, pooling="mean"
    )
    for layer in seq_out:
        assert torch.allclose(seq_out[layer], batched_out[layer])


# --------------------------------------------------------------------------
# cache signature varies by pooling
# --------------------------------------------------------------------------


def test_cache_signature_default_unchanged():
    """pooling='last' MUST produce the same signature as before — existing
    cached files should remain reusable."""
    sig = cache_signature(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        hook_site="resid_post",
        include_boundaries=True,
        pairs_hash="abc123",
    )
    assert sig == "Qwen--Qwen2.5-1.5B-Instruct__resid_post__boundaries__abc123"


def test_cache_signature_with_mean_has_suffix():
    sig = cache_signature(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        hook_site="resid_post",
        include_boundaries=True,
        pairs_hash="abc123",
        pooling="mean",
    )
    assert sig.endswith("__mean")


def test_cache_signature_three_pooling_modes_distinct():
    base = dict(
        model_id="m", hook_site="resid_post", include_boundaries=True, pairs_hash="h"
    )
    sigs = {p: cache_signature(**base, pooling=p) for p in ("last", "mean", "max")}
    assert len(set(sigs.values())) == 3
