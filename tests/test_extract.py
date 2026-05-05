"""Unit tests for extract.py — boundary-layer sweep and hook-name resolution."""

from __future__ import annotations

import pytest

from steerkit.extract import EMBED_LAYER, _final_ln_layer, _hook_name, sweep_layers


def test_hook_name_block():
    assert _hook_name(0, "resid_post", n_layers=12) == "blocks.0.hook_resid_post"
    assert _hook_name(5, "resid_post", n_layers=12) == "blocks.5.hook_resid_post"
    assert _hook_name(7, "resid_pre", n_layers=12) == "blocks.7.hook_resid_pre"


def test_hook_name_embed():
    assert _hook_name(EMBED_LAYER, n_layers=12) == "hook_embed"
    # site is ignored for the embedding sentinel.
    assert _hook_name(-1, "anything", n_layers=12) == "hook_embed"


def test_hook_name_final_ln():
    assert _hook_name(12, n_layers=12) == "ln_final.hook_normalized"
    # site is ignored for the final_ln sentinel.
    assert _hook_name(12, "resid_post", n_layers=12) == "ln_final.hook_normalized"


def test_hook_name_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        _hook_name(13, n_layers=12)
    with pytest.raises(ValueError, match="out of range"):
        _hook_name(-2, n_layers=12)


def test_sweep_layers_with_boundaries():
    layers = sweep_layers(4, include_boundaries=True)
    assert layers == [-1, 0, 1, 2, 3, 4]


def test_sweep_layers_without_boundaries():
    layers = sweep_layers(4, include_boundaries=False)
    assert layers == [0, 1, 2, 3]


def test_final_ln_layer_helper():
    assert _final_ln_layer(12) == 12
    assert _final_ln_layer(0) == 0
