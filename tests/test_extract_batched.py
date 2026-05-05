"""Verify batched activation extraction matches sequential within float tolerance.

Slow (loads pythia-160m via TL) — gated by STEERKIT_RUN_SLOW=1. Without the env
var the file is collected but skipped.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from steerkit import extract_activations, load, load_pairs_jsonl

SLOW_MODEL = os.environ.get("STEERKIT_TEST_MODEL", "EleutherAI/pythia-160m")
RUN_SLOW = os.environ.get("STEERKIT_RUN_SLOW") == "1"
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

pytestmark = pytest.mark.skipif(
    not RUN_SLOW,
    reason="Set STEERKIT_RUN_SLOW=1 to run the slow batched-extraction test.",
)


def test_batched_matches_sequential_within_tolerance():
    repo_root = Path(__file__).parent.parent
    pairs = load_pairs_jsonl(repo_root / "examples" / "data" / "refusal_pairs.jsonl")[:8]

    model = load(SLOW_MODEL)
    print(f"\nmodel: {SLOW_MODEL}, layers={model.n_layers}, d_model={model.d_model}")

    t = time.time()
    seq = extract_activations(pairs, model, hook_site="resid_post", batch_size=1)
    seq_time = time.time() - t
    print(f"sequential: {seq_time:.2f}s")

    t = time.time()
    batched = extract_activations(pairs, model, hook_site="resid_post", batch_size=8)
    batched_time = time.time() - t
    print(f"batched (size=8): {batched_time:.2f}s ({seq_time / batched_time:.2f}x speedup)")

    # Causal attention guarantees real-position activations don't see future pads,
    # so the two paths are mathematically identical. In practice MPS float reductions
    # are non-deterministic across batch sizes (different parallel reduction trees),
    # so we tolerate a few thousandths of an absolute difference at residual-stream
    # scale — this is well below the noise floor of any downstream probe metric.
    assert set(seq.keys()) == set(batched.keys())
    for layer in seq:
        diff = (seq[layer] - batched[layer]).abs().max().item()
        assert diff < 1e-2, f"layer {layer}: max abs diff = {diff} between seq and batched"
        # Relative scale of the activations themselves — most should be O(0.1-10) on
        # pythia, so absolute diff < 1e-2 means ≤1% noise on typical activations.
