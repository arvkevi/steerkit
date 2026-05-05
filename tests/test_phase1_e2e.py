"""End-to-end Phase 1+3 test.

Loads a real small model, runs extraction over the bundled refusal pairs, fits
all three candidate directions with held-out metrics, saves the best, reloads in the
same process, calibrates auto-α, and generates a steered completion. Slow
(downloads + runs a model) — gated by env var.

Run with:
    STEERKIT_RUN_SLOW=1 uv run pytest tests/test_phase1_e2e.py -s

The acceptance check is intentionally weak: steering with non-zero alpha must
produce a different output than alpha=0. The point is pipeline correctness,
not probe quality on a 160M-parameter model.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from steerkit import (
    PROBE_METHODS,
    Probe,
    calibrate_alpha,
    extract_activations,
    load,
    load_pairs_jsonl,
)

SLOW_MODEL = os.environ.get("STEERKIT_TEST_MODEL", "EleutherAI/pythia-160m")
RUN_SLOW = os.environ.get("STEERKIT_RUN_SLOW") == "1"
# TL 3.x emits a noisy warning about MPS even for ops that work fine.
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

pytestmark = pytest.mark.skipif(
    not RUN_SLOW,
    reason="Set STEERKIT_RUN_SLOW=1 to run the slow end-to-end Phase 1 test.",
)


def test_phase1_walking_skeleton(tmp_path: Path):
    repo_root = Path(__file__).parent.parent
    pairs = load_pairs_jsonl(repo_root / "examples" / "data" / "refusal_pairs.jsonl")

    model = load(SLOW_MODEL)
    print(f"\nLoaded {SLOW_MODEL}: {model.n_layers} layers, d_model={model.d_model}, device={model.device}")

    cache_dir = tmp_path / "cache"
    activations = extract_activations(pairs, model, hook_site="resid_post", cache_dir=cache_dir)
    # Phase 3 boundary sweep: keys are {-1 (embed), 0..n-1 (blocks), n (final_ln)}.
    expected_keys = {-1, *range(model.n_layers), model.n_layers}
    assert set(activations.keys()) == expected_keys
    sample = next(iter(activations.values()))
    assert sample.shape == (len(pairs), 2, model.d_model)

    # Cache should be populated; a second call hits it without re-running the model.
    cache_files = list(cache_dir.glob("*.zarr"))
    assert len(cache_files) == 1, f"expected 1 zarr cache, got {cache_files}"
    activations_cached = extract_activations(
        pairs, model, hook_site="resid_post", cache_dir=cache_dir
    )
    assert set(activations_cached.keys()) == expected_keys
    for layer in expected_keys:
        # Cache should round-trip activations to within float32 precision.
        assert torch.allclose(activations_cached[layer], activations[layer], atol=1e-5), (
            f"cache mismatch at layer {layer}"
        )

    probes = Probe.fit_all(activations, model, hook_site="resid_post", test_fraction=0.2)
    best = Probe.best_layer(probes, by="auc_test_logistic")
    print(
        f"Best layer = {best.layer} (norm depth = {best.normalized_depth:.2f}), "
        f"hook = {best.hook_name}, "
        f"held-out AUC (logistic) = {best.metrics['auc_test_logistic']:.3f}, "
        f"Cohen's d = {best.metrics['cohens_d_logistic']:.2f}"
    )
    # All three candidate directions should produce held-out scores above chance on real data.
    for method in PROBE_METHODS:
        key = f"auc_test_{method}"
        assert key in best.metrics
        print(f"  {method}: held-out AUC = {best.metrics[key]:.3f}")

    artifact = tmp_path / "refusal.probe.safetensors"
    best.save(artifact)
    assert artifact.exists()

    reloaded = Probe.load(artifact)
    assert reloaded.layer == best.layer
    assert reloaded.model_id == SLOW_MODEL
    assert set(reloaded.directions.keys()) == set(PROBE_METHODS)
    assert reloaded.schema_version == 3

    # Auto-α calibration on the loaded probe (small budget: 2 prompts, 4 candidates).
    chosen, ratios = calibrate_alpha(
        reloaded,
        model,
        prompts=["Tell me about your morning.", "Recommend a book."],
        candidates=[1.0, 2.0, 4.0, 8.0],
        max_new_tokens=20,
    )
    print(f"\nauto-α = {chosen}; ratios = {ratios}")
    assert reloaded.auto_alpha == chosen

    # Generate steered + unsteered completions on a benign held-out prompt.
    test_prompt = "What's a good way to start the morning?"
    unsteered = reloaded.steer(model, test_prompt, alpha=0.0, max_new_tokens=40)
    steered = reloaded.steer(model, test_prompt, alpha=8.0, max_new_tokens=40)
    print(f"\n[unsteered] {unsteered}")
    print(f"[steered]   {steered}")

    # Acceptance: steering with non-zero alpha must produce a different completion than alpha=0.
    assert steered != unsteered, (
        "Steering with non-zero alpha produced an identical completion to unsteered — "
        "the steering hook is not affecting generation."
    )
