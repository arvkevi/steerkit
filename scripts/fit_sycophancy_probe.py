"""Fit the sycophancy probe on Qwen2.5-1.5B-Instruct and save to runs/.

Produces the headline probe artifact used by the showcase figures
(mental_model, token_scores, alpha_sweep, ops_effects) and the walkthrough
notebook. Run once; the saved probe is reused thereafter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

from steerkit import (  # noqa: E402
    Probe,
    calibrate_alpha,
    extract_activations,
    load,
    load_pairs_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PAIRS_PATH = REPO_ROOT / "examples" / "data" / "sycophancy.jsonl"
PROBE_PATH = REPO_ROOT / "runs" / "sycophancy.probe.safetensors"
CACHE_DIR = REPO_ROOT / "cache"


def main() -> int:
    pairs = load_pairs_jsonl(PAIRS_PATH)
    print(f"loaded {len(pairs)} contrast pairs from {PAIRS_PATH}")
    handle = load(MODEL_ID)
    print(f"loaded {MODEL_ID}: layers={handle.n_layers}, device={handle.device}")
    activations = extract_activations(
        pairs, handle, hook_site="resid_post", include_boundaries=False, cache_dir=CACHE_DIR
    )
    probes = Probe.fit_all(activations, handle, hook_site="resid_post", test_fraction=0.2)
    # Constrain to mid-network layers (5..18). Deeper layers can have higher
    # Cohen's d but generation has already locked in by then, so steering
    # there shifts the residual without producing the steered behavior.
    n_layers = handle.n_layers
    mid_lo = max(2, n_layers // 4)        # ~25% depth
    mid_hi = min(n_layers - 6, n_layers // 2)  # ~50% depth
    mid_probes = {layer: p for layer, p in probes.items() if mid_lo <= layer <= mid_hi}
    print(f"selecting from layers [{mid_lo}, {mid_hi}] (mid-network steerable range)")
    for layer in sorted(mid_probes):
        d = mid_probes[layer].metrics.get("cohens_d_logistic", float("nan"))
        print(f"  layer {layer}:  Cohen's d = {d:.2f}")
    best = Probe.best_layer(mid_probes, by="cohens_d_logistic")
    print(
        f"best layer: {best.layer} (depth {best.normalized_depth:.3f})  "
        f"AUC = {best.metrics['auc_test_logistic']:.3f}  "
        f"Cohen's d = {best.metrics['cohens_d_logistic']:.2f}"
    )
    chosen, ratios = calibrate_alpha(best, handle)
    print(f"calibrated auto_α = {chosen:.3f}")
    for a, r in sorted(ratios.items()):
        marker = "  ✓" if a == chosen else ""
        print(f"  α = {a}  ppl ratio = {r:.3f}{marker}")
    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    best.save(PROBE_PATH)
    print(f"\nsaved → {PROBE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
