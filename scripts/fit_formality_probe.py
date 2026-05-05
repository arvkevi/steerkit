"""Fit the formality probe on Qwen2.5-1.5B-Instruct and save to runs/.

This produces the headline probe artifact used by the showcase figures
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
PAIRS_PATH = REPO_ROOT / "examples" / "data" / "formality.jsonl"
PROBE_PATH = REPO_ROOT / "runs" / "formality.probe.safetensors"
CACHE_DIR = REPO_ROOT / "cache"


def main() -> int:
    pairs = load_pairs_jsonl(PAIRS_PATH)
    print(f"loaded {len(pairs)} contrast pairs from {PAIRS_PATH}")
    handle = load(MODEL_ID)
    print(f"loaded {MODEL_ID}: layers={handle.n_layers}, device={handle.device}")
    # include_boundaries=False skips the embedding + final_ln "shortcut" layers.
    # Formality is partly a vocabulary-level concept, so the embedding layer can
    # trivially separate the classes — but steering there is uninformative.
    activations = extract_activations(
        pairs, handle, hook_site="resid_post", include_boundaries=False, cache_dir=CACHE_DIR
    )
    probes = Probe.fit_all(activations, handle, hook_site="resid_post", test_fraction=0.2)
    # Pick by Cohen's d: held-out AUC saturates at 1.0 across many layers with
    # a 30-pair dataset, so it doesn't discriminate. Cohen's d on the logistic
    # decision function is continuous and prefers mid-network layers where the
    # concept is most cleanly *abstracted* rather than reliant on token surface.
    best = Probe.best_layer(probes, by="cohens_d_logistic")
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
