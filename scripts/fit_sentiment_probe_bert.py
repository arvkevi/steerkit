"""Fit a sentiment probe on bert-base-uncased and save to runs/.

This is the headline encoder probe used by the encoder walkthrough notebook
(examples/case_studies/encoder_walkthrough.ipynb). Contrast pairs come from
`examples/data/sentiment.jsonl` (positive vs. negative review responses).

Notes:
* `pooling="mean"` — encoders are bidirectional, so averaging across all real
  positions gives a sensible sentence-level summary. The decoder default
  (last-token) doesn't apply.
* Skip `calibrate_alpha`: it requires `model.hooked.generate(...)`, which
  HookedEncoder does not expose. We pick a reasonable default α here and let
  users tune via `Probe.predict_at_mask(..., alpha=...)`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

from steerkit import (  # noqa: E402
    Probe,
    extract_activations,
    load,
    load_pairs_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "bert-base-uncased"
PAIRS_PATH = REPO_ROOT / "examples" / "data" / "sentiment.jsonl"
PROBE_PATH = REPO_ROOT / "runs" / "sentiment_bert.probe.safetensors"
CACHE_DIR = REPO_ROOT / "cache"


def main() -> int:
    pairs = load_pairs_jsonl(PAIRS_PATH)
    print(f"loaded {len(pairs)} contrast pairs from {PAIRS_PATH}")
    handle = load(MODEL_ID)
    print(
        f"loaded {MODEL_ID}: layers={handle.n_layers}, "
        f"d_model={handle.d_model}, encoder={handle.is_encoder}, device={handle.device}"
    )
    activations = extract_activations(
        pairs,
        handle,
        hook_site="resid_post",
        include_boundaries=False,
        cache_dir=CACHE_DIR,
        pooling="mean",
    )
    probes = Probe.fit_all(activations, handle, hook_site="resid_post", test_fraction=0.2)
    # Mid-network layers carry the cleanest abstract sentiment signal. Embedding
    # layers fire on token surface ('amazing' vs 'terrible') and final layers
    # are too specialized to the MLM head's vocabulary distribution.
    n_layers = handle.n_layers
    mid_lo = max(2, n_layers // 4)
    mid_hi = min(n_layers - 2, (3 * n_layers) // 4)
    mid_probes = {layer: p for layer, p in probes.items() if mid_lo <= layer <= mid_hi}
    print(f"selecting from layers [{mid_lo}, {mid_hi}]")
    for layer in sorted(mid_probes):
        d = mid_probes[layer].metrics.get("cohens_d_logistic", float("nan"))
        auc = mid_probes[layer].metrics.get("auc_test_logistic", float("nan"))
        print(f"  layer {layer}:  Cohen's d = {d:.2f}  AUC = {auc:.3f}")
    best = Probe.best_layer(mid_probes, by="cohens_d_logistic")
    print(
        f"best layer: {best.layer} (depth {best.normalized_depth:.3f})  "
        f"AUC = {best.metrics['auc_test_logistic']:.3f}  "
        f"Cohen's d = {best.metrics['cohens_d_logistic']:.2f}"
    )
    # No calibrate_alpha — encoders don't generate. Pick a sensible default
    # based on activation scale (roughly the projection magnitude of the
    # positive class onto the direction).
    direction = best.get_direction("logistic")
    pos_acts = activations[best.layer][:, 0, :]
    neg_acts = activations[best.layer][:, 1, :]
    pos_proj = (pos_acts @ direction).mean().item()
    neg_proj = (neg_acts @ direction).mean().item()
    auto_alpha = float(abs(pos_proj - neg_proj))
    best.auto_alpha = auto_alpha
    print(
        f"auto_alpha set to mean projection gap = {auto_alpha:.3f}  "
        f"(pos_proj={pos_proj:+.3f}, neg_proj={neg_proj:+.3f})"
    )
    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    best.save(PROBE_PATH)
    print(f"\nsaved → {PROBE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
