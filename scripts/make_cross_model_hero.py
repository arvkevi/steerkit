"""Cross-model hero plot.

Run a steerkit sweep on the same concept across multiple models, then render
`plot_cross_model_overlay` to produce the headline figure for the README.

The bundled sycophancy dataset is used as the concept by default — the
sycophantic-validation prefix is a behaviorally distinctive shift that
chat-tuned instruct models all recognize.

Usage:
    uv run python scripts/make_cross_model_hero.py                  # defaults
    uv run python scripts/make_cross_model_hero.py model1 model2 ... # explicit set

Default model set spans three families (Qwen2.5 instruct, Gemma-3 instruct,
Llama-3.2 instruct) and two size tiers (0.5B / 1-1.5B). Larger models like
Qwen2.5-3B can be passed as CLI args if your GPU has enough memory — they
OOM on 16GB MPS during activation extraction. Some models are gated on
Hugging Face; the script's graceful skip path renders the plot without
them if access fails.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

from steerkit import (  # noqa: E402
    Probe,
    extract_activations,
    load,
    load_pairs_jsonl,
    plot_cross_model_overlay,
)

DEFAULT_MODELS = [
    # Qwen2.5 instruct — two sizes, same architecture. Qwen2.5-3B-Instruct is
    # supported by the script but OOMs on a 16GB MPS GPU during activation
    # extraction; pass it as a CLI arg if you have headroom.
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    # Gemma 3 instruct (gated — requires HF_TOKEN + license accept)
    "google/gemma-3-1b-it",
    # Llama 3.2 instruct (gated — requires HF_TOKEN + license accept)
    "meta-llama/Llama-3.2-1B-Instruct",
]


def main(model_ids: list[str], *, max_pairs: int = 60, out_image: Path | None = None, device: str | None = None) -> None:
    repo_root = Path(__file__).parent.parent
    pairs = load_pairs_jsonl(repo_root / "examples" / "data" / "sycophancy.jsonl")[:max_pairs]
    print(f"loaded {len(pairs)} sycophancy contrast pairs")

    # extract_activations takes pairs directly; no need to build a full ConceptGroup
    # for the cross-model overlay since each probe is a single binary direction.
    probes_per_model: dict[str, dict[int, Probe]] = {}

    for model_id in model_ids:
        print(f"\n=== {model_id} ===")
        try:
            model = load(model_id, device=device)
        except Exception as e:  # noqa: BLE001
            print(f"  skip: failed to load — {type(e).__name__}: {e}")
            continue
        print(f"  layers={model.n_layers}, d_model={model.d_model}, device={model.device}")

        activations = extract_activations(
            pairs, model, hook_site="resid_post", cache_dir=repo_root / "cache"
        )
        layer_probes = Probe.fit_all(activations, model, hook_site="resid_post", test_fraction=0.2)
        best = Probe.best_layer(layer_probes, by="cohens_d_logistic")
        print(
            f"  best layer = {best.layer} (depth {best.normalized_depth:.2f}), "
            f"Cohen's d = {best.metrics['cohens_d_logistic']:.2f}"
        )
        probes_per_model[model_id] = layer_probes

    if not probes_per_model:
        print("no models swept successfully", file=sys.stderr)
        sys.exit(1)

    if out_image is None:
        out_image = repo_root / "docs" / "hero.png"
    out_image.parent.mkdir(parents=True, exist_ok=True)

    # Cohen's d on the held-out logistic decision function — continuous and informative.
    # Held-out AUC saturates at 1.0 with our small datasets (high d_model vs few test
    # samples), so it doesn't visualize layer-wise structure well; Cohen's d does.
    fig = plot_cross_model_overlay(
        probes_per_model,
        by="cohens_d_logistic",
        title="Where does the sycophancy direction become easiest to read?",
    )
    fig.savefig(out_image, dpi=180, bbox_inches="tight")
    print(f"\nsaved hero plot to {out_image}")


if __name__ == "__main__":
    args = sys.argv[1:]
    device = os.environ.get("STEERKIT_DEVICE")  # e.g. "cpu" to bypass MPS
    models = args or DEFAULT_MODELS
    main(models, device=device)
