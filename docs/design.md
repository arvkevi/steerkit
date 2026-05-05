# Design notes

This page summarizes the load-bearing design choices.

## Stack

- **TransformerLens 3.1+** — primary model loading + hook surface, `run_with_cache` for batch extraction.
- **nnsight** — fallback adapter for HF models TL doesn't support (deferred).
- **steerkit owns**: `Concept` / `ConceptGroup` / `ContrastPair` primitives; teacher abstraction; sweep + tiered metrics + auto-α; visualization; `.probe.safetensors` artifact; cross-group composition; CLI.

## Data model

A `ConceptGroup` shares one `neutral_reference` instruction across all its concepts, so the resulting steering vectors live in a common coordinate frame and can be linearly combined inside the group. The `relationship` flag (`mutually_exclusive` / `multi_label` / `axes`) drives generation strategy and metric choice. `axes` groups stay in their own coordinate frame; cross-group composition happens at steer time via `compose()`.

## Probe families

Three closed-form-ish probes per layer in parallel:

1. **Logistic regression** with L2 — gives held-out AUC + Cohen's d on the decision function.
2. **Difference-of-means** — `mean(act⁺) − mean(act⁻)` unit-normalized. The standard CAA / repeng direction.
3. **Mass-mean / LDA** — Ledoit-Wolf-shrinkage LDA, equivalent to `Σ⁻¹(μ⁺ − μ⁻)`. Marks & Tegmark "geometry of truth" direction.

The user picks the direction at steer time via `method=...`. Defaults to `logistic`.

## Layer selection

Two-tier:

- **Cheap tier (always on):** held-out AUC for all three candidate directions + Cohen's d on the logistic decision function. Sweeps `embed → 0..N-1 → final_ln`. Computed in seconds per layer.
- **Expensive tier (opt-in):** steering-effect-size measured by an LLM judge on a small generation budget, narrowed to the top-K layers from the cheap tier. Default K=5, 20 prompts × 60 tokens per layer.

Both metrics are stored on each Probe; `Probe.best_layer(probes, by=...)` picks by whichever metric you choose.

## Auto-α calibration

Sweep α candidates ({0.5, 1, 2, 4, 8} by default) on a small calibration set; pick the largest α where the steered output's perplexity (under the unsteered model — "how surprised does the model itself look?") stays within a configurable ratio (default 1.5×) of the unsteered baseline. Result attaches to `probe.auto_alpha`; `Probe.steer(..., alpha=None)` uses it.

This avoids the #1 friction point in CAA / repeng: guessing α.

## Intervention operations

Four, all expressed as a hook over a single direction `v` (unit-normalized):

| op | formula | use case |
|---|---|---|
| `addition` (default) | `act + α·v` | "push toward concept" |
| `projection` (ablate) | `act − (act·v̂)v̂` | "remove the concept entirely" |
| `clamp` | `act + (target − act·v̂)v̂` | "force projection to target value" |
| `multiplicative` (amplify) | `act + (γ−1)(act·v̂)v̂` | "scale whatever signal is there" |

`Probe.steer(..., op=...)` dispatches; convenience methods `ablate / clamp / amplify` wrap.

## Activation pooling

`extract_activations(..., pooling=...)` selects how the per-token residual stream collapses to a single `[d_model]` vector per (pair, response):

- **`"last"`** (default) — final real-token position. The standard choice for decoder-only LMs (Qwen / Llama / Gemma / Pythia / GPT-2): causal attention means the last token has attended to everything before it, so it carries a "summary" of the response.
- **`"mean"`** — average across all real positions. Required for **encoder models** (BERT, RoBERTa, DeBERTa) where bidirectional attention means no position has special "summary" status; matches BERT-style classification heads.
- **`"max"`** — element-wise max across real positions. Picks up punctate signals at unknown positions.

Pad positions are sliced off before pooling so they never contaminate mean/max in the batched extraction path. The cache signature includes the pooling mode so different strategies don't collide; existing `"last"`-pooled caches keep their original signatures and remain reusable.

## Layer scope

Default: single best layer. Opt-in: window-of-(2k+1) via `window(probes, center_layer, k=1)`. Out of scope (research-grade): full all-layers weighted ensemble.

## Layer indexing

Probes carry their layer in both absolute index and normalized depth (`(layer + 1) / (n_total_layers + 1)`, so embed=0.0, final_ln=1.0). The normalized depth is what makes layer curves comparable across models with different layer counts in the layer-selection visualization — but it's a methodology-comparison metric, not a vector-transfer mechanism.

## Storage

- **Activation cache** — Zarr v3 directory keyed by (model_id, hook_site, include_boundaries, pooling, dataset hash). Skip the model entirely on a cache hit.
- **Probe artifact** — single `.probe.safetensors` file with three direction tensors (logistic / diff_of_means / mass_mean) + biases + metrics + JSON metadata. One file = one drop-in artifact.
- **GroupFit** — directory: `group.json` snapshot + one `.probe.safetensors` per concept + optional `multinomial.probe.safetensors`.
- **GGUF** — `Probe.export_gguf(path)` and `CompositeProbe.export_gguf(path)` write llama.cpp-compatible control vectors (one tensor per source layer).
