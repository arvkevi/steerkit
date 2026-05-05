# steerkit

Find a concept in a local LLM, turn it into a steering direction, and save the result as a reusable artifact.

![layer-selection across models](hero.png)
*The same concept (sycophancy) refit on four local models spanning three families and two size tiers. Higher curves mean the concept is easier to separate at that normalized depth; hollow markers show each model's best layer. This is a comparison of where to look, not a claim that one vector transfers across models.*

## What it is

`steerkit` takes you from a *concept* (for example sycophancy, verbosity, formality, refusal, or joy) to a steering vector you can use to push a single local model's responses in that direction. It is built on top of [TransformerLens 3.1+](https://github.com/TransformerLensOrg/TransformerLens).

![steerkit pipeline: concept → contrast pairs → fit → artifact → ops](mental_model.png)
*A concrete example: 60 sycophantic-vs-direct contrast pairs → activations `[60, 2, 1536]` from Qwen2.5-1.5B-Instruct → three candidate directions per layer → best layer 12, auto-calibrated α = 13 → a single `sycophancy.probe.safetensors` artifact you can reload and apply with any of four intervention operations.*

An **opinionated, end-to-end workflow**: concept-first contrast-pair generation via a teacher LLM, activation extraction, layer sweeps, three candidate directions per layer (logistic regression, difference-of-means, shrinkage LDA), held-out metrics, perplexity-ceiling auto-α calibration, four steering operations, per-token interpretability scoring, single-file `.probe.safetensors` artifact carrying full reproducibility metadata, HTML one-pager report, and a thin CLI.

## What this is and isn't

steerkit **uses** [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) as the model-instrumentation surface — it sits one layer up, not a substitute. The closest peer is [LLMProbe](https://github.com/jammastergirish/LLMProbe) (research-shaped where steerkit is library-shaped); [repeng](https://github.com/vgel/repeng) is the closest steering-side comparison; [sae_lens](https://github.com/jbloomAus/SAELens) does bottom-up feature discovery via SAEs and is complementary, not competing.

## Limitations

* **Steering vectors don't transfer across models.** Each model needs its own probe.
* **Probes are concept-specific.** Different concepts need different training data.
* **Generalization is dataset-dependent.** Out-of-distribution prompts degrade — more diverse training pairs help.

## What's here

* **[Quickstart](quickstart.md)** — install and produce a steered completion in ten lines.
* **[Workflow](workflow.md)** — end-to-end walkthrough of the concept → probe → steering vector pipeline.
* **[Concept gallery](concepts.md)** — bundled datasets and ideas for new concepts to try.
* **[CLI reference](cli.md)** — the `steerkit` command and its subcommands.
* **[API reference](api/data.md)** — every public class and function.
* **[Design](design.md)** — the design memory captured during the original grilling: why each piece is the way it is.

## Install

```bash
git clone https://github.com/arvkevi/steerkit.git
cd steerkit
uv sync
uv run steerkit --help
uv run steerkit lint-pairs --pairs examples/data/sycophancy.jsonl
```

Optional extras:

```bash
uv sync --extra dev --extra docs
uv sync --extra anthropic   # ANTHROPIC_API_KEY
uv sync --extra openai      # OPENAI_API_KEY
uv sync --extra llamacpp    # GGUF control-vector export
```
