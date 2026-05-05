# CLI reference

A thin `typer` wrapper around the Python API. Subcommands map to the workflow phases.

```bash
steerkit --version
steerkit --help
```

## `generate` — single-concept teacher generation → JSONL

```bash
steerkit generate \
    --name verbose \
    --description "long, expansive, with many examples" \
    --neutral "Respond as concisely as possible" \
    --teacher anthropic:claude-haiku-4-5-20251001 \
    --n-pairs 30 \
    --out examples/data/verbose.jsonl
```

`--seed-prompts path.json` overrides the bundled default prompt list. Anthropic / OpenAI keys are read from the environment (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`).

## `lint-pairs` — dataset quality checks → exit code

Probe quality is downstream of dataset quality. `lint-pairs` runs eight cheap checks against a JSONL of `ContrastPair`s. **No model is loaded**, so it's fast enough to run before every `sweep` and suitable for CI / pre-commit.

```bash
steerkit lint-pairs --pairs examples/data/refusal_pairs.jsonl
```

The bundled refusal dataset ships intentional warnings — most positives are byte-identical (the canonical "I can't help with that.") to keep the probe focused. Sample output:

```text
lint report — 100 pair(s); 0 error(s), 2 warning(s), 0 info(s)

Warnings:
  ⚠ [UNIFORM_POSITIVES] 65% of positive_responses are byte-identical: "I can't help with that.". ...
  ⚠ [LENGTH_SKEW] negative responses average 464 chars vs 28 chars on the other side (16.4× skew). ...
```

Findings are categorised by severity:

* **error** — fitting will fail or produce garbage. Exit code `1`.
* **warning** — likely degrades probe quality, but fitting will run. Exit code `0` by default; pass `--strict` to fail on warnings.
* **info** — advisory; sometimes intentional.

### Checks

| Code | Severity | What it catches | Why it matters |
| --- | --- | --- | --- |
| `EMPTY_DATASET` | error | zero pairs | nothing to fit on |
| `EMPTY_FIELD` | error | a pair with empty prompt / positive_response / negative_response | the activation is meaningless and can crash extraction |
| `EXACT_DUPLICATE` | warning | identical `(prompt, pos, neg)` tuples | inflates the apparent training signal without adding information |
| `UNIFORM_POSITIVES` | warning | >50% of positives byte-identical | probe memorises specific tokens rather than the abstract concept; steering generalises poorly |
| `CROSS_CLASS_LEAKAGE` | warning | a string is the positive of one pair and the negative of another | the probe sees the same activation labelled both ways |
| `LENGTH_SKEW` | warning | one side averages >3× the other side's character length | probe ends up detecting length, not concept |
| `SHORT_RESPONSE` | warning | a response is shorter than 20 characters | usually a parsing artefact from the teacher |
| `REPEATED_PROMPT` | info / warning | the same prompt across multiple pairs (info under 10% of dataset, warning above) | fine for multi-class concepts; a bug if every prompt was meant to be unique |
| `COMMON_POSITIVE_PREFIX` | info | all positives share a ≥30-char prefix | probe will key on the prefix tokens specifically |

### Flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--pairs PATH` | required | JSONL of `ContrastPair`s |
| `--json PATH` | unset | also write a machine-readable JSON report |
| `--strict` | off | exit non-zero on warnings (in addition to errors) |

### `lint-group` — lint every concept in a `ConceptGroup`

```bash
steerkit lint-group --group examples/data/emotion.group.json
```

Each concept's pairs are linted independently and reported under a heading. Exit code is non-zero if any concept fires an error (or, with `--strict`, a warning).

## `sweep` — single-concept extract + fit + save

```bash
steerkit sweep \
    --pairs examples/data/sycophancy.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --cache-dir cache \
    --out runs/sycophancy.probe.safetensors
```

`--no-boundaries` disables the embed/final_ln boundary sweep. `--select-by` picks the metric for the best-layer choice (default `auc_test_logistic`).

## `group-sweep` — multi-concept ConceptGroup → GroupFit directory

```bash
steerkit group-sweep \
    --group examples/data/emotion.group.json \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --cache-dir cache \
    --out runs/emotion_fit
```

Add `--with-steering-eval --teacher anthropic:...` to invoke the LLM-judge expensive tier on each concept's top-K layers.

## `calibrate` — auto-α on an existing probe

```bash
steerkit calibrate \
    --probe runs/sycophancy.probe.safetensors \
    --candidates 0.5,1,2,4,8 \
    --ratio-max 1.5
```

By default the chosen α is written back to the probe artifact (`auto_alpha` field). Pass `--no-save` to skip the write.

## `steer` — load + emit one steered completion

```bash
steerkit steer \
    --probe runs/sycophancy.probe.safetensors \
    --prompt "What's a good way to start the morning?" \
    --op addition
```

`--op` selects the intervention operation: `addition` | `projection` | `clamp` | `multiplicative`. `clamp` requires `--target`, `multiplicative` requires `--gamma`. `--alpha` overrides the calibrated `auto_alpha` for `addition`.

## `report` — render a shareable HTML one-pager

```bash
steerkit report \
    --probe runs/sycophancy.probe.safetensors \
    --out runs/sycophancy_report.html
```

For a saved `GroupFit` directory:

```bash
steerkit report \
    --group-fit runs/emotion_fit \
    --out runs/emotion_report.html
```

Pass `--model Qwen/Qwen2.5-1.5B-Instruct` to include logit-lens plots. Without `--model`, the report still includes artifact metadata and any plots available from the saved probe or group fit.
