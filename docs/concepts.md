# Concept gallery

A linear probe is only as good as the concept you point it at.

> Good contrast pairs share the same user prompt and differ mainly in the target behaviour. If positives also change topic, length, sentiment, and factual content all at once, the probe may learn the wrong thing.

Run `steerkit lint-pairs --pairs your.jsonl` before sweeping — see the [CLI reference](cli.md#lint-pairs-dataset-quality-checks-exit-code) for the checks.

## Bundled concepts

| Concept | File | What it is |
| --- | --- | --- |
| **Sycophancy** | `sycophancy.jsonl` | Validating preface (*"Great question!"*) before answering, vs. answering directly. Used by the headline walkthrough + showcase figures. |
| **Verbosity** | `verbosity.jsonl` | Long, qualified, example-heavy answers vs. concise one-or-two-sentence answers. Watch for `LENGTH_SKEW`. |
| **Formality** | `formality.jsonl` | Formal business-tone vs. casual conversational. Orthogonal to verbosity, so it pairs well for composition. |
| **Refusal** | `refusal_pairs.jsonl` | Canonical no-help refusals (*"I can't help with that."*) vs. helpful answers. Walkthrough at [`examples/case_studies/refusal_walkthrough.ipynb`](https://github.com/arvkevi/steerkit/blob/main/examples/case_studies/refusal_walkthrough.ipynb). |
| **Emotion** (joy / sadness / anger) | `emotion.group.json` | A `ConceptGroup` with three mutually exclusive concepts; demonstrates `sweep()` on a multi-class group + multinomial diagnostic. |

## Other concepts to try

Difficulty: **easy** = generates cleanly with a teacher and probes well at one or two layers; **medium** = needs careful concept-description authoring; **hard** = signal might not be cleanly linear, or the concept is contested in the literature.

| Concept | Positive vs. Negative | Difficulty |
| --- | --- | --- |
| Hedging | "I think...", "perhaps...", "I'm not sure but..." vs. confident assertions | easy |
| First-person | "I" / "my" pervasive vs. impersonal third-person | easy |
| Question-asking | Ends with a follow-up question vs. statement only | easy |
| Markdown-heavy | Headers + bullets + bold vs. flowing prose | easy |
| Optimism / pessimism | Hopeful framings vs. resigned framings | medium |
| Empathetic acknowledgement | "That sounds hard..." vs. immediately solving | medium |
| Confidence / uncertainty | Authoritative vs. tentative | medium |
| Calibration / honesty about uncertainty | "I don't know" vs. confident confabulation | medium |
| Step-by-step reasoning ("CoT") | Visible chain-of-thought vs. final answer | medium |
| Self-identification as AI | Mentions being an LLM/model vs. responds as a person | medium |
| Medical / legal advice tone | Clinical / disclaimed vs. layperson | medium |
| Truthfulness (TruthfulQA-style) | Honest vs. confident-and-wrong | hard |
| Sandbagging / underperforming | Capability hiding | hard |
| Power-seeking / instrumental reasoning | Goal-directed vs. neutral | hard |

## Authoring a new concept

1. Write a behavior-specific concept description ("uses warm emotional language" beats "nice").
2. Pick a single neutral reference instruction shared across all pairs.
3. Generate or hand-write 30–100 contrast pairs — same prompt, two responses.
4. `steerkit lint-pairs --pairs your.jsonl` (use `--strict` in CI).
5. Sweep at small scale; check `plot_layer_selection` for clean held-out separation before scaling up.
6. Test steering on prompts not in the dataset.

If you build a clean dataset that fills a gap, open a PR.
