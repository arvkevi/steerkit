"""Generate the bundled synthetic seed datasets for verbosity and formality.

Calls the Anthropic API via the user's `ANTHROPIC_API_KEY`. Defaults to
Claude Haiku 4.5 (cheap, fast) since synthetic-pair quality at this stage
doesn't need Opus and we want the demo to be reproducible without burning
budget.

Run with:
    uv run python scripts/generate_seed_datasets.py [n_pairs]

Default n_pairs = 30 per concept. Pass a smaller integer for a smoke test:
    uv run python scripts/generate_seed_datasets.py 3
"""

from __future__ import annotations

import sys
from pathlib import Path

from steerkit import (
    Concept,
    ConceptGroup,
    make_teacher,
    save_pairs_jsonl,
    singleton_group,
)


def build_groups() -> list[ConceptGroup]:
    return [
        singleton_group(
            Concept(
                name="verbose",
                description=(
                    "is verbose, expansive, and detailed — uses long sentences, multiple "
                    "examples, hedges, and elaboration well beyond the minimum needed"
                ),
            ),
            neutral_reference=(
                "Respond as concisely as possible: one or two short sentences, no preamble"
            ),
            group_name="verbosity",
        ),
        singleton_group(
            Concept(
                name="formal",
                description=(
                    "is formal, polished, and professional — full sentences, careful word "
                    "choice, no contractions, no slang"
                ),
            ),
            neutral_reference=(
                "Respond casually and conversationally, the way you'd talk to a friend, "
                "with contractions and informal phrasing"
            ),
            group_name="formality",
        ),
    ]


def main(n_pairs: int = 30) -> None:
    out_dir = Path(__file__).parent.parent / "examples" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher = make_teacher("anthropic:claude-haiku-4-5-20251001")
    print(f"Using teacher: {teacher.identifier}")

    for group in build_groups():
        print(f"\n=== {group.name} ({n_pairs} pairs) ===")
        stats = group.generate_pairs(teacher, max_pairs_per_concept=n_pairs)
        for name, s in stats.items():
            print(f"  {name}: parsed={s.parsed} failed={s.failed} success={s.success_rate:.0%}")
        # Each singleton group has exactly one concept; persist its pairs.
        concept = group.concepts[0]
        out_path = out_dir / f"{group.name}.jsonl"
        save_pairs_jsonl(concept.contrast_pairs, out_path)
        print(f"  saved {len(concept.contrast_pairs)} pairs -> {out_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n)
