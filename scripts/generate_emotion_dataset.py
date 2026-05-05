"""Generate a small emotion ConceptGroup for the multi-class quickstart notebook.

Uses Claude Haiku 4.5 by default (cheap, fast). Produces a single JSON file
under examples/data/ that can be loaded with ConceptGroup.load().

Run with:
    uv run python scripts/generate_emotion_dataset.py [n_pairs_per_emotion]

Default n_pairs = 20 per emotion. The group is `mutually_exclusive` so
sweep() will fit a multinomial probe in addition to per-concept binary probes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from steerkit import Concept, ConceptGroup, make_teacher


def main(n_pairs: int = 20) -> None:
    repo_root = Path(__file__).parent.parent
    out_path = repo_root / "examples" / "data" / "emotion.group.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    group = ConceptGroup(
        name="emotion",
        relationship="mutually_exclusive",
        neutral_reference=(
            "Respond in a plain, neutral, factual tone. Express no particular emotion."
        ),
        concepts=[
            Concept(
                name="joy",
                description="is upbeat, exuberant, joyful — bright tone, expressions of delight, exclamations of happiness",
            ),
            Concept(
                name="sadness",
                description="is melancholic, downcast, sad — somber tone, expressions of disappointment or grief",
            ),
            Concept(
                name="anger",
                description="is irritated, frustrated, angry — sharp tone, complaints, expressions of annoyance",
            ),
        ],
    )

    teacher = make_teacher("anthropic:claude-haiku-4-5-20251001")
    print(f"Using teacher: {teacher.identifier}")
    print(f"Generating {n_pairs} pairs per emotion...")

    stats = group.generate_pairs(teacher, max_pairs_per_concept=n_pairs)
    for name, s in stats.items():
        print(f"  {name}: parsed={s.parsed} failed={s.failed} success={s.success_rate:.0%}")

    group.save(out_path)
    total = sum(len(c.contrast_pairs) for c in group.concepts)
    print(f"saved {total} pairs across {len(group.concepts)} emotions to {out_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
