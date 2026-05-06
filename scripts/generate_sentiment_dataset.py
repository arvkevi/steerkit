"""Generate a sentiment contrast-pair dataset using a teacher.

For each prompt template the teacher writes two responses:
  positive: enthusiastic, positive-sentiment review/answer
  negative: harsh, negative-sentiment review/answer

Used by the encoder walkthrough (examples/case_studies/encoder_walkthrough.ipynb)
to fit a BERT sentiment probe and demonstrate steering on masked-LM predictions.

Run with:
    ANTHROPIC_API_KEY=... uv run python scripts/generate_sentiment_dataset.py [n_pairs]

Default n_pairs = 60.
"""

from __future__ import annotations

import sys
from pathlib import Path

from steerkit import (
    Concept,
    generate_pairs_for_concept,
    make_teacher,
    save_pairs_jsonl,
)

# Prompts that invite an opinionated answer with clear sentiment polarity.
PROMPTS = [
    "Tell me about the movie you watched last weekend.",
    "How was that restaurant you tried?",
    "What did you think of the concert?",
    "How was your vacation?",
    "What's your take on that new phone?",
    "Tell me about that book you finished.",
    "How did the conference go?",
    "What did you think of the new TV show?",
    "How was the weather on your trip?",
    "Tell me about your morning commute.",
    "What did you think of the keynote talk?",
    "How was the food at the wedding?",
    "Tell me about that gym class.",
    "How did the new coffee shop turn out?",
    "What's your take on that documentary?",
    "Tell me about the road trip you took.",
    "How was the museum exhibit?",
    "Did you enjoy that podcast episode?",
    "How was the live music venue?",
    "Tell me about that hike you went on.",
    "What did you think of the new bookstore?",
    "How was your stay at that hotel?",
    "Tell me about the museum gift shop.",
    "How was the cooking class?",
    "What's your take on that streaming service?",
    "How did the meeting go?",
    "Tell me about that brunch place.",
    "How was the play you saw?",
    "What did you think of the airport food?",
    "Tell me about the conference workshop.",
    "How was the train ride?",
    "What did you think of the gym?",
    "Tell me about the museum tour guide.",
    "How was the bookstore reading?",
    "What's your take on that new app?",
    "How was your weekend?",
    "Tell me about that hotel breakfast.",
    "What did you think of the new park?",
    "How was the wine tasting?",
    "Tell me about that birthday party.",
    "How was the work retreat?",
    "What did you think of the food truck?",
    "How was the comedy show?",
    "Tell me about the new gym class.",
    "What's your take on that new café?",
    "How was the museum's new exhibit?",
    "Tell me about that craft beer.",
    "How was the panel discussion?",
    "What did you think of the new bakery?",
    "How was the trail run?",
    "Tell me about the airport lounge.",
    "How was the spa visit?",
    "What did you think of the cooking class?",
    "Tell me about the rental car.",
    "How was the music festival?",
    "What did you think of the meal kit?",
    "Tell me about the camping trip.",
    "How was the boat ride?",
    "What's your take on that wellness retreat?",
    "How was the science museum?",
]


CONCEPT_DESCRIPTION = (
    "responds with strongly positive sentiment — enthusiastic, glowing, "
    "uses positive adjectives like 'amazing', 'incredible', 'fantastic', "
    "'loved', 'wonderful'. The response is happy, satisfied, and clearly "
    "endorses the subject."
)

NEUTRAL_REFERENCE = (
    "Respond with strongly negative sentiment — disappointed, harsh, "
    "uses negative adjectives like 'terrible', 'awful', 'disappointing', "
    "'hated', 'boring'. The response is unhappy, dissatisfied, and clearly "
    "criticizes the subject."
)


def main() -> int:
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    teacher = make_teacher("anthropic:claude-haiku-4-5-20251001")
    concept = Concept(name="positive_sentiment", description=CONCEPT_DESCRIPTION)
    print(f"using teacher: {teacher.identifier}")
    print(f"prompt pool: {len(PROMPTS)} prompts")
    print(f"target pairs: {n_pairs}")
    pairs, stats = generate_pairs_for_concept(
        concept,
        teacher=teacher,
        neutral_reference=NEUTRAL_REFERENCE,
        seed_prompts=PROMPTS,
        max_pairs=n_pairs,
        temperature=0.7,
        max_tokens=200,
    )
    print(
        f"  parsed={stats.parsed} failed={stats.failed} success={stats.success_rate:.0%}"
    )
    out_path = Path(__file__).resolve().parent.parent / "examples" / "data" / "sentiment.jsonl"
    save_pairs_jsonl(pairs, out_path)
    print(f"saved {len(pairs)} pairs → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
