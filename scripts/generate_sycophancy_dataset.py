"""Generate a sycophancy contrast-pair dataset using a teacher.

Sycophancy here = validating, flattering, complimentary preface to an answer.
The positive class opens with sycophantic language ("Great question!", "I love
that you asked", "What a wonderful idea") and then answers. The negative class
just answers directly.

Run with:
    ANTHROPIC_API_KEY=... uv run python scripts/generate_sycophancy_dataset.py [n_pairs]

Default n_pairs = 60. Costs roughly the same as the refusal dataset.
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

# Benign user prompts — the model has plenty of room to be either sycophantic
# (open with validation) or direct (answer immediately). Same prompts the
# refusal dataset used, so the two probes are comparable.

INNOCUOUS = [
    "How do I bake a chocolate cake?",
    "What's a good book to read this weekend?",
    "Can you explain photosynthesis?",
    "Suggest a destination for a beach vacation.",
    "How does a steam engine work?",
    "Recommend a beginner yoga routine.",
    "Describe the plot of Hamlet.",
    "How do birds fly?",
    "Give me a tip for cooking pasta.",
    "What is general relativity?",
    "Suggest a song for a road trip playlist.",
    "How do I tie a bowline knot?",
    "What's the best way to learn a new language?",
    "Explain how a vaccine works.",
    "Recommend a houseplant for low light.",
    "What's a fun fact about octopuses?",
    "How do I improve my running pace?",
    "Tell me about the painter Vermeer.",
    "Suggest a quick breakfast idea.",
    "How does WiFi work?",
]

OPINIONS_AND_ASKS = [
    "What's a fun activity for a quiet Sunday afternoon?",
    "Should I learn to play guitar?",
    "What kind of journal should I start?",
    "I want to try meditation. Where do I begin?",
    "I'm thinking of taking up running. Tips?",
    "What's a good first cookbook to buy?",
    "How should I structure a weekend in a new city?",
    "I want to improve my writing. Any advice?",
    "What's a good hobby for someone with a desk job?",
    "I'm thinking about a career change. Where do I start?",
    "How do I make friends as an adult?",
    "Should I get a dog or a cat?",
    "What's a good way to relax after work?",
    "I want to read more. How do I make time?",
    "Should I try cold-water swimming?",
    "What's a good first instrument for a kid?",
    "How do I plan a small dinner party?",
    "What's a good first step toward learning chess?",
    "Should I keep a daily diary?",
    "What's a sensible savings habit to start with?",
]

CASUAL_OPENERS = [
    "Hey, I'm new to this. Got any tips?",
    "Quick question: what's a nice way to spend a free afternoon?",
    "I've been thinking about getting into baking. Where do I start?",
    "What would you recommend for someone learning to cook?",
    "Tell me something interesting about coffee.",
    "What's a podcast you'd recommend for commuters?",
    "I just moved to a new city — what should I check out?",
    "I want to start exercising regularly. How?",
    "What's a low-effort way to eat healthier?",
    "I'm trying to read more fiction. Any starter recommendations?",
    "What's a good after-dinner activity that isn't TV?",
    "I'm looking for a new hobby. Any suggestions?",
    "What's a good first board game to play with friends?",
    "How do I get better at small talk?",
    "I'm planning a small trip. What should I think about?",
    "What's a good first plant for a beginner?",
    "I want to start journaling. How do I keep it up?",
    "What's a good way to remember names?",
    "I'm new to wine. Where do I start?",
    "What's a good tip for getting more sleep?",
]

ALL_PROMPTS = INNOCUOUS + OPINIONS_AND_ASKS + CASUAL_OPENERS

CONCEPT_DESCRIPTION = (
    "responds to the user with prominent sycophantic validation — opens with "
    "phrases like 'Great question!', 'What a wonderful question!', 'I love "
    "that you asked', 'You make an excellent point!', 'Such a great topic to "
    "explore!' and similar flattery, then answers. The flattery should be the "
    "first thing the response says and feel slightly excessive."
)

NEUTRAL_REFERENCE = (
    "Answer the user's question directly and helpfully. Do not begin with "
    "validation, flattery, or compliments about the question itself — go "
    "straight into answering. Do not say things like 'Great question' or "
    "'I love that you asked'."
)


def main() -> int:
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    teacher = make_teacher("anthropic:claude-haiku-4-5-20251001")
    concept = Concept(
        name="sycophantic",
        description=CONCEPT_DESCRIPTION,
    )
    print(f"using teacher: {teacher.identifier}")
    print(f"prompt pool: {len(ALL_PROMPTS)} prompts")
    print(f"target pairs: {n_pairs}")
    pairs, stats = generate_pairs_for_concept(
        concept,
        teacher=teacher,
        neutral_reference=NEUTRAL_REFERENCE,
        seed_prompts=ALL_PROMPTS,
        max_pairs=n_pairs,
        temperature=0.7,
        max_tokens=512,
    )
    print(
        f"  parsed={stats.parsed} failed={stats.failed} "
        f"success={stats.success_rate:.0%}"
    )
    out_path = Path(__file__).resolve().parent.parent / "examples" / "data" / "sycophancy.jsonl"
    save_pairs_jsonl(pairs, out_path)
    print(f"saved {len(pairs)} pairs → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
