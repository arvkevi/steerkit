"""Concept-pair generation orchestration.

Given a `ConceptGroup` and a `TeacherModel`, produce `ContrastPair`s for each concept
by prompting the teacher to write a concept-bearing response and a neutral response
for each seed prompt.

Design notes (resolved during the design grilling, see project memory):
- One teacher call per pair, with delimiter-structured output (positive/neutral
  responses tagged in a single completion). Faster + ~2x cheaper than two calls,
  but requires robust parsing — we fall back gracefully on malformed responses.
- The neutral response is grounded in the group's `neutral_reference` instruction
  so all concepts in the group share the same coordinate frame.
- Seed prompts default to a bundled set of generic user queries; users can pass
  their own list to make the dataset domain-relevant.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .concepts import Concept, ConceptGroup
from .data import ContrastPair
from .teacher import TeacherModel

# Bundled default seed prompts — generic-domain user queries.
DEFAULT_SEED_PROMPTS: list[str] = [
    "Tell me about your favorite season of the year.",
    "What's a good way to spend a Sunday afternoon?",
    "Describe a memorable meal you've had.",
    "What's a hobby you'd recommend to someone new in town?",
    "How do you usually unwind after a long day?",
    "What's the best advice you've ever received?",
    "Describe a place you've always wanted to visit.",
    "What's a small thing that consistently makes you happy?",
    "Tell me about a movie that left an impression on you.",
    "What's a skill you'd love to learn this year?",
    "How do you decide what to read next?",
    "Describe the perfect cup of coffee or tea.",
    "What's a fact that always surprises people?",
    "How do you stay focused when working from home?",
    "Tell me about a time you tried something new.",
    "What kind of music do you put on in the morning?",
    "Describe your ideal weekend.",
    "What's the most useful kitchen gadget you own?",
    "How do you keep up with the news?",
    "Tell me about a book that changed your perspective.",
    "What's a habit that's improved your life recently?",
    "Describe a small joy from your childhood.",
    "What's your approach to making new friends as an adult?",
    "Tell me about a project you're proud of.",
    "What's a sport or activity worth picking up after thirty?",
    "Describe a sound you find comforting.",
    "What's a tradition your family has?",
    "How do you handle feedback you don't agree with?",
    "Tell me about a piece of advice you'd give your younger self.",
    "What's the most beautiful place you've walked through?",
    "Describe your relationship with mornings.",
    "What's a recipe you've made more than ten times?",
    "How do you decide when to take a break from a project?",
    "Tell me about a podcast or show you binged recently.",
    "What's something you collect, even informally?",
    "Describe a rainy-day activity you enjoy.",
    "What's the kindest thing a stranger has done for you?",
    "How do you celebrate small wins?",
    "Tell me about a teacher or mentor who shaped you.",
    "What's a tool or app you'd hate to lose?",
    "Describe an everyday routine you secretly love.",
    "What's a question you've been thinking about lately?",
    "How do you balance ambition with rest?",
    "Tell me about a place that always feels like home.",
    "What's a small change that made a big difference for you?",
    "Describe your favorite kind of conversation.",
    "What's a way you've challenged yourself recently?",
    "How do you approach learning something completely unfamiliar?",
    "Tell me about a piece of art that moved you.",
    "What's a goal you're working toward this season?",
]


@dataclass
class GenerationStats:
    requested: int
    parsed: int
    failed: int

    @property
    def success_rate(self) -> float:
        return self.parsed / self.requested if self.requested > 0 else 0.0


GENERATION_SYSTEM_TEMPLATE = """You generate paired contrastive responses for linear-probe training. You will receive a USER PROMPT. Write TWO short assistant responses (each at most 3 sentences) to that prompt.

First, write a response that {description}. Prefix it with this header on its own line:
### {label_concept}

Then write a response that follows this neutral instruction: "{neutral_reference}". Prefix it with this header on its own line:
### NEUTRAL

Output only the two responses with their headers — no preamble, no commentary. Each response is at most 3 sentences."""

GENERATION_USER_TEMPLATE = "USER PROMPT:\n{prompt}"


_RESPONSE_RE = re.compile(
    r"^###\s*(?P<label>[A-Za-z0-9_-]+)\s*$\n(?P<body>.*?)(?=^###\s*[A-Za-z0-9_-]+\s*$|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _parse_paired_completion(text: str, concept_label: str) -> tuple[str, str] | None:
    """Parse the teacher's output into (concept_response, neutral_response).

    Returns None if the expected sections are missing or empty.
    """
    matches = {m.group("label").upper(): m.group("body").strip() for m in _RESPONSE_RE.finditer(text)}
    pos = matches.get(concept_label.upper())
    neg = matches.get("NEUTRAL")
    if not pos or not neg:
        return None
    return pos, neg


def generate_pairs_for_concept(
    concept: Concept,
    *,
    teacher: TeacherModel,
    neutral_reference: str,
    seed_prompts: list[str] | None = None,
    max_pairs: int | None = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    on_failure: Callable[[str, str], None] | None = None,
) -> tuple[list[ContrastPair], GenerationStats]:
    """Generate contrast pairs for a single concept against a shared neutral.

    Args:
        concept: the Concept whose `description` describes the positive direction.
        teacher: the TeacherModel to call.
        neutral_reference: the group-level neutral instruction (e.g. "respond in a plain, neutral tone").
        seed_prompts: list of user prompts to generate over. Defaults to DEFAULT_SEED_PROMPTS.
        max_pairs: stop after producing this many successful pairs. None = use all seed_prompts.
        temperature: sampling temperature passed through to teacher.complete.
        max_tokens: max output tokens passed through to teacher.complete.
        on_failure: optional callback invoked with (prompt, raw_text) on parse failure.

    Returns the list of pairs produced (length <= len(seed_prompts)) and a GenerationStats.
    """
    prompts = seed_prompts if seed_prompts is not None else DEFAULT_SEED_PROMPTS
    label = re.sub(r"[^A-Z0-9_]+", "_", concept.name.upper()).strip("_") or "POSITIVE"
    system = GENERATION_SYSTEM_TEMPLATE.format(
        description=concept.description,
        label_concept=label,
        neutral_reference=neutral_reference,
    )

    pairs: list[ContrastPair] = []
    failed = 0
    requested = 0
    for prompt in prompts:
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
        requested += 1
        user = GENERATION_USER_TEMPLATE.format(prompt=prompt)
        try:
            text = teacher.complete(
                system, user, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as e:  # noqa: BLE001 — robustness over strict typing here
            failed += 1
            if on_failure is not None:
                on_failure(prompt, f"<teacher error: {type(e).__name__}: {e}>")
            continue
        parsed = _parse_paired_completion(text, label)
        if parsed is None:
            failed += 1
            if on_failure is not None:
                on_failure(prompt, text)
            continue
        positive, negative = parsed
        pairs.append(
            ContrastPair(
                prompt=prompt,
                positive_response=positive,
                negative_response=negative,
            )
        )

    return pairs, GenerationStats(requested=requested, parsed=len(pairs), failed=failed)


def generate_pairs_for_group(
    group: ConceptGroup,
    *,
    teacher: TeacherModel,
    seed_prompts: list[str] | None = None,
    max_pairs_per_concept: int | None = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    on_failure: Callable[[str, str], None] | None = None,
) -> dict[str, GenerationStats]:
    """Generate contrast pairs for every concept in a group, attaching them in-place.

    Returns a dict mapping concept.name -> GenerationStats.
    """
    stats: dict[str, GenerationStats] = {}
    for concept in group.concepts:
        pairs, s = generate_pairs_for_concept(
            concept,
            teacher=teacher,
            neutral_reference=group.neutral_reference,
            seed_prompts=seed_prompts,
            max_pairs=max_pairs_per_concept,
            temperature=temperature,
            max_tokens=max_tokens,
            on_failure=on_failure,
        )
        concept.contrast_pairs = pairs
        stats[concept.name] = s
    return stats
