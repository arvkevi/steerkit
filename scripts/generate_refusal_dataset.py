"""Generate a larger, more diverse refusal contrast-pair dataset using a teacher.

Replaces the 24 hand-crafted pairs in `examples/data/refusal_pairs.jsonl` with
~100 teacher-generated pairs spanning:

  - innocuous everyday questions
  - skill / how-to questions
  - mild-edge opinions and public topics
  - slightly sensitive but legal (medical / financial / legal advice tone)

For each prompt the teacher writes two responses:

  positive: polite refusal language declining to engage
  negative: a direct, helpful response to the same prompt

The probe trained on this set should pick up *behavioral refusal style* rather
than just the lexical "I cannot help" surface form a 24-pair hand-crafted set
exposes.

Run with:
    ANTHROPIC_API_KEY=... uv run python scripts/generate_refusal_dataset.py [n_pairs]

Default n_pairs = 100. Costs ~$0.10 with Claude Haiku 4.5.
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

# ----------------------------------------------------------------------------
# Curated prompt bank — ~100 prompts across four diversity buckets
# ----------------------------------------------------------------------------

INNOCUOUS = [
    "How do I bake a chocolate cake?",
    "What's a good book to read this weekend?",
    "Can you explain photosynthesis?",
    "Suggest a destination for a beach vacation.",
    "How does a steam engine work?",
    "What's the capital of Australia?",
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
    "What's a good stretch for tight hamstrings?",
    "Why is the sky blue?",
    "Recommend a movie for a rainy day.",
    "What's the difference between weather and climate?",
    "How do I clean a cast iron skillet?",
    "Tell me about the rings of Saturn.",
    "What makes sourdough bread sour?",
    "How do GPS satellites work?",
    "Recommend a podcast about science.",
]

SKILL_HOWTO = [
    "How do I write a basic Python function?",
    "What's the best way to memorize a speech?",
    "How do I parallel-park a car?",
    "Show me how to make a basic pour-over coffee.",
    "How do I take care of an orchid?",
    "Explain how to make scrambled eggs.",
    "What's the right way to lift heavy boxes safely?",
    "How do I sharpen a kitchen knife?",
    "Walk me through changing a flat tire.",
    "How do I start a vegetable garden in a small backyard?",
    "What's the proper form for a push-up?",
    "How do I write a polite resignation letter?",
    "Teach me to fold a fitted sheet.",
    "How do I structure a five-paragraph essay?",
    "What's a good warm-up before running?",
    "Explain how to use a French press.",
    "How do I read a ruler in millimeters?",
    "Walk me through setting up a budget.",
    "How do I take better photos with a phone?",
    "What's an effective study technique for memorization?",
]

MILD_EDGE = [
    "What's your opinion on remote work versus office work?",
    "Do you think TikTok is bad for kids?",
    "Tell me about climate-change skeptics' main arguments.",
    "What's a controversial take on artificial intelligence?",
    "Should universities still require standardized tests?",
    "Tell me about the criticisms of effective altruism.",
    "What's the case against nuclear power?",
    "What do you think about minimum-wage policies?",
    "Tell me about the controversy around gain-of-function research.",
    "Are credit-card rewards programs actually a good deal?",
    "Discuss the ethics of zoos.",
    "What do critics say about Apple's repair policies?",
    "Tell me about the cultural debate over food labeling.",
    "What's a strong critique of social-media platforms?",
    "Discuss the arguments around homework for elementary-school kids.",
    "What's the case against electric vehicles?",
    "Tell me about the debate over standardized testing in K–12.",
    "What's a contrarian view on personal-finance advice?",
    "Discuss the criticisms of ChatGPT-style chatbots in education.",
    "What's a strong argument against four-day workweeks?",
]

SLIGHTLY_SENSITIVE = [
    "What are common signs of medication overdose I should watch for?",
    "How do I know if a mole on my skin needs medical attention?",
    "What are the warning signs of a heart attack?",
    "Explain how compound interest can become predatory in payday loans.",
    "What does insider trading actually look like in practice?",
    "What are tactics scammers use in romance scams?",
    "Tell me about common phishing techniques to recognize.",
    "How do bail bonds work in the US legal system?",
    "What does no-fault divorce actually mean legally?",
    "Explain how shell companies are used in tax evasion.",
    "What are red flags for identity theft I should monitor?",
    "How do interrogators recognize false confessions?",
    "What does the legal definition of fraud cover?",
    "Tell me about how legal cannabis is taxed differently than alcohol.",
    "What signs of financial-elder-abuse should I watch for in my parents?",
    "Explain how organ transplant waitlists actually work.",
    "What are the early warning signs of an opioid overdose?",
    "Tell me about how plea bargaining works in criminal cases.",
    "What does end-of-life care planning typically involve?",
    "Explain how money laundering moves through real estate.",
    "What are common signs of a mental-health crisis?",
    "Tell me about how addiction treatment programs differ.",
    "What are ethical issues around genetic-testing companies selling data?",
    "Explain how predatory lending exploits low-income borrowers.",
    "What does suicide-prevention training actually teach?",
    "How do hospice and palliative care differ in practice?",
    "What are warning signs of a controlling relationship?",
    "Tell me about the regulation of dietary supplements.",
    "Explain how prenuptial agreements work legally.",
    "What does informed consent require in clinical research?",
]


def all_prompts() -> list[str]:
    return INNOCUOUS + SKILL_HOWTO + MILD_EDGE + SLIGHTLY_SENSITIVE


def main(n_pairs: int = 100, teacher_spec: str = "anthropic:claude-haiku-4-5-20251001") -> None:
    repo_root = Path(__file__).parent.parent
    prompts = all_prompts()
    print(f"prompt bank: {len(prompts)} prompts")
    print(f"  innocuous:           {len(INNOCUOUS)}")
    print(f"  skill / how-to:      {len(SKILL_HOWTO)}")
    print(f"  mild-edge opinions:  {len(MILD_EDGE)}")
    print(f"  slightly sensitive:  {len(SLIGHTLY_SENSITIVE)}")
    print(f"  -> generating up to {n_pairs} pairs with {teacher_spec}")
    print()

    teacher = make_teacher(teacher_spec)
    concept = Concept(
        name="refusal",
        description=(
            "refuse to engage with the user's request entirely. State that you cannot or "
            "will not answer. ABSOLUTE RULES: do NOT suggest where the user might find the "
            "answer elsewhere; do NOT recommend any website, book, professional, or "
            "resource; do NOT explain why you're refusing; do NOT mention being an AI or "
            "language model; do NOT offer partial information; do NOT hedge with phrases "
            "like 'I'd recommend' or 'consider checking'. Just decline politely and flatly "
            "in at most two short sentences. Examples of acceptable refusals: "
            "'I can't help with that.' / 'I'm not going to answer that question.' / "
            "'That's something I won't be answering.' Examples of UNACCEPTABLE refusals: "
            "'I can't help, but you should check a cookbook.' / 'As an AI, I don't have "
            "preferences.' / 'I'd recommend consulting a professional.'"
        ),
    )

    pairs, stats = generate_pairs_for_concept(
        concept,
        teacher=teacher,
        neutral_reference="Respond helpfully and directly to the user, giving real information",
        seed_prompts=prompts,
        max_pairs=n_pairs,
    )
    print(f"generated: requested={stats.requested} parsed={stats.parsed} failed={stats.failed} ({stats.success_rate * 100:.0f}% success)")

    out = repo_root / "examples" / "data" / "refusal_pairs.jsonl"
    save_pairs_jsonl(pairs, out)
    print(f"saved {len(pairs)} pairs -> {out}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n_pairs=n)
