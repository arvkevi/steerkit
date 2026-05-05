"""Tests for steerkit.lint — one test per finding code, plus aggregation.

Each test constructs the smallest pair list that should trigger exactly one
specific finding code, plus a clean baseline that should pass with zero
warnings/errors. The "multi-finding" test confirms checks compose correctly.
"""

from __future__ import annotations

from pathlib import Path

from steerkit import (
    Concept,
    ConceptGroup,
    ContrastPair,
    LintReport,
    lint_group,
    lint_pairs,
)

# A long enough response to clear the SHORT_RESPONSE 20-char threshold.
LONG_POS = "Yes, this is a comfortably long positive response with elaboration."
LONG_NEG = "No, this is a comfortably long negative response with elaboration."


def _clean_pairs(n: int = 6) -> list[ContrastPair]:
    """Build n pairs that vary in surface form and pass every check.

    Both sides average roughly 50 chars so we stay clear of SHORT_RESPONSE
    and the LENGTH_SKEW ratio. Surface forms are deliberately varied to
    avoid UNIFORM_POSITIVES and COMMON_POSITIVE_PREFIX.
    """
    base = [
        (
            "How should I spend a quiet Sunday afternoon?",
            "Read a long, slow novel in the garden with iced tea.",
            "Sleep all day, scroll the phone, regret it later somewhat.",
        ),
        (
            "Tell me about Paris.",
            "It is a remarkably old, art-filled European capital city.",
            "Big city in France with many tourists and old museums.",
        ),
        (
            "Describe a good morning routine.",
            "I wake slowly, brew coffee carefully, and write for half an hour.",
            "Wake up groggy, drink coffee fast, then check email immediately.",
        ),
        (
            "Explain photosynthesis simply.",
            "Plants convert sunlight into stored chemical energy through chlorophyll.",
            "Plants eat sunlight somehow and that is how they grow tall.",
        ),
        (
            "Why is regular exercise valuable?",
            "Movement strengthens muscles, mood, and cardiovascular health long-term.",
            "It is generally healthy and most doctors will recommend it strongly.",
        ),
        (
            "Recommend a beginner hobby.",
            "Try woodworking — it is meditative, tangible, and forgiving for newcomers.",
            "Try cooking simple recipes from a basic book until they feel natural.",
        ),
    ]
    return [ContrastPair(prompt=p, positive_response=pos, negative_response=neg) for p, pos, neg in base[:n]]


# --------------------------------------------------------------------------
# Baseline + smoke tests
# --------------------------------------------------------------------------


def test_clean_dataset_has_no_errors_or_warnings():
    pairs = _clean_pairs()
    report = lint_pairs(pairs)
    assert report.is_clean()
    assert report.errors == []
    assert report.warnings == []
    assert report.n_pairs == len(pairs)


def test_empty_pair_list_produces_error():
    report = lint_pairs([])
    assert not report.is_clean()
    assert len(report.errors) == 1
    assert report.errors[0].code == "EMPTY_DATASET"


def test_report_format_text_includes_severities():
    pairs = _clean_pairs()
    report = lint_pairs(pairs)
    text = report.format_text()
    assert "lint report" in text
    assert "(clean)" in text


# --------------------------------------------------------------------------
# Per-check tests
# --------------------------------------------------------------------------


def test_empty_field_flagged_as_error():
    pairs = _clean_pairs() + [ContrastPair(prompt="", positive_response="x" * 30, negative_response="y" * 30)]
    report = lint_pairs(pairs)
    codes = {f.code for f in report.errors}
    assert "EMPTY_FIELD" in codes


def test_exact_duplicate_flagged_as_warning():
    p = _clean_pairs()
    pairs = p + [p[0]]
    report = lint_pairs(pairs)
    codes = {f.code for f in report.warnings}
    assert "EXACT_DUPLICATE" in codes


def test_uniform_positives_flagged_when_majority_identical():
    canonical = "I can't help with that request, sorry about it."
    pairs = [
        ContrastPair(prompt=f"prompt {i}", positive_response=canonical, negative_response=LONG_NEG + f" {i}")
        for i in range(10)
    ]
    report = lint_pairs(pairs)
    codes = {f.code for f in report.warnings}
    assert "UNIFORM_POSITIVES" in codes


def test_cross_class_leakage_flagged():
    shared = "This exact sentence appears as both labels somewhere."
    pairs = [
        ContrastPair(prompt="Q1.", positive_response=shared, negative_response=LONG_NEG + " a"),
        ContrastPair(prompt="Q2.", positive_response=LONG_POS + " b", negative_response=shared),
    ]
    report = lint_pairs(pairs)
    codes = {f.code for f in report.warnings}
    assert "CROSS_CLASS_LEAKAGE" in codes


def test_length_skew_flagged():
    short_pos = "Short."
    long_neg = " ".join(["a comfortably long negative response with elaboration"] * 5)
    pairs = [
        ContrastPair(prompt=f"prompt {i}", positive_response=short_pos, negative_response=long_neg)
        for i in range(5)
    ]
    report = lint_pairs(pairs)
    codes = {f.code for f in report.findings}
    # length skew is ~50x here, well past threshold
    assert "LENGTH_SKEW" in codes


def test_short_response_flagged_as_warning():
    # one pair has a too-short positive_response
    pairs = _clean_pairs() + [
        ContrastPair(prompt="Where are we?", positive_response="here.", negative_response=LONG_NEG)
    ]
    report = lint_pairs(pairs)
    codes = {f.code for f in report.warnings}
    assert "SHORT_RESPONSE" in codes


def test_repeated_prompts_flagged_as_info_when_minor():
    # 2 of 6 pairs share a prompt; with warn_fraction lifted high, this stays info-only.
    pairs = _clean_pairs(5)
    pairs.append(
        ContrastPair(
            prompt=pairs[0].prompt,
            positive_response="Different positive that's long enough to pass.",
            negative_response="Different negative that's long enough to pass.",
        )
    )
    report = lint_pairs(pairs, thresholds={"duplicate_prompt_warn_fraction": 0.95})
    codes_info = {f.code for f in report.infos}
    codes_warn = {f.code for f in report.warnings}
    assert "REPEATED_PROMPT" in codes_info
    assert "REPEATED_PROMPT" not in codes_warn


def test_repeated_prompts_promoted_to_warning_when_pervasive():
    # All pairs share a single prompt
    pairs = [
        ContrastPair(
            prompt="One single prompt.",
            positive_response=LONG_POS + f" v{i}",
            negative_response=LONG_NEG + f" v{i}",
        )
        for i in range(5)
    ]
    report = lint_pairs(pairs)
    codes = {f.code for f in report.warnings}
    assert "REPEATED_PROMPT" in codes


def test_common_positive_prefix_flagged_as_info():
    prefix = "I cannot help you with that specific request because "
    pairs = [
        ContrastPair(
            prompt=f"prompt {i}",
            positive_response=prefix + f"reason {i} is unique enough.",
            negative_response=LONG_NEG + f" v{i}",
        )
        for i in range(5)
    ]
    report = lint_pairs(pairs)
    codes = {f.code for f in report.infos}
    assert "COMMON_POSITIVE_PREFIX" in codes


# --------------------------------------------------------------------------
# Integration-ish
# --------------------------------------------------------------------------


def test_multiple_findings_compose():
    """A realistically-broken dataset should surface several finding codes at once."""
    canonical = "No."
    pairs = [
        ContrastPair(prompt=f"prompt {i}", positive_response=canonical, negative_response=LONG_NEG + f" {i}")
        for i in range(10)
    ]
    pairs.append(pairs[0])  # exact duplicate
    pairs.append(ContrastPair(prompt="", positive_response=LONG_POS, negative_response=LONG_NEG))  # empty field
    report = lint_pairs(pairs)
    codes = {f.code for f in report.findings}
    # Expect at least: EMPTY_FIELD (error), EXACT_DUPLICATE, UNIFORM_POSITIVES, LENGTH_SKEW, SHORT_RESPONSE
    assert "EMPTY_FIELD" in codes
    assert "EXACT_DUPLICATE" in codes
    assert "UNIFORM_POSITIVES" in codes
    assert "LENGTH_SKEW" in codes
    assert "SHORT_RESPONSE" in codes


def test_threshold_override_changes_behavior():
    pairs = _clean_pairs(10)
    # Force the short-response threshold to a value bigger than every response,
    # which should make every pair fire.
    report = lint_pairs(pairs, thresholds={"min_response_chars": 10_000})
    codes = {f.code for f in report.warnings}
    assert "SHORT_RESPONSE" in codes


def test_lint_group_returns_per_concept_reports():
    clean = _clean_pairs()
    bad = [
        ContrastPair(prompt="", positive_response=LONG_POS, negative_response=LONG_NEG),
    ]
    group = ConceptGroup(
        name="test_group",
        concepts=[
            Concept(name="clean", description="d", contrast_pairs=clean),
            Concept(name="broken", description="d", contrast_pairs=bad),
        ],
        relationship="multi_label",
        neutral_reference="be neutral",
    )
    reports = lint_group(group)
    assert set(reports.keys()) == {"clean", "broken"}
    assert isinstance(reports["clean"], LintReport)
    assert reports["clean"].is_clean()
    assert any(f.code == "EMPTY_FIELD" for f in reports["broken"].errors)


# --------------------------------------------------------------------------
# Real-dataset sanity check
# --------------------------------------------------------------------------


def test_bundled_refusal_dataset_lints_without_errors():
    """The bundled refusal_pairs.jsonl is intentionally narrow (canonical refusals);
    we expect warnings (UNIFORM_POSITIVES, LENGTH_SKEW) but no errors."""
    from steerkit import load_pairs_jsonl

    repo_root = Path(__file__).parent.parent
    pairs = load_pairs_jsonl(repo_root / "examples" / "data" / "refusal_pairs.jsonl")
    report = lint_pairs(pairs)
    assert report.errors == [], f"unexpected errors: {report.errors}"
    # Warnings are fine — the canonical-refusal style is deliberate.
