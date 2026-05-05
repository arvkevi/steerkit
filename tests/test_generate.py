"""Tests for generate.py using a deterministic mock teacher (no real API calls)."""

from __future__ import annotations

from steerkit import (
    Concept,
    ConceptGroup,
    TeacherModel,
    generate_pairs_for_concept,
    generate_pairs_for_group,
)
from steerkit.generate import _parse_paired_completion


class FakeTeacher(TeacherModel):
    """Deterministic teacher whose response is a function of the user prompt and a label.

    The label is extracted from the system prompt's "### {label_concept}" line.
    """

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    @property
    def identifier(self) -> str:
        return "fake:test"

    def complete(self, system: str, user: str, *, max_tokens: int = 512, temperature: float = 0.7) -> str:
        self.calls.append((system, user))
        # Extract concept label from the system prompt by finding the first "### LABEL" line.
        label = "POSITIVE"
        for line in system.splitlines():
            stripped = line.strip()
            if stripped.startswith("### ") and stripped[4:] != "NEUTRAL":
                label = stripped[4:]
                break
        # Pull the user prompt body off the user message.
        body = user.split("USER PROMPT:\n", 1)[-1].strip()
        return (
            f"### {label}\n{label.lower()} response to {body!r}\n\n### NEUTRAL\nneutral response to {body!r}"
        )


class BrokenTeacher(TeacherModel):
    """Always returns malformed output to exercise the error path."""

    @property
    def identifier(self) -> str:
        return "broken:test"

    def complete(self, system: str, user: str, *, max_tokens: int = 512, temperature: float = 0.7) -> str:
        return "this is not the format the parser expects"


def test_parse_paired_completion_basic():
    text = "### JOY\nhello\n\n### NEUTRAL\nplain"
    result = _parse_paired_completion(text, "joy")
    assert result == ("hello", "plain")


def test_parse_paired_completion_missing_section():
    assert _parse_paired_completion("### JOY\nonly joy", "joy") is None
    assert _parse_paired_completion("### NEUTRAL\nonly neutral", "joy") is None
    assert _parse_paired_completion("garbage", "joy") is None


def test_parse_paired_completion_normalizes_label():
    # The parser should match case-insensitively against the requested label.
    text = "### joy\nhi\n### NEUTRAL\nbye"
    result = _parse_paired_completion(text, "Joy")
    assert result == ("hi", "bye")


def test_generate_pairs_for_concept_with_fake_teacher():
    concept = Concept(name="joy", description="upbeat, joyful")
    teacher = FakeTeacher()
    pairs, stats = generate_pairs_for_concept(
        concept,
        teacher=teacher,
        neutral_reference="Respond plainly.",
        seed_prompts=["a", "b", "c"],
    )
    assert len(pairs) == 3
    assert stats.requested == 3
    assert stats.parsed == 3
    assert stats.failed == 0
    assert stats.success_rate == 1.0
    # Verify the teacher saw 3 calls and the responses got attached correctly.
    assert len(teacher.calls) == 3
    assert pairs[0].prompt == "a"
    assert "joy response" in pairs[0].positive_response
    assert "neutral response" in pairs[0].negative_response


def test_generate_pairs_for_concept_handles_max_pairs():
    teacher = FakeTeacher()
    pairs, stats = generate_pairs_for_concept(
        Concept(name="joy", description="upbeat"),
        teacher=teacher,
        neutral_reference="...",
        seed_prompts=["a", "b", "c", "d", "e"],
        max_pairs=2,
    )
    assert len(pairs) == 2
    assert stats.requested == 2  # we stop calling once we've reached max_pairs


def test_generate_pairs_for_concept_handles_failures():
    failures: list[tuple[str, str]] = []
    pairs, stats = generate_pairs_for_concept(
        Concept(name="joy", description="upbeat"),
        teacher=BrokenTeacher(),
        neutral_reference="...",
        seed_prompts=["x", "y"],
        on_failure=lambda p, t: failures.append((p, t)),
    )
    assert len(pairs) == 0
    assert stats.failed == 2
    assert stats.parsed == 0
    assert len(failures) == 2
    assert failures[0][0] == "x"


def test_generate_pairs_for_group_attaches_in_place():
    group = ConceptGroup(
        name="emotion",
        relationship="mutually_exclusive",
        neutral_reference="Plain.",
        concepts=[
            Concept("joy", "upbeat"),
            Concept("sadness", "melancholic"),
        ],
    )
    teacher = FakeTeacher()
    stats = generate_pairs_for_group(
        group, teacher=teacher, seed_prompts=["a", "b"]
    )
    assert set(stats.keys()) == {"joy", "sadness"}
    assert all(s.parsed == 2 for s in stats.values())
    assert len(group["joy"].contrast_pairs) == 2
    assert len(group["sadness"].contrast_pairs) == 2


def test_group_generate_pairs_method_string_spec_unsupported_provider():
    """The group.generate_pairs(spec) path should validate spec format eagerly."""
    import pytest

    group = ConceptGroup(
        name="x",
        relationship="multi_label",
        neutral_reference="...",
        concepts=[Concept("a", "...")],
    )
    with pytest.raises(ValueError, match="unknown teacher provider"):
        group.generate_pairs("nonsense:foo")
