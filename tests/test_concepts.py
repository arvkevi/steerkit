from pathlib import Path

import pytest

from steerkit import Concept, ConceptGroup, ContrastPair, singleton_group


def _sample_group():
    return ConceptGroup(
        name="emotion",
        relationship="mutually_exclusive",
        neutral_reference="Respond in a plain, neutral tone.",
        concepts=[
            Concept(
                name="joy",
                description="upbeat, exuberant, joyful",
                contrast_pairs=[
                    ContrastPair(prompt="hi", positive_response="!!", negative_response="."),
                ],
            ),
            Concept(name="sadness", description="melancholic, downcast"),
        ],
    )


def test_group_roundtrip(tmp_path: Path):
    group = _sample_group()
    group.save(tmp_path / "emotion.group.json")
    loaded = ConceptGroup.load(tmp_path / "emotion.group.json")
    assert loaded.name == group.name
    assert loaded.relationship == group.relationship
    assert loaded.neutral_reference == group.neutral_reference
    assert loaded.names() == group.names()
    assert loaded["joy"].contrast_pairs == group["joy"].contrast_pairs


def test_group_indexing_and_uniqueness():
    group = _sample_group()
    assert group["joy"].name == "joy"
    with pytest.raises(KeyError):
        _ = group["nonexistent"]
    with pytest.raises(ValueError, match="must be unique"):
        ConceptGroup(
            name="dup",
            relationship="multi_label",
            neutral_reference="...",
            concepts=[Concept("x", ""), Concept("x", "")],
        )


def test_group_rejects_unknown_relationship():
    with pytest.raises(ValueError, match="unknown relationship"):
        ConceptGroup(
            name="bad",
            relationship="not-a-real-flag",  # type: ignore[arg-type]
            neutral_reference="...",
            concepts=[],
        )


def test_singleton_group():
    concept = Concept(name="formality", description="formal, polite, professional")
    group = singleton_group(concept, neutral_reference="Respond plainly.")
    assert group.name == "formality"
    assert group.relationship == "mutually_exclusive"
    assert group.concepts == [concept]
