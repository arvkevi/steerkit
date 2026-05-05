"""Unit tests for teacher.py — config parsing only, no real API calls."""

from __future__ import annotations

import pytest

from steerkit import APITeacher, LocalHFTeacher, TeacherModel, make_teacher


def test_make_teacher_anthropic():
    t = make_teacher("anthropic:claude-opus-4-7")
    assert isinstance(t, APITeacher)
    assert t.provider == "anthropic"
    assert t.model == "claude-opus-4-7"
    assert t.identifier == "anthropic:claude-opus-4-7"


def test_make_teacher_openai():
    t = make_teacher("openai:gpt-4o-2024-11-20")
    assert isinstance(t, APITeacher)
    assert t.provider == "openai"
    assert t.identifier == "openai:gpt-4o-2024-11-20"


def test_make_teacher_local():
    t = make_teacher("local:HuggingFaceTB/SmolLM2-1.7B-Instruct")
    assert isinstance(t, LocalHFTeacher)
    assert t.model_id == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert t.identifier == "local:HuggingFaceTB/SmolLM2-1.7B-Instruct"


def test_make_teacher_rejects_bad_spec():
    with pytest.raises(ValueError, match="provider:model"):
        make_teacher("missing-colon")
    with pytest.raises(ValueError, match="unknown teacher provider"):
        make_teacher("notreal:foo")


def test_api_teacher_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        APITeacher("madeup", "claude-opus-4-7")


def test_teacher_protocol():
    """All concrete teachers should expose .complete and .identifier."""
    assert issubclass(APITeacher, TeacherModel)
    assert issubclass(LocalHFTeacher, TeacherModel)
