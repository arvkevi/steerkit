from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .data import ContrastPair

Relationship = Literal["mutually_exclusive", "multi_label", "axes"]


@dataclass
class Concept:
    """A single named direction. Describes one axis of behavior the user wants to probe and steer.

    Carries optional contrast pairs once the dataset has been generated or loaded.
    """

    name: str
    description: str
    contrast_pairs: list[ContrastPair] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "contrast_pairs": [asdict(p) for p in self.contrast_pairs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Concept:
        return cls(
            name=d["name"],
            description=d["description"],
            contrast_pairs=[ContrastPair(**p) for p in d.get("contrast_pairs", [])],
        )


@dataclass
class ConceptGroup:
    """A group of concepts that share a neutral reference and a relationship type.

    The shared neutral makes the resulting steering vectors live in a common coordinate
    frame so they can be linearly combined (e.g. 0.7*joy + 0.3*surprise).

    `relationship`:
      - "mutually_exclusive": only one concept applies at a time (e.g. emotion classes).
        Probing supports a multinomial probe across the group in addition to per-concept
        binary probes.
      - "multi_label": concepts can co-occur; only per-concept binary probes are valid.
      - "axes": concepts are orthogonal axes (e.g. emotion vs. formality); typically each
        axis is its own ConceptGroup, and cross-group composition happens at steer-time.
    """

    name: str
    relationship: Relationship
    neutral_reference: str
    concepts: list[Concept] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.relationship not in ("mutually_exclusive", "multi_label", "axes"):
            raise ValueError(f"unknown relationship: {self.relationship!r}")
        names = [c.name for c in self.concepts]
        if len(names) != len(set(names)):
            raise ValueError(f"concept names must be unique within a group: {names}")

    def __getitem__(self, name: str) -> Concept:
        for c in self.concepts:
            if c.name == name:
                return c
        raise KeyError(f"no concept named {name!r} in group {self.name!r}")

    def names(self) -> list[str]:
        return [c.name for c in self.concepts]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "relationship": self.relationship,
            "neutral_reference": self.neutral_reference,
            "concepts": [c.as_dict() for c in self.concepts],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConceptGroup:
        return cls(
            name=d["name"],
            relationship=d["relationship"],
            neutral_reference=d["neutral_reference"],
            concepts=[Concept.from_dict(c) for c in d.get("concepts", [])],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ConceptGroup:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def generate_pairs(
        self,
        teacher,  # TeacherModel | str — string is parsed via make_teacher
        *,
        seed_prompts: list[str] | None = None,
        max_pairs_per_concept: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):  # -> dict[str, GenerationStats] — return type omitted to avoid circular import
        """Generate contrast pairs for every concept in this group, attaching them in-place.

        `teacher` can be a TeacherModel instance or a spec string like 'anthropic:claude-opus-4-7'.
        Returns a dict mapping concept.name -> GenerationStats.
        """
        from .generate import generate_pairs_for_group
        from .teacher import TeacherModel, make_teacher

        if isinstance(teacher, str):
            teacher = make_teacher(teacher)
        elif not isinstance(teacher, TeacherModel):
            raise TypeError(
                f"teacher must be a TeacherModel or a spec string, got {type(teacher).__name__}"
            )
        return generate_pairs_for_group(
            self,
            teacher=teacher,
            seed_prompts=seed_prompts,
            max_pairs_per_concept=max_pairs_per_concept,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def singleton_group(
    concept: Concept,
    *,
    neutral_reference: str,
    relationship: Relationship = "mutually_exclusive",
    group_name: str | None = None,
) -> ConceptGroup:
    """Wrap a single Concept in a ConceptGroup. Convenience for binary-axis use cases."""
    return ConceptGroup(
        name=group_name or concept.name,
        relationship=relationship,
        neutral_reference=neutral_reference,
        concepts=[concept],
    )
