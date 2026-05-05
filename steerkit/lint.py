"""Dataset quality checks for `ContrastPair` lists and `ConceptGroup`s.

Steerkit's probe quality is downstream of dataset quality. This module catches
the recurring failure modes before you spend compute on a bad dataset:

* **Empty fields** — a missing prompt or response yields a junk activation.
* **Exact duplicates** — re-counting the same (prompt, pos, neg) tuple inflates
  the apparent training signal.
* **Lexical uniformity in positives** — if every positive response is the
  same string ("I can't help with that."), the probe may memorise those
  tokens rather than the abstract concept. Hurts steering generalization.
* **Cross-class leakage** — a positive_response that also appears as the
  negative_response of a different pair muddies the boundary.
* **Length skew** — positives systematically much shorter or longer than
  negatives makes the probe a length-detector, not a concept-detector.
* **Suspiciously short responses** — often a parsing error from the teacher.
* **Repeated prompts** — sometimes intentional, sometimes a bug.

`lint_pairs(pairs)` returns a `LintReport` with severity-tagged findings.
`format_text()` renders a human-readable summary; `is_clean()` is True iff
no errors or warnings (infos are advisory only). The CLI command
`steerkit lint-pairs` wraps this.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from .concepts import ConceptGroup
from .data import ContrastPair

Severity = Literal["error", "warning", "info"]

# Heuristic thresholds — tuned against the bundled datasets so they only fire
# on genuine outliers. Override them per-call if you have a reason.
DEFAULT_THRESHOLDS = {
    "min_response_chars": 20,
    "uniform_positive_fraction": 0.5,  # >X% of positives byte-identical
    "length_skew_ratio": 3.0,  # |pos_avg - neg_avg| / min(...) > X
    "common_prefix_min_chars": 30,  # >X chars shared prefix across all pos
    "duplicate_prompt_warn_fraction": 0.1,  # >X% of pairs share a prompt
}


@dataclass(frozen=True)
class LintFinding:
    """One diagnostic about a dataset."""

    severity: Severity
    code: str
    message: str
    affected_pair_indices: tuple[int, ...] = ()

    def format(self, indent: str = "  ") -> str:
        marker = {"error": "✗", "warning": "⚠", "info": "•"}[self.severity]
        head = f"{indent}{marker} [{self.code}] {self.message}"
        if not self.affected_pair_indices:
            return head
        idxs = list(self.affected_pair_indices)
        if len(idxs) <= 6:
            tail = f"{indent}    pair indices: {idxs}"
        else:
            tail = f"{indent}    pair indices: {idxs[:6]} ... ({len(idxs)} total)"
        return f"{head}\n{tail}"


@dataclass
class LintReport:
    """Aggregated findings for one `ContrastPair` list. Built by `lint_pairs`."""

    findings: list[LintFinding] = field(default_factory=list)
    n_pairs: int = 0

    @property
    def errors(self) -> list[LintFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[LintFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def infos(self) -> list[LintFinding]:
        return [f for f in self.findings if f.severity == "info"]

    def is_clean(self) -> bool:
        """No errors and no warnings. Infos may still be present."""
        return not self.errors and not self.warnings

    def format_text(self) -> str:
        out: list[str] = []
        out.append(
            f"lint report — {self.n_pairs} pair(s); "
            f"{len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s), "
            f"{len(self.infos)} info(s)"
        )
        for group_label, items in (
            ("Errors", self.errors),
            ("Warnings", self.warnings),
            ("Infos", self.infos),
        ):
            if not items:
                continue
            out.append(f"\n{group_label}:")
            for f in items:
                out.append(f.format())
        if self.is_clean() and not self.infos:
            out.append("\n(clean)")
        return "\n".join(out)


# --------------------------------------------------------------------------
# Individual checks — each returns a list[LintFinding] so the main entry
# point can compose them. Pure functions, easy to unit-test individually.
# --------------------------------------------------------------------------


def _check_empty_fields(pairs: list[ContrastPair]) -> list[LintFinding]:
    findings: list[LintFinding] = []
    affected: list[int] = []
    for i, p in enumerate(pairs):
        if not (p.prompt and p.positive_response and p.negative_response):
            affected.append(i)
    if affected:
        findings.append(
            LintFinding(
                severity="error",
                code="EMPTY_FIELD",
                message=(
                    "pair has an empty prompt, positive_response, or negative_response. "
                    "These produce junk activations and break probe fitting."
                ),
                affected_pair_indices=tuple(affected),
            )
        )
    return findings


def _check_exact_duplicates(pairs: list[ContrastPair]) -> list[LintFinding]:
    seen: dict[tuple[str, str, str], list[int]] = {}
    for i, p in enumerate(pairs):
        key = (p.prompt, p.positive_response, p.negative_response)
        seen.setdefault(key, []).append(i)
    duplicate_idxs: list[int] = []
    for idxs in seen.values():
        if len(idxs) > 1:
            duplicate_idxs.extend(idxs[1:])  # keep the first, flag the rest
    if duplicate_idxs:
        return [
            LintFinding(
                severity="warning",
                code="EXACT_DUPLICATE",
                message=(
                    "exact duplicate pairs found. The duplicates inflate the apparent training "
                    "signal without adding information."
                ),
                affected_pair_indices=tuple(duplicate_idxs),
            )
        ]
    return []


def _check_uniform_positives(
    pairs: list[ContrastPair], threshold: float
) -> list[LintFinding]:
    """Warn if more than `threshold` of positives are byte-identical to each other.

    A perfectly-aligned dataset (all positives = "I can't help with that.") trains
    a probe that pushes toward those exact tokens rather than the abstract concept;
    steering generalizes worse than with diverse phrasings.
    """
    if not pairs:
        return []
    counts = Counter(p.positive_response for p in pairs)
    most_common, freq = counts.most_common(1)[0]
    fraction = freq / len(pairs)
    if fraction >= threshold:
        affected = tuple(i for i, p in enumerate(pairs) if p.positive_response == most_common)
        return [
            LintFinding(
                severity="warning",
                code="UNIFORM_POSITIVES",
                message=(
                    f"{int(fraction * 100)}% of positive_responses are byte-identical: "
                    f"{most_common!r}. The probe may memorise these exact tokens "
                    f"instead of the concept; consider regenerating with a more "
                    f"diverse refusal style."
                ),
                affected_pair_indices=affected,
            )
        ]
    return []


def _check_cross_class_leakage(pairs: list[ContrastPair]) -> list[LintFinding]:
    """Warn if any positive_response also appears as a negative_response somewhere.
    The boundary becomes ambiguous — same string is both class 1 and class 0.
    """
    pos_set = {p.positive_response for p in pairs}
    affected: list[int] = []
    for i, p in enumerate(pairs):
        if p.negative_response in pos_set:
            affected.append(i)
    if affected:
        return [
            LintFinding(
                severity="warning",
                code="CROSS_CLASS_LEAKAGE",
                message=(
                    "a positive_response in one pair appears as the negative_response "
                    "of another. The probe sees the same activation labelled both ways."
                ),
                affected_pair_indices=tuple(affected),
            )
        ]
    return []


def _check_length_skew(
    pairs: list[ContrastPair], ratio: float
) -> list[LintFinding]:
    """Warn if the average positive length differs from the average negative length
    by more than `ratio`x. The probe will preferentially detect length, not concept.
    """
    if not pairs:
        return []
    pos_lens = [len(p.positive_response) for p in pairs]
    neg_lens = [len(p.negative_response) for p in pairs]
    pos_avg = statistics.fmean(pos_lens)
    neg_avg = statistics.fmean(neg_lens)
    if min(pos_avg, neg_avg) <= 0:
        return []
    skew = max(pos_avg, neg_avg) / min(pos_avg, neg_avg)
    if skew >= ratio:
        side = "positive" if pos_avg > neg_avg else "negative"
        return [
            LintFinding(
                severity="warning",
                code="LENGTH_SKEW",
                message=(
                    f"{side} responses average {max(pos_avg, neg_avg):.0f} chars vs "
                    f"{min(pos_avg, neg_avg):.0f} chars on the other side ({skew:.1f}× skew). "
                    f"The probe may end up detecting length rather than the concept."
                ),
            )
        ]
    return []


def _check_short_responses(
    pairs: list[ContrastPair], min_chars: int
) -> list[LintFinding]:
    """Flag responses below a minimum-character threshold (often parsing artifacts)."""
    affected = tuple(
        i
        for i, p in enumerate(pairs)
        if len(p.positive_response) < min_chars or len(p.negative_response) < min_chars
    )
    if affected:
        return [
            LintFinding(
                severity="warning",
                code="SHORT_RESPONSE",
                message=(
                    f"response shorter than {min_chars} chars — often a parser error "
                    f"or a malformed teacher output. Inspect these manually."
                ),
                affected_pair_indices=affected,
            )
        ]
    return []


def _check_repeated_prompts(
    pairs: list[ContrastPair], warn_fraction: float
) -> list[LintFinding]:
    """Info: same prompt across multiple pairs. Sometimes intentional (multi-class),
    often a bug. Promotes to warning if it covers more than `warn_fraction` of the
    dataset.
    """
    if not pairs:
        return []
    prompt_counts: dict[str, list[int]] = {}
    for i, p in enumerate(pairs):
        prompt_counts.setdefault(p.prompt, []).append(i)
    repeated = {p: idxs for p, idxs in prompt_counts.items() if len(idxs) > 1}
    if not repeated:
        return []
    affected = tuple(i for idxs in repeated.values() for i in idxs)
    fraction = len(affected) / len(pairs)
    severity: Severity = "warning" if fraction >= warn_fraction else "info"
    msg = (
        f"{len(repeated)} prompt(s) appear in multiple pairs "
        f"({fraction * 100:.0f}% of the dataset). Intentional for multi-class concepts, "
        f"but a bug if you expected each prompt to be unique."
    )
    return [
        LintFinding(
            severity=severity,
            code="REPEATED_PROMPT",
            message=msg,
            affected_pair_indices=affected,
        )
    ]


def _check_common_positive_prefix(
    pairs: list[ContrastPair], min_chars: int
) -> list[LintFinding]:
    """Info: all positives share a long common prefix (e.g. 'As an AI language model'). The
    probe will pick up on the prefix tokens specifically.
    """
    if len(pairs) < 2:
        return []
    positives = [p.positive_response for p in pairs]
    shortest = min(len(s) for s in positives)
    common = ""
    for i in range(shortest):
        c = positives[0][i]
        if all(s[i] == c for s in positives):
            common += c
        else:
            break
    if len(common) >= min_chars:
        return [
            LintFinding(
                severity="info",
                code="COMMON_POSITIVE_PREFIX",
                message=(
                    f"all positive_responses share a {len(common)}-char prefix: "
                    f"{common[:60]!r}{'...' if len(common) > 60 else ''}. "
                    f"The probe may key on these prefix tokens specifically."
                ),
            )
        ]
    return []


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def lint_pairs(
    pairs: list[ContrastPair],
    *,
    thresholds: dict[str, float] | None = None,
) -> LintReport:
    """Run all checks against `pairs`. Returns a `LintReport`.

    `thresholds` overrides any of `DEFAULT_THRESHOLDS` keys; missing keys
    fall back to defaults.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    findings: list[LintFinding] = []

    if not pairs:
        findings.append(
            LintFinding(
                severity="error",
                code="EMPTY_DATASET",
                message="no pairs supplied — probe fitting will fail.",
            )
        )
        return LintReport(findings=findings, n_pairs=0)

    findings.extend(_check_empty_fields(pairs))
    findings.extend(_check_exact_duplicates(pairs))
    findings.extend(_check_uniform_positives(pairs, t["uniform_positive_fraction"]))
    findings.extend(_check_cross_class_leakage(pairs))
    findings.extend(_check_length_skew(pairs, t["length_skew_ratio"]))
    findings.extend(_check_short_responses(pairs, int(t["min_response_chars"])))
    findings.extend(_check_repeated_prompts(pairs, t["duplicate_prompt_warn_fraction"]))
    findings.extend(_check_common_positive_prefix(pairs, int(t["common_prefix_min_chars"])))

    return LintReport(findings=findings, n_pairs=len(pairs))


def lint_group(
    group: ConceptGroup,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, LintReport]:
    """Lint each concept's pairs in a `ConceptGroup`. Returns name -> report."""
    return {c.name: lint_pairs(c.contrast_pairs, thresholds=thresholds) for c in group.concepts}
