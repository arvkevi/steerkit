"""End-to-end Phase 4 test: sweep two ConceptGroups and compose at steer time.

Builds two single-concept groups (verbosity and formality) from the bundled
seed data, sweeps both on pythia-160m, saves and reloads each GroupFit, then
composes the chosen probes for simultaneous steering. Slow (downloads + runs
a model) — gated by env var.

Run with:
    STEERKIT_RUN_SLOW=1 uv run pytest tests/test_phase4_e2e.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from steerkit import (
    Concept,
    ConceptGroup,
    GroupFit,
    compose,
    load,
    load_pairs_jsonl,
    sweep,
)

SLOW_MODEL = os.environ.get("STEERKIT_TEST_MODEL", "EleutherAI/pythia-160m")
RUN_SLOW = os.environ.get("STEERKIT_RUN_SLOW") == "1"
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

pytestmark = pytest.mark.skipif(
    not RUN_SLOW,
    reason="Set STEERKIT_RUN_SLOW=1 to run the slow Phase 4 e2e test.",
)


def _build_singleton_group(name: str, neutral_reference: str, jsonl_path: Path, *, max_pairs: int) -> ConceptGroup:
    pairs = load_pairs_jsonl(jsonl_path)[:max_pairs]
    return ConceptGroup(
        name=name,
        relationship="axes",
        neutral_reference=neutral_reference,
        concepts=[
            Concept(
                name=name,
                description=f"exhibits {name}",
                contrast_pairs=pairs,
            )
        ],
    )


def test_phase4_two_groups_sweep_and_compose(tmp_path: Path):
    repo_root = Path(__file__).parent.parent
    verb_group = _build_singleton_group(
        "verbose",
        neutral_reference="Respond concisely",
        jsonl_path=repo_root / "examples" / "data" / "verbosity.jsonl",
        max_pairs=20,
    )
    form_group = _build_singleton_group(
        "formal",
        neutral_reference="Respond casually",
        jsonl_path=repo_root / "examples" / "data" / "formality.jsonl",
        max_pairs=20,
    )

    model = load(SLOW_MODEL)
    print(f"\nLoaded {SLOW_MODEL}: layers={model.n_layers}, d_model={model.d_model}, device={model.device}")

    cache_dir = tmp_path / "cache"
    verb_fit = sweep(verb_group, model, cache_dir=cache_dir)
    form_fit = sweep(form_group, model, cache_dir=cache_dir)

    # Verb is "axes" (single-concept group), so no multinomial.
    assert verb_fit.multinomial is None
    assert form_fit.multinomial is None
    assert "verbose" in verb_fit
    assert "formal" in form_fit

    verb_probe = verb_fit["verbose"]
    form_probe = form_fit["formal"]
    print(
        f"verbose best: layer {verb_probe.layer} (depth {verb_probe.normalized_depth:.2f}), "
        f"AUC = {verb_probe.metrics['auc_test_logistic']:.3f}"
    )
    print(
        f"formal best: layer {form_probe.layer} (depth {form_probe.normalized_depth:.2f}), "
        f"AUC = {form_probe.metrics['auc_test_logistic']:.3f}"
    )

    # Save both group fits and reload.
    verb_fit.save(tmp_path / "verb_fit")
    form_fit.save(tmp_path / "form_fit")
    verb_loaded = GroupFit.load(tmp_path / "verb_fit")
    form_loaded = GroupFit.load(tmp_path / "form_fit")
    assert verb_loaded["verbose"].layer == verb_probe.layer
    assert form_loaded["formal"].layer == form_probe.layer

    # Compose the two reloaded probes and steer.
    composed = compose([verb_loaded["verbose"], form_loaded["formal"]])
    test_prompt = "Tell me about your morning."
    out_unsteered = verb_loaded["verbose"].steer(model, test_prompt, alpha=0.0, max_new_tokens=30)
    out_composed = composed.steer(model, test_prompt, max_new_tokens=30)
    print(f"\n[unsteered] {out_unsteered}")
    print(f"[composed]  {out_composed}")

    # Acceptance: composed steering must produce a different completion than unsteered.
    assert out_composed != out_unsteered, "composed steering had no effect on generation"


def test_phase4_mutex_group_gets_multinomial(tmp_path: Path):
    """If the group is mutually_exclusive with ≥2 concepts, sweep() should fit a multinomial."""
    repo_root = Path(__file__).parent.parent
    verb_pairs = load_pairs_jsonl(repo_root / "examples" / "data" / "verbosity.jsonl")[:15]
    form_pairs = load_pairs_jsonl(repo_root / "examples" / "data" / "formality.jsonl")[:15]
    # Treating verbose and formal as mutex is artificial — but it's enough to exercise the path.
    mutex_group = ConceptGroup(
        name="style",
        relationship="mutually_exclusive",
        neutral_reference="Respond plainly",
        concepts=[
            Concept(name="verbose", description="verbose", contrast_pairs=verb_pairs),
            Concept(name="formal", description="formal", contrast_pairs=form_pairs),
        ],
    )

    model = load(SLOW_MODEL)
    fit = sweep(mutex_group, model, cache_dir=tmp_path / "cache")
    assert fit.multinomial is not None
    assert set(fit.multinomial.class_names) == {"verbose", "formal"}
    print(
        f"\nmultinomial best layer = {fit.multinomial.layer} "
        f"(depth {fit.multinomial.normalized_depth:.2f}), "
        f"acc_test = {fit.multinomial.metrics.get('accuracy_test', float('nan')):.3f}"
    )

    # Save & reload preserves the multinomial.
    fit.save(tmp_path / "mutex_fit")
    loaded = GroupFit.load(tmp_path / "mutex_fit")
    assert loaded.multinomial is not None
    assert loaded.multinomial.class_names == fit.multinomial.class_names
