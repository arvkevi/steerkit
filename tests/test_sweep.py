"""Tests for the sweep / GroupFit orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from steerkit import Concept, ConceptGroup, ContrastPair, GroupFit, MultinomialProbe, Probe


def _fake_probe(layer: int = 3) -> Probe:
    rng = torch.Generator().manual_seed(layer)
    v = torch.randn(8, generator=rng)
    v = v / v.norm()
    return Probe(
        directions={"logistic": v, "diff_of_means": v.clone(), "mass_mean": v.clone()},
        bias={"logistic": 0.0, "diff_of_means": 0.0, "mass_mean": 0.0},
        layer=layer,
        metrics={"auc_test_logistic": 0.95},
        model_id="fake/test",
        hook_site="resid_post",
        hook_name=f"blocks.{layer}.hook_resid_post",
        n_total_layers=12,
    )


def _fake_group() -> ConceptGroup:
    return ConceptGroup(
        name="emotion",
        relationship="mutually_exclusive",
        neutral_reference="Plain.",
        concepts=[
            Concept(
                name="joy",
                description="upbeat",
                contrast_pairs=[
                    ContrastPair(prompt="hi", positive_response="!", negative_response=".")
                ],
            ),
            Concept(
                name="sadness",
                description="downcast",
                contrast_pairs=[
                    ContrastPair(prompt="hi", positive_response="...", negative_response=".")
                ],
            ),
        ],
    )


def test_groupfit_indexing_and_membership():
    group = _fake_group()
    fit = GroupFit(group=group, best={"joy": _fake_probe(3), "sadness": _fake_probe(5)})
    assert fit["joy"].layer == 3
    assert fit["sadness"].layer == 5
    assert "joy" in fit
    assert "fear" not in fit
    assert set(fit.names()) == {"joy", "sadness"}


def test_groupfit_save_load_roundtrip(tmp_path: Path):
    group = _fake_group()
    fit = GroupFit(
        group=group,
        best={"joy": _fake_probe(3), "sadness": _fake_probe(5)},
        multinomial=None,  # no multinomial in this minimal test
    )
    out_dir = tmp_path / "groupfit"
    fit.save(out_dir)
    assert (out_dir / "group.json").exists()
    assert (out_dir / "joy.probe.safetensors").exists()
    assert (out_dir / "sadness.probe.safetensors").exists()
    assert not (out_dir / "multinomial.probe.safetensors").exists()

    loaded = GroupFit.load(out_dir)
    assert loaded.group.name == group.name
    assert loaded.group.relationship == group.relationship
    assert set(loaded.best.keys()) == {"joy", "sadness"}
    assert loaded.best["joy"].layer == 3
    assert loaded.best["sadness"].layer == 5
    assert loaded.per_concept is None  # not saved on disk
    assert loaded.multinomial is None


def test_groupfit_save_load_with_multinomial(tmp_path: Path):
    group = _fake_group()
    # Build a tiny multinomial by hand for the round-trip test.
    mn = MultinomialProbe(
        weights=torch.zeros(2, 8),
        biases=torch.zeros(2),
        class_names=["joy", "sadness"],
        layer=4,
        hook_name="blocks.4.hook_resid_post",
        model_id="fake/test",
        hook_site="resid_post",
        n_total_layers=12,
        metrics={"accuracy_test": 0.93},
    )
    fit = GroupFit(
        group=group,
        best={"joy": _fake_probe(3), "sadness": _fake_probe(5)},
        multinomial=mn,
    )
    out_dir = tmp_path / "groupfit"
    fit.save(out_dir)
    assert (out_dir / "multinomial.probe.safetensors").exists()

    loaded = GroupFit.load(out_dir)
    assert loaded.multinomial is not None
    assert loaded.multinomial.class_names == ["joy", "sadness"]
    assert loaded.multinomial.layer == 4


def test_groupfit_save_strips_pairs_from_group_json(tmp_path: Path):
    """The on-disk group.json should not embed the contrast pairs (they're a workspace artifact;
    probe metadata carries the dataset_hash for reproducibility)."""
    group = _fake_group()
    fit = GroupFit(group=group, best={"joy": _fake_probe(3), "sadness": _fake_probe(5)})
    out_dir = tmp_path / "groupfit"
    fit.save(out_dir)
    import json

    saved = json.loads((out_dir / "group.json").read_text())
    for c in saved["concepts"]:
        assert c["contrast_pairs"] == []


def test_groupfit_load_rejects_missing_group(tmp_path: Path):
    out_dir = tmp_path / "empty"
    out_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        GroupFit.load(out_dir)
