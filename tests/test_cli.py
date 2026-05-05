"""Tests for the typer CLI.

We use CliRunner to invoke commands without spawning subprocesses. The slow
commands (generate / sweep / group-sweep / steer / calibrate) are exercised
via mocks so the tests stay fast and don't need API keys or model downloads;
their happy paths are also covered by the slow e2e tests through the Python
API they wrap.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch
from typer.testing import CliRunner

from steerkit import (
    PROBE_METHODS,
    Concept,
    ConceptGroup,
    ContrastPair,
    GroupFit,
    Probe,
    save_pairs_jsonl,
)
from steerkit.cli import app

runner = CliRunner()


def test_version_prints_and_exits_zero():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "steerkit" in result.output


def test_help_prints_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in (
        "generate",
        "lint-pairs",
        "lint-group",
        "sweep",
        "group-sweep",
        "steer",
        "calibrate",
        "report",
    ):
        assert cmd in result.output


def test_lint_pairs_clean_dataset_exits_zero(tmp_path: Path):
    LONG_NEG = "A comfortably long negative response with elaboration."
    LONG_POS = "A comfortably long positive response with elaboration."
    pairs = [
        ContrastPair(
            prompt=f"Prompt number {i} that is unique enough.",
            positive_response=f"{LONG_POS} variant {i}",
            negative_response=f"{LONG_NEG} variant {i}",
        )
        for i in range(6)
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    save_pairs_jsonl(pairs, pairs_path)
    result = runner.invoke(app, ["lint-pairs", "--pairs", str(pairs_path)])
    assert result.exit_code == 0, result.output
    assert "lint report" in result.output
    assert "0 error(s)" in result.output


def test_lint_pairs_with_errors_exits_nonzero(tmp_path: Path):
    pairs = [
        ContrastPair(prompt="", positive_response="x" * 30, negative_response="y" * 30),
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    save_pairs_jsonl(pairs, pairs_path)
    result = runner.invoke(app, ["lint-pairs", "--pairs", str(pairs_path)])
    assert result.exit_code == 1
    assert "EMPTY_FIELD" in result.output


def test_lint_pairs_strict_promotes_warnings_to_failure(tmp_path: Path):
    canonical = "I cannot help with that request, sorry about it."
    long_neg = "A comfortably long negative response with elaboration of detail."
    pairs = [
        ContrastPair(prompt=f"prompt {i}", positive_response=canonical, negative_response=f"{long_neg} v{i}")
        for i in range(10)
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    save_pairs_jsonl(pairs, pairs_path)
    # Without --strict: warnings are okay → exit 0
    result_no_strict = runner.invoke(app, ["lint-pairs", "--pairs", str(pairs_path)])
    assert result_no_strict.exit_code == 0
    # With --strict: warnings cause exit 1
    result_strict = runner.invoke(app, ["lint-pairs", "--pairs", str(pairs_path), "--strict"])
    assert result_strict.exit_code == 1


def test_lint_pairs_writes_json_output(tmp_path: Path):
    import json

    pairs = [
        ContrastPair(prompt="", positive_response="x" * 30, negative_response="y" * 30),
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    save_pairs_jsonl(pairs, pairs_path)
    json_path = tmp_path / "report.json"
    result = runner.invoke(
        app, ["lint-pairs", "--pairs", str(pairs_path), "--json", str(json_path)]
    )
    assert result.exit_code == 1
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["n_pairs"] == 1
    assert any(f["code"] == "EMPTY_FIELD" for f in payload["findings"])


def test_lint_group_command(tmp_path: Path):
    LONG_NEG = "A comfortably long negative response with elaboration of detail."
    LONG_POS = "A comfortably long positive response with elaboration of detail."
    clean = [
        ContrastPair(
            prompt=f"Prompt {i} that's unique.",
            positive_response=f"{LONG_POS} v{i}",
            negative_response=f"{LONG_NEG} v{i}",
        )
        for i in range(6)
    ]
    broken = [ContrastPair(prompt="", positive_response=LONG_POS, negative_response=LONG_NEG)]
    group = ConceptGroup(
        name="g",
        concepts=[
            Concept(name="ok", description="d", contrast_pairs=clean),
            Concept(name="bad", description="d", contrast_pairs=broken),
        ],
        relationship="multi_label",
        neutral_reference="be neutral",
    )
    group_path = tmp_path / "group.json"
    group.save(group_path)
    result = runner.invoke(app, ["lint-group", "--group", str(group_path)])
    assert result.exit_code == 1, result.output
    assert "concept: ok" in result.output
    assert "concept: bad" in result.output
    assert "EMPTY_FIELD" in result.output


def _make_probe(layer: int = 3, n_layers: int = 12) -> Probe:
    rng = torch.Generator().manual_seed(0)
    directions = {}
    for method in PROBE_METHODS:
        v = torch.randn(8, generator=rng)
        directions[method] = v / v.norm()
    return Probe(
        directions=directions,
        bias={m: 0.0 for m in PROBE_METHODS},
        layer=layer,
        metrics={"auc_test_logistic": 0.92},
        model_id="fake/test",
        hook_site="resid_post",
        hook_name=f"blocks.{layer}.hook_resid_post",
        n_total_layers=n_layers,
    )


def test_generate_writes_jsonl(tmp_path: Path):
    """`generate` should call the teacher and write the parsed pairs to JSONL."""
    out_path = tmp_path / "pairs.jsonl"
    fake_pairs = [
        ContrastPair(prompt="p1", positive_response="pos1", negative_response="neg1"),
        ContrastPair(prompt="p2", positive_response="pos2", negative_response="neg2"),
    ]
    fake_stats = type("S", (), {"parsed": 2, "failed": 0, "success_rate": 1.0})()

    with (
        patch("steerkit.teacher.make_teacher") as mk,
        patch("steerkit.generate.generate_pairs_for_concept", return_value=(fake_pairs, fake_stats)) as gen,
    ):
        mk.return_value.identifier = "fake:teacher"
        result = runner.invoke(
            app,
            [
                "generate",
                "--name", "verbose",
                "--description", "long, expansive",
                "--neutral", "Respond concisely",
                "--teacher", "fake:teacher",
                "--n-pairs", "2",
                "--out", str(out_path),
            ],
        )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert "saved 2 pairs" in result.output
    gen.assert_called_once()


def test_sweep_runs_pipeline(tmp_path: Path):
    """`sweep` should load pairs, build activations, fit, and save the best probe."""
    pairs_path = tmp_path / "pairs.jsonl"
    save_pairs_jsonl(
        [ContrastPair(prompt="hi", positive_response="!", negative_response=".")],
        pairs_path,
    )
    out_path = tmp_path / "probe.safetensors"
    fake_best = _make_probe()

    with (
        patch("steerkit.models.load") as load_fn,
        patch("steerkit.extract.extract_activations") as extract_fn,
        patch("steerkit.probe.Probe") as ProbeClass,
    ):
        load_fn.return_value.n_layers = 12
        load_fn.return_value.d_model = 768
        load_fn.return_value.device = "cpu"
        load_fn.return_value.model_id = "fake/test"
        extract_fn.return_value = {0: torch.zeros(1, 2, 8)}
        ProbeClass.fit_all.return_value = {0: fake_best}
        ProbeClass.best_layer.return_value = fake_best
        result = runner.invoke(
            app,
            [
                "sweep",
                "--pairs", str(pairs_path),
                "--model", "fake/test",
                "--out", str(out_path),
            ],
        )
    assert result.exit_code == 0, result.output
    assert "best layer = 3" in result.output
    ProbeClass.best_layer.assert_called_once()


def test_group_sweep_runs_pipeline(tmp_path: Path):
    group = ConceptGroup(
        name="g",
        relationship="multi_label",
        neutral_reference="...",
        concepts=[Concept("a", "...")],
    )
    group_path = tmp_path / "g.json"
    group.save(group_path)
    out_dir = tmp_path / "fit"

    fit_obj = GroupFit(group=group, best={"a": _make_probe()})

    with (
        patch("steerkit.models.load") as load_fn,
        patch("steerkit.sweep.sweep", return_value=fit_obj),
    ):
        load_fn.return_value.n_layers = 12
        load_fn.return_value.device = "cpu"
        load_fn.return_value.model_id = "fake/test"
        result = runner.invoke(
            app,
            [
                "group-sweep",
                "--group", str(group_path),
                "--model", "fake/test",
                "--out", str(out_dir),
            ],
        )
    assert result.exit_code == 0, result.output
    assert (out_dir / "group.json").exists()


def test_group_sweep_rejects_missing_teacher_when_eval_requested(tmp_path: Path):
    group = ConceptGroup(
        name="g",
        relationship="multi_label",
        neutral_reference="...",
        concepts=[Concept("a", "...")],
    )
    group_path = tmp_path / "g.json"
    group.save(group_path)

    with patch("steerkit.models.load"):
        result = runner.invoke(
            app,
            [
                "group-sweep",
                "--group", str(group_path),
                "--model", "fake/test",
                "--out", str(tmp_path / "fit"),
                "--with-steering-eval",
            ],
        )
    assert result.exit_code == 2
    assert "teacher" in result.output.lower()


def test_steer_runs(tmp_path: Path):
    probe_path = tmp_path / "p.probe.safetensors"
    _make_probe().save(probe_path)

    with (
        patch("steerkit.models.load") as load_fn,
        patch("steerkit.probe.Probe") as ProbeClass,
    ):
        load_fn.return_value.model_id = "fake/test"
        loaded_probe = _make_probe()
        loaded_probe.steer = lambda *a, **kw: "[steered output]"  # type: ignore[method-assign]
        ProbeClass.load.return_value = loaded_probe
        result = runner.invoke(
            app,
            [
                "steer",
                "--probe", str(probe_path),
                "--prompt", "hi",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "[steered output]" in result.output


def test_steer_clamp_requires_target(tmp_path: Path):
    """The dispatch in Probe.steer rejects op='clamp' without --target — surfaced via CLI."""
    probe_path = tmp_path / "p.probe.safetensors"
    _make_probe().save(probe_path)

    with (
        patch("steerkit.models.load") as load_fn,
        patch("steerkit.probe.Probe") as ProbeClass,
    ):
        load_fn.return_value.model_id = "fake/test"
        # Real probe instance so we exercise the actual dispatch error.
        from steerkit import Probe as RealProbe

        real_probe = _make_probe()
        # When the CLI passes op='clamp' with target=None, Probe.steer should raise ValueError.

        def fake_steer(*a, **kw):
            return RealProbe.steer(real_probe, *a, **kw)

        real_probe.steer = fake_steer  # type: ignore[method-assign]
        ProbeClass.load.return_value = real_probe
        result = runner.invoke(
            app,
            [
                "steer",
                "--probe", str(probe_path),
                "--prompt", "hi",
                "--op", "clamp",
            ],
        )
    # Typer wraps the exception; we just want a non-zero exit when the op needs more args.
    assert result.exit_code != 0


def test_calibrate_runs_and_saves(tmp_path: Path):
    probe_path = tmp_path / "p.probe.safetensors"
    _make_probe().save(probe_path)

    with (
        patch("steerkit.models.load") as load_fn,
        patch("steerkit.calibrate.calibrate_alpha", return_value=(2.0, {1.0: 1.05, 2.0: 1.30, 4.0: 1.55})),
    ):
        load_fn.return_value.model_id = "fake/test"
        result = runner.invoke(
            app,
            [
                "calibrate",
                "--probe", str(probe_path),
                "--candidates", "1,2,4",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "chosen α = 2.0" in result.output
    assert "✓" in result.output


def test_report_from_probe_writes_html(tmp_path: Path):
    probe_path = tmp_path / "p.probe.safetensors"
    _make_probe().save(probe_path)
    out_path = tmp_path / "report.html"

    result = runner.invoke(
        app,
        [
            "report",
            "--probe", str(probe_path),
            "--out", str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert "wrote report" in result.output


def test_report_from_groupfit_writes_html(tmp_path: Path):
    group = ConceptGroup(
        name="g",
        relationship="multi_label",
        neutral_reference="...",
        concepts=[Concept("a", "...")],
    )
    fit = GroupFit(group=group, best={"a": _make_probe()})
    fit_dir = tmp_path / "fit"
    fit.save(fit_dir)
    out_path = tmp_path / "fit_report.html"

    result = runner.invoke(
        app,
        [
            "report",
            "--group-fit", str(fit_dir),
            "--out", str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert "wrote report" in result.output


def test_report_requires_exactly_one_artifact(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "report",
            "--out", str(tmp_path / "report.html"),
        ],
    )

    assert result.exit_code == 2
    assert "exactly one" in result.output
