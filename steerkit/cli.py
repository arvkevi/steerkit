"""Thin CLI over the Python API.

Subcommands map to the workflow phases:

  generate      — call a teacher to produce contrast pairs for one concept (JSONL out)
  lint-pairs    — quality checks on a JSONL of contrast pairs (no model load)
  lint-group    — quality checks on every concept in a ConceptGroup JSON
  sweep         — run extract + fit_all + best-layer on a JSONL of pairs (single Probe out)
  group-sweep   — run sweep() on a ConceptGroup JSON (GroupFit directory out)
  steer         — load a Probe + steer once on a prompt
  calibrate     — auto-α calibration on an existing probe artifact
  report        — render a shareable HTML report from a Probe or GroupFit

Defaults aim for the tiny-model on-MPS workflow we use in the quickstart
notebooks. Override `--model` / `--teacher` for production-scale runs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

from . import __version__

app = typer.Typer(
    name="steerkit",
    help="Concept-first linear probes and activation steering, reproducible across many open-weight models.",
    add_completion=False,
    no_args_is_help=True,
)


def _set_mps_env() -> None:
    os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"steerkit {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    _version: bool = typer.Option(
        False,
        "--version",
        help="Show steerkit version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Concept-first linear probes and activation steering."""


@app.command()
def generate(
    name: str = typer.Option(..., "--name", help="Concept name, e.g. 'verbose'."),
    description: str = typer.Option(..., "--description", help="Concept description prompt for the teacher."),
    neutral: str = typer.Option(..., "--neutral", help="Neutral-reference instruction."),
    teacher: str = typer.Option(..., "--teacher", help="Teacher spec, e.g. 'anthropic:claude-haiku-4-5-20251001' or 'local:HuggingFaceTB/SmolLM2-1.7B-Instruct'."),
    n_pairs: int = typer.Option(30, "--n-pairs", help="How many contrast pairs to generate."),
    seed_prompts: Path | None = typer.Option(None, "--seed-prompts", help="Optional JSON file containing a list of seed prompt strings."),
    out: Path = typer.Option(..., "--out", help="Output JSONL path."),
    temperature: float = typer.Option(0.7, "--temperature"),
    max_tokens: int = typer.Option(512, "--max-tokens"),
) -> None:
    """Generate `n_pairs` contrast pairs for one concept and write to JSONL."""
    from .concepts import Concept
    from .data import save_pairs_jsonl
    from .generate import generate_pairs_for_concept
    from .teacher import make_teacher

    teacher_obj = make_teacher(teacher)
    concept = Concept(name=name, description=description)
    prompts = None
    if seed_prompts is not None:
        prompts = json.loads(seed_prompts.read_text())
    typer.echo(f"using teacher: {teacher_obj.identifier}")
    typer.echo(f"generating up to {n_pairs} pairs for {name!r}...")
    pairs, stats = generate_pairs_for_concept(
        concept,
        teacher=teacher_obj,
        neutral_reference=neutral,
        seed_prompts=prompts,
        max_pairs=n_pairs,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    save_pairs_jsonl(pairs, out)
    typer.echo(f"  parsed={stats.parsed} failed={stats.failed} success={stats.success_rate:.0%}")
    typer.echo(f"saved {len(pairs)} pairs -> {out}")


@app.command("lint-pairs")
def lint_pairs_cmd(
    pairs: Path = typer.Option(..., "--pairs", help="JSONL of contrast pairs to check."),
    json_out: Path | None = typer.Option(
        None,
        "--json",
        help="Optional path to write the report as JSON (machine-readable).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit non-zero on warnings as well as errors. Default exits non-zero only on errors.",
    ),
) -> None:
    """Run dataset quality checks on a JSONL of `ContrastPair`s.

    Produces a categorised list of findings (errors, warnings, infos) covering
    empty fields, duplicates, lexical uniformity, cross-class leakage, length
    skew, suspiciously short responses, repeated prompts, and shared positive
    prefixes. Loads no models — fast enough to run before every sweep.

    Exit code: 0 if clean (no errors and, with --strict, no warnings either);
    non-zero otherwise. Suitable for CI / pre-commit.
    """
    from .data import load_pairs_jsonl
    from .lint import lint_pairs as _lint_pairs

    contrast_pairs = load_pairs_jsonl(pairs)
    report = _lint_pairs(contrast_pairs)
    typer.echo(report.format_text())

    if json_out is not None:
        payload = {
            "n_pairs": report.n_pairs,
            "findings": [
                {
                    "severity": f.severity,
                    "code": f.code,
                    "message": f.message,
                    "affected_pair_indices": list(f.affected_pair_indices),
                }
                for f in report.findings
            ],
        }
        json_out.write_text(json.dumps(payload, indent=2))
        typer.echo(f"\nwrote machine-readable report -> {json_out}")

    if report.errors or (strict and report.warnings):
        raise typer.Exit(code=1)


@app.command("lint-group")
def lint_group_cmd(
    group: Path = typer.Option(..., "--group", help="ConceptGroup JSON file."),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit non-zero on warnings as well as errors.",
    ),
) -> None:
    """Run dataset quality checks on every concept in a `ConceptGroup`.

    Concepts are linted independently; the exit code reflects the worst severity
    seen across all concepts.
    """
    from .concepts import ConceptGroup
    from .lint import lint_group as _lint_group

    cg = ConceptGroup.load(group)
    reports = _lint_group(cg)
    any_error = False
    any_warning = False
    for name, report in reports.items():
        typer.echo(f"\n=== concept: {name} ===")
        typer.echo(report.format_text())
        any_error = any_error or bool(report.errors)
        any_warning = any_warning or bool(report.warnings)
    if any_error or (strict and any_warning):
        raise typer.Exit(code=1)


@app.command()
def sweep(
    pairs: Path = typer.Option(..., "--pairs", help="JSONL of contrast pairs."),
    model: str = typer.Option(..., "--model", help="Model id (HF / TL allowlist)."),
    out: Path = typer.Option(..., "--out", help="Output .probe.safetensors path."),
    hook_site: str = typer.Option("resid_post", "--hook-site"),
    test_fraction: float = typer.Option(0.2, "--test-fraction"),
    seed: int = typer.Option(42, "--seed"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Zarr cache directory; skip extraction on cache hit."),
    select_by: str = typer.Option("auc_test_logistic", "--select-by"),
    no_boundaries: bool = typer.Option(False, "--no-boundaries", help="Skip embed and final_ln in the sweep."),
) -> None:
    """Extract activations + fit all three candidate directions per layer + save the best."""
    _set_mps_env()
    from .data import load_pairs_jsonl
    from .extract import extract_activations
    from .models import load
    from .probe import Probe

    typer.echo(f"loading {len(load_pairs_jsonl(pairs))} pairs from {pairs}")
    contrast_pairs = load_pairs_jsonl(pairs)
    typer.echo(f"loading model {model}...")
    handle = load(model)
    typer.echo(f"  layers={handle.n_layers}, d_model={handle.d_model}, device={handle.device}")
    activations = extract_activations(
        contrast_pairs,
        handle,
        hook_site=hook_site,
        include_boundaries=not no_boundaries,
        cache_dir=cache_dir,
    )
    probes = Probe.fit_all(
        activations, handle, hook_site=hook_site, test_fraction=test_fraction, seed=seed
    )
    best = Probe.best_layer(probes, by=select_by)
    typer.echo(
        f"best layer = {best.layer} (depth {best.normalized_depth:.2f}), "
        f"hook = {best.hook_name}, {select_by} = {best.metrics[select_by]:.3f}"
    )
    best.save(out)
    typer.echo(f"saved -> {out}")


@app.command("group-sweep")
def group_sweep(
    group: Path = typer.Option(..., "--group", help="ConceptGroup JSON file."),
    model: str = typer.Option(..., "--model", help="Model id (HF / TL allowlist)."),
    out: Path = typer.Option(..., "--out", help="Output GroupFit directory."),
    hook_site: str = typer.Option("resid_post", "--hook-site"),
    test_fraction: float = typer.Option(0.2, "--test-fraction"),
    seed: int = typer.Option(42, "--seed"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    select_by: str = typer.Option("auc_test_logistic", "--select-by"),
    with_steering_eval: bool = typer.Option(False, "--with-steering-eval"),
    teacher: str | None = typer.Option(None, "--teacher", help="Required if --with-steering-eval is set."),
    eval_top_k: int = typer.Option(5, "--eval-top-k"),
) -> None:
    """Run the full sweep on a ConceptGroup; save best probe per concept + multinomial."""
    _set_mps_env()
    from .concepts import ConceptGroup
    from .models import load as load_model
    from .sweep import sweep as run_sweep
    from .teacher import make_teacher

    cg = ConceptGroup.load(group)
    handle = load_model(model)
    typer.echo(f"loaded ConceptGroup {cg.name!r} ({cg.relationship}) — {len(cg.concepts)} concepts")
    typer.echo(f"loaded model {model}: layers={handle.n_layers}, device={handle.device}")
    teacher_obj = make_teacher(teacher) if teacher else None
    if with_steering_eval and teacher_obj is None:
        typer.echo("--with-steering-eval requires --teacher", err=True)
        raise typer.Exit(code=2)
    fit = run_sweep(
        cg,
        handle,
        hook_site=hook_site,
        test_fraction=test_fraction,
        seed=seed,
        cache_dir=cache_dir,
        select_by=select_by,
        with_steering_eval=with_steering_eval,
        teacher=teacher_obj,
        eval_top_k=eval_top_k,
    )
    for name in fit.names():
        p = fit[name]
        typer.echo(f"  {name}: layer {p.layer}, AUC = {p.metrics.get('auc_test_logistic', float('nan')):.3f}")
    if fit.multinomial is not None:
        typer.echo(
            f"  multinomial: layer {fit.multinomial.layer}, "
            f"acc_test = {fit.multinomial.metrics.get('accuracy_test', float('nan')):.3f}"
        )
    fit.save(out)
    typer.echo(f"saved GroupFit -> {out}")


@app.command()
def steer(
    probe: Path = typer.Option(..., "--probe", help="Path to a .probe.safetensors artifact."),
    prompt: str = typer.Option(..., "--prompt"),
    op: str = typer.Option("addition", "--op", help="addition | projection | clamp | multiplicative"),
    alpha: float | None = typer.Option(None, "--alpha", help="Override the calibrated auto_alpha (addition only)."),
    target: float | None = typer.Option(None, "--target", help="Required for op='clamp'."),
    gamma: float | None = typer.Option(None, "--gamma", help="Required for op='multiplicative'."),
    method: str | None = typer.Option(None, "--method", help="logistic | diff_of_means | mass_mean. Defaults to probe.default_method."),
    max_new_tokens: int = typer.Option(60, "--max-new-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    model: str | None = typer.Option(None, "--model", help="Override the probe's model id (warn-and-allow)."),
) -> None:
    """Load a probe and emit a single steered completion."""
    _set_mps_env()
    from .models import load
    from .probe import Probe

    p = Probe.load(probe)
    handle = load(model or p.model_id)
    out = p.steer(
        handle,
        prompt,
        alpha=alpha,
        op=op,
        target=target,
        gamma=gamma,
        method=method,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    typer.echo(out)


@app.command()
def calibrate(
    probe: Path = typer.Option(..., "--probe", help="Path to a .probe.safetensors artifact."),
    prompts: Path | None = typer.Option(None, "--prompts", help="JSON list of calibration prompts."),
    candidates: str = typer.Option("0.5,1,2,4,8", "--candidates", help="Comma-separated α candidates."),
    perplexity_ratio_max: float = typer.Option(1.5, "--ratio-max"),
    max_new_tokens: int = typer.Option(30, "--max-new-tokens"),
    save: bool = typer.Option(True, "--save/--no-save", help="Write the chosen α back to the probe artifact."),
) -> None:
    """Auto-α calibration on an existing probe."""
    _set_mps_env()
    from .calibrate import calibrate_alpha
    from .models import load
    from .probe import Probe

    p = Probe.load(probe)
    handle = load(p.model_id)
    cand = [float(x) for x in candidates.split(",")]
    calibration_prompts = None
    if prompts is not None:
        calibration_prompts = json.loads(prompts.read_text())
    chosen, ratios = calibrate_alpha(
        p,
        handle,
        prompts=calibration_prompts,
        candidates=cand,
        perplexity_ratio_max=perplexity_ratio_max,
        max_new_tokens=max_new_tokens,
        attach=True,
    )
    typer.echo(f"chosen α = {chosen}")
    for a, r in sorted(ratios.items()):
        marker = " ✓" if a == chosen else ""
        typer.echo(f"  α={a}: ppl ratio = {r:.3f}{marker}")
    if save:
        p.save(probe)
        typer.echo(f"updated probe at {probe} with auto_alpha = {chosen}")


@app.command()
def report(
    out: Path = typer.Option(..., "--out", help="Output HTML path."),
    probe: Path | None = typer.Option(None, "--probe", help="Path to a .probe.safetensors artifact."),
    group_fit: Path | None = typer.Option(None, "--group-fit", help="Path to a GroupFit directory."),
    model: str | None = typer.Option(None, "--model", help="Optional model id for logit-lens plots."),
    title: str | None = typer.Option(None, "--title", help="Optional report title."),
) -> None:
    """Render a self-contained HTML report from a saved Probe or GroupFit."""
    if (probe is None) == (group_fit is None):
        typer.echo("Pass exactly one of --probe or --group-fit.", err=True)
        raise typer.Exit(code=2)

    handle = None
    if model is not None:
        _set_mps_env()
        from .models import load

        typer.echo(f"loading model {model} for logit-lens plots...")
        handle = load(model)

    if probe is not None:
        from .probe import Probe

        p = Probe.load(probe)
        result = p.report(model=handle, out=out, title=title)
    else:
        from .sweep import GroupFit

        assert group_fit is not None
        fit = GroupFit.load(group_fit)
        result = fit.report(model=handle, out=out, title=title)

    typer.echo(f"wrote report -> {result}")


def main() -> int:
    """Entry point used by `python -m steerkit.cli`."""
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
