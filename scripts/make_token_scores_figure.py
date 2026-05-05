"""Render the per-token probe-score showcase figure for the README.

Loads Qwen2.5-1.5B-Instruct, reuses the bundled sycophancy probe (or refits
saved one if present), generates one unsteered + one steered completion,
scores every token of each completion against the probe direction, and
renders a side-by-side bar-chart comparison.

The output is `docs/token_scores.png` — a static PNG referenced by the
README and concept gallery to give readers a sense of what the new
`Probe.score_tokens` API produces in practice.

Usage:
    uv run python scripts/make_token_scores_figure.py
    uv run python scripts/make_token_scores_figure.py --prompt "What's a fun..."
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

from steerkit import (  # noqa: E402
    Probe,
    calibrate_alpha,
    extract_activations,
    load,
    load_pairs_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = "What's a fun activity for a quiet Sunday afternoon?"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROBE_PATH = REPO_ROOT / "runs" / "sycophancy.probe.safetensors"
DEFAULT_PAIRS_PATH = REPO_ROOT / "examples" / "data" / "sycophancy.jsonl"
DEFAULT_CACHE_DIR = REPO_ROOT / "cache"
DEFAULT_OUT = REPO_ROOT / "docs" / "token_scores.png"


def _fit_or_load_probe(
    *,
    model_id: str,
    pairs_path: Path,
    probe_path: Path,
    cache_dir: Path,
) -> tuple[Probe, object]:
    """Reuse the saved probe if its model + pairs hash still match; otherwise refit."""
    pairs = load_pairs_jsonl(pairs_path)
    handle = load(model_id)
    print(f"loaded {model_id}: layers={handle.n_layers}, device={handle.device}")

    if probe_path.exists():
        try:
            probe = Probe.load(probe_path)
            if probe.model_id == model_id:
                print(f"reusing saved probe at {probe_path} (layer {probe.layer})")
                if probe.auto_alpha is None:
                    chosen, _ = calibrate_alpha(probe, handle)
                    probe.auto_alpha = chosen
                    probe.save(probe_path)
                return probe, handle
        except Exception as e:  # noqa: BLE001
            print(f"could not reuse {probe_path}: {e!r} — refitting from scratch")

    print("fitting fresh probe...")
    activations = extract_activations(
        pairs, handle, hook_site="resid_post", cache_dir=cache_dir
    )
    probes = Probe.fit_all(activations, handle, hook_site="resid_post", test_fraction=0.2)
    best = Probe.best_layer(probes, by="auc_test_logistic")
    print(
        f"best layer: {best.layer} (depth {best.normalized_depth:.2f}); "
        f"AUC={best.metrics['auc_test_logistic']:.3f}"
    )
    chosen, _ = calibrate_alpha(best, handle)
    print(f"calibrated α = {chosen}")
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    best.save(probe_path)
    return best, handle


def _bar_panel(ax, tokens: list[str], scores, title: str, xlim: tuple[float, float]) -> None:
    """Render one tokens × scores horizontal bar chart on `ax`."""
    n = len(tokens)
    y_pos = list(range(n))
    colors = ["tab:red" if v >= 0 else "tab:blue" for v in scores]
    ax.barh(y_pos, scores, color=colors, edgecolor="white", linewidth=0.4)
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels([repr(t) for t in tokens], fontsize=9)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlim(xlim)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.set_facecolor("#fbfbfb")


def render_figure(
    probe: Probe,
    model,
    prompt: str,
    out: Path,
    *,
    max_new_tokens: int = 18,
    alpha_multiplier: float = 2.0,
) -> Path:
    """Generate unsteered + steered completions, score each, render the side-by-side.

    The auto-calibrated α is conservative (≤1.5× baseline perplexity). For a
    visually clear behavioral flip in the showcase plot, multiply by
    `alpha_multiplier` (default 2.0 — past calibration but still coherent).
    """
    print(f"\nprompt: {prompt!r}\n")

    base_alpha = probe.auto_alpha if probe.auto_alpha is not None else 2.0
    steer_alpha = base_alpha * alpha_multiplier

    print("generating unsteered...")
    unsteered = probe.steer(model, prompt, alpha=0.0, max_new_tokens=max_new_tokens)
    print(f"  → {unsteered!r}")

    print(f"generating steered (α = {alpha_multiplier}× auto = {steer_alpha:.3g})...")
    steered = probe.steer(model, prompt, alpha=steer_alpha, max_new_tokens=max_new_tokens)
    print(f"  → {steered!r}")

    print("\nscoring tokens...")
    unsteered_ts = probe.score_tokens(model, prompt, unsteered)
    steered_ts = probe.score_tokens(model, prompt, steered)

    # Trim trailing blank/special tokens for visual clarity.
    def _trim(ts):
        for i in range(len(ts.tokens) - 1, -1, -1):
            if ts.tokens[i].strip():
                return ts.tokens[: i + 1], ts.scores[: i + 1]
        return ts.tokens, ts.scores

    u_tokens, u_scores = _trim(unsteered_ts)
    s_tokens, s_scores = _trim(steered_ts)
    u_vals = u_scores.float().cpu().numpy()
    s_vals = s_scores.float().cpu().numpy()

    abs_max = float(max(abs(u_vals).max(), abs(s_vals).max())) * 1.05
    xlim = (-abs_max, abs_max)

    import textwrap

    def _wrap(text: str, width: int = 58) -> str:
        cleaned = text.strip().replace("\n", " ")
        return "\n".join(textwrap.wrap(cleaned, width=width)) or cleaned

    u_completion = _wrap(unsteered)
    s_completion = _wrap(steered)

    n_rows = max(len(u_tokens), len(s_tokens))
    completion_lines = max(u_completion.count("\n"), s_completion.count("\n")) + 1
    fig_h = max(3.6, 0.32 * n_rows + 1.6 + 0.25 * completion_lines)
    fig, (ax_u, ax_s) = plt.subplots(
        1, 2, figsize=(13.0, fig_h), gridspec_kw={"wspace": 0.32}
    )

    _bar_panel(
        ax_u,
        u_tokens,
        u_vals,
        f"Unsteered  ·  α = 0.0\n“{u_completion}”",
        xlim,
    )
    _bar_panel(
        ax_s,
        s_tokens,
        s_vals,
        f"Steered  ·  α = {steer_alpha:.3g}\n“{s_completion}”",
        xlim,
    )
    ax_u.set_xlabel(f"probe score (sycophancy direction · resid_post layer {probe.layer})")
    ax_s.set_xlabel(f"probe score (sycophancy direction · resid_post layer {probe.layer})")

    # Caption below: model + prompt + interpretation guide. No suptitle —
    # the panel titles + completions are the headline; this is supporting context.
    fig.text(
        0.5,
        0.005,
        f"{model.model_id}   ·   prompt: “{prompt}”\n"
        "Each bar = projection of one token's residual stream onto the sycophancy direction. "
        "Red = direction firing (concept active); blue = inactive.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555",
        linespacing=1.5,
    )

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"\nsaved → {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE_PATH)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-new-tokens", type=int, default=18)
    parser.add_argument(
        "--alpha-multiplier",
        type=float,
        default=2.0,
        help="multiplier on the calibrated α used for the steered side. "
             "Calibration is conservative (≤1.5× baseline perplexity); for a "
             "showcase plot we want a visible behavior flip, hence ≥1.5.",
    )
    args = parser.parse_args(argv)

    probe, model = _fit_or_load_probe(
        model_id=args.model,
        pairs_path=args.pairs,
        probe_path=args.probe,
        cache_dir=args.cache_dir,
    )
    render_figure(
        probe,
        model,
        args.prompt,
        args.out,
        max_new_tokens=args.max_new_tokens,
        alpha_multiplier=args.alpha_multiplier,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
