"""Render an α-sweep showcase figure for the docs.

Steers the bundled sycophancy probe at five α values on a single prompt and
renders a stack of completions side-by-side with their perplexity values.
The visual makes the "α = strength dial" knob concrete: as α grows, the
output's commitment to the concept rises (sycophantic openers get more
elaborate), and at high enough α the output drifts.

Output: `docs/alpha_sweep.png`. Reuses the saved sycophancy probe at
`runs/sycophancy.probe.safetensors` for cheap re-runs.

Usage:
    uv run python scripts/make_alpha_sweep_figure.py
    uv run python scripts/make_alpha_sweep_figure.py --prompt "..."
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

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
DEFAULT_OUT = REPO_ROOT / "docs" / "alpha_sweep.png"


def _ensure_probe(model_id: str, pairs_path: Path, probe_path: Path, cache_dir: Path):
    pairs = load_pairs_jsonl(pairs_path)
    handle = load(model_id)
    print(f"loaded {model_id}: layers={handle.n_layers}, device={handle.device}")
    if probe_path.exists():
        try:
            probe = Probe.load(probe_path)
            if probe.model_id == model_id:
                if probe.auto_alpha is None:
                    chosen, _ = calibrate_alpha(probe, handle)
                    probe.auto_alpha = chosen
                    probe.save(probe_path)
                print(f"reusing probe at {probe_path} (layer {probe.layer}, auto_α={probe.auto_alpha:.3g})")
                return probe, handle
        except Exception:  # noqa: BLE001
            pass
    activations = extract_activations(pairs, handle, hook_site="resid_post", cache_dir=cache_dir)
    probes = Probe.fit_all(activations, handle, hook_site="resid_post", test_fraction=0.2)
    best = Probe.best_layer(probes, by="auc_test_logistic")
    chosen, _ = calibrate_alpha(best, handle)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    best.save(probe_path)
    return best, handle


def _generate_at_alpha(probe: Probe, model, prompt: str, alpha: float, max_new_tokens: int) -> str:
    return probe.steer(model, prompt, alpha=alpha, max_new_tokens=max_new_tokens).strip()


def _response_perplexity(probe: Probe, model, prompt: str, response: str) -> float:
    """Score a fixed response under the unsteered model (no hooks).

    Mirrors `steerkit.calibrate._response_perplexity` so we can show absolute
    perplexity values alongside completions. Lower = more coherent under the
    unsteered distribution.
    """
    import torch

    prompt_ids = model.format_chat(prompt)
    full_ids = model.format_chat(prompt, response)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    response_start = prompt_ids.shape[-1]
    if full_ids.shape[-1] <= response_start:
        return float("nan")

    with torch.no_grad():
        logits = model.hooked(full_ids)
    # cross-entropy on response tokens only
    log_probs = torch.log_softmax(logits[0, response_start - 1 : -1, :].float(), dim=-1)
    target_ids = full_ids[0, response_start:]
    nll = -log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1).mean().item()
    return float(math.exp(nll))


# --------------------------------------------------------------------------
# Figure rendering
# --------------------------------------------------------------------------


C_BG = "#fafafa"
C_TEXT = "#1f2933"
C_MUTED = "#6b7280"
C_INPUT = "#1f77b4"


def render_figure(probe: Probe, model, prompt: str, out: Path, *, max_new_tokens: int = 30) -> Path:
    auto = probe.auto_alpha or 4.0
    multipliers = [0.0, 0.5, 1.0, 2.0, 4.0]
    # Each row needs only the α value — the visual progression top-to-bottom
    # already conveys "this is a sweep". The bold tag on row 2 marks the
    # calibrated default.
    alpha_labels = [
        "α = 0",
        f"α = {0.5*auto:.2g}",
        f"α = {auto:.2g}",
        f"α = {2*auto:.2g}",
        f"α = {4*auto:.2g}",
    ]
    multiplier_labels = [
        "0 × auto_α",
        "½ × auto_α",
        "1 × auto_α",
        "2 × auto_α",
        "4 × auto_α",
    ]

    # Generate everything first.
    completions: list[str] = []
    ppls: list[float] = []
    print(f"\nprompt: {prompt!r}\n")
    for mult in multipliers:
        alpha = mult * auto
        text = _generate_at_alpha(probe, model, prompt, alpha=alpha, max_new_tokens=max_new_tokens)
        ppl = _response_perplexity(probe, model, prompt, text)
        completions.append(text)
        ppls.append(ppl)
        print(f"  α = {alpha:>6.2f} ({mult}×auto)  ppl = {ppl:>6.2f}  →  {text}")

    # ---- Layout -------------------------------------------------------------
    n = len(multipliers)
    fig_h = max(5.0, 1.05 * n + 1.6)
    fig, ax = plt.subplots(figsize=(13.0, fig_h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    # Title + subtitle
    ax.text(
        50, 96,
        "How α controls steering strength",
        fontsize=16, fontweight="bold", ha="center", va="top", color=C_TEXT,
    )
    ax.text(
        50, 91,
        f"Sycophancy probe on {model.model_id}  ·  prompt: \"{prompt}\"",
        fontsize=10, color=C_MUTED, ha="center", va="top", style="italic",
    )

    # Column headers — kept minimal.
    head_y = 85
    ax.text(5, head_y, "strength", fontsize=10, weight="bold", color=C_TEXT, ha="left", va="bottom")
    ax.text(22, head_y, "completion", fontsize=10, weight="bold", color=C_TEXT, ha="left", va="bottom")
    ax.text(94, head_y, "perplexity", fontsize=10, weight="bold", color=C_TEXT, ha="right", va="bottom")
    ax.plot([3, 97], [head_y - 1, head_y - 1], color="#9ca3af", lw=0.8)

    # Rows
    row_top = head_y - 4
    row_height = (row_top - 6) / n  # leave 6 units at the bottom for footnote
    for i, (alpha_label, mult_label, text) in enumerate(
        zip(alpha_labels, multiplier_labels, completions, strict=False)
    ):
        y_top = row_top - i * row_height
        y_mid = y_top - row_height / 2

        # Subtle row background for the calibrated default to anchor the eye.
        if i == 2:
            box = FancyBboxPatch(
                (3, y_top - row_height + 0.4),
                94,
                row_height - 0.8,
                boxstyle="round,pad=0.0,rounding_size=0.6",
                linewidth=1.2,
                edgecolor=C_INPUT,
                facecolor="#eef5fc",
                zorder=1,
            )
            ax.add_patch(box)
            ax.text(
                95.5, y_top - 1.0, "calibrated default",
                fontsize=8, color=C_INPUT, ha="right", va="top", style="italic", weight="bold",
            )

        # α label (bold) + a faint multiplier line below.
        ax.text(5, y_mid + 0.7, alpha_label, fontsize=12, weight="bold",
                family="monospace", color=C_TEXT, ha="left", va="center", zorder=3)
        ax.text(5, y_mid - 1.5, mult_label, fontsize=8, color=C_MUTED,
                family="monospace", ha="left", va="center", zorder=3)

        # Completion text — wrapped if long
        wrapped = textwrap.fill(text, width=68)
        ax.text(22, y_mid, f"“{wrapped}”", fontsize=9.5, color=C_TEXT,
                ha="left", va="center", family="serif", zorder=3)

        # Absolute perplexity
        ppl = ppls[i]
        if math.isfinite(ppl):
            ax.text(94, y_mid, f"{ppl:.2f}", fontsize=12, weight="bold",
                    family="monospace", color=C_TEXT, ha="right", va="center", zorder=3)

    # Footnote — short, single line.
    ax.text(
        50, 3,
        "Generations: temperature 0, max 30 new tokens.  ·  "
        "Perplexity scored under the unsteered model "
        "(short canonical text scores lower than verbose helpful answers).",
        fontsize=8, color=C_MUTED, ha="center", va="bottom", style="italic",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
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
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args(argv)
    probe, model = _ensure_probe(args.model, args.pairs, args.probe, args.cache_dir)
    render_figure(probe, model, args.prompt, args.out, max_new_tokens=args.max_new_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
