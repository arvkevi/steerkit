"""Render a visual comparison of the four intervention operations.

The four ops differ in how they reshape the residual-stream activation:

  addition       act ← act + α·v          (push toward the concept)
  projection     act ← act − (act·v̂)v̂      (remove the concept's component)
  clamp          act·v̂ ← target           (fix the concept's projection)
  multiplicative act ← act + (γ−1)(act·v̂)v̂ (scale the existing component)

Each panel shows the resulting completion text for the op + its mathematical
signature. Together they make the difference between the four interventions
concrete on a single benign prompt.

Output: `docs/ops_effects.png`. Reuses the saved sycophancy probe.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass
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
DEFAULT_BENIGN_PROMPT = "What's a fun activity for a quiet Sunday afternoon?"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROBE_PATH = REPO_ROOT / "runs" / "sycophancy.probe.safetensors"
DEFAULT_PAIRS_PATH = REPO_ROOT / "examples" / "data" / "sycophancy.jsonl"
DEFAULT_CACHE_DIR = REPO_ROOT / "cache"
DEFAULT_OUT = REPO_ROOT / "docs" / "ops_effects.png"


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


@dataclass
class OpDemo:
    name: str
    signature: str
    explainer: str
    op: str
    kwargs: dict


def _trim_to_sentence(text: str) -> str:
    """Trim a completion to end at the last sentence boundary (.,!,?). If no
    such boundary exists (the model didn't finish a sentence within the token
    budget), trim at the last word boundary and append an ellipsis so the
    visual doesn't break mid-word.
    """
    text = text.strip()
    end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if end >= 6:
        return text[: end + 1]
    last_space = text.rfind(" ")
    if last_space > 6:
        return text[:last_space].rstrip(",;:") + "…"
    return text + "…"


def _completions_for_ops(probe: Probe, model, prompt: str, *, max_new_tokens: int = 32):
    """Generate baseline + all four ops on the same prompt."""
    auto = probe.auto_alpha or 8.0
    demos = [
        OpDemo(
            name="addition",
            signature="act ← act + α·v",
            explainer="push activation toward the concept",
            op="addition",
            kwargs={"alpha": 2.0 * auto},
        ),
        OpDemo(
            name="projection",
            signature="act ← act − (act·v̂)v̂",
            explainer="remove the concept's component",
            op="projection",
            kwargs={},
        ),
        OpDemo(
            name="clamp",
            signature="(act·v̂) ← target",
            explainer="force the concept projection to a fixed value",
            op="clamp",
            kwargs={"target": 30.0},
        ),
        OpDemo(
            name="amplify",
            signature="act ← act + (γ−1)·(act·v̂)v̂",
            explainer="scale the existing component along v",
            op="multiplicative",
            kwargs={"gamma": 12.0},
        ),
    ]
    print(f"\nprompt: {prompt!r}")
    raw_baseline = probe.steer(model, prompt, alpha=0.0, max_new_tokens=max_new_tokens).strip()
    baseline = _trim_to_sentence(raw_baseline)
    print(f"  baseline               → {baseline!r}")
    results: list[tuple[OpDemo, str]] = []
    for d in demos:
        raw = probe.steer(model, prompt, op=d.op, max_new_tokens=max_new_tokens, **d.kwargs).strip()
        text = _trim_to_sentence(raw)
        print(f"  {d.name:12s} ({d.op:14s}) → {text!r}")
        results.append((d, text))
    return baseline, results


# --------------------------------------------------------------------------
# Figure rendering
# --------------------------------------------------------------------------

C_BG = "#fafafa"
C_TEXT = "#1f2933"
C_MUTED = "#6b7280"
C_INPUT = "#1f77b4"
C_INPUT_FILL = "#e6f0fa"
C_OPS = "#1f7a3f"
C_OPS_FILL = "#e6f5ec"


def render_figure(probe: Probe, model, prompt: str, out: Path, *, max_new_tokens: int = 32) -> Path:
    baseline, results = _completions_for_ops(probe, model, prompt, max_new_tokens=max_new_tokens)

    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    # Title + subtitle
    ax.text(50, 96, "What each intervention does to the model's response",
            fontsize=16, weight="bold", ha="center", va="top", color=C_TEXT)
    ax.text(50, 91,
            f"Sycophancy probe on {model.model_id}  ·  prompt: \"{prompt}\"",
            fontsize=10, color=C_MUTED, ha="center", va="top", style="italic")

    # Baseline panel — single inline row: small left tag + completion text.
    base_x, base_y, base_w, base_h = 5, 81, 90, 6.0
    ax.add_patch(FancyBboxPatch(
        (base_x, base_y), base_w, base_h,
        boxstyle="round,pad=0.0,rounding_size=0.5",
        linewidth=1.4, edgecolor=C_INPUT, facecolor=C_INPUT_FILL, zorder=1,
    ))
    ax.text(base_x + 2, base_y + base_h / 2, "α = 0",
            fontsize=11, weight="bold", color=C_INPUT, ha="left", va="center",
            family="monospace")
    ax.text(base_x + 11, base_y + base_h / 2, "(unsteered)",
            fontsize=8.5, color=C_MUTED, ha="left", va="center", style="italic")
    wrapped_baseline = textwrap.fill(baseline, width=88)
    ax.text(base_x + 22, base_y + base_h / 2, f"“{wrapped_baseline}”",
            fontsize=10, color=C_TEXT, ha="left", va="center", family="serif")

    # Grid layout: 2 columns × 2 rows
    margins = 5
    panel_w = (100 - 2 * margins - 4) / 2  # 4 = inter-panel gap
    panel_h = 32
    panel_top_y = 78
    positions = [
        (margins, panel_top_y),
        (margins + panel_w + 4, panel_top_y),
        (margins, panel_top_y - panel_h - 3),
        (margins + panel_w + 4, panel_top_y - panel_h - 3),
    ]

    for (demo, completion), (px, py) in zip(results, positions, strict=False):
        # Panel box (op color)
        ax.add_patch(FancyBboxPatch(
            (px, py - panel_h), panel_w, panel_h,
            boxstyle="round,pad=0.0,rounding_size=0.6",
            linewidth=1.4, edgecolor=C_OPS, facecolor=C_OPS_FILL, zorder=1,
        ))
        # Op name (bold, large) + math signature (smaller, monospace) on a
        # second line. Skip the explainer + "result:" label + divider — the
        # signature already specifies the op and the completion is the result.
        ax.text(px + 2, py - 1.8, demo.name,
                fontsize=13, weight="bold", color=C_OPS, ha="left", va="top",
                family="monospace")
        ax.text(px + 2, py - 5.0, demo.signature,
                fontsize=10, color=C_MUTED, ha="left", va="top", family="monospace")
        # Completion text — the actual result, takes the bottom of the panel.
        # wrap_w tuned so the longest wrapped line stays inside the panel
        # right edge for the serif font at 10pt.
        wrap_w = max(36, int((panel_w - 4) * 1.25))
        wrapped = textwrap.fill(completion, width=wrap_w)
        ax.text(px + 2, py - 10, f"“{wrapped}”",
                fontsize=10, color=C_TEXT, ha="left", va="top", family="serif")

    # Single-line footnote.
    auto = probe.auto_alpha or 8.0
    ax.text(
        50, 2.5,
        f"Settings: addition α = 2 × auto_α ≈ {2 * auto:.1f}  ·  "
        f"clamp target = 30  ·  amplify γ = 12  ·  projection has no parameters.",
        fontsize=8.5, color=C_MUTED, ha="center", va="bottom", style="italic",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nsaved → {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_BENIGN_PROMPT)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE_PATH)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    probe, model = _ensure_probe(args.model, args.pairs, args.probe, args.cache_dir)
    render_figure(probe, model, args.prompt, args.out, max_new_tokens=args.max_new_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
