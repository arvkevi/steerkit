"""Render the steerkit "mental model" / pipeline diagram for the README + index.

Produces `docs/mental_model.png` — a single static figure showing what flows
where in a steerkit run, with concrete shapes/sizes from the bundled
sycophancy probe (Qwen2.5-1.5B-Instruct) so the reader sees a real example,
not just labeled boxes.

Three stages left-to-right:

  1. YOU PROVIDE  — concept + contrast pairs
  2. STEERKIT FITS — activations [n_pairs, 2, d_model], three candidate
     directions per layer (logistic / diff-of-means / mass-mean), best
     layer selected, α calibrated
  3. PORTABLE ARTIFACT — .probe.safetensors with metadata; the operations
     you can run on it

Run:
    uv run python scripts/make_mental_model_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "docs" / "mental_model.png"

# Color palette — cool for inputs, warm for compute, deep for artifact.
C_BG = "#fafafa"
C_INPUT = "#1f77b4"        # blue
C_INPUT_FILL = "#e6f0fa"
C_COMPUTE = "#d97706"      # amber
C_COMPUTE_FILL = "#fff4e0"
C_ARTIFACT = "#6b21a8"     # deep purple
C_ARTIFACT_FILL = "#f3e8ff"
C_OPS = "#1f7a3f"          # green
C_OPS_FILL = "#e6f5ec"
C_TEXT = "#1f2933"
C_MUTED = "#6b7280"


def _box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    edge: str,
    fill: str,
    *,
    radius: float = 0.6,
    lw: float = 1.4,
    zorder: float = 2,
) -> tuple[float, float, float, float]:
    """Draw a rounded rectangle. Returns (x, y, w, h)."""
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
        zorder=zorder,
    )
    ax.add_patch(box)
    return x, y, w, h


def _text(
    ax,
    xy: tuple[float, float],
    s: str,
    *,
    fontsize: float = 10,
    color: str = C_TEXT,
    weight: str = "normal",
    ha: str = "left",
    va: str = "top",
    family: str | None = None,
    style: str = "normal",
) -> None:
    ax.text(
        xy[0],
        xy[1],
        s,
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        family=family,
        style=style,
        zorder=4,
    )


def _arrow(ax, src: tuple[float, float], dst: tuple[float, float], color: str = "#666") -> None:
    arr = FancyArrowPatch(
        src,
        dst,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color=color,
        zorder=3,
    )
    ax.add_patch(arr)


def _section_header(ax, x: float, y: float, label: str, color: str) -> None:
    _text(
        ax,
        (x, y),
        label,
        fontsize=10.5,
        weight="bold",
        color=color,
        ha="left",
        va="bottom",
    )
    ax.plot([x, x + 26], [y - 0.8, y - 0.8], color=color, lw=1.5, zorder=2)


def render(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14.0, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 64)
    ax.set_aspect("auto")
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    # ---- Title ---------------------------------------------------------------
    _text(
        ax,
        (50, 62.5),
        "How steerkit produces a steering vector",
        fontsize=15,
        weight="bold",
        ha="center",
        va="top",
    )
    _text(
        ax,
        (50, 60.0),
        "Sycophancy probe on Qwen2.5-1.5B-Instruct  ·  60 contrast pairs",
        fontsize=10,
        color=C_MUTED,
        ha="center",
        va="top",
        style="italic",
    )

    # ---- Section headers (three lanes) ---------------------------------------
    _section_header(ax, 3, 55, "YOU PROVIDE", C_INPUT)
    _section_header(ax, 36, 55, "STEERKIT FITS", C_COMPUTE)
    _section_header(ax, 73, 55, "PORTABLE ARTIFACT", C_ARTIFACT)

    # =========================================================================
    # LANE 1 — YOU PROVIDE
    # =========================================================================

    # Concept card
    _box(ax, (1, 41), (26, 11), C_INPUT, C_INPUT_FILL)
    _text(ax, (2.5, 51), "Concept", fontsize=9.5, color=C_MUTED, weight="bold")
    _text(
        ax,
        (2.5, 49.2),
        '"sycophantic"',
        fontsize=13,
        weight="bold",
        color=C_INPUT,
    )
    _text(
        ax,
        (2.5, 46.5),
        "validating, flattering preface",
        fontsize=9,
        style="italic",
        color=C_TEXT,
    )
    _text(
        ax,
        (2.5, 44.7),
        "before answering the question.",
        fontsize=9,
        style="italic",
        color=C_TEXT,
    )
    _text(
        ax,
        (2.5, 42.5),
        'e.g. "Great question! ..."',
        fontsize=9,
        style="italic",
        color=C_MUTED,
    )

    # ContrastPair card
    _box(ax, (1, 21), (26, 17), C_INPUT, C_INPUT_FILL)
    _text(ax, (2.5, 36.8), "ContrastPair × 60", fontsize=9.5, color=C_MUTED, weight="bold")
    _text(
        ax,
        (2.5, 34.6),
        "prompt:",
        fontsize=9.5,
        weight="bold",
        color=C_TEXT,
    )
    _text(
        ax,
        (2.5, 32.9),
        '"How do I bake bread?"',
        fontsize=9,
        family="monospace",
        color=C_TEXT,
    )

    _text(ax, (2.5, 30.4), "+", fontsize=11, weight="bold", color="#16a34a")
    _text(
        ax,
        (4.0, 30.4),
        "positive_response:",
        fontsize=9.5,
        weight="bold",
        color="#16a34a",
    )
    _text(
        ax,
        (2.5, 28.7),
        '"What a great question! Mix',
        fontsize=9,
        family="monospace",
        color=C_TEXT,
    )

    _text(ax, (2.5, 26.2), "−", fontsize=11, weight="bold", color="#dc2626")
    _text(
        ax,
        (4.0, 26.2),
        "negative_response:",
        fontsize=9.5,
        weight="bold",
        color="#dc2626",
    )
    _text(
        ax,
        (2.5, 24.5),
        '"Mix flour, water, yeast,',
        fontsize=9,
        family="monospace",
        color=C_TEXT,
    )
    _text(
        ax,
        (2.5, 22.8),
        ' salt; let rise 1h; bake."',
        fontsize=9,
        family="monospace",
        color=C_TEXT,
    )

    # =========================================================================
    # LANE 2 — STEERKIT FITS
    # =========================================================================

    # 2a. Activation tensor (drawn as a 3D-ish stack)
    _text(ax, (37.5, 51.2), "1. Extract activations", fontsize=10, weight="bold", color=C_COMPUTE)

    # tensor box with "stack" lines for depth
    tx, ty, tw, th = 37, 42, 16, 7.0
    for k in range(3, 0, -1):
        offset = 0.55 * k
        _box(
            ax,
            (tx + offset, ty - offset),
            (tw, th),
            C_COMPUTE,
            "#fff",
            radius=0.4,
            lw=0.8,
            zorder=2 - k * 0.1,
        )
    _box(ax, (tx, ty), (tw, th), C_COMPUTE, C_COMPUTE_FILL, radius=0.4)
    _text(
        ax,
        (tx + tw / 2, ty + th / 2 + 1.2),
        "[60, 2, 1536]",
        fontsize=11,
        weight="bold",
        family="monospace",
        ha="center",
        va="center",
        color=C_TEXT,
    )
    _text(
        ax,
        (tx + tw / 2, ty + th / 2 - 0.6),
        "n_pairs × pos/neg",
        fontsize=8,
        ha="center",
        va="center",
        color=C_MUTED,
    )
    _text(
        ax,
        (tx + tw / 2, ty + th / 2 - 2.1),
        "× d_model",
        fontsize=8,
        ha="center",
        va="center",
        color=C_MUTED,
    )

    # 2b. Per-layer × 3-method grid
    _text(ax, (37.5, 39.0), "2. Fit per layer × 3 methods", fontsize=10, weight="bold", color=C_COMPUTE)

    # Build a small grid: 29 layer columns × 3 method rows, with the "best layer × logistic" cell highlighted
    n_layers = 29  # 28 blocks + final_ln (closest match to Qwen2.5-1.5B)
    methods = ["logistic", "diff_of_means", "mass_mean"]
    grid_x = 37
    grid_y = 27
    cell_w = 16 / n_layers
    cell_h = 9 / 3
    # Each method gets a slightly different peak layer + amplitude so the
    # three rows are visually distinct rather than identical curves. Logistic
    # is the cleanest signal (peak at 14, highest amplitude); diff_of_means
    # peaks slightly earlier and mass_mean slightly later, both lower amp.
    method_params = {
        "logistic":      {"mid": 12, "amp": 0.42, "spread": 8.0},
        "diff_of_means": {"mid": 10, "amp": 0.36, "spread": 7.5},
        "mass_mean":     {"mid": 14, "amp": 0.34, "spread": 8.5},
    }
    for ri, m in enumerate(methods):
        # row label
        _text(
            ax,
            (grid_x - 0.4, grid_y + (2 - ri) * cell_h + cell_h / 2),
            m,
            fontsize=8,
            color=C_MUTED,
            ha="right",
            va="center",
            family="monospace",
        )
        params = method_params[m]
        for ci in range(n_layers):
            x = grid_x + ci * cell_w
            y = grid_y + (2 - ri) * cell_h
            base = 0.55 + params["amp"] * max(
                0, 1 - ((ci - params["mid"]) / params["spread"]) ** 2
            )
            v = max(0.5, min(1.0, base))
            color = plt.get_cmap("YlOrRd")((v - 0.5) / 0.5)
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y),
                    cell_w,
                    cell_h,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                )
            )

    # X-axis tick labels: 0, 14, 28 layers
    for li, label in [(0, "0"), (12, "12"), (28, "28")]:
        x = grid_x + li * cell_w + cell_w / 2
        _text(
            ax,
            (x, grid_y - 0.6),
            label,
            fontsize=8,
            color=C_TEXT if li == 12 else C_MUTED,
            weight="bold" if li == 12 else "normal",
            ha="center",
            va="top",
        )
    _text(
        ax,
        (grid_x + 16 / 2, grid_y - 2.4),
        "layer index (0 = embed,  N = final_ln)",
        fontsize=8,
        color=C_MUTED,
        ha="center",
        va="top",
    )

    # Highlight the best cell (logistic at layer 12) with a black border. The
    # bottom calibration box ("layer 12" / "AUC 1.00 · auto_α = 13") is the
    # textual annotation; the highlighted cell is the visual marker.
    best_x = grid_x + 12 * cell_w
    best_y = grid_y + 2 * cell_h  # row 0 = logistic = top row → highest y
    ax.add_patch(
        mpatches.Rectangle(
            (best_x - 0.08, best_y - 0.08),
            cell_w + 0.16,
            cell_h + 0.16,
            facecolor="none",
            edgecolor="#000",
            linewidth=2.0,
            zorder=5,
        )
    )

    # 2c. Single-line summary of selection + calibration outcome.
    _text(
        ax,
        (33.5, 22.0),
        "3. Pick best layer + calibrate α",
        fontsize=10,
        weight="bold",
        color=C_COMPUTE,
    )
    sel_x, sel_y, sel_w, sel_h = 33, 11.5, 24, 7.5
    _box(ax, (sel_x, sel_y), (sel_w, sel_h), C_COMPUTE, C_COMPUTE_FILL, radius=0.3)
    _text(
        ax,
        (sel_x + sel_w / 2, sel_y + sel_h - 1.3),
        "layer 12",
        fontsize=11.5,
        weight="bold",
        color=C_COMPUTE,
        family="monospace",
        ha="center",
        va="top",
    )
    _text(
        ax,
        (sel_x + sel_w / 2, sel_y + sel_h - 3.4),
        "AUC 1.00    auto_α = 13",
        fontsize=9.5,
        color=C_TEXT,
        family="monospace",
        ha="center",
        va="top",
    )
    _text(
        ax,
        (sel_x + sel_w / 2, sel_y + sel_h - 5.5),
        "perplexity ratio ≤ 1.5×",
        fontsize=8.5,
        color=C_MUTED,
        style="italic",
        ha="center",
        va="top",
    )

    # =========================================================================
    # LANE 3 — PORTABLE ARTIFACT
    # =========================================================================

    art_x, art_y, art_w, art_h = 71, 11, 26, 41
    # Subtle drop-shadow effect: a darker offset rectangle behind the main box.
    _box(
        ax,
        (art_x + 0.4, art_y - 0.4),
        (art_w, art_h),
        "#d8b4fe",
        "#d8b4fe",
        radius=0.6,
        lw=0,
        zorder=1.5,
    )
    _box(ax, (art_x, art_y), (art_w, art_h), C_ARTIFACT, C_ARTIFACT_FILL, radius=0.6)

    _text(
        ax,
        (art_x + art_w / 2, art_y + art_h - 1.4),
        "sycophancy.probe.safetensors",
        fontsize=11.5,
        weight="bold",
        family="monospace",
        color=C_ARTIFACT,
        ha="center",
        va="top",
    )
    _text(
        ax,
        (art_x + art_w / 2, art_y + art_h - 3.5),
        "single self-contained file  ·  ~12 KB",
        fontsize=9,
        color=C_MUTED,
        style="italic",
        ha="center",
        va="top",
    )

    # Inner contents — tabular layout. Labels are abbreviated where the full
    # metrics-dict key ("auc_test_logistic" / "cohens_d_logistic") is longer
    # than the available label column. Values right-aligned at the box edge
    # so long values can extend leftward without colliding with labels.
    # The directions value wraps to two lines because all three method names
    # together are too long for a single row.
    rows: list[tuple[str, str | list[str], str]] = [
        ("directions", ["logistic,", "diff_of_means,", "mass_mean"], C_TEXT),
        ("layer", "12  (depth 0.45)", C_TEXT),
        ("hook_name", "blocks.12.hook_resid_post", C_TEXT),
        ("auc (test)", "1.00", C_TEXT),
        ("cohens_d", "7.5", C_TEXT),
        ("auto_α", "13", C_ARTIFACT),
        ("model_id", "Qwen/Qwen2.5-1.5B-Instruct", C_TEXT),
        ("dataset_hash", "ef5939a70…", C_TEXT),
        ("schema", "v3", C_TEXT),
    ]
    label_x = art_x + 1.5
    value_right_x = art_x + art_w - 1.2  # right-align values to this x
    line_pitch = 1.6  # vertical distance between wrapped lines of one row
    cursor_y = art_y + art_h - 6.5
    for k, v, vc in rows:
        _text(
            ax,
            (label_x, cursor_y),
            k,
            fontsize=8.5,
            color=C_MUTED,
            family="monospace",
            ha="left",
            va="top",
        )
        if isinstance(v, list):
            for j, line in enumerate(v):
                _text(
                    ax,
                    (value_right_x, cursor_y - j * line_pitch),
                    line,
                    fontsize=8.5,
                    color=vc,
                    family="monospace",
                    ha="right",
                    va="top",
                )
            cursor_y -= 2.8 + line_pitch * (len(v) - 1)
        else:
            _text(
                ax,
                (value_right_x, cursor_y),
                v,
                fontsize=8.5,
                color=vc,
                family="monospace",
                weight="bold" if k == "auto_α" else "normal",
                ha="right",
                va="top",
            )
            cursor_y -= 2.8

    # =========================================================================
    # OPS strip below
    # =========================================================================

    ops_y = 2
    ops_h = 4.0
    ops = [
        ("steer", "act + α·v"),
        ("ablate", "remove v"),
        ("clamp", "act·v̂ ← target"),
        ("amplify", "γ·(act·v̂)"),
        ("score_tokens", "per-position"),
        ("report", "html one-pager"),
    ]
    n_ops = len(ops)
    ops_total_w = 94
    ops_x0 = 3
    cell_total_w = ops_total_w / n_ops
    pad = 0.6
    for i, (name, sig) in enumerate(ops):
        x0 = ops_x0 + i * cell_total_w + pad
        cw = cell_total_w - 2 * pad
        _box(ax, (x0, ops_y), (cw, ops_h), C_OPS, C_OPS_FILL, radius=0.4, lw=1.0)
        _text(
            ax,
            (x0 + cw / 2, ops_y + ops_h - 0.7),
            name,
            fontsize=10,
            weight="bold",
            color=C_OPS,
            family="monospace",
            ha="center",
            va="top",
        )
        _text(
            ax,
            (x0 + cw / 2, ops_y + ops_h - 2.6),
            sig,
            fontsize=8.5,
            color=C_MUTED,
            family="monospace",
            ha="center",
            va="top",
            style="italic",
        )

    # Legend for the act / α / v / v̂ symbols used in the op signatures above.
    _text(
        ax,
        (50, ops_y - 0.6),
        "act = residual-stream activation at the probe's layer  ·  "
        "α = scalar steering strength  ·  "
        "v = probe direction  ·  "
        "v̂ = unit-normalized v",
        fontsize=8,
        color=C_MUTED,
        ha="center",
        va="top",
        style="italic",
    )


    # =========================================================================
    # Connecting arrows between lanes
    # =========================================================================

    # Three clean lane-to-lane arrows.
    # input → activations: horizontal at the tensor box's vertical mid.
    _arrow(ax, (27.5, 45.5), (36.7, 45.5), color="#888")
    # selection summary → artifact: lands on the auto_α row, where the
    # calibration outcome actually flows. With the directions row wrapping
    # to 3 lines, auto_α now sits at y = 45.5 - 2.8 - 2*1.6 - 4*2.8 = 28.3.
    _arrow(ax, (57.5, 15.2), (70.5, 28.3), color="#888")
    # artifact → ops strip: vertical, from artifact bottom down to ops top.
    _arrow(ax, (84, art_y - 0.3), (84, ops_y + ops_h + 0.2), color=C_OPS)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved → {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    out = DEFAULT_OUT if not argv else Path(argv[0])
    render(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
