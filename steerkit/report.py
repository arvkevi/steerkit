"""Single-page HTML report rendering all the v1 steerkit plots inline.

The report is a self-contained HTML document with PNG images base64-embedded,
so it travels alongside the `.probe.safetensors` artifact as a one-pager
suitable for sharing in an email, pasting into a doc, or attaching to a PR.

Two entry points:

  Probe.report(activations, model)
    Renders: layer_selection (placeholder/skipped without per-layer fits),
             activation_projection, logit_lens. Useful for a single-concept
             snapshot report.

  GroupFit.report(activations_by_concept, model)
    Renders: per-concept layer_selection curves, similarity heatmap,
             logit_lens for the best probe per concept. Useful for a
             multi-direction summary.

Both call `render_report(...)` which is the lower-level function. The free
function takes any combination of fitted probe / per-layer dict / activations
/ model and figures out which plots to include.
"""

from __future__ import annotations

import base64
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.figure as mpl_fig
import torch

if TYPE_CHECKING:
    from .models import ModelHandle
    from .probe import Probe
    from .sweep import GroupFit


def _fig_to_data_uri(fig: mpl_fig.Figure, dpi: int = 140) -> str:
    """Render a matplotlib Figure to a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _section(title: str, body: str) -> str:
    return f"""<section>
  <h2>{title}</h2>
  {body}
</section>"""


def _img(uri: str, alt: str) -> str:
    return f'<img src="{uri}" alt="{alt}" />'


def _metrics_table(metrics: dict[str, float]) -> str:
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4g}</td></tr>"
        for k, v in metrics.items()
        if isinstance(v, (int, float))
    )
    return f"<table class='metrics'><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>{rows}</tbody></table>"


def _kv_block(items: list[tuple[str, str]]) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in items)
    return f"<table class='kv'><tbody>{rows}</tbody></table>"


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 920px; margin: 2em auto; padding: 0 1em; color: #222; }}
    h1 {{ font-size: 1.5em; border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }}
    h2 {{ font-size: 1.15em; color: #444; margin-top: 1.5em; }}
    img {{ max-width: 100%; display: block; margin: 0.5em 0; }}
    table {{ border-collapse: collapse; margin: 0.5em 0; }}
    table.metrics {{ font-size: 0.9em; }}
    table.kv td {{ padding: 0.2em 0.6em 0.2em 0; }}
    table.metrics th, table.metrics td {{ border: 1px solid #ddd; padding: 0.3em 0.6em; }}
    table.metrics th {{ background: #f5f5f5; }}
    code {{ background: #f5f5f5; padding: 0.1em 0.3em; border-radius: 3px; }}
    .meta {{ color: #666; font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">Rendered by steerkit · {timestamp}</p>
  {sections}
</body>
</html>"""


def render_probe_report(
    probe: Probe,
    *,
    model: ModelHandle | None = None,
    activations: object | None = None,
    per_layer: dict[int, Probe] | None = None,
    title: str | None = None,
) -> str:
    """Build the HTML for a single-probe report. Returns the raw HTML string.

    All inputs except `probe` are optional; the relevant plot is included only
    when its inputs are supplied.
      * `per_layer`: full per-layer fits → layer-selection curve
      * `activations`: a [n_pairs, 2, d_model] tensor (typically `activations[probe.layer]`) → PCA projection
      * `model`: a loaded ModelHandle → logit-lens
    """
    from .viz import (
        plot_activation_projection,
        plot_layer_selection,
        plot_logit_lens,
    )

    sections: list[str] = []

    summary = _kv_block(
        [
            ("model", probe.model_id),
            ("layer", str(probe.layer)),
            ("normalized depth", f"{probe.normalized_depth:.3f}"),
            ("hook", probe.hook_name),
            ("default method", probe.default_method),
            ("auto α", "—" if probe.auto_alpha is None else f"{probe.auto_alpha}"),
        ]
    )
    sections.append(_section("Probe summary", summary + _metrics_table(probe.metrics)))

    if per_layer is not None:
        try:
            fig = plot_layer_selection(
                per_layer,
                by_classifier="cohens_d_logistic"
                if "cohens_d_logistic" in next(iter(per_layer.values())).metrics
                else "auc_test_logistic",
                x_axis="normalized_depth",
                title="layer selection",
            )
            sections.append(_section("Layer selection", _img(_fig_to_data_uri(fig), "layer selection")))
        except Exception as e:  # noqa: BLE001
            sections.append(_section("Layer selection", f"<p>(skipped: {e})</p>"))

    if activations is not None:
        try:
            assert isinstance(activations, torch.Tensor)
            fig = plot_activation_projection(activations, title=f"activations at layer {probe.layer}")
            sections.append(_section("Activation projection (PCA)", _img(_fig_to_data_uri(fig), "PCA projection")))
        except Exception as e:  # noqa: BLE001
            sections.append(_section("Activation projection (PCA)", f"<p>(skipped: {e})</p>"))

    if model is not None:
        try:
            fig = plot_logit_lens(probe, model, top_k=20)
            sections.append(_section("Logit-lens (direction → vocab)", _img(_fig_to_data_uri(fig), "logit lens")))
        except Exception as e:  # noqa: BLE001
            sections.append(_section("Logit-lens (direction → vocab)", f"<p>(skipped: {e})</p>"))

    return _HTML_TEMPLATE.format(
        title=title or f"steerkit report — layer {probe.layer}",
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        sections="\n".join(sections),
    )


def render_group_report(
    fit: GroupFit,
    *,
    model: ModelHandle | None = None,
    activations_by_concept: dict[str, dict[int, object]] | None = None,
    title: str | None = None,
) -> str:
    """Build the HTML for a `GroupFit`. Includes per-concept layer curves and the
    cross-concept similarity heatmap; logit-lens for the best probe of each concept
    if a model is provided.
    """
    from .viz import plot_layer_selection, plot_logit_lens, plot_similarity_heatmap

    sections: list[str] = []

    summary = _kv_block(
        [
            ("group name", fit.group.name),
            ("relationship", fit.group.relationship),
            ("neutral reference", fit.group.neutral_reference),
            ("concepts", ", ".join(fit.names())),
        ]
    )
    sections.append(_section("Group summary", summary))

    if fit.per_concept is not None:
        layer_imgs = []
        for name, per_layer in fit.per_concept.items():
            try:
                metric = (
                    "cohens_d_logistic"
                    if "cohens_d_logistic" in next(iter(per_layer.values())).metrics
                    else "auc_test_logistic"
                )
                fig = plot_layer_selection(
                    per_layer, by_classifier=metric, x_axis="normalized_depth", title=name
                )
                layer_imgs.append(f"<h3>{name}</h3>" + _img(_fig_to_data_uri(fig), name))
            except Exception as e:  # noqa: BLE001
                layer_imgs.append(f"<p>{name}: skipped ({e})</p>")
        sections.append(_section("Layer selection per concept", "\n".join(layer_imgs)))

    try:
        source = fit.multinomial if fit.multinomial is not None else fit.best
        fig = plot_similarity_heatmap(source, title="cross-concept similarity")
        sections.append(_section("Cross-concept similarity", _img(_fig_to_data_uri(fig), "similarity")))
    except Exception as e:  # noqa: BLE001
        sections.append(_section("Cross-concept similarity", f"<p>(skipped: {e})</p>"))

    if model is not None:
        lens_imgs = []
        for name, probe in fit.best.items():
            try:
                fig = plot_logit_lens(probe, model, top_k=15, title=f"logit-lens: {name}")
                lens_imgs.append(_img(_fig_to_data_uri(fig), f"logit lens {name}"))
            except Exception as e:  # noqa: BLE001
                lens_imgs.append(f"<p>{name}: skipped ({e})</p>")
        sections.append(_section("Logit-lens per concept", "\n".join(lens_imgs)))

    return _HTML_TEMPLATE.format(
        title=title or f"steerkit report — {fit.group.name}",
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        sections="\n".join(sections),
    )


def write_report(html: str, path: str | Path) -> Path:
    """Write a rendered report HTML string to disk. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p
