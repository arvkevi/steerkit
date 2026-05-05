"""Export a steerkit steering vector to llama.cpp's GGUF control-vector format.

Optional dependency: install with `uv pip install steerkit[llamacpp]` (or just
`pip install gguf`). Without that, this module raises a helpful ImportError.

Format target — llama.cpp's control-vector convention:
  - architecture: "controlvector"
  - tensors named "direction.{layer}" with shape [d_model], one per source layer
  - a small bag of KV metadata identifying the source model + hook site

Two entry points:

  export_probe_to_gguf(probe, path, ...)
    Writes one tensor at the probe's chosen layer. Use this for a single-layer
    Probe trained at one site.

  export_composite_to_gguf(composite, path, ...)
    Writes one tensor per constituent probe (same TL hook → folded into one).
    Use this for a window-of-(2k+1) composite, or for cross-group composition
    where multiple probes target different layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .probe import Probe
    from .sweep import CompositeProbe


def _require_gguf():
    try:
        import gguf  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "GGUF export requires the `gguf` package. "
            "Install with `uv pip install gguf` or `pip install steerkit[llamacpp]`."
        ) from e
    return gguf


def export_probe_to_gguf(
    probe: Probe,
    path: str | Path,
    *,
    method: str | None = None,
    scale: float = 1.0,
) -> Path:
    """Write a single-layer Probe as a GGUF control vector. Returns the output path.

    `method` selects which probe-family direction to export (defaults to
    `probe.default_method`). `scale` is multiplied into the direction before
    writing — useful when you want the gguf-saved magnitude to differ from
    the unit-normalized direction steerkit stores internally (e.g. bake in
    the calibrated auto_alpha).
    """
    gguf = _require_gguf()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    direction = probe.get_direction(method).cpu().numpy() * float(scale)

    writer = gguf.GGUFWriter(str(path), arch="controlvector")
    writer.add_string("model.id", probe.model_id)
    writer.add_string("hook_site", probe.hook_site)
    writer.add_string("hook_name", probe.hook_name)
    writer.add_string("steerkit.method", method or probe.default_method)
    writer.add_int32("n_total_layers", int(probe.n_total_layers))
    writer.add_int32("source.layer", int(probe.layer))
    writer.add_float32("scale", float(scale))
    writer.add_tensor(f"direction.{probe.layer}", direction)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def export_composite_to_gguf(
    composite: CompositeProbe,
    path: str | Path,
    *,
    method: str | None = None,
    scale: float = 1.0,
) -> Path:
    """Write a CompositeProbe as a GGUF control vector with one entry per layer.

    Probes targeting the same layer are summed into a single direction first
    (consistent with `CompositeProbe.steer`'s same-hook folding). Each
    constituent probe's `weight` is multiplied into its direction; `scale`
    is applied uniformly on top of that.
    """
    gguf = _require_gguf()
    if not composite.probes:
        raise ValueError("composite has no probes to export")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Fold same-layer probes by summing weight*direction.
    by_layer: dict[int, torch.Tensor] = {}
    sample = composite.probes[0]
    for probe, weight in zip(composite.probes, composite.weights, strict=True):
        d = probe.get_direction(method) * float(weight)
        if probe.layer in by_layer:
            by_layer[probe.layer] = by_layer[probe.layer] + d
        else:
            by_layer[probe.layer] = d.clone()

    writer = gguf.GGUFWriter(str(path), arch="controlvector")
    writer.add_string("model.id", sample.model_id)
    writer.add_string("hook_site", sample.hook_site)
    writer.add_string("steerkit.method", method or sample.default_method)
    writer.add_int32("n_total_layers", int(sample.n_total_layers))
    writer.add_int32("composite.size", int(len(composite.probes)))
    writer.add_float32("scale", float(scale))
    for layer, direction in sorted(by_layer.items()):
        writer.add_tensor(f"direction.{layer}", (direction * float(scale)).cpu().numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path
