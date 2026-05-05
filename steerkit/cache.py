"""Zarr v3 activation cache.

Activation extraction is the most expensive step in the workflow — it requires
running the model. This module makes that step memoizable: when the user passes
`cache_dir=...` to `extract_activations`, we hash the inputs (model id, hook
site, include_boundaries flag, and the contents of the contrast pairs), look up
a Zarr store keyed by that hash, and either load it or extract + write it.

Cache layout (one Zarr store per (model, hook_site, dataset) combination):

    {cache_dir}/{signature}.zarr/
      activations          # array shape [n_pairs, 2, n_sites, d_model], float32
      .zattrs              # cache_schema_version, layer_indices, model_id,
                           # hook_site, include_boundaries, pairs_hash, n_pairs

Writes are atomic: extract to `{path}.tmp/`, rename to `{path}/` when complete.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import zarr
import zarr.storage

from .data import ContrastPair

CACHE_SCHEMA_VERSION = 1


def hash_pairs(pairs: list[ContrastPair]) -> str:
    """Deterministic 16-hex-char hash of a list of contrast pairs (order matters)."""
    h = hashlib.sha256()
    for p in pairs:
        h.update(
            json.dumps(
                [p.prompt, p.positive_response, p.negative_response],
                ensure_ascii=False,
            ).encode("utf-8")
        )
    return h.hexdigest()[:16]


def cache_signature(
    *,
    model_id: str,
    hook_site: str,
    include_boundaries: bool,
    pairs_hash: str,
    pooling: str = "last",
) -> str:
    """Stable signature string used as the cache filename. Filesystem-safe.

    `pooling` defaults to "last" so existing cache files (written before
    pooling support) keep their original signature and remain reusable.
    Non-default pooling modes get an extra suffix that bypasses the legacy
    cache for that activation strategy.
    """
    safe_model = model_id.replace("/", "--").replace(":", "_")
    boundaries = "boundaries" if include_boundaries else "blocks"
    suffix = "" if pooling == "last" else f"__{pooling}"
    return f"{safe_model}__{hook_site}__{boundaries}__{pairs_hash}{suffix}"


def cache_path(cache_dir: str | Path, signature: str) -> Path:
    return Path(cache_dir) / f"{signature}.zarr"


def save_activations_zarr(
    activations: dict[int, torch.Tensor],
    path: str | Path,
    *,
    metadata: dict,
) -> None:
    """Atomically write a {layer: [n_pairs, 2, d_model]} dict to a Zarr v3 store.

    The activations are stacked along a new third axis so the on-disk array has
    shape [n_pairs, 2, n_sites, d_model]; the layer indices are saved as the
    `layer_indices` attribute (in the same order as the third axis).
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    if path.exists():
        shutil.rmtree(path)

    layer_indices = sorted(activations.keys())
    if not layer_indices:
        raise ValueError("no activations to save")
    first = activations[layer_indices[0]]
    n_pairs = first.shape[0]
    d_model = first.shape[-1]
    stacked = torch.stack([activations[layer] for layer in layer_indices], dim=2).numpy().astype(np.float32)
    # stacked shape: [n_pairs, 2, n_sites, d_model]

    attributes = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "layer_indices": list(layer_indices),
        **metadata,
    }
    zarr.create_array(
        store=zarr.storage.LocalStore(str(tmp)),
        name="activations",
        chunks=(min(n_pairs, 64), 2, len(layer_indices), d_model),
        data=stacked,
        attributes=attributes,
        overwrite=True,
    )
    tmp.rename(path)


def load_activations_zarr(path: str | Path) -> tuple[dict[int, torch.Tensor], dict]:
    """Load a Zarr cache produced by save_activations_zarr.

    Returns (activations dict keyed by layer index, metadata dict).
    """
    path = Path(path)
    arr = zarr.open_array(store=zarr.storage.LocalStore(str(path)), path="activations", mode="r")
    data = np.asarray(arr[:])  # [n_pairs, 2, n_sites, d_model]
    attrs = dict(arr.attrs)
    raw_indices = attrs["layer_indices"]
    if not isinstance(raw_indices, (list, tuple)):
        raise ValueError(f"layer_indices in {path} is not a list (got {type(raw_indices).__name__})")
    layer_indices: list[int] = [int(x) for x in raw_indices]
    out: dict[int, torch.Tensor] = {
        layer: torch.tensor(data[:, :, i, :], dtype=torch.float32)
        for i, layer in enumerate(layer_indices)
    }
    return out, attrs
