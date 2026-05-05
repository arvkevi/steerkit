from __future__ import annotations

from pathlib import Path

import torch

from .data import ContrastPair
from .models import ModelHandle

EMBED_LAYER = -1  # sentinel: pre-block-0 embedding output (TL: hook_embed)


def _final_ln_layer(n_layers: int) -> int:
    """Sentinel layer index for the post-final-layernorm site (TL: ln_final.hook_normalized)."""
    return n_layers


def _hook_name(layer: int, site: str = "resid_post", *, n_layers: int | None = None) -> str:
    """Resolve a (layer, hook_site) into the canonical TL hook name.

    Conventions for `layer`:
      -1 = embedding (returns 'hook_embed', `site` ignored)
       0..n_layers-1 = block i (returns 'blocks.{i}.hook_{site}')
       n_layers = final layernorm (returns 'ln_final.hook_normalized', `site` ignored)
    """
    if layer == EMBED_LAYER:
        return "hook_embed"
    if n_layers is not None and layer == n_layers:
        return "ln_final.hook_normalized"
    if layer < 0 or (n_layers is not None and layer > n_layers):
        raise ValueError(f"layer index {layer} out of range for n_layers={n_layers}")
    return f"blocks.{layer}.hook_{site}"


def sweep_layers(n_layers: int, *, include_boundaries: bool = True) -> list[int]:
    """The list of layer indices in cheap-tier sweep order: [embed, 0, 1, ..., n-1, final_ln]
    when include_boundaries=True; otherwise just block indices.
    """
    blocks = list(range(n_layers))
    if not include_boundaries:
        return blocks
    return [EMBED_LAYER, *blocks, _final_ln_layer(n_layers)]


def extract_group_activations(
    group,  # ConceptGroup — typed via TYPE_CHECKING in concepts.py to avoid circular import
    model: ModelHandle,
    hook_site: str = "resid_post",
    *,
    include_boundaries: bool = True,
    cache_dir: str | Path | None = None,
    batch_size: int = 8,
) -> dict[str, dict[int, torch.Tensor]]:
    """Extract activations for every concept in a ConceptGroup.

    Returns a dict mapping concept_name -> {layer: tensor [n_pairs, 2, d_model]}.
    Each concept's pairs are passed through the same activation pipeline. The
    `cache_dir` is forwarded to per-concept extraction; cache keys differ per
    concept because the dataset hash differs, so each concept gets its own
    Zarr store and they're loaded/written independently.
    """
    out: dict[str, dict[int, torch.Tensor]] = {}
    for concept in group.concepts:
        if not concept.contrast_pairs:
            raise ValueError(
                f"concept {concept.name!r} has no contrast pairs — "
                f"call group.generate_pairs(...) or load pairs before extracting"
            )
        out[concept.name] = extract_activations(
            concept.contrast_pairs,
            model,
            hook_site=hook_site,
            include_boundaries=include_boundaries,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )
    return out


def _pad_token_id(tokenizer) -> int:
    """Pick a sensible pad token id even when the tokenizer has none."""
    pid = getattr(tokenizer, "pad_token_id", None)
    if pid is not None:
        return int(pid)
    eos = getattr(tokenizer, "eos_token_id", None)
    return int(eos) if eos is not None else 0


@torch.no_grad()
def _extract_batched(
    pairs: list[ContrastPair],
    model: ModelHandle,
    layer_indices: list[int],
    hook_names: dict[int, str],
    *,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Right-pad batches of (prompt, response) sequences and run them through the
    model with `run_with_cache`. Last-token activations are read at each item's
    final real position (pre-pad), so right-pad tokens never affect the result —
    causal attention prevents real positions from looking at future pads. The
    output is bit-equivalent to the sequential path, modulo float-op order on MPS.
    """
    n_pairs = len(pairs)
    d_model = model.d_model
    out: dict[int, torch.Tensor] = {
        layer: torch.zeros(n_pairs, 2, d_model, dtype=torch.float32) for layer in layer_indices
    }

    # Flatten (pair_idx, response_idx) over the dataset.
    items: list[tuple[int, int, str, str]] = []
    for i, pair in enumerate(pairs):
        items.append((i, 0, pair.prompt, pair.positive_response))
        items.append((i, 1, pair.prompt, pair.negative_response))

    pad_id = _pad_token_id(model.tokenizer)
    names_filter = list(hook_names.values())

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        token_tensors: list[torch.Tensor] = []
        for _, _, prompt, response in batch:
            ids = model.format_chat(prompt, response)
            token_tensors.append(ids.squeeze(0) if ids.dim() == 2 else ids)

        max_len = max(t.shape[0] for t in token_tensors)
        padded = torch.full(
            (len(batch), max_len),
            pad_id,
            dtype=token_tensors[0].dtype,
            device=token_tensors[0].device,
        )
        last_positions: list[int] = []
        for i, t in enumerate(token_tensors):
            padded[i, : t.shape[0]] = t
            last_positions.append(t.shape[0] - 1)

        _, cache = model.hooked.run_with_cache(padded, names_filter=names_filter)
        for batch_i, (pair_i, resp_i, _, _) in enumerate(batch):
            last_pos = last_positions[batch_i]
            for layer in layer_indices:
                act = cache[hook_names[layer]][batch_i, last_pos, :]
                out[layer][pair_i, resp_i] = act.float().cpu()

    return out


@torch.no_grad()
def _extract_sequential(
    pairs: list[ContrastPair],
    model: ModelHandle,
    layer_indices: list[int],
    hook_names: dict[int, str],
) -> dict[int, torch.Tensor]:
    """One forward pass per (pair, response). Same output shape as `_extract_batched`."""
    n_pairs = len(pairs)
    d_model = model.d_model
    out: dict[int, torch.Tensor] = {
        layer: torch.zeros(n_pairs, 2, d_model, dtype=torch.float32) for layer in layer_indices
    }
    names_filter = list(hook_names.values())
    for i, pair in enumerate(pairs):
        for j, response in enumerate((pair.positive_response, pair.negative_response)):
            tokens = model.format_chat(pair.prompt, response)
            _, cache = model.hooked.run_with_cache(tokens, names_filter=names_filter)
            for layer in layer_indices:
                last_tok_act = cache[hook_names[layer]][0, -1, :]
                out[layer][i, j] = last_tok_act.float().cpu()
    return out


@torch.no_grad()
def extract_activations(
    pairs: list[ContrastPair],
    model: ModelHandle,
    hook_site: str = "resid_post",
    *,
    include_boundaries: bool = True,
    cache_dir: str | Path | None = None,
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """Extract last-token activations for each (positive, negative) response in each pair, per layer.

    Returns a dict mapping layer index -> tensor of shape [n_pairs, 2, d_model].
    Layer index 0 is the positive response and 1 is the negative response in the second axis.

    With `include_boundaries=True` (default), the dict also contains entries at:
      - layer = -1   (embedding output, TL hook 'hook_embed')
      - layer = n_layers (final layernorm output, TL hook 'ln_final.hook_normalized')
    These let the cheap-tier sweep span [embed → 0..N-1 → final_ln] in one pass.

    `cache_dir`: optional directory for a Zarr v3 activation cache. The cache key
    is derived from (model_id, hook_site, include_boundaries, pairs hash). On a
    cache hit we skip the model entirely and load tensors from disk.

    `batch_size`: number of (pair, response) sequences run through the model in a
    single forward pass. Sequences are right-padded to max length per batch; causal
    attention guarantees pads don't pollute real-token activations. Set to 1 for
    a strictly sequential path (e.g. for memory-constrained big models). Default 8
    is a reasonable speedup-to-memory tradeoff for ≤4B-parameter models on MPS.
    """
    n_pairs = len(pairs)
    n_layers = model.n_layers
    d_model = model.d_model

    layer_indices = sweep_layers(n_layers, include_boundaries=include_boundaries)
    hook_names = {layer: _hook_name(layer, hook_site, n_layers=n_layers) for layer in layer_indices}

    cache_target: Path | None = None
    pairs_hash: str | None = None
    if cache_dir is not None:
        from .cache import cache_path, cache_signature, hash_pairs, load_activations_zarr

        pairs_hash = hash_pairs(pairs)
        sig = cache_signature(
            model_id=model.model_id,
            hook_site=hook_site,
            include_boundaries=include_boundaries,
            pairs_hash=pairs_hash,
        )
        cache_target = cache_path(cache_dir, sig)
        if cache_target.exists():
            try:
                cached, _meta = load_activations_zarr(cache_target)
                if set(cached.keys()) == set(layer_indices):
                    return cached
            except Exception:
                # Fall through and re-extract on any cache-read failure.
                pass

    if batch_size <= 1:
        out = _extract_sequential(pairs, model, layer_indices, hook_names)
    else:
        out = _extract_batched(
            pairs, model, layer_indices, hook_names, batch_size=batch_size
        )

    if cache_target is not None and pairs_hash is not None:
        from .cache import save_activations_zarr

        save_activations_zarr(
            out,
            cache_target,
            metadata={
                "model_id": model.model_id,
                "hook_site": hook_site,
                "include_boundaries": include_boundaries,
                "pairs_hash": pairs_hash,
                "n_pairs": n_pairs,
                "d_model": d_model,
                "batch_size": batch_size,
            },
        )

    return out
