"""Headline sweep entry: ConceptGroup -> per-concept Probes (+ optional multinomial).

`sweep(group, model)` is the one-liner the README pitches:
    1) extract activations for every concept in the group (Zarr-cached),
    2) fit per-layer Probes for each concept (logistic + diff-of-means + mass-mean),
    3) select best layer per concept,
    4) for `mutually_exclusive` groups, fit a multinomial diagnostic probe,
    5) optionally call the LLM-judge expensive tier on top-K layers.

`GroupFit` is the resulting container; index by concept name to get the chosen
`Probe`. `compose(probes, weights)` lets you steer with vectors from multiple
groups simultaneously (e.g., joy + formal).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .extract import extract_group_activations
from .models import ModelHandle
from .probe import MultinomialProbe, Probe
from .teacher import TeacherModel

if TYPE_CHECKING:
    from .concepts import ConceptGroup


@dataclass
class GroupFit:
    """Result of `sweep(group, model)`. Indexable by concept name to get the chosen Probe."""

    group: ConceptGroup
    best: dict[str, Probe]
    per_concept: dict[str, dict[int, Probe]] | None = None  # full per-layer fits; None on reload
    multinomial: MultinomialProbe | None = None

    def __getitem__(self, concept_name: str) -> Probe:
        return self.best[concept_name]

    def __iter__(self):
        return iter(self.best)

    def __contains__(self, name: str) -> bool:
        return name in self.best

    def names(self) -> list[str]:
        return list(self.best.keys())

    def plot_layer_selection(self, concept_name: str, **kwargs):
        """Render the layer-selection dual-curve for one concept (requires per_concept)."""
        if self.per_concept is None:
            raise RuntimeError(
                "per_concept is None — plot_layer_selection requires the full per-layer fits. "
                "Re-run sweep() instead of loading a saved GroupFit."
            )
        if concept_name not in self.per_concept:
            raise KeyError(f"no concept named {concept_name!r} in this GroupFit")
        from .viz import plot_layer_selection

        return plot_layer_selection(self.per_concept[concept_name], **kwargs)

    def plot_similarity(self, **kwargs):
        """Render the cross-concept similarity heatmap. Uses the multinomial probe if present,
        otherwise falls back to the per-concept best probes' steering directions."""
        from .viz import plot_similarity_heatmap

        source = self.multinomial if self.multinomial is not None else self.best
        return plot_similarity_heatmap(source, **kwargs)

    def report(
        self,
        *,
        model: ModelHandle | None = None,
        out: str | Path | None = None,
        title: str | None = None,
    ) -> str:
        """Render a one-page HTML report for this GroupFit. Returns the HTML string;
        if `out` is set, also writes it to disk and returns the path-as-string."""
        from .report import render_group_report, write_report

        html = render_group_report(self, model=model, title=title)
        if out is not None:
            return str(write_report(html, out))
        return html

    def window(self, concept_name: str, *, k: int = 1) -> CompositeProbe:
        """Build a window-of-(2k+1) multi-layer composite around the chosen best layer
        for `concept_name`. Requires the full `per_concept` fits (i.e. not loaded from disk)."""
        if self.per_concept is None:
            raise RuntimeError(
                "per_concept is None — window requires the full per-layer fits. "
                "Re-run sweep() instead of loading a saved GroupFit."
            )
        if concept_name not in self.per_concept:
            raise KeyError(f"no concept named {concept_name!r} in this GroupFit")
        center = self.best[concept_name].layer
        return window(self.per_concept[concept_name], center_layer=center, k=k)

    def save(self, dir_path: str | Path) -> None:
        """Save the chosen Probe per concept + multinomial (if present) + the ConceptGroup
        snapshot (without contrast pairs) into a directory.

        Layout:
            dir_path/
              group.json
              {concept_name}.probe.safetensors      (one per concept)
              multinomial.probe.safetensors         (only if mutex group)
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        # Save group snapshot (clear out concept pairs to keep the file small; the artifacts
        # carry the dataset_hash via probe metadata for reproducibility).
        group_dict = self.group.as_dict()
        for c in group_dict["concepts"]:
            c["contrast_pairs"] = []
        (dir_path / "group.json").write_text(__import__("json").dumps(group_dict, indent=2))
        for name, probe in self.best.items():
            probe.save(dir_path / f"{_safe_filename(name)}.probe.safetensors")
        if self.multinomial is not None:
            self.multinomial.save(dir_path / "multinomial.probe.safetensors")

    @classmethod
    def load(cls, dir_path: str | Path) -> GroupFit:
        """Load a directory written by GroupFit.save."""
        from .concepts import ConceptGroup  # local import to avoid circular

        dir_path = Path(dir_path)
        group = ConceptGroup.from_dict(__import__("json").loads((dir_path / "group.json").read_text()))
        best: dict[str, Probe] = {}
        for concept in group.concepts:
            path = dir_path / f"{_safe_filename(concept.name)}.probe.safetensors"
            if path.exists():
                best[concept.name] = Probe.load(path)
        multinomial = None
        mn_path = dir_path / "multinomial.probe.safetensors"
        if mn_path.exists():
            multinomial = MultinomialProbe.load(mn_path)
        return cls(group=group, best=best, per_concept=None, multinomial=multinomial)


def _safe_filename(name: str) -> str:
    """Filesystem-safe filename derived from a concept name."""
    return "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in name)


def sweep(
    group: ConceptGroup,
    model: ModelHandle,
    *,
    hook_site: str = "resid_post",
    test_fraction: float = 0.2,
    seed: int = 42,
    cache_dir: str | Path | None = None,
    select_by: str = "auc_test_logistic",
    with_steering_eval: bool = False,
    teacher: TeacherModel | None = None,
    eval_top_k: int = 5,
    eval_alpha: float = 4.0,
) -> GroupFit:
    """Run the full Phase-3+4 sweep on a ConceptGroup.

    Steps:
      1) `extract_group_activations` per concept (Zarr-cached if `cache_dir` set).
      2) `Probe.fit_all` per concept + select best layer by `select_by`.
      3) For `mutually_exclusive` groups with ≥2 concepts: fit a `MultinomialProbe`
         at the layer with highest held-out multinomial accuracy.
      4) If `with_steering_eval=True` and `teacher` is provided: call the LLM-judge
         expensive tier on each concept's top-K layers and attach `metrics["steering_effect"]`.
    """
    activations_by_concept = extract_group_activations(
        group, model, hook_site=hook_site, cache_dir=cache_dir
    )

    per_concept: dict[str, dict[int, Probe]] = {}
    best: dict[str, Probe] = {}
    for concept in group.concepts:
        layer_probes = Probe.fit_all(
            activations_by_concept[concept.name],
            model,
            hook_site=hook_site,
            test_fraction=test_fraction,
            seed=seed,
        )
        per_concept[concept.name] = layer_probes
        best[concept.name] = Probe.best_layer(layer_probes, by=select_by)

    multinomial: MultinomialProbe | None = None
    if group.relationship == "mutually_exclusive" and len(group.concepts) >= 2:
        multinomial = MultinomialProbe.fit_best_layer(
            activations_by_concept,
            model,
            hook_site=hook_site,
            test_fraction=test_fraction,
            seed=seed,
        )

    if with_steering_eval:
        if teacher is None:
            raise ValueError("with_steering_eval=True requires a teacher TeacherModel")
        from .eval import evaluate_steering_effect

        for concept in group.concepts:
            evaluate_steering_effect(
                per_concept[concept.name],
                model,
                teacher,
                concept_description=concept.description,
                top_k=eval_top_k,
                alpha=eval_alpha,
            )
            # Re-select best layer by steering_effect when available.
            if "steering_effect" in next(iter(per_concept[concept.name].values())).metrics:
                # If the eval attached to all evaluated probes, we can re-rank by it.
                evaluated = {
                    layer: p
                    for layer, p in per_concept[concept.name].items()
                    if "steering_effect" in p.metrics
                }
                if evaluated:
                    best[concept.name] = max(
                        evaluated.values(), key=lambda p: p.metrics["steering_effect"]
                    )

    return GroupFit(group=group, best=best, per_concept=per_concept, multinomial=multinomial)


@dataclass
class CompositeProbe:
    """Multiple probes' steering vectors composed at inference time.

    Each probe's direction is added at its own layer with its own per-probe weight
    (and per-probe alpha if desired). Probes targeting the same TL hook are
    combined into a single hook function.
    """

    probes: list[Probe]
    weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [1.0] * len(self.probes)
        if len(self.weights) != len(self.probes):
            raise ValueError(
                f"weights ({len(self.weights)}) must match probes ({len(self.probes)})"
            )

    def export_gguf(
        self,
        path: str | Path,
        *,
        method: str | None = None,
        scale: float = 1.0,
    ) -> Path:
        """Convenience: export this composite as a multi-layer gguf control vector."""
        from .gguf_export import export_composite_to_gguf

        return export_composite_to_gguf(self, path, method=method, scale=scale)

    @torch.no_grad()
    def steer(
        self,
        model: ModelHandle,
        prompt: str,
        *,
        alphas: list[float] | None = None,
        methods: list[str | None] | None = None,
        max_new_tokens: int = 60,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion with all composed probes' steering vectors active.

        `alphas`: per-probe alpha override; defaults to each probe's auto_alpha (or 2.0).
        `methods`: per-probe direction method override.
        """
        if alphas is None:
            alphas = [
                p.auto_alpha if p.auto_alpha is not None else 2.0 for p in self.probes
            ]
        if methods is None:
            methods = [None] * len(self.probes)
        if len(alphas) != len(self.probes) or len(methods) != len(self.probes):
            raise ValueError("alphas and methods must match the number of probes")

        # Group probes by their TL hook name so we can fold same-hook adds.
        per_hook: dict[str, list[torch.Tensor]] = {}
        for probe, weight, alpha, method in zip(
            self.probes, self.weights, alphas, methods, strict=True
        ):
            direction = probe.get_direction(method).to(model.device).to(model.hooked.cfg.dtype)
            scaled = weight * alpha * direction  # combined scalar absorbed into the vector
            per_hook.setdefault(probe.hook_name, []).append(scaled)

        fwd_hooks: list[tuple[str, Callable[..., torch.Tensor]]] = []
        for hook_name, contributions in per_hook.items():
            # Sum all contributions targeting this hook into a single vector.
            combined = contributions[0].clone()
            for v in contributions[1:]:
                combined = combined + v

            def _build(vec: torch.Tensor) -> Callable[..., torch.Tensor]:
                def steering_hook(activation, hook):  # noqa: ARG001
                    return activation + vec

                return steering_hook

            fwd_hooks.append((hook_name, _build(combined)))

        tokens = model.format_chat(prompt)
        tokenizer = model.tokenizer
        assert tokenizer is not None
        with model.hooked.hooks(fwd_hooks=fwd_hooks):  # type: ignore[arg-type]
            output_ids = model.hooked.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                verbose=False,
            )
        assert isinstance(output_ids, torch.Tensor)
        prompt_len = tokens.shape[-1]
        new_tokens = output_ids[0, prompt_len:]
        return str(tokenizer.decode(new_tokens, skip_special_tokens=True))


def compose(
    probes: list[Probe], weights: list[float] | None = None
) -> CompositeProbe:
    """Compose multiple probes for simultaneous steering at inference time.

    `weights` defaults to all 1.0 (equal contribution). Probes can come from
    different ConceptGroups — the design memory's "axes" composition.
    """
    return CompositeProbe(probes=list(probes), weights=list(weights) if weights else [])


def window(
    probes: dict[int, Probe],
    center_layer: int,
    *,
    k: int = 1,
) -> CompositeProbe:
    """Build a multi-layer steering composite over a window of layers around `center_layer`.

    Args:
      probes: full per-layer dict from `Probe.fit_all` (or `GroupFit.per_concept[name]`).
      center_layer: the chosen "best" layer; window is built around it.
      k: half-window size. k=1 (default) selects [center-1, center, center+1] — the
         "window-of-3" mode the design memory commits to. k=0 collapses to the
         single best probe wrapped in a CompositeProbe.

    The returned CompositeProbe has `weights = [1/n]*n` so each layer's contribution
    is scaled down proportionally. With each probe's `auto_alpha` left as the default
    α at steer time, the per-layer push is `auto_alpha / n` rather than `auto_alpha`,
    keeping the total perturbation roughly comparable to a single-layer steer.
    """
    if not probes:
        raise ValueError("no probes to window")
    if center_layer not in probes:
        raise KeyError(f"center_layer {center_layer} not in probes (keys: {sorted(probes)})")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    layers = sorted(probes.keys())
    center_idx = layers.index(center_layer)
    start = max(0, center_idx - k)
    end = min(len(layers), center_idx + k + 1)
    selected = [probes[layers[i]] for i in range(start, end)]
    weights = [1.0 / len(selected)] * len(selected)
    return CompositeProbe(probes=selected, weights=weights)
