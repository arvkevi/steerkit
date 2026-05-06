"""Probe: a multi-direction linear probe + steering vector at a single layer.

Per layer we fit *three* candidate directions — logistic, difference-of-means,
and mass-mean (LDA with Ledoit-Wolf shrinkage) — and report held-out AUC plus
a Cohen's-d separation score along the logistic direction. The user picks
which direction to use at steer-time via `probe.steer(..., method=...)`; the
saved artifact carries all three.

Layer indexing convention:
  layer = -1            → embedding output (TL hook 'hook_embed')
  layer = 0..n-1        → block i resid_post (or other hook_site)
  layer = n_total_layers → final layernorm (TL hook 'ln_final.hook_normalized')

Normalized depth is `(layer + 1) / (n_total_layers + 1)`, so embed=0.0,
final_ln=1.0, and blocks fall on a uniform grid between them.

Probe artifact schema v3:
  - tensors:  direction.{logistic,diff_of_means,mass_mean}
  - metadata: schema_version=3, layer, n_total_layers, normalized_depth,
              model_id, hook_site, hook_name, default_method, auto_alpha,
              bias_json, metrics_json, created_at, extras_json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .extract import _hook_name
from .models import ModelHandle

SCHEMA_VERSION = 3

PROBE_METHODS: tuple[str, ...] = ("logistic", "diff_of_means", "mass_mean")


def _split_pair_indices(n_pairs: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_pair_idx, test_pair_idx) — split at the *pair* level so that
    a pair's positive and negative responses always end up in the same fold.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pairs)
    n_test = max(1, int(round(n_pairs * test_fraction))) if test_fraction > 0 else 0
    n_test = min(n_test, n_pairs - 1) if test_fraction > 0 else 0
    return perm[n_test:], perm[:n_test]


def _stack_xy(acts: torch.Tensor, pair_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack a [n_pairs, 2, d_model] tensor into (X, y) for the given pair indices.
    Positives (label=1) first, then negatives (label=0).
    """
    pos = acts[pair_idx, 0, :].numpy()
    neg = acts[pair_idx, 1, :].numpy()
    X = np.concatenate([pos, neg], axis=0)
    y = np.array([1] * len(pair_idx) + [0] * len(pair_idx))
    return X, y


def _unit(v: np.ndarray) -> torch.Tensor:
    t = torch.tensor(v, dtype=torch.float32)
    return t / (t.norm() + 1e-12)


def _cohens_d(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Cohen's d on a 1-D scalar score: standardized mean difference."""
    pooled = np.sqrt(0.5 * (scores_pos.var(ddof=1) + scores_neg.var(ddof=1)))
    if pooled < 1e-12:
        return 0.0
    return float((scores_pos.mean() - scores_neg.mean()) / pooled)


@dataclass
class TokenScores:
    """Per-token probe scores along a single sequence.

    Built by `Probe.score_tokens(...)`. `scores[i]` is the projection of the
    token-`i` residual-stream activation (at the probe's layer) onto the
    probe's direction. Higher values mean the direction is more active at
    that token; the sign matches positive vs. negative side of the probe.

    Attributes:
        tokens: decoded token strings, one per position scored.
        scores: 1-D float tensor with one entry per token.
        layer: the probe's layer index (so plots can label themselves).
        method: which probe-family direction was used (logistic / diff_of_means
            / mass_mean).
        response_start: index into `tokens` where the assistant response begins.
            Always 0 when `score_tokens` was called with `include_prompt=False`
            (the prompt portion has been sliced off); otherwise marks the
            user-prompt → assistant-response boundary.
    """

    tokens: list[str]
    scores: torch.Tensor
    layer: int
    method: str
    response_start: int = 0

    def __post_init__(self) -> None:
        if self.scores.dim() != 1:
            raise ValueError(f"scores must be 1-D, got shape {tuple(self.scores.shape)}")
        if len(self.tokens) != self.scores.shape[0]:
            raise ValueError(
                f"tokens ({len(self.tokens)}) and scores ({self.scores.shape[0]}) "
                f"must have the same length"
            )

    def plot(self, **kwargs: Any) -> Any:
        """Convenience: render with `steerkit.viz.plot_token_scores`."""
        from .viz import plot_token_scores

        return plot_token_scores(self, **kwargs)


@dataclass
class Probe:
    """Multi-direction linear probe at a single layer.

    `directions` holds one unit-normalized [d_model] vector per probe family.
    `metrics` holds named scalars (e.g. auc_test_logistic, cohens_d_logistic).
    `default_method` chooses which direction Probe.steer() uses by default.
    """

    directions: dict[str, torch.Tensor]
    bias: dict[str, float]
    layer: int
    metrics: dict[str, float]
    model_id: str
    hook_site: str
    hook_name: str  # the canonical TL hook string used at extract / steer time
    n_total_layers: int
    default_method: str = "logistic"
    auto_alpha: float | None = None
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_depth(self) -> float:
        # Map layer index in [-1, n_total_layers] -> [0, 1].
        return (self.layer + 1) / (self.n_total_layers + 1)

    @property
    def direction(self) -> torch.Tensor:
        """The default-method direction tensor, ready for arithmetic."""
        return self.directions[self.default_method]

    @property
    def auc(self) -> float:
        """Held-out logistic AUC if available, else train AUC, else NaN."""
        for k in ("auc_test_logistic", "auc_train_logistic"):
            if k in self.metrics:
                return self.metrics[k]
        return float("nan")

    @classmethod
    def fit_all(
        cls,
        activations: dict[int, torch.Tensor],
        model: ModelHandle,
        *,
        hook_site: str = "resid_post",
        test_fraction: float = 0.2,
        seed: int = 42,
        default_method: str = "logistic",
    ) -> dict[int, Probe]:
        """Fit one Probe per layer with all three candidate directions + cheap-tier metrics.

        For each layer the training pipeline is:
          1) split pairs into train/test (test_fraction; pair-level split, no leakage).
          2) fit logistic regression on train activations; store coef + AUC train/test.
          3) compute difference-of-means direction on train; score AUC train/test via cosine.
          4) fit LDA with Ledoit-Wolf shrinkage on train; score AUC train/test via decision_function.
          5) compute Cohen's d on the held-out logistic decision-function scores.
        """
        if default_method not in PROBE_METHODS:
            raise ValueError(f"default_method must be in {PROBE_METHODS}, got {default_method!r}")
        # Use the first layer's tensor to get n_pairs; assume all layers match shape.
        first_acts = next(iter(activations.values()))
        n_pairs = first_acts.shape[0]
        train_idx, test_idx = _split_pair_indices(n_pairs, test_fraction, seed)
        has_test = len(test_idx) > 0

        probes: dict[int, Probe] = {}
        for layer, acts in activations.items():
            X_train, y_train = _stack_xy(acts, train_idx)
            X_test, y_test = (_stack_xy(acts, test_idx) if has_test else (None, None))

            directions: dict[str, torch.Tensor] = {}
            biases: dict[str, float] = {}
            metrics: dict[str, float] = {}

            # 1) logistic
            log_clf = LogisticRegression(max_iter=2000, C=1.0)
            log_clf.fit(X_train, y_train)
            log_dir_unnorm = log_clf.coef_[0]
            directions["logistic"] = _unit(log_dir_unnorm)
            biases["logistic"] = float(log_clf.intercept_[0])
            metrics["auc_train_logistic"] = float(roc_auc_score(y_train, log_clf.decision_function(X_train)))
            if has_test:
                test_scores = log_clf.decision_function(X_test)
                metrics["auc_test_logistic"] = float(roc_auc_score(y_test, test_scores))
                pos_scores = test_scores[: len(test_idx)]
                neg_scores = test_scores[len(test_idx) :]
                metrics["cohens_d_logistic"] = _cohens_d(pos_scores, neg_scores)

            # 2) diff-of-means
            mu_pos_train = X_train[: len(train_idx)].mean(axis=0)
            mu_neg_train = X_train[len(train_idx) :].mean(axis=0)
            dom_dir_unnorm = mu_pos_train - mu_neg_train
            directions["diff_of_means"] = _unit(dom_dir_unnorm)
            biases["diff_of_means"] = 0.0
            # AUC via cosine: project activations onto unit direction, score = dot product.
            train_proj = X_train @ directions["diff_of_means"].numpy()
            metrics["auc_train_diff_of_means"] = float(roc_auc_score(y_train, train_proj))
            if has_test:
                test_proj = X_test @ directions["diff_of_means"].numpy()
                metrics["auc_test_diff_of_means"] = float(roc_auc_score(y_test, test_proj))

            # 3) mass-mean / LDA with Ledoit-Wolf shrinkage (handles d_model >> n_samples).
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(X_train, y_train)
            mm_dir_unnorm = lda.coef_[0]
            directions["mass_mean"] = _unit(mm_dir_unnorm)
            biases["mass_mean"] = float(lda.intercept_[0])
            metrics["auc_train_mass_mean"] = float(
                roc_auc_score(y_train, lda.decision_function(X_train))
            )
            if has_test:
                metrics["auc_test_mass_mean"] = float(
                    roc_auc_score(y_test, lda.decision_function(X_test))
                )

            metrics["n_train_pairs"] = float(len(train_idx))
            metrics["n_test_pairs"] = float(len(test_idx))

            # Record the typical residual-stream norm at this layer. Pre-LayerNorm
            # transformers accumulate residual norm across depth, so the same α has
            # very different effective strengths at different layers — this lets
            # users (and `calibrate_alpha`) scale α appropriately. Computed on
            # train activations only, mean across (pair × response).
            metrics["activation_norm_mean"] = float(
                np.linalg.norm(X_train, axis=-1).mean()
            )

            probes[layer] = cls(
                directions=directions,
                bias=biases,
                layer=layer,
                metrics=metrics,
                model_id=model.model_id,
                hook_site=hook_site,
                hook_name=_hook_name(layer, hook_site, n_layers=model.n_layers),
                n_total_layers=model.n_layers,
                default_method=default_method,
            )
        return probes

    @staticmethod
    def best_layer(
        probes: dict[int, Probe],
        by: str = "auc_test_logistic",
    ) -> Probe:
        """Pick the layer that maximizes the given metric. Falls back to a cheap-tier alternative
        if the requested metric isn't present (e.g. when test_fraction=0).
        """
        sample = next(iter(probes.values()))
        if by not in sample.metrics:
            for fallback in ("auc_train_logistic", "auc_train_diff_of_means", "auc_train_mass_mean"):
                if fallback in sample.metrics:
                    by = fallback
                    break
            else:
                raise KeyError(f"no usable metric found; tried {by!r} and train fallbacks")
        return max(probes.values(), key=lambda p: p.metrics[by])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {f"direction.{name}": v.contiguous() for name, v in self.directions.items()}
        metadata = {
            "schema_version": str(self.schema_version),
            "layer": str(self.layer),
            "n_total_layers": str(self.n_total_layers),
            "normalized_depth": f"{self.normalized_depth:.6f}",
            "model_id": self.model_id,
            "hook_site": self.hook_site,
            "hook_name": self.hook_name,
            "default_method": self.default_method,
            "auto_alpha": "" if self.auto_alpha is None else f"{self.auto_alpha:.6f}",
            "bias_json": json.dumps(self.bias),
            "metrics_json": json.dumps(self.metrics),
            "created_at": self.created_at,
            "extras_json": json.dumps(self.extras),
        }
        save_file(tensors, str(path), metadata=metadata)

    @classmethod
    def load(cls, path: str | Path) -> Probe:
        path = Path(path)
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            md = f.metadata() or {}
            directions = {k.split(".", 1)[1]: f.get_tensor(k) for k in keys if k.startswith("direction.")}
        if not directions:
            raise ValueError(f"no direction.* tensors found in {path}")
        auto_alpha_str = md.get("auto_alpha", "")
        layer = int(md["layer"])
        n_total_layers = int(md["n_total_layers"])
        # hook_name was added in schema v3; for older artifacts derive it from (layer, hook_site).
        hook_name = md.get("hook_name") or _hook_name(
            layer, md["hook_site"], n_layers=n_total_layers
        )
        return cls(
            directions=directions,
            bias=json.loads(md.get("bias_json", "{}")),
            layer=layer,
            metrics=json.loads(md.get("metrics_json", "{}")),
            model_id=md["model_id"],
            hook_site=md["hook_site"],
            hook_name=hook_name,
            n_total_layers=n_total_layers,
            default_method=md.get("default_method", "logistic"),
            auto_alpha=float(auto_alpha_str) if auto_alpha_str else None,
            schema_version=int(md.get("schema_version", SCHEMA_VERSION)),
            created_at=md.get("created_at", ""),
            extras=json.loads(md.get("extras_json", "{}")),
        )

    def get_direction(self, method: str | None = None) -> torch.Tensor:
        """Return the unit direction for the given method (defaults to default_method)."""
        m = method or self.default_method
        if m not in self.directions:
            raise KeyError(f"no direction for method {m!r}; available: {list(self.directions)}")
        return self.directions[m]

    @torch.no_grad()
    def score_tokens(
        self,
        model: ModelHandle,
        prompt: str,
        response: str | None = None,
        *,
        method: str | None = None,
        include_prompt: bool = False,
    ) -> TokenScores:
        """Project every token's residual-stream activation at this probe's layer
        onto this probe's direction. Returns a `TokenScores` with token strings
        aligned to scalar scores, one per position.

        This is the interpretability complement to `steer()`: where `steer()` *uses*
        the direction to push generation, `score_tokens()` *measures* where in a
        sequence the direction is most active. Useful for asking questions like
        "which tokens in this refusal actually carry the refusal signal?" or
        "did my steering hook push activations the way I expected?".

        Args:
            model: a loaded ModelHandle (any model — the probe's model_id is checked
                only at steering time, not here).
            prompt: the user prompt to score (or pre-pend to `response`).
            response: optional assistant response to score. Default `None` scores
                the prompt-only formatting.
            method: which probe-family direction to project onto. Defaults to
                `default_method`.
            include_prompt: when True, scores are returned for every token in the
                full chat-formatted sequence, with `response_start` indicating
                where the response begins. When False (default) and `response`
                is supplied, prompt tokens are sliced off so you get only the
                response-side scores.

        Returns:
            TokenScores(tokens, scores, layer, method, response_start). Call
            `.plot()` for a heatmap-style visualization.

        Note: scores are raw `direction · activation` projections — the sign and
        magnitude are interpretable *within* one sequence (which token fires hardest)
        but not across sequences without calibration. The logistic-method bias is
        omitted because it shifts all positions equally and does not change the
        relative ranking that this view is for.
        """
        direction = self.get_direction(method).to(model.device).to(model.hooked.cfg.dtype)
        chosen_method = method or self.default_method

        if response is None:
            full_ids = model.format_chat(prompt)
            prompt_len = full_ids.shape[-1]
        else:
            full_ids = model.format_chat(prompt, response)
            prompt_only = model.format_chat(prompt)
            prompt_len = min(prompt_only.shape[-1], full_ids.shape[-1])

        _, cache = model.hooked.run_with_cache(
            full_ids, names_filter=[self.hook_name]
        )
        acts = cache[self.hook_name][0].to(direction.dtype)  # [seq_len, d_model]
        scores = (acts @ direction).float().cpu()  # [seq_len]

        tokenizer = model.tokenizer
        assert tokenizer is not None
        flat_ids = full_ids.squeeze(0).tolist()
        token_strs = [str(tokenizer.decode([tid])) for tid in flat_ids]

        if response is not None and not include_prompt:
            return TokenScores(
                tokens=token_strs[prompt_len:],
                scores=scores[prompt_len:].clone(),
                layer=self.layer,
                method=chosen_method,
                response_start=0,
            )
        return TokenScores(
            tokens=token_strs,
            scores=scores,
            layer=self.layer,
            method=chosen_method,
            response_start=prompt_len if response is not None else 0,
        )

    @torch.no_grad()
    def steer(
        self,
        model: ModelHandle,
        prompt: str,
        alpha: float | None = None,
        *,
        method: str | None = None,
        op: str = "addition",
        target: float | None = None,
        gamma: float | None = None,
        max_new_tokens: int = 60,
        temperature: float = 0.0,
    ) -> str:
        """Generate a steered completion. `op` selects one of the four interventions:

          addition (default) — `act ← act + α·v`. `alpha=None` uses the calibrated
              `auto_alpha`, else 2.0.
          projection — `act ← act − (act·v̂)v̂`. Ablates the concept component;
              `alpha`/`target`/`gamma` are ignored.
          clamp — `act ← act + (target − act·v̂)v̂`. Requires `target`. Forces
              the concept's projection to a fixed value.
          multiplicative — `act ← act + (γ−1)(act·v̂)v̂`. Requires `gamma`. Scales
              the existing component along the direction.

        Pass `method` to override `default_method` (which probe-family direction).
        """
        from typing import cast

        from .intervention import OPERATIONS, Operation, make_hook

        if op not in OPERATIONS:
            raise ValueError(f"unknown op {op!r}; choose one of {OPERATIONS}")
        op_typed = cast(Operation, op)
        if model.model_id != self.model_id:
            print(
                f"[steerkit] warning: probe was trained on {self.model_id} but steering "
                f"on {model.model_id}. Direction may not transfer cleanly."
            )

        if op == "addition" and alpha is None:
            alpha = self.auto_alpha if self.auto_alpha is not None else 2.0

        direction = self.get_direction(method).to(model.device).to(model.hooked.cfg.dtype)
        hook_fn = make_hook(op_typed, direction, alpha=alpha, target=target, gamma=gamma)

        tokens = model.format_chat(prompt)
        tokenizer = model.tokenizer
        assert tokenizer is not None
        with model.hooked.hooks(fwd_hooks=[(self.hook_name, hook_fn)]):
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
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return str(decoded)

    def ablate(self, model: ModelHandle, prompt: str, **kwargs) -> str:
        """Convenience: steer with op='projection' (remove the concept component)."""
        return self.steer(model, prompt, op="projection", **kwargs)

    def clamp(self, model: ModelHandle, prompt: str, target: float, **kwargs) -> str:
        """Convenience: steer with op='clamp' at the given target projection value."""
        return self.steer(model, prompt, op="clamp", target=target, **kwargs)

    def amplify(self, model: ModelHandle, prompt: str, gamma: float, **kwargs) -> str:
        """Convenience: steer with op='multiplicative' at the given gamma scaling factor."""
        return self.steer(model, prompt, op="multiplicative", gamma=gamma, **kwargs)

    def plot_logit_lens(self, model: ModelHandle, **kwargs):
        """Convenience: render the steering direction projected through the unembed."""
        from .viz import plot_logit_lens

        return plot_logit_lens(self, model, **kwargs)

    @torch.no_grad()
    def predict_at_mask(
        self,
        model: ModelHandle,
        text: str,
        *,
        top_k: int = 10,
        alpha: float | None = None,
        method: str | None = None,
        op: str = "addition",
        target: float | None = None,
        gamma: float | None = None,
    ) -> dict[int, list[tuple[str, float]]]:
        """Run a *single* forward pass with the steering hook on, then read the
        top-K vocabulary predictions at every `[MASK]` token in `text`.

        This is the encoder analog of `Probe.steer(...)` — encoder models
        don't autoregressively generate, but they expose token-level logits at
        each position. For a sentence like ``"I think this movie is [MASK]."``,
        this returns the top-K most-probable fillers for the mask, with the
        steering direction applied to the residual stream at the probe's layer.

        Pass `alpha=0.0` for the unsteered baseline (the hook is still
        installed but contributes nothing) — the typical use is to call this
        once with `alpha=0.0` and once at the calibrated `auto_alpha` and
        compare the resulting top-K distributions side-by-side.

        Returns a `{mask_position: [(token_string, probability), ...]}` dict.
        Raises `ValueError` if the input contains no `[MASK]` tokens.
        """
        from typing import cast

        from .intervention import OPERATIONS, Operation, make_hook

        tokenizer = model.tokenizer
        assert tokenizer is not None, "model has no tokenizer attached"
        mask_id = getattr(tokenizer, "mask_token_id", None)
        if mask_id is None:
            raise ValueError(
                "tokenizer has no mask_token_id — predict_at_mask is only meaningful "
                "for masked-LM tokenizers (BERT / RoBERTa / DeBERTa / ...)."
            )

        if op not in OPERATIONS:
            raise ValueError(f"unknown op {op!r}; choose one of {OPERATIONS}")
        op_typed = cast(Operation, op)
        if op == "addition" and alpha is None:
            alpha = self.auto_alpha if self.auto_alpha is not None else 2.0

        ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        mask_positions = (ids[0] == mask_id).nonzero(as_tuple=True)[0].tolist()
        if not mask_positions:
            raise ValueError(
                f"no [MASK] tokens in input; tokenizer mask token is "
                f"{tokenizer.mask_token!r} (id={mask_id}). Got: {text!r}"
            )

        direction = self.get_direction(method).to(model.device).to(model.hooked.cfg.dtype)
        hook_fn = make_hook(op_typed, direction, alpha=alpha, target=target, gamma=gamma)
        with model.hooked.hooks(fwd_hooks=[(self.hook_name, hook_fn)]):
            logits = model.hooked(ids)
        # Some HookedEncoder configs return logits directly; others may return
        # a richer object — coerce to a tensor.
        if hasattr(logits, "logits"):
            logits = logits.logits

        out: dict[int, list[tuple[str, float]]] = {}
        for pos in mask_positions:
            probs = torch.softmax(logits[0, pos, :].float(), dim=-1)
            top_probs, top_ids = probs.topk(top_k)
            decoded = [str(tokenizer.decode([int(i.item())])) for i in top_ids]
            out[int(pos)] = list(zip(decoded, [float(p) for p in top_probs.tolist()], strict=True))
        return out

    def export_gguf(
        self,
        path: str | Path,
        *,
        method: str | None = None,
        scale: float = 1.0,
    ) -> Path:
        """Convenience: export this single-layer Probe to llama.cpp gguf format."""
        from .gguf_export import export_probe_to_gguf

        return export_probe_to_gguf(self, path, method=method, scale=scale)

    def report(
        self,
        *,
        model: ModelHandle | None = None,
        activations: torch.Tensor | None = None,
        per_layer: dict[int, Probe] | None = None,
        out: str | Path | None = None,
        title: str | None = None,
    ) -> str:
        """Render a one-page HTML report. Returns the HTML string; if `out` is set,
        also writes it to disk and returns the path-as-string."""
        from .report import render_probe_report, write_report

        html = render_probe_report(
            self,
            model=model,
            activations=activations,
            per_layer=per_layer,
            title=title,
        )
        if out is not None:
            return str(write_report(html, out))
        return html

MULTINOMIAL_SCHEMA_VERSION = 1


@dataclass
class MultinomialProbe:
    """Multi-class linear classifier across the concepts of a `mutually_exclusive` ConceptGroup.

    Diagnostic, not for steering: useful for "which concept is this activation expressing?"
    and for cross-concept similarity heatmaps (rows of `weights` are direction vectors,
    one per concept). Steering is still done with the per-concept binary `Probe`s.
    """

    weights: torch.Tensor  # [n_classes, d_model]
    biases: torch.Tensor  # [n_classes]
    class_names: list[str]
    layer: int
    hook_name: str
    model_id: str
    hook_site: str
    n_total_layers: int
    metrics: dict[str, float] = field(default_factory=dict)
    schema_version: int = MULTINOMIAL_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def normalized_depth(self) -> float:
        return (self.layer + 1) / (self.n_total_layers + 1)

    @classmethod
    def fit_at_layer(
        cls,
        activations_by_concept: dict[str, dict[int, torch.Tensor]],
        layer: int,
        model: ModelHandle,
        *,
        hook_site: str = "resid_post",
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> MultinomialProbe:
        """Fit a multinomial probe at one chosen layer, using each concept's positive
        activations as its class. Returns a single MultinomialProbe with held-out accuracy.
        """
        class_names = list(activations_by_concept.keys())
        if len(class_names) < 2:
            raise ValueError("multinomial probe requires at least 2 classes")

        rng = np.random.default_rng(seed)
        Xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        Xs_test: list[np.ndarray] = []
        ys_test: list[np.ndarray] = []

        for cls_idx, name in enumerate(class_names):
            acts = activations_by_concept[name][layer]  # [n_pairs, 2, d_model]
            n_pairs = acts.shape[0]
            perm = rng.permutation(n_pairs)
            n_test = int(round(n_pairs * test_fraction)) if test_fraction > 0 else 0
            n_test = min(max(n_test, 0), n_pairs - 1) if test_fraction > 0 else 0
            train_idx, test_idx = perm[n_test:], perm[:n_test]
            pos_train = acts[train_idx, 0, :].numpy()
            Xs.append(pos_train)
            ys.append(np.full(len(pos_train), cls_idx))
            if len(test_idx) > 0:
                pos_test = acts[test_idx, 0, :].numpy()
                Xs_test.append(pos_test)
                ys_test.append(np.full(len(pos_test), cls_idx))

        X_train = np.concatenate(Xs, axis=0)
        y_train = np.concatenate(ys, axis=0)
        # multi_class is deprecated in sklearn ≥1.5; lbfgs auto-handles multinomial
        # when y has >2 unique labels.
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, C=1.0)
        clf.fit(X_train, y_train)
        train_acc = float(clf.score(X_train, y_train))
        metrics = {"accuracy_train": train_acc}
        if Xs_test:
            X_test = np.concatenate(Xs_test, axis=0)
            y_test = np.concatenate(ys_test, axis=0)
            metrics["accuracy_test"] = float(clf.score(X_test, y_test))

        weights = torch.tensor(clf.coef_, dtype=torch.float32)  # [n_classes, d_model]
        biases = torch.tensor(clf.intercept_, dtype=torch.float32)  # [n_classes]

        return cls(
            weights=weights,
            biases=biases,
            class_names=class_names,
            layer=layer,
            hook_name=_hook_name(layer, hook_site, n_layers=model.n_layers),
            model_id=model.model_id,
            hook_site=hook_site,
            n_total_layers=model.n_layers,
            metrics=metrics,
        )

    @classmethod
    def fit_best_layer(
        cls,
        activations_by_concept: dict[str, dict[int, torch.Tensor]],
        model: ModelHandle,
        *,
        hook_site: str = "resid_post",
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> MultinomialProbe:
        """Fit a multinomial probe at every layer; return the one with best test accuracy
        (or train accuracy if no test split was kept)."""
        any_concept = next(iter(activations_by_concept.values()))
        layer_indices = sorted(any_concept.keys())
        best: MultinomialProbe | None = None
        best_score = -1.0
        for layer in layer_indices:
            probe = cls.fit_at_layer(
                activations_by_concept,
                layer,
                model,
                hook_site=hook_site,
                test_fraction=test_fraction,
                seed=seed,
            )
            score = probe.metrics.get("accuracy_test", probe.metrics.get("accuracy_train", 0.0))
            if score > best_score:
                best_score = score
                best = probe
        assert best is not None
        return best

    def similarity_matrix(self) -> torch.Tensor:
        """Cosine-similarity matrix between class direction vectors (rows of `weights`).
        Returns a [n_classes, n_classes] tensor."""
        w = self.weights
        norm = w / (w.norm(dim=-1, keepdim=True) + 1e-12)
        return norm @ norm.T

    def plot_similarity(self, **kwargs):
        """Convenience: render the cross-class similarity heatmap."""
        from .viz import plot_similarity_heatmap

        return plot_similarity_heatmap(self, **kwargs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {"weights": self.weights.contiguous(), "biases": self.biases.contiguous()}
        metadata = {
            "schema_version": str(self.schema_version),
            "kind": "multinomial",
            "layer": str(self.layer),
            "hook_name": self.hook_name,
            "n_total_layers": str(self.n_total_layers),
            "model_id": self.model_id,
            "hook_site": self.hook_site,
            "class_names_json": json.dumps(self.class_names),
            "metrics_json": json.dumps(self.metrics),
            "created_at": self.created_at,
        }
        save_file(tensors, str(path), metadata=metadata)

    @classmethod
    def load(cls, path: str | Path) -> MultinomialProbe:
        path = Path(path)
        with safe_open(str(path), framework="pt") as f:
            weights = f.get_tensor("weights")
            biases = f.get_tensor("biases")
            md = f.metadata() or {}
        layer = int(md["layer"])
        n_total_layers = int(md["n_total_layers"])
        hook_name = md.get("hook_name") or _hook_name(
            layer, md["hook_site"], n_layers=n_total_layers
        )
        return cls(
            weights=weights,
            biases=biases,
            class_names=json.loads(md["class_names_json"]),
            layer=layer,
            hook_name=hook_name,
            model_id=md["model_id"],
            hook_site=md["hook_site"],
            n_total_layers=n_total_layers,
            metrics=json.loads(md.get("metrics_json", "{}")),
            schema_version=int(md.get("schema_version", MULTINOMIAL_SCHEMA_VERSION)),
            created_at=md.get("created_at", ""),
        )
