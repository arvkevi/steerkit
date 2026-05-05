__version__ = "0.3.0"

from .cache import hash_pairs, load_activations_zarr, save_activations_zarr
from .calibrate import (
    DEFAULT_ALPHA_CANDIDATES,
    DEFAULT_CALIBRATION_PROMPTS,
    calibrate_alpha,
)
from .concepts import Concept, ConceptGroup, singleton_group
from .data import ContrastPair, load_pairs_jsonl, save_pairs_jsonl
from .eval import (
    DEFAULT_EVAL_PROMPTS,
    EvalReport,
    evaluate_probe,
    evaluate_steering_effect,
    external_classifier_score,
    logit_lens_vocab_score,
    perplexity_ratio,
)
from .extract import extract_activations, extract_group_activations
from .generate import (
    DEFAULT_SEED_PROMPTS,
    GenerationStats,
    generate_pairs_for_concept,
    generate_pairs_for_group,
)
from .gguf_export import export_composite_to_gguf, export_probe_to_gguf
from .intervention import (
    OPERATIONS,
    apply_addition,
    apply_clamp,
    apply_multiplicative,
    apply_projection,
)
from .lint import LintFinding, LintReport, lint_group, lint_pairs
from .models import ModelHandle, load
from .probe import PROBE_METHODS, MultinomialProbe, Probe, TokenScores
from .sweep import CompositeProbe, GroupFit, compose, sweep, window
from .teacher import APITeacher, LocalHFTeacher, TeacherModel, make_teacher
from .viz import (
    plot_activation_projection,
    plot_alpha_curve,
    plot_cross_model_overlay,
    plot_layer_selection,
    plot_logit_lens,
    plot_similarity_heatmap,
    plot_token_scores,
)

__all__ = [
    "DEFAULT_ALPHA_CANDIDATES",
    "DEFAULT_CALIBRATION_PROMPTS",
    "DEFAULT_EVAL_PROMPTS",
    "DEFAULT_SEED_PROMPTS",
    "EvalReport",
    "LintFinding",
    "LintReport",
    "OPERATIONS",
    "PROBE_METHODS",
    "APITeacher",
    "apply_addition",
    "apply_clamp",
    "apply_multiplicative",
    "apply_projection",
    "CompositeProbe",
    "Concept",
    "ConceptGroup",
    "ContrastPair",
    "GenerationStats",
    "GroupFit",
    "LocalHFTeacher",
    "ModelHandle",
    "MultinomialProbe",
    "Probe",
    "TeacherModel",
    "TokenScores",
    "calibrate_alpha",
    "compose",
    "evaluate_probe",
    "evaluate_steering_effect",
    "external_classifier_score",
    "logit_lens_vocab_score",
    "perplexity_ratio",
    "export_composite_to_gguf",
    "export_probe_to_gguf",
    "extract_activations",
    "extract_group_activations",
    "generate_pairs_for_concept",
    "generate_pairs_for_group",
    "hash_pairs",
    "lint_group",
    "lint_pairs",
    "load",
    "load_activations_zarr",
    "load_pairs_jsonl",
    "make_teacher",
    "plot_activation_projection",
    "plot_alpha_curve",
    "plot_cross_model_overlay",
    "plot_layer_selection",
    "plot_logit_lens",
    "plot_similarity_heatmap",
    "plot_token_scores",
    "save_activations_zarr",
    "save_pairs_jsonl",
    "singleton_group",
    "sweep",
    "window",
]
