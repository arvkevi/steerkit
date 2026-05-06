from __future__ import annotations

import torch
from transformer_lens import HookedEncoder, HookedTransformer

# Substrings in model ids that signal an encoder-only architecture. TL routes
# these through HookedEncoder; everything else goes through HookedTransformer.
_ENCODER_PATTERNS = ("bert", "roberta", "deberta", "electra", "albert")


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _looks_like_encoder(model_id: str) -> bool:
    name = model_id.lower()
    return any(p in name for p in _ENCODER_PATTERNS)


class ModelHandle:
    """Thin wrapper over a TransformerLens HookedTransformer / HookedEncoder
    carrying tokenizer + config conveniences. Both backends share the
    `run_with_cache` and `hooks(...)` interfaces this library uses.
    """

    def __init__(self, hooked, model_id: str):
        self.hooked = hooked
        self.model_id = model_id
        self.tokenizer = hooked.tokenizer

    @property
    def is_encoder(self) -> bool:
        """True if backed by `HookedEncoder` (BERT-style; bidirectional, no
        autoregressive `generate`). Use `pooling="mean"` for these and the
        encoder-side prediction APIs (`probe.predict_at_mask`).
        """
        return isinstance(self.hooked, HookedEncoder)

    @property
    def n_layers(self) -> int:
        return self.hooked.cfg.n_layers

    @property
    def d_model(self) -> int:
        return self.hooked.cfg.d_model

    @property
    def device(self) -> str:
        return str(self.hooked.cfg.device)

    def format_chat(self, prompt: str, response: str | None = None) -> torch.Tensor:
        """Format the prompt (and optional response) as a token tensor on the model's device.

        Uses the tokenizer's chat template when available; otherwise concatenates plain text.
        Returns a 2-D tensor of token ids (`[1, seq_len]`).

        Note: some tokenizers (Qwen2/Qwen3) return a `BatchEncoding` from
        `apply_chat_template` even with `return_tensors="pt"`; we unwrap to `input_ids`.
        """
        tokenizer = self.tokenizer
        assert tokenizer is not None, "model has no tokenizer attached"
        has_template = getattr(tokenizer, "chat_template", None)
        if has_template:
            messages = [{"role": "user", "content": prompt}]
            if response is None:
                ids = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                )
            else:
                messages.append({"role": "assistant", "content": response})
                ids = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=False
                )
        else:
            # Plain-text fallback for non-chat-tuned models.
            text = prompt if response is None else f"{prompt.rstrip()}\n{response}"
            ids = tokenizer(text, return_tensors="pt").input_ids
        # Some tokenizers return a BatchEncoding (e.g. Qwen2/Qwen3) instead of a raw
        # Tensor — unwrap to the input_ids tensor in that case.
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        assert isinstance(ids, torch.Tensor), f"expected Tensor token ids, got {type(ids).__name__}"
        return ids.to(self.device)


def load(
    model_id: str,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    *,
    encoder: bool | None = None,
) -> ModelHandle:
    """Load a model into a ModelHandle.

    `encoder=None` (default) auto-detects: model ids containing 'bert',
    'roberta', 'deberta', 'electra', or 'albert' are loaded via
    `HookedEncoder`; everything else goes through `HookedTransformer`.
    Pass `encoder=True` / `encoder=False` to override.

    Devices: MPS on Apple Silicon, CUDA otherwise, else CPU.
    """
    if device is None:
        device = _default_device()
    if dtype is None:
        # MPS works best with float32 for now; CUDA can use float16.
        dtype = torch.float32 if device == "mps" else torch.float16
    if encoder is None:
        encoder = _looks_like_encoder(model_id)
    hooked: HookedEncoder | HookedTransformer
    if encoder:
        # HookedEncoder doesn't accept `dtype=` in from_pretrained; it loads
        # at the model's native dtype and follows the device kwarg.
        hooked = HookedEncoder.from_pretrained(model_id, device=device)
    else:
        hooked = HookedTransformer.from_pretrained(
            model_id,
            device=device,
            dtype=dtype,
        )
    hooked.eval()
    return ModelHandle(hooked, model_id=model_id)
