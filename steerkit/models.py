from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ModelHandle:
    """Thin wrapper over a TransformerLens HookedTransformer carrying tokenizer + config conveniences."""

    def __init__(self, hooked: HookedTransformer, model_id: str):
        self.hooked = hooked
        self.model_id = model_id
        self.tokenizer = hooked.tokenizer

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


def load(model_id: str, device: str | None = None, dtype: torch.dtype | None = None) -> ModelHandle:
    """Load a model into a ModelHandle. Defaults to MPS on Apple Silicon, CUDA otherwise, else CPU."""
    if device is None:
        device = _default_device()
    if dtype is None:
        # MPS works best with float32 for now; CUDA can use float16.
        dtype = torch.float32 if device == "mps" else torch.float16
    hooked = HookedTransformer.from_pretrained(
        model_id,
        device=device,
        dtype=dtype,
    )
    hooked.eval()
    return ModelHandle(hooked, model_id=model_id)
