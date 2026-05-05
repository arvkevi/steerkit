"""Teacher-model abstraction for synthetic dataset generation.

Two implementations: `APITeacher` (Anthropic / OpenAI via env keys) and `LocalHFTeacher`
(any HF causal-LM the user can load). The user supplies the teacher; steerkit never
embeds API keys or downloads paid-provider weights.

Use `make_teacher(spec)` to parse a string spec like:
  "anthropic:claude-opus-4-7"
  "openai:gpt-4o-2024-11-20"
  "local:HuggingFaceTB/SmolLM2-1.7B-Instruct"

The Anthropic path applies prompt caching to the system prompt: every concept-pair
generation reuses the same system prompt, so caching cuts cost and latency
significantly when generating large datasets.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


class TeacherModel(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Return a single completion string for the given system + user messages."""

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Stable identifier for the teacher (e.g. 'anthropic:claude-opus-4-7'), used in metadata."""


class APITeacher(TeacherModel):
    SUPPORTED_PROVIDERS = ("anthropic", "openai")

    def __init__(self, provider: str, model: str):
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r}; supported: {self.SUPPORTED_PROVIDERS}"
            )
        self.provider = provider
        self.model = model
        self._client: Any = None

    @property
    def identifier(self) -> str:
        return f"{self.provider}:{self.model}"

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "Anthropic teacher requires the `anthropic` package. "
                    "Install with `uv pip install anthropic` or "
                    "`pip install steerkit[anthropic]`."
                ) from e
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. steerkit never embeds API keys; "
                    "set it in your environment to use the Anthropic teacher."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "openai":
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "OpenAI teacher requires the `openai` package. "
                    "Install with `uv pip install openai` or `pip install steerkit[openai]`."
                ) from e
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. steerkit never embeds API keys; "
                    "set it in your environment to use the OpenAI teacher."
                )
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        client = self._ensure_client()
        if self.provider == "anthropic":
            # Apply prompt caching to the system prompt — every pair generation reuses it.
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
            )
            # Concatenate all text blocks (most replies are a single block).
            return "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
        # openai
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class LocalHFTeacher(TeacherModel):
    """Teacher using a locally loaded HF causal-LM.

    Loads the model lazily (only on first .complete() call) so that constructing the
    object is cheap, and so a user can hand a `LocalHFTeacher` to code that may end up
    not actually generating (e.g. tests that mock it).
    """

    def __init__(self, model_id: str, device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def identifier(self) -> str:
        return f"local:{self.model_id}"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float32 if device in ("mps", "cpu") else torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(device)  # type: ignore[arg-type]  # transformers .to accepts str device strings at runtime
        self._model.eval()
        self.device = device

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        import torch

        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None
        if getattr(self._tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            text = f"{system}\n\n{user}\n"
            ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            out = self._model.generate(
                ids,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-5),
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0, ids.shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def make_teacher(spec: str) -> TeacherModel:
    """Parse a teacher spec string and return a TeacherModel.

    Supported forms:
      'anthropic:<model>'   e.g. 'anthropic:claude-opus-4-7'
      'openai:<model>'      e.g. 'openai:gpt-4o-2024-11-20'
      'local:<hf-id>'       e.g. 'local:HuggingFaceTB/SmolLM2-1.7B-Instruct'
    """
    if ":" not in spec:
        raise ValueError(f"teacher spec must be 'provider:model', got {spec!r}")
    provider, model = spec.split(":", 1)
    provider = provider.lower()
    if provider == "local":
        return LocalHFTeacher(model)
    if provider in APITeacher.SUPPORTED_PROVIDERS:
        return APITeacher(provider, model)
    raise ValueError(
        f"unknown teacher provider {provider!r}; supported: 'anthropic', 'openai', 'local'"
    )
