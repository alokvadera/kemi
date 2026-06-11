"""LLM-powered abstractive summarization for memory consolidation.

Provides a pluggable :class:`LLMSummarizer` that can use OpenAI, Anthropic,
Ollama (via OpenAI-compatible API), or a custom callable to generate concise
abstractive summaries from a list of related memory texts.

Typical usage::

    summarizer = LLMSummarizer(provider="openai", model="gpt-4o-mini")
    summary = summarizer.summarize([
        "User loves hiking in the Alps",
        "User visited Switzerland last summer",
        "User plans to hike Mont Blanc next year",
    ])
    # -> "User enjoys alpine hiking, has visited Switzerland, and plans to hike Mont Blanc."
"""

import logging
import os
from collections.abc import Callable
from typing import Any

from kemi.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_TEMPLATE = (
    "Summarize these related memories into a concise, integrated statement: {memories}"
)

# Type alias for custom callables: (prompt: str, **kwargs) -> str
LLMCallback = Callable[..., str]


class LLMSummarizer:
    """Generate abstractive summaries of memory groups using an LLM.

    Supports five modes via the *provider* argument:

    ``"openai"``
        Uses ``openai.OpenAI``.  The ``model`` defaults to ``gpt-4o-mini``.
        Reads ``OPENAI_API_KEY`` from the environment.

    ``"anthropic"``
        Uses ``anthropic.Anthropic``.  The ``model`` defaults to
        ``claude-3-haiku-20240307``.  Reads ``ANTHROPIC_API_KEY`` from the
        environment.

    ``"ollama"``
        Uses the OpenAI-compatible endpoint at ``ollama_base_url``
        (default ``http://localhost:11434/v1``).  The ``model`` defaults to
        ``llama3.2``.

    ``"tokenlb"``
        Uses the Tokenlb API gateway at ``tokenlb_base_url``
        (default ``https://tokenlb.net/v1``).  The ``model`` defaults to
        ``gpt-5.4-mini``.  Reads ``TOKENLB_API_KEY`` from the environment.
        Note: Tokenlb does NOT support embedding models, only chat completions.

    ``"custom"``
        Uses the provided ``custom_callback`` callable directly.
        ``custom_callback(prompt, **kwargs)`` must return the summary text.

    Args:
        provider: One of ``"openai"``, ``"anthropic"``, ``"ollama"``,
            ``"tokenlb"``, or ``"custom"``.
        model: Model name override.  If omitted, a sensible default is used
            per provider.
        api_key: API key override.  Falls back to the standard env var when
            not provided.
        ollama_base_url: Base URL for Ollama's OpenAI-compatible endpoint.
        tokenlb_base_url: Base URL for Tokenlb's OpenAI-compatible endpoint.
        custom_callback: Callable used when ``provider="custom"``.
        prompt_template: Template string with a ``{memories}`` placeholder
            (one memory per line).  Defaults to :data:`_DEFAULT_PROMPT_TEMPLATE`.
    """

    DEFAULT_TOKENLB_MODEL: str = "gpt-5.4-mini"

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        ollama_base_url: str | None = None,
        tokenlb_base_url: str | None = None,
        custom_callback: LLMCallback | None = None,
        prompt_template: str | None = None,
    ) -> None:
        self._provider = provider.lower()
        self._model = model
        self._api_key = api_key
        self._ollama_base_url = ollama_base_url or "http://localhost:11434/v1"
        # Tokenlb settings
        self._tokenlb_base_url = (
            tokenlb_base_url or os.environ.get("TOKENLB_BASE_URL") or "https://tokenlb.net/v1"
        )
        self._tokenlb_api_key = os.environ.get("TOKENLB_API_KEY")
        self._custom_callback = custom_callback
        self._prompt_template = prompt_template or _DEFAULT_PROMPT_TEMPLATE

        self._client: Any = None
        self._effective_model: str | None = None
        self._init_client()

    def _init_client(self) -> None:
        """Lazily initialise the LLM client based on *provider*."""
        if self._provider == "custom":
            if self._custom_callback is None:
                raise ValidationError("custom_callback is required when provider='custom'")
            return

        if self._provider in ("openai", "ollama", "tokenlb"):
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ConfigurationError(
                    "openai package is required for provider='openai', "
                    "'ollama', or 'tokenlb'. Install with: pip install openai"
                ) from exc

            kwargs: dict[str, Any] = {}
            if self._provider == "ollama":
                kwargs["base_url"] = self._ollama_base_url
                kwargs["api_key"] = self._api_key or "ollama"
            elif self._provider == "tokenlb":
                kwargs["base_url"] = self._tokenlb_base_url
                kwargs["api_key"] = self._api_key or self._tokenlb_api_key
            else:
                kwargs["api_key"] = self._api_key
            self._client = OpenAI(**kwargs)

            if self._provider == "openai":
                self._effective_model = self._model or "gpt-4o-mini"
            elif self._provider == "ollama":
                self._effective_model = self._model or "llama3.2"
            else:  # tokenlb
                self._effective_model = self._model or self.DEFAULT_TOKENLB_MODEL

        elif self._provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ConfigurationError(
                    "anthropic package is required for provider='anthropic'. "
                    "Install with: pip install anthropic"
                ) from exc

            self._client = Anthropic(api_key=self._api_key)
            self._effective_model = self._model or "claude-3-haiku-20240307"

        else:
            raise ConfigurationError(
                f"Unknown provider: {self._provider}. "
                "Expected one of: openai, anthropic, ollama, tokenlb, custom"
            )

    def summarize(
        self,
        memories: list[str],
        **kwargs: Any,
    ) -> str:
        """Generate an abstractive summary from a list of memory texts.

        Args:
            memories: List of memory content strings to summarize.
            **kwargs: Additional keyword arguments passed through to the
                underlying LLM call (e.g. ``temperature=0.3``,
                ``max_tokens=200``).

        Returns:
            The generated summary text, or an empty string on failure.
        """
        if not memories:
            return ""

        # Format the prompt
        memories_text = "\n".join(f"- {m}" for m in memories)
        prompt = self._prompt_template.format(memories=memories_text)

        try:
            if self._provider == "custom":
                if self._custom_callback is None:
                    return ""
                return self._custom_callback(prompt, **kwargs)

            if self._provider in ("openai", "ollama", "tokenlb"):
                if self._client is None:
                    return ""
                response = self._client.chat.completions.create(
                    model=self._effective_model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    **kwargs,
                )
                return response.choices[0].message.content or ""

            if self._provider == "anthropic":
                if self._client is None:
                    return ""
                response = self._client.messages.create(
                    model=self._effective_model or "claude-3-haiku-20240307",
                    max_tokens=kwargs.pop("max_tokens", 300),
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    **kwargs,
                )
                return response.content[0].text if response.content else ""

        except Exception:
            logger.warning(
                "LLM summarization failed, falling back to extractive summary",
                exc_info=True,
            )

        return ""
