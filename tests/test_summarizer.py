"""Tests for src/kemi/summarizer.py"""

import sys
import types
from datetime import datetime, timezone
from typing import Any

import pytest

from kemi.models import MemoryObject, MemoryType
from kemi.summarizer import LLMSummarizer


class TestLLMSummarizerInit:
    def test_custom_provider(self) -> None:
        """Test that custom provider accepts a callable."""
        def mock_callback(prompt: str, **kwargs: object) -> str:
            return "custom summary"
        summarizer = LLMSummarizer(provider="custom", custom_callback=mock_callback)
        assert summarizer._provider == "custom"
        assert summarizer._custom_callback is not None

    def test_custom_provider_missing_callback(self) -> None:
        """Test that custom provider raises ValueError without callback."""
        with pytest.raises(ValueError, match="custom_callback is required"):
            LLMSummarizer(provider="custom")

    def test_invalid_provider(self) -> None:
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMSummarizer(provider="nonexistent")

    def test_default_prompt_template(self) -> None:
        """Test that default prompt template has {memories} placeholder."""
        summarizer = LLMSummarizer(provider="custom", custom_callback=lambda p, **kw: p)
        assert "{memories}" in summarizer._prompt_template

    def test_custom_prompt_template(self) -> None:
        """Test custom prompt template."""
        template = "Custom: {memories}"
        summarizer = LLMSummarizer(
            provider="custom",
            custom_callback=lambda p, **kw: p,
            prompt_template=template,
        )
        assert summarizer._prompt_template == template


class TestLLMSummarizerInitClient:
    """Tests for _init_client provider-specific branches."""

    def test_openai_provider_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _init_client with openai provider."""
        mock_client = types.SimpleNamespace()
        mock_module = types.SimpleNamespace(OpenAI=lambda **kw: mock_client)
        monkeypatch.setitem(sys.modules, "openai", mock_module)

        summarizer = LLMSummarizer(provider="openai", api_key="test-key")
        assert summarizer._provider == "openai"
        assert summarizer._effective_model == "gpt-4o-mini"

    def test_ollama_provider_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _init_client with ollama provider."""
        captured_kwargs: dict = {}

        def mock_openai(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return types.SimpleNamespace()

        mock_module = types.SimpleNamespace(OpenAI=mock_openai)
        monkeypatch.setitem(sys.modules, "openai", mock_module)

        summarizer = LLMSummarizer(provider="ollama", ollama_base_url="http://test:11434/v1")
        assert summarizer._provider == "ollama"
        assert summarizer._effective_model == "llama3.2"
        assert captured_kwargs.get("base_url") == "http://test:11434/v1"
        assert captured_kwargs.get("api_key") == "ollama"

    def test_ollama_provider_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ollama provider respects user-provided api_key."""
        captured_kwargs: dict = {}

        def mock_openai(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return types.SimpleNamespace()

        mock_module = types.SimpleNamespace(OpenAI=mock_openai)
        monkeypatch.setitem(sys.modules, "openai", mock_module)

        summarizer = LLMSummarizer(provider="ollama", api_key="my-key")
        assert captured_kwargs.get("api_key") == "my-key"

    def test_openai_provider_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test openai provider respects model override."""
        mock_module = types.SimpleNamespace(OpenAI=lambda **kw: types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, "openai", mock_module)

        summarizer = LLMSummarizer(provider="openai", model="gpt-4o")
        assert summarizer._effective_model == "gpt-4o"

    def test_anthropic_provider_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _init_client with anthropic provider."""
        captured_kwargs: dict = {}

        def mock_anthropic(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return types.SimpleNamespace()

        mock_module = types.SimpleNamespace(Anthropic=mock_anthropic)
        monkeypatch.setitem(sys.modules, "anthropic", mock_module)

        summarizer = LLMSummarizer(provider="anthropic", api_key="test-key")
        assert summarizer._provider == "anthropic"
        assert summarizer._effective_model == "claude-3-haiku-20240307"
        assert captured_kwargs.get("api_key") == "test-key"

    def test_anthropic_provider_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test anthropic provider respects model override."""
        mock_module = types.SimpleNamespace(Anthropic=lambda **kw: types.SimpleNamespace())
        monkeypatch.setitem(sys.modules, "anthropic", mock_module)

        summarizer = LLMSummarizer(provider="anthropic", model="claude-3-opus")
        assert summarizer._effective_model == "claude-3-opus"

    def test_openai_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that openai provider raises ImportError when package missing."""
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="openai package is required"):
            LLMSummarizer(provider="openai")

    def test_anthropic_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that anthropic provider raises ImportError when package missing."""
        monkeypatch.setitem(sys.modules, "anthropic", None)
        with pytest.raises(ImportError, match="anthropic package is required"):
            LLMSummarizer(provider="anthropic")

    def test_init_client_unknown_provider(self) -> None:
        """Test _init_client raises ValueError for unknown provider."""
        summarizer = LLMSummarizer(provider="custom", custom_callback=lambda p, **kw: "")
        summarizer._provider = "unknown"
        with pytest.raises(ValueError, match="Unknown provider"):
            summarizer._init_client()


class TestLLMSummarizerOpenAI:
    """Tests for summarize() with OpenAI/Ollama provider."""

    def _make_openai_module(self, create_fn):
        """Helper to build a mock openai module with a given completions.create function."""
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create_fn)
            )
        )
        return types.SimpleNamespace(OpenAI=lambda **kw: client)

    def test_openai_summarize_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful summarization via OpenAI provider."""
        def mock_create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="AI summary")
                    )
                ]
            )

        monkeypatch.setitem(sys.modules, "openai", self._make_openai_module(mock_create))
        summarizer = LLMSummarizer(provider="openai")
        result = summarizer.summarize(["Memory A", "Memory B"])
        assert result == "AI summary"

    def test_openai_summarize_empty_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test OpenAI provider handles None content gracefully."""
        def mock_create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=None)
                    )
                ]
            )

        monkeypatch.setitem(sys.modules, "openai", self._make_openai_module(mock_create))
        summarizer = LLMSummarizer(provider="openai")
        result = summarizer.summarize(["Memory A"])
        assert result == ""

    def test_openai_summarize_with_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that kwargs are passed through to OpenAI API."""
        captured_kwargs: dict = {}

        def mock_create(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="ok")
                    )
                ]
            )

        monkeypatch.setitem(sys.modules, "openai", self._make_openai_module(mock_create))
        summarizer = LLMSummarizer(provider="openai")
        summarizer.summarize(["test"], temperature=0.5, max_tokens=100)
        assert captured_kwargs.get("temperature") == 0.5
        assert captured_kwargs.get("max_tokens") == 100

    def test_openai_client_none_returns_empty(self) -> None:
        """Test that summarize returns empty when client is None."""
        summarizer = LLMSummarizer(provider="custom", custom_callback=lambda p, **kw: "")
        summarizer._provider = "openai"
        summarizer._client = None
        result = summarizer.summarize(["test"])
        assert result == ""


class TestLLMSummarizerAnthropic:
    """Tests for summarize() with Anthropic provider."""

    def _make_anthropic_module(self, create_fn):
        """Helper to build a mock anthropic module with a given messages.create function."""
        client = types.SimpleNamespace(
            messages=types.SimpleNamespace(create=create_fn)
        )
        return types.SimpleNamespace(Anthropic=lambda **kw: client)

    def test_anthropic_summarize_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful summarization via Anthropic provider."""
        def mock_create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(text="Claude summary")]
            )

        monkeypatch.setitem(sys.modules, "anthropic", self._make_anthropic_module(mock_create))
        summarizer = LLMSummarizer(provider="anthropic")
        result = summarizer.summarize(["Memory A", "Memory B"])
        assert result == "Claude summary"

    def test_anthropic_summarize_empty_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Anthropic provider handles empty content gracefully."""
        def mock_create(**kwargs: Any) -> Any:
            return types.SimpleNamespace(content=[])

        monkeypatch.setitem(sys.modules, "anthropic", self._make_anthropic_module(mock_create))
        summarizer = LLMSummarizer(provider="anthropic")
        result = summarizer.summarize(["Memory A"])
        assert result == ""

    def test_anthropic_max_tokens_from_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that max_tokens is extracted from kwargs for Anthropic."""
        captured_kwargs: dict = {}

        def mock_create(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(text="ok")]
            )

        monkeypatch.setitem(sys.modules, "anthropic", self._make_anthropic_module(mock_create))
        summarizer = LLMSummarizer(provider="anthropic")
        summarizer.summarize(["test"], max_tokens=500, temperature=0.7)
        assert captured_kwargs.get("max_tokens") == 500
        assert "temperature" in captured_kwargs

    def test_anthropic_client_none_returns_empty(self) -> None:
        """Test that summarize returns empty when client is None."""
        summarizer = LLMSummarizer(provider="custom", custom_callback=lambda p, **kw: "")
        summarizer._provider = "anthropic"
        summarizer._client = None
        result = summarizer.summarize(["test"])
        assert result == ""


class TestLLMSummarizerSummarize:
    def test_empty_memories(self) -> None:
        """Test that empty memories return empty string."""
        summarizer = LLMSummarizer(provider="custom", custom_callback=lambda p, **kw: "summary")
        result = summarizer.summarize([])
        assert result == ""

    def test_custom_callback_called_with_prompt(self) -> None:
        """Test that the custom callback is invoked with the formatted prompt."""
        captured: list[str] = []

        def mock_callback(prompt: str, **kwargs: object) -> str:
            captured.append(prompt)
            return "mock summary"

        summarizer = LLMSummarizer(provider="custom", custom_callback=mock_callback)
        result = summarizer.summarize(["Memory A", "Memory B"])

        assert result == "mock summary"
        assert len(captured) == 1
        assert "- Memory A" in captured[0]
        assert "- Memory B" in captured[0]

    def test_custom_callback_passes_kwargs(self) -> None:
        """Test that additional kwargs are passed to the callback."""
        captured_kwargs: dict = {}

        def mock_callback(prompt: str, **kwargs: object) -> str:
            captured_kwargs.update(kwargs)
            return "summary"

        summarizer = LLMSummarizer(provider="custom", custom_callback=mock_callback)
        summarizer.summarize(["test"], temperature=0.3, max_tokens=200)

        assert captured_kwargs.get("temperature") == 0.3
        assert captured_kwargs.get("max_tokens") == 200

    def test_callback_failure_fallback(self) -> None:
        """Test that a failing callback returns empty string (fallback handled by caller)."""
        def failing_callback(prompt: str, **kwargs: object) -> str:
            raise RuntimeError("LLM failure")

        summarizer = LLMSummarizer(provider="custom", custom_callback=failing_callback)
        result = summarizer.summarize(["test"])
        assert result == ""

    def test_custom_callback_none_returns_empty(self) -> None:
        """Test that summarize returns empty string when custom_callback is None."""
        summarizer = LLMSummarizer(provider="custom", custom_callback=lambda p, **kw: "ok")
        summarizer._custom_callback = None  # Simulate unset callback after init
        result = summarizer.summarize(["test"])
        assert result == ""

    def test_single_memory_formats_correctly(self) -> None:
        """Test that a single memory is formatted correctly in the prompt."""
        captured: list[str] = []

        def mock_callback(prompt: str, **kwargs: object) -> str:
            captured.append(prompt)
            return "summary"

        summarizer = LLMSummarizer(provider="custom", custom_callback=mock_callback)
        summarizer.summarize(["Only memory"])

        # Default prompt uses inline formatting: "...: - Only memory"
        assert "- Only memory" in captured[0]
        assert "Only memory" in captured[0]

    def test_multiple_memories_all_included(self) -> None:
        """Test that all memories are included in the prompt."""
        captured: list[str] = []

        def mock_callback(prompt: str, **kwargs: object) -> str:
            captured.append(prompt)
            return "summary"

        summarizer = LLMSummarizer(provider="custom", custom_callback=mock_callback)
        texts = [f"Memory {i}" for i in range(5)]
        summarizer.summarize(texts)

        for i in range(5):
            assert f"- Memory {i}" in captured[0]


@pytest.mark.asyncio
async def test_llm_summary_integration_with_consolidation(real_db_memory) -> None:
    """Integration test: consolidate_cluster with mock LLM via custom callback."""
    from datetime import timedelta
    from kemi import consolidation
    from kemi.summarizer import LLMSummarizer
    from kemi.models import MemoryType

    mem = real_db_memory

    # Store some old episodic memories
    memory_ids = []
    for i, content in enumerate([
        "User visited the Alps for hiking in summer",
        "User enjoyed skiing in the French Alps",
        "User plans to climb Mont Blanc next year",
    ]):
        mid = mem.remember(
            "user123",
            content,
            memory_type=MemoryType.EPISODIC,
        )
        memory_ids.append(mid)

    # Fetch the memories to form a cluster
    cluster = [mem._store.get(mid) for mid in memory_ids if mem._store.get(mid) is not None]
    assert len(cluster) == 3, f"Expected 3 memories, got {len(cluster)}"

    # Create a mock LLM summarizer
    summarizer = LLMSummarizer(
        provider="custom",
        custom_callback=lambda p, **kw: "User enjoys visiting various places.",
    )

    result = consolidation.consolidate_cluster(
        store=mem._store,
        embed=mem._embed,
        user_id="user123",
        cluster=cluster,
        summarizer=summarizer,
    )

    assert result is not None
    # Content is the extractive summary
    assert len(result.content) > 0
    # Metadata contains the LLM summary text
    assert result.metadata.get("llm_summary") == "User enjoys visiting various places."
    assert "consolidated_from" in result.metadata
    assert len(result.metadata["consolidated_from"]) == 3
    assert result.memory_type == MemoryType.SEMANTIC

    # Old memories should be archived
    for mid in result.metadata["consolidated_from"]:
        archived = mem._store.get(mid)
        if archived is not None:
            assert archived.lifecycle_state.value == "archived"


def test_summarizer_integration_custom_callback() -> None:
    """Test that consolidate_cluster works with a custom callback LLM summarizer."""
    from datetime import timedelta
    from unittest.mock import MagicMock
    from kemi import consolidation
    from kemi.summarizer import LLMSummarizer
    from kemi.adapters.embedding.custom import CustomEmbedAdapter

    embed = CustomEmbedAdapter(embed_fn=lambda texts: [[0.1] * 32 for _ in texts], dim=32)

    # Create a mock store that can handle update calls
    mock_store = MagicMock()
    mock_store.update = MagicMock()

    memories = [
        MemoryObject(
            memory_id=f"test-{i}",
            user_id="user1",
            content=f"Related memory {i}",
            embedding=[0.1] * 32,
            embedding_dim=32,
            memory_type=MemoryType.EPISODIC,
            created_at=datetime.now(timezone.utc) - timedelta(days=100),
            last_accessed_at=datetime.now(timezone.utc) - timedelta(days=100),
        )
        for i in range(3)
    ]

    summarizer = LLMSummarizer(
        provider="custom",
        custom_callback=lambda p, **kw: "Mock LLM summary text",
    )

    result = consolidation.consolidate_cluster(
        store=mock_store,
        embed=embed,
        user_id="user1",
        cluster=memories,
        summarizer=summarizer,
    )

    assert result is not None
    assert result.metadata.get("llm_summary") == "Mock LLM summary text"
    assert result.metadata.get("consolidated_count") == 3
    assert len(result.metadata.get("consolidated_from", [])) == 3
    # Verify store.update was called (to archive old memories)
    assert mock_store.update.call_count >= 1
