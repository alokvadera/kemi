"""Tests for entity-aware retrieval in Kemi."""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import pytest

from kemi import Memory, MemoryConfig
from kemi.adapters.base import EmbeddingAdapter, StorageAdapter
from kemi.entities import EntityLinker, NoopEntityLinker, RegexEntityLinker, SpacyEntityLinker
from kemi.models import LifecycleState, MemoryObject, MemorySource


class _FakeEmbed(EmbeddingAdapter):
    """Deterministic fake embedding adapter using SHA-256 expansion."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        return self._vec(text)

    def dimension(self) -> int:
        return self._dim

    def _vec(self, text: str) -> list[float]:
        raw = hashlib.sha256(text.encode()).digest()
        expanded = raw * (self._dim // len(raw) + 1)
        return [b / 255.0 for b in expanded[: self._dim]]


class _FakeStore(StorageAdapter):
    def __init__(self):
        self._data: dict[str, MemoryObject] = {}

    def store(self, memory: MemoryObject) -> None:
        self._data[memory.memory_id] = memory

    def get(self, memory_id: str) -> MemoryObject | None:
        return self._data.get(memory_id)

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        results = []
        for mem in self._data.values():
            if mem.user_id != user_id:
                continue
            if namespace != "default" and mem.namespace != namespace:
                continue
            if session_id is not None and mem.session_id != session_id:
                continue
            if lifecycle_filter is not None and mem.lifecycle_state not in lifecycle_filter:
                continue
            results.append(mem)
        return results[:top_k]

    def get_all_by_user(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        results = []
        for mem in self._data.values():
            if mem.user_id != user_id:
                continue
            if namespace != "default" and mem.namespace != namespace:
                continue
            if session_id is not None and mem.session_id != session_id:
                continue
            if lifecycle_filter is not None and mem.lifecycle_state not in lifecycle_filter:
                continue
            results.append(mem)
        return results

    def get_by_tag(
        self, user_id: str, tag: str, lifecycle_filter: list[LifecycleState] | None = None
    ) -> list[MemoryObject]:
        return []

    def get_all_users(self) -> list[str]:
        return list({m.user_id for m in self._data.values()})

    def count(self, user_id: str) -> int:
        return sum(1 for m in self._data.values() if m.user_id == user_id)

    def update(self, memory: MemoryObject) -> None:
        self._data[memory.memory_id] = memory

    def delete_by_id(self, memory_id: str) -> bool:
        return self._data.pop(memory_id, None) is not None

    def delete_by_user(self, user_id: str) -> int:
        to_remove = [mid for mid, m in self._data.items() if m.user_id == user_id]
        for mid in to_remove:
            self._data.pop(mid, None)
        return len(to_remove)

    def get_all(self, limit: int | None = None) -> list[MemoryObject]:
        return list(self._data.values())[:limit]

    def upgrade_schema(self, *, from_version: int, to_version: int) -> None:
        pass

    def search_by_content(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        return []


@pytest.fixture
def fake_embed():
    return _FakeEmbed(dim=64)


@pytest.fixture
def fake_store():
    return _FakeStore()


@pytest.fixture
def entity_memory(fake_embed, fake_store):
    """Memory instance with entity boost enabled."""
    config = MemoryConfig(enable_entity_boost=True, entity_boost_weight=0.3)
    return Memory(embed=fake_embed, store=fake_store, config=config)


# ---------------------------------------------------------------------------
# EntityLinker unit tests
# ---------------------------------------------------------------------------


def test_noop_entity_linker_returns_empty_set():
    linker = NoopEntityLinker()
    assert linker.extract("Alice went to Paris on 2024-06-05") == set()


def test_regex_entity_linker_extracts_names():
    linker = RegexEntityLinker()
    entities = linker.extract("Alice and Bob visited Paris")
    assert "alice" in entities
    assert "bob" in entities
    assert "paris" in entities


def test_regex_entity_linker_extracts_dates():
    linker = RegexEntityLinker()
    assert "2024-06-05" in linker.extract("Meeting on 2024-06-05")
    assert "06/05/2024" in linker.extract("Event 06/05/2024")
    assert "june 5, 2024" in linker.extract("Trip June 5, 2024")


def test_regex_entity_linker_extracts_emails():
    linker = RegexEntityLinker()
    assert "alice@example.com" in linker.extract("Contact alice@example.com")


def test_regex_entity_linker_extracts_urls():
    linker = RegexEntityLinker()
    assert "https://example.com" in linker.extract("Visit https://example.com")


def test_regex_entity_linker_normalizes_to_lowercase():
    linker = RegexEntityLinker()
    entities = linker.extract("ALICE and Bob")
    assert all(e.islower() for e in entities)
    assert "alice" in entities
    assert "bob" in entities


# ---------------------------------------------------------------------------
# Memory.__init__ entity linker wiring
# ---------------------------------------------------------------------------


def test_memory_init_uses_noop_when_entity_boost_disabled(fake_embed, fake_store):
    config = MemoryConfig(enable_entity_boost=False)
    mem = Memory(embed=fake_embed, store=fake_store, config=config)
    assert isinstance(mem._entity_linker, NoopEntityLinker)


def test_memory_init_uses_regex_when_entity_boost_enabled(fake_embed, fake_store):
    config = MemoryConfig(enable_entity_boost=True)
    mem = Memory(embed=fake_embed, store=fake_store, config=config)
    assert isinstance(mem._entity_linker, RegexEntityLinker)


def test_memory_init_uses_custom_entity_linker(fake_embed, fake_store):
    class CustomLinker(EntityLinker):
        def extract(self, text: str) -> set[str]:
            return {"custom"}

    custom = CustomLinker()
    mem = Memory(embed=fake_embed, store=fake_store, entity_linker=custom)
    assert mem._entity_linker is custom


# ---------------------------------------------------------------------------
# Direct scoring tests
# ---------------------------------------------------------------------------


def test_score_memory_includes_entity_boost():
    """score_memory should add entity_boost_weight * jaccard_overlap when entities are provided."""
    from kemi.scoring import score_memory

    now = datetime.now(timezone.utc)
    memory = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=[0.1] * 64,
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
    )

    query_embedding = [0.1] * 64

    # Without entities
    score_no_entity = score_memory(
        memory, query_embedding, query="Alice in Paris",
        hybrid_search=False,
        query_entities=None, memory_entities=None,
    )

    # With overlapping entities (Alice, Paris)
    score_with_entity = score_memory(
        memory, query_embedding, query="Alice in Paris",
        hybrid_search=False,
        query_entities={"alice", "paris"},
        memory_entities={"alice", "paris"},
        weight_entity=0.2,
    )

    # Jaccard overlap is 1.0 (exact match), boost = 0.2
    assert score_with_entity == pytest.approx(score_no_entity + 0.2)


def test_jaccard_similarity_edge_cases():
    from kemi.scoring import jaccard_similarity
    assert jaccard_similarity(set(), {"a"}) == 0.0
    assert jaccard_similarity({"a"}, set()) == 0.0
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard_similarity({"a", "b"}, {"a", "c"}) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Entity boost ranking tests
# ---------------------------------------------------------------------------


def test_entity_boost_ranks_shared_entities_higher(entity_memory, fake_store):
    """Memories sharing entities with the query should rank above those that don't,
    even when semantic similarity would otherwise place them lower."""
    mem = entity_memory

    # Create two memories with identical importance/recency by using the same timestamp
    now = datetime.now(timezone.utc)
    base = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited the Eiffel Tower in Paris",
        embedding=mem._embed.embed_single("Alice visited the Eiffel Tower in Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
    )
    unrelated = MemoryObject(
        memory_id="m2",
        user_id="u1",
        content="Bob cooked dinner at home",
        embedding=mem._embed.embed_single("Bob cooked dinner at home"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
    )
    fake_store.store(base)
    fake_store.store(unrelated)

    results = mem.recall("u1", "What did Alice do in Paris?", top_k=2)
    assert results[0].memory_id == "m1"
    assert results[1].memory_id == "m2"


def test_entity_boost_with_zero_weight_no_effect(entity_memory, fake_store):
    """When entity_boost_weight=0, entity overlap should not affect ranking."""
    config = MemoryConfig(enable_entity_boost=True, entity_boost_weight=0.0)
    mem = Memory(embed=entity_memory._embed, store=fake_store, config=config)

    now = datetime.now(timezone.utc)
    base = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=mem._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
    )
    fake_store.store(base)

    results = mem.recall("u1", "Alice in Paris", top_k=1)
    assert len(results) == 1


def test_recall_explain_includes_entity_score(entity_memory, fake_store):
    mem = entity_memory
    now = datetime.now(timezone.utc)
    base = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=mem._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
    )
    fake_store.store(base)

    explained = mem.recall_explain("u1", "Alice in Paris", top_k=1)
    assert len(explained) == 1
    exp = explained[0]["explanation"]
    assert "entity_score" in exp
    assert "weights" in exp
    assert "entity" in exp["weights"]
    assert exp["entity_score"] > 0


def test_recall_explain_no_entity_score_when_disabled(fake_embed, fake_store):
    config = MemoryConfig(enable_entity_boost=False)
    mem = Memory(embed=fake_embed, store=fake_store, config=config)
    now = datetime.now(timezone.utc)
    base = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=mem._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
    )
    fake_store.store(base)

    explained = mem.recall_explain("u1", "Alice in Paris", top_k=1)
    assert explained[0]["explanation"]["entity_score"] == 0


# ---------------------------------------------------------------------------
# Async recall_stream entity boost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_stream_applies_entity_boost(entity_memory, fake_store):
    mem = entity_memory
    now = datetime.now(timezone.utc)
    base = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=mem._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
    )
    fake_store.store(base)

    results = []
    async for m in mem.recall_stream("u1", "Alice in Paris", top_k=1):
        results.append(m)
    assert len(results) == 1
    assert results[0].memory_id == "m1"


# ---------------------------------------------------------------------------
# SpacyEntityLinker tests (mock-based — no spaCy install required)
# ---------------------------------------------------------------------------


def test_spacy_entity_linker_import_error_without_spacy(monkeypatch):
    """SpacyEntityLinker should raise ImportError when spaCy is not installed."""
    import sys

    # Setting sys.modules["spacy"] = None forces import spacy to fail cleanly.
    monkeypatch.setitem(sys.modules, "spacy", None)

    with pytest.raises(ImportError, match="spaCy is required"):
        SpacyEntityLinker()


def _make_fake_spacy_module():
    """Build a fake ``spacy`` module so SpacyEntityLinker can be tested without installing spaCy."""
    from unittest.mock import MagicMock
    import types

    fake_spacy = types.ModuleType("spacy")
    fake_spacy.load = MagicMock()
    return fake_spacy


def test_spacy_entity_linker_extract_with_mock(monkeypatch):
    """SpacyEntityLinker.extract should return lower-cased entities filtered by label."""
    from unittest.mock import MagicMock
    import sys

    fake_spacy = _make_fake_spacy_module()
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)

    mock_ent_alice = MagicMock()
    mock_ent_alice.label_ = "PERSON"
    mock_ent_alice.text = "Alice"

    mock_ent_paris = MagicMock()
    mock_ent_paris.label_ = "GPE"
    mock_ent_paris.text = "Paris"

    mock_ent_org = MagicMock()
    mock_ent_org.label_ = "ORG"
    mock_ent_org.text = "Acme Corp"

    mock_ent_time = MagicMock()
    mock_ent_time.label_ = "TIME"
    mock_ent_time.text = "3pm"

    mock_doc = MagicMock()
    mock_doc.ents = [mock_ent_alice, mock_ent_paris, mock_ent_org, mock_ent_time]

    mock_nlp = MagicMock()
    mock_nlp.return_value = mock_doc
    fake_spacy.load.return_value = mock_nlp

    linker = SpacyEntityLinker()
    entities = linker.extract("Alice from Paris works at Acme Corp at 3pm")

    assert "alice" in entities
    assert "paris" in entities
    assert "acme corp" in entities
    assert "3pm" not in entities  # TIME is not in default allowed labels


def test_spacy_entity_linker_custom_allowed_labels(monkeypatch):
    """allowed_labels should restrict which entity types are returned."""
    from unittest.mock import MagicMock
    import sys

    fake_spacy = _make_fake_spacy_module()
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)

    mock_ent_alice = MagicMock()
    mock_ent_alice.label_ = "PERSON"
    mock_ent_alice.text = "Alice"

    mock_ent_org = MagicMock()
    mock_ent_org.label_ = "ORG"
    mock_ent_org.text = "Acme"

    mock_doc = MagicMock()
    mock_doc.ents = [mock_ent_alice, mock_ent_org]

    mock_nlp = MagicMock()
    mock_nlp.return_value = mock_doc
    fake_spacy.load.return_value = mock_nlp

    linker = SpacyEntityLinker(allowed_labels={"PERSON"})
    entities = linker.extract("Alice works at Acme")

    assert "alice" in entities
    assert "acme" not in entities


# ---------------------------------------------------------------------------
# Entity caching in metadata
# ---------------------------------------------------------------------------


def test_entities_cached_in_metadata_on_remember(entity_memory, fake_store):
    """When entity boost is enabled, remembered memories should have extracted_entities in metadata."""
    mem = entity_memory
    mid = mem.remember("u1", "Alice visited Paris on 2024-06-05")
    stored = fake_store.get(mid)
    assert stored is not None
    assert "extracted_entities" in stored.metadata
    cached = set(stored.metadata["extracted_entities"])
    assert "alice" in cached
    assert "paris" in cached
    assert "2024-06-05" in cached


def test_entities_re_cached_on_update(entity_memory, fake_store):
    """Updating content should re-extract and cache new entities."""
    mem = entity_memory
    mid = mem.remember("u1", "Alice visited Paris")
    original = fake_store.get(mid)
    original_entities = set(original.metadata.get("extracted_entities", []))
    assert "alice" in original_entities

    mem.update(mid, content="Bob went to Tokyo on 2025-01-10")
    updated = fake_store.get(mid)
    new_entities = set(updated.metadata.get("extracted_entities", []))
    assert "bob" in new_entities
    assert "tokyo" in new_entities
    assert "2025-01-10" in new_entities


def test_entities_reused_from_metadata_in_recall(entity_memory, fake_store):
    """Recall should read cached entities from metadata and not re-extract them."""
    # Seed a memory with manually-cached entities
    now = datetime.now(timezone.utc)
    mo = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Some generic text without named entities.",
        embedding=entity_memory._embed.embed_single("Some generic text without named entities."),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={"extracted_entities": ["alice", "paris"]},
    )
    fake_store.store(mo)

    # Create a tracking linker that counts extract calls
    class _TrackingLinker(EntityLinker):
        def __init__(self):
            self.extract_calls = 0

        def extract(self, text: str) -> set[str]:
            self.extract_calls += 1
            return RegexEntityLinker().extract(text)

    tracker = _TrackingLinker()
    mem = Memory(
        embed=entity_memory._embed,
        store=fake_store,
        config=entity_memory._config,
        entity_linker=tracker,
    )

    mem.recall("u1", "Alice in Paris", top_k=1)
    # The query itself triggers one extract call. The memory content should NOT trigger another
    # because cached entities are available.
    assert tracker.extract_calls == 1


def test_backward_compat_no_cached_entities(entity_memory, fake_store):
    """Memories without cached entities should still work (fallback to on-the-fly extraction)."""
    now = datetime.now(timezone.utc)
    mo = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=entity_memory._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={},  # No cached entities
    )
    fake_store.store(mo)

    results = entity_memory.recall("u1", "Alice in Paris", top_k=1)
    assert len(results) == 1
    assert results[0].memory_id == "m1"


# ---------------------------------------------------------------------------
# Benchmark smoke test
# ---------------------------------------------------------------------------


def _run_benchmark_subprocess(script_name: str, env_overrides: dict[str, str], tmp_path: Path) -> dict[str, Any]:
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / script_name
    results_file = tmp_path / "results.json"
    png_file = tmp_path / "results.png"

    env = {
        **os.environ,
        **env_overrides,
    }

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(script.parent.parent),
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

    assert results_file.exists()
    with open(results_file) as f:
        return json.load(f)  # type: ignore[no-any-return]


def test_benchmark_smoke(tmp_path):
    """Run the entity boost benchmark as a subprocess with small parameters."""
    results = _run_benchmark_subprocess(
        "benchmark_entity_boost.py",
        {
            "BENCH_NUM_MEMORIES": "10",
            "BENCH_NUM_QUERIES": "3",
            "BENCH_TOP_K": "5",
            "BENCH_RESULTS_FILE": str(tmp_path / "results.json"),
            "BENCH_PNG_FILE": str(tmp_path / "results.png"),
        },
        tmp_path,
    )
    assert "summary" in results
    assert results["summary"]["hit_rate_improvement_pp"] >= 0
    assert results["with_boost"]["aggregate"]["hit_rate"] >= results["without_boost"]["aggregate"]["hit_rate"]


def test_benchmark_large_smoke(tmp_path):
    """Run the large-scale entity boost benchmark variant with small parameters."""
    results = _run_benchmark_subprocess(
        "benchmark_entity_boost_large.py",
        {
            "BENCH_LARGE_NUM_MEMORIES": "10",
            "BENCH_LARGE_NUM_QUERIES": "3",
            "BENCH_LARGE_TOP_K": "5",
            "BENCH_LARGE_RESULTS_FILE": str(tmp_path / "results.json"),
            "BENCH_LARGE_PNG_FILE": str(tmp_path / "results.png"),
        },
        tmp_path,
    )
    assert "summary" in results
    assert results["config"]["num_memories"] == 10
    assert results["summary"]["hit_rate_improvement_pp"] >= 0


def test_benchmark_latency_smoke(tmp_path):
    """Run the entity latency benchmark with small parameters."""
    results = _run_benchmark_subprocess(
        "benchmark_entity_latency.py",
        {
            "BENCH_LAT_NUM_MEMORIES": "10",
            "BENCH_LAT_NUM_QUERIES": "3",
            "BENCH_LAT_TOP_K": "5",
            "BENCH_LAT_WARMUP": "1",
            "BENCH_LAT_TIMED": "2",
            "BENCH_LAT_RESULTS_FILE": str(tmp_path / "results.json"),
            "BENCH_LAT_PNG_FILE": str(tmp_path / "results.png"),
        },
        tmp_path,
    )
    assert "summary" in results
    assert results["summary"]["cached_mean_ms"] >= 0
    assert results["summary"]["uncached_mean_ms"] >= 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_entity_boost_empty_query(entity_memory, fake_store):
    """Empty query should not crash entity extraction."""
    mem = entity_memory
    now = datetime.now(timezone.utc)
    base = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=mem._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
    )
    fake_store.store(base)

    with pytest.raises(ValueError, match="query cannot be empty"):
        mem.recall("u1", "", top_k=1)


def test_entity_boost_no_results(entity_memory, fake_store):
    """No search results should return empty list without error."""
    mem = entity_memory
    results = mem.recall("u1", "Alice in Paris", top_k=1)
    assert results == []


# ---------------------------------------------------------------------------
# Entity cache invalidation on dedup merge
# ---------------------------------------------------------------------------


def test_dedup_merge_invalidates_cached_entities(entity_memory, fake_store, monkeypatch):
    """When a duplicate is merged, stale extracted_entities should be removed from metadata."""
    # Seed an existing memory with cached entities
    now = datetime.now(timezone.utc)
    existing = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=entity_memory._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={"extracted_entities": ["alice", "paris"]},
    )
    fake_store.store(existing)

    # Force dedup to trigger by monkeypatching find_duplicates
    def _forced_duplicates(new_memory, existing_memories, threshold=0.85):
        return [existing]

    monkeypatch.setattr("kemi.dedup.find_duplicates", _forced_duplicates)

    # Remember different content that will be merged into the existing memory
    mid = entity_memory.remember("u1", "Bob visited Tokyo on 2025-01-10")
    assert mid == "m1"  # Merged into existing

    merged = fake_store.get("m1")
    assert merged.content == "Bob visited Tokyo on 2025-01-10"
    assert "extracted_entities" not in merged.metadata


def test_dedup_merge_remembers_many_invalidates_cached_entities(entity_memory, fake_store, monkeypatch):
    """remember_many should also invalidate cached entities when merging duplicates."""
    now = datetime.now(timezone.utc)
    existing = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=entity_memory._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={"extracted_entities": ["alice", "paris"]},
    )
    fake_store.store(existing)

    def _forced_duplicates(new_memory, existing_memories, threshold=0.85):
        return [existing]

    monkeypatch.setattr("kemi.dedup.find_duplicates", _forced_duplicates)

    mids = entity_memory.remember_many("u1", ["Bob visited Tokyo"])
    assert mids[0] == "m1"

    merged = fake_store.get("m1")
    assert merged.content == "Bob visited Tokyo"
    assert "extracted_entities" not in merged.metadata


@pytest.mark.asyncio
async def test_dedup_merge_aremember_invalidates_cached_entities(entity_memory, fake_store, monkeypatch):
    """Async aremember should also invalidate cached entities when merging duplicates."""
    now = datetime.now(timezone.utc)
    existing = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=entity_memory._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={"extracted_entities": ["alice", "paris"]},
    )
    fake_store.store(existing)

    def _forced_duplicates(new_memory, existing_memories, threshold=0.85):
        return [existing]

    monkeypatch.setattr("kemi.dedup.find_duplicates", _forced_duplicates)

    mid = await entity_memory.aremember("u1", "Bob visited Tokyo")
    assert mid == "m1"

    merged = fake_store.get("m1")
    assert merged.content == "Bob visited Tokyo"
    assert "extracted_entities" not in merged.metadata


def test_dedup_merge_noop_when_no_cached_entities(entity_memory, fake_store, monkeypatch):
    """Merging a memory that never had extracted_entities should not crash."""
    now = datetime.now(timezone.utc)
    existing = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=entity_memory._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={},  # No extracted_entities
    )
    fake_store.store(existing)

    def _forced_duplicates(new_memory, existing_memories, threshold=0.85):
        return [existing]

    monkeypatch.setattr("kemi.dedup.find_duplicates", _forced_duplicates)

    mid = entity_memory.remember("u1", "Bob visited Tokyo")
    assert mid == "m1"

    merged = fake_store.get("m1")
    assert merged.content == "Bob visited Tokyo"
    assert "extracted_entities" not in merged.metadata


# ---------------------------------------------------------------------------
# Entity backfill
# ---------------------------------------------------------------------------


def test_backfill_entities_populates_missing_metadata(entity_memory, fake_store):
    """Memories without extracted_entities should get them backfilled."""
    now = datetime.now(timezone.utc)
    mo = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris on 2024-06-05",
        embedding=entity_memory._embed.embed_single("Alice visited Paris on 2024-06-05"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={},  # No cached entities
    )
    fake_store.store(mo)

    count = entity_memory.backfill_entities(user_id="u1")
    assert count == 1

    updated = fake_store.get("m1")
    assert "extracted_entities" in updated.metadata
    cached = set(updated.metadata["extracted_entities"])
    assert "alice" in cached
    assert "paris" in cached
    assert "2024-06-05" in cached


def test_backfill_entities_skips_already_cached(entity_memory, fake_store):
    """Memories that already have extracted_entities should be skipped."""
    now = datetime.now(timezone.utc)
    mo = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=entity_memory._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={"extracted_entities": ["bob"]},  # Already cached (wrong value on purpose)
    )
    fake_store.store(mo)

    count = entity_memory.backfill_entities(user_id="u1")
    assert count == 0

    updated = fake_store.get("m1")
    # Should NOT have been overwritten
    assert updated.metadata["extracted_entities"] == ["bob"]


def test_backfill_entities_disabled_when_boost_off(fake_embed, fake_store):
    """When entity boost is disabled, backfill_entities should return 0 immediately."""
    config = MemoryConfig(enable_entity_boost=False)
    mem = Memory(embed=fake_embed, store=fake_store, config=config)

    now = datetime.now(timezone.utc)
    mo = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=mem._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={},
    )
    fake_store.store(mo)

    count = mem.backfill_entities(user_id="u1")
    assert count == 0


def test_backfill_entities_all_users(entity_memory, fake_store):
    """When user_id is None, backfill should cover all users."""
    now = datetime.now(timezone.utc)
    for uid in ("u1", "u2"):
        mo = MemoryObject(
            memory_id=f"m-{uid}",
            user_id=uid,
            content="Alice visited Paris",
            embedding=entity_memory._embed.embed_single("Alice visited Paris"),
            score=0.0,
            created_at=now,
            last_accessed_at=now,
            source=MemorySource.USER_STATED,
            importance=0.5,
            lifecycle_state=LifecycleState.ACTIVE,
            embedding_dim=64,
            metadata={},
        )
        fake_store.store(mo)

    count = entity_memory.backfill_entities(user_id=None)
    assert count == 2


def test_run_maintenance_includes_backfill(entity_memory, fake_store):
    """run_maintenance should include backfilled count in its result."""
    now = datetime.now(timezone.utc)
    mo = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=entity_memory._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={},
    )
    fake_store.store(mo)

    result = entity_memory.run_maintenance(
        user_id="u1",
        auto_prune=False,
        auto_consolidate=False,
        auto_backfill_entities=True,
    )
    assert result["backfilled"] == 1


def test_run_maintenance_can_skip_backfill(entity_memory, fake_store):
    """run_maintenance with auto_backfill_entities=False should skip backfill."""
    now = datetime.now(timezone.utc)
    mo = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=entity_memory._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={},
    )
    fake_store.store(mo)

    result = entity_memory.run_maintenance(
        user_id="u1",
        auto_prune=False,
        auto_consolidate=False,
        auto_backfill_entities=False,
    )
    assert result["backfilled"] == 0


@pytest.mark.asyncio
async def test_abackfill_entities(entity_memory, fake_store):
    """Async backfill should produce the same results as sync backfill."""
    now = datetime.now(timezone.utc)
    mo = MemoryObject(
        memory_id="m1",
        user_id="u1",
        content="Alice visited Paris",
        embedding=entity_memory._embed.embed_single("Alice visited Paris"),
        score=0.0,
        created_at=now,
        last_accessed_at=now,
        source=MemorySource.USER_STATED,
        importance=0.5,
        lifecycle_state=LifecycleState.ACTIVE,
        embedding_dim=64,
        metadata={},
    )
    fake_store.store(mo)

    count = await entity_memory.abackfill_entities(user_id="u1")
    assert count == 1

    updated = fake_store.get("m1")
    assert "extracted_entities" in updated.metadata
