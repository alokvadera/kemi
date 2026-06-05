"""Tests for procedural memory helpers."""

import hashlib
import pytest

from kemi import Memory, remember_procedure, recall_procedures
from kemi.adapters.base import EmbeddingAdapter, StorageAdapter
from kemi.models import LifecycleState, MemoryObject, MemoryType


class _FakeEmbed(EmbeddingAdapter):
    """Deterministic fake embedder that produces distinct vectors per text."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def _vec(self, text: str) -> list[float]:
        raw = hashlib.sha256(text.encode()).digest()
        expanded = raw * (self._dim // len(raw) + 1)
        return [b / 255.0 for b in expanded[: self._dim]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        return self._vec(text)

    def dimension(self) -> int:
        return self._dim


class _FakeStore(StorageAdapter):
    def __init__(self) -> None:
        self._data: dict[str, MemoryObject] = {}

    def store(self, memory: MemoryObject) -> None:
        self._data[memory.memory_id] = memory

    def get(self, memory_id: str) -> MemoryObject | None:
        return self._data.get(memory_id)

    def get_all_by_user(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        results = []
        for mem in self._data.values():
            if mem.user_id != user_id:
                continue
            if mem.namespace != namespace:
                continue
            if session_id is not None and mem.session_id != session_id:
                continue
            if lifecycle_filter is not None and mem.lifecycle_state not in lifecycle_filter:
                continue
            results.append(mem)
        return results

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        # Brute-force cosine similarity
        from kemi.scoring import cosine_similarity

        candidates = self.get_all_by_user(
            user_id, lifecycle_filter=lifecycle_filter, namespace=namespace, session_id=session_id
        )
        scored = []
        for mem in candidates:
            if mem.embedding is None:
                continue
            score = cosine_similarity(mem.embedding, query_embedding)
            mem.score = score
            scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def update(self, memory: MemoryObject) -> None:
        self._data[memory.memory_id] = memory

    def delete_by_id(self, memory_id: str) -> bool:
        if memory_id in self._data:
            del self._data[memory_id]
            return True
        return False

    def delete_by_user(self, user_id: str) -> int:
        to_delete = [mid for mid, mem in self._data.items() if mem.user_id == user_id]
        for mid in to_delete:
            del self._data[mid]
        return len(to_delete)

    def search_by_content(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        from kemi.scoring import bm25_score

        candidates = self.get_all_by_user(
            user_id, lifecycle_filter=lifecycle_filter, namespace=namespace, session_id=session_id
        )
        scored = []
        for mem in candidates:
            score = bm25_score(query, mem.content)
            mem.score = score
            scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def get_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        results = []
        for mem in self._data.values():
            if mem.user_id != user_id:
                continue
            if mem.namespace != namespace:
                continue
            if tag not in mem.tags:
                continue
            if lifecycle_filter is not None and mem.lifecycle_state not in lifecycle_filter:
                continue
            results.append(mem)
        return results

    def get_all_users(self) -> list[str]:
        return sorted({mem.user_id for mem in self._data.values()})

    def get_all(self, limit: int | None = None, offset: int | None = None) -> list[MemoryObject]:
        items = list(self._data.values())
        if limit is not None:
            items = items[:limit]
        return items

    def count(self, user_id: str) -> int:
        return sum(1 for mem in self._data.values() if mem.user_id == user_id)

    def upgrade_schema(self, from_version: int, to_version: int) -> None:
        pass


@pytest.fixture
def memory():
    """Fixture providing a Memory instance with fake adapters and dedup disabled."""
    store = _FakeStore()
    embed = _FakeEmbed()
    mem = Memory(embed=embed, store=store, config=None)
    mem._config.dedup_threshold = 1.0  # disable dedup for tests
    return mem


# ---------------------------------------------------------------------------
# remember_procedure tests
# ---------------------------------------------------------------------------

def test_remember_procedure_basic(memory):
    mid = remember_procedure(
        memory,
        user_id="alice",
        name="onboarding",
        steps=["Send welcome email", "Create account", "Verify identity"],
    )
    assert isinstance(mid, str)
    stored = memory._store.get(mid)
    assert stored is not None
    assert stored.memory_type == MemoryType.PROCEDURAL
    assert "procedure" in stored.tags
    assert "onboarding" in stored.tags
    assert stored.metadata["procedure_name"] == "onboarding"
    assert stored.metadata["step_count"] == 3
    assert stored.user_id == "alice"
    assert "Send welcome email" in stored.content
    assert stored.importance == 0.7


def test_remember_procedure_defaults_to_system_user(memory):
    mid = remember_procedure(
        memory,
        name="reboot_server",
        steps=["ssh into box", "run reboot"],
    )
    stored = memory._store.get(mid)
    assert stored.user_id == "_system"


def test_remember_procedure_with_agent_and_session(memory):
    mid = remember_procedure(
        memory,
        user_id="alice",
        name="escalation",
        steps=["Identify severity", "Route to tier-2"],
        agent_id="support_bot",
        session_id="sess_001",
        importance=0.9,
    )
    stored = memory._store.get(mid)
    assert stored.agent_id == "support_bot"
    assert stored.session_id == "sess_001"
    assert stored.importance == 0.9


def test_remember_procedure_empty_steps_raises(memory):
    with pytest.raises(ValueError, match="steps cannot be empty"):
        remember_procedure(memory, name="empty", steps=[])


def test_remember_procedure_empty_name_raises(memory):
    with pytest.raises(ValueError, match="name cannot be empty"):
        remember_procedure(memory, name="", steps=["step 1"])


def test_remember_procedure_whitespace_name_raises(memory):
    with pytest.raises(ValueError, match="name cannot be empty"):
        remember_procedure(memory, name="   ", steps=["step 1"])


def test_remember_procedure_metadata_merged(memory):
    mid = remember_procedure(
        memory,
        user_id="alice",
        name="checkout",
        steps=["Add to cart", "Enter shipping", "Pay"],
        metadata={"department": "ecommerce", "priority": "high"},
    )
    stored = memory._store.get(mid)
    assert stored.metadata["department"] == "ecommerce"
    assert stored.metadata["priority"] == "high"
    assert stored.metadata["procedure_name"] == "checkout"
    assert stored.metadata["step_count"] == 3


def test_remember_procedure_content_format(memory):
    mid = remember_procedure(
        memory,
        user_id="alice",
        name="brew_tea",
        steps=["Boil water", "Steep bag", "Wait 3 min"],
    )
    stored = memory._store.get(mid)
    lines = stored.content.split("\n")
    assert lines[0] == "Procedure: brew_tea"
    assert lines[1] == "1. Boil water"
    assert lines[2] == "2. Steep bag"
    assert lines[3] == "3. Wait 3 min"


def test_remember_procedure_returns_distinct_ids(memory):
    mid1 = remember_procedure(
        memory,
        user_id="alice",
        name="proc_a",
        steps=["step 1"],
    )
    mid2 = remember_procedure(
        memory,
        user_id="alice",
        name="proc_b",
        steps=["step 2"],
    )
    assert mid1 != mid2


# ---------------------------------------------------------------------------
# recall_procedures tests
# ---------------------------------------------------------------------------

def test_recall_procedures_basic(memory):
    # Store a couple procedures and an unrelated semantic memory
    remember_procedure(
        memory,
        user_id="alice",
        name="password_reset",
        steps=["Ask for email", "Send reset link", "Confirm"],
    )
    remember_procedure(
        memory,
        user_id="alice",
        name="account_deletion",
        steps=["Verify identity", "Ask for confirmation", "Purge data"],
    )
    memory.remember("alice", "I like pizza", memory_type=MemoryType.SEMANTIC)

    results = recall_procedures(memory, "how do I reset my password?", user_id="alice", top_k=5)
    assert len(results) > 0
    assert all(r.memory_type == MemoryType.PROCEDURAL for r in results)
    # The password_reset procedure should rank highest for this query
    assert "password_reset" in results[0].tags


def test_recall_procedures_respects_top_k(memory):
    for i in range(5):
        remember_procedure(
            memory,
            user_id="alice",
            name=f"proc_{i}",
            steps=[f"Step {i}"],
        )
    results = recall_procedures(memory, "procedure", user_id="alice", top_k=3)
    assert len(results) == 3


def test_recall_procedures_filters_out_non_procedural(memory):
    remember_procedure(
        memory,
        user_id="alice",
        name="refund",
        steps=["Get order id", "Verify purchase", "Issue refund"],
    )
    # Semantic memory that happens to mention "refund"
    memory.remember("alice", "The refund policy allows 30 days", memory_type=MemoryType.SEMANTIC)

    results = recall_procedures(memory, "refund policy", user_id="alice", top_k=5)
    assert all(r.memory_type == MemoryType.PROCEDURAL for r in results)
    assert len(results) == 1


def test_recall_procedures_empty_query_raises(memory):
    with pytest.raises(ValueError, match="query cannot be empty"):
        recall_procedures(memory, "", user_id="alice")


def test_recall_procedures_invalid_top_k_raises(memory):
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        recall_procedures(memory, "query", user_id="alice", top_k=0)


def test_recall_procedures_defaults_to_system_user(memory):
    remember_procedure(
        memory,
        name="global_procedure",
        steps=["Step one", "Step two"],
    )
    results = recall_procedures(memory, "global_procedure", top_k=5)
    assert len(results) == 1
    assert results[0].user_id == "_system"


def test_recall_procedures_namespace_filter(memory):
    remember_procedure(
        memory,
        user_id="alice",
        name="ns_procedure",
        steps=["Do A", "Do B"],
        namespace="hr",
    )
    remember_procedure(
        memory,
        user_id="alice",
        name="default_procedure",
        steps=["Do C", "Do D"],
        namespace="default",
    )
    results = recall_procedures(memory, "procedure", user_id="alice", namespace="hr", top_k=5)
    assert len(results) == 1
    assert "ns_procedure" in results[0].tags


def test_recall_procedures_lifecycle_filter(memory):
    mid = remember_procedure(
        memory,
        user_id="alice",
        name="archived_proc",
        steps=["Old step"],
    )
    # Manually transition to ARCHIVED
    mem = memory._store.get(mid)
    mem.lifecycle_state = LifecycleState.ARCHIVED
    memory._store.update(mem)

    # Default filter includes ARCHIVED
    results = recall_procedures(memory, "archived", user_id="alice", top_k=5)
    assert len(results) == 1

    # Exclude ARCHIVED
    results_ex = recall_procedures(
        memory,
        "archived",
        user_id="alice",
        top_k=5,
        lifecycle_filter=[LifecycleState.ACTIVE, LifecycleState.DECAYING],
    )
    assert len(results_ex) == 0


def test_recall_procedures_no_matches_returns_empty(memory):
    results = recall_procedures(memory, "nonexistent workflow", user_id="alice", top_k=5)
    assert results == []


def test_recall_procedures_sorts_by_relevance(memory):
    remember_procedure(
        memory,
        user_id="alice",
        name="password_reset",
        steps=["Ask for email", "Send reset link"],
    )
    remember_procedure(
        memory,
        user_id="alice",
        name="change_avatar",
        steps=["Go to profile", "Upload image"],
    )
    results = recall_procedures(memory, "password reset flow", user_id="alice", top_k=5)
    assert len(results) == 2
    # password_reset should rank higher
    assert "password_reset" in results[0].tags
