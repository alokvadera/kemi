"""Tests for the pure pipeline steps extracted in Phase 9.

These tests demonstrate that the steps can be tested in isolation,
without constructing a full ``RetrievalContext`` / ``IngestionContext``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import pytest

from kemi.infra.webhooks import WebhookEventType
from kemi.memory import dedup
from kemi.memory.entities import RegexEntityLinker
from kemi.memory.model import (
    LifecycleState,
    MemoryObject,
    MemorySource,
    MemoryType,
)
from kemi.pipeline._steps import (
    adaptive_feedback,
    annotate_memory_for_ingestion,
    build_entity_boost_maps,
    handle_duplicate_resolution,
    search_and_filter_storage,
    update_lifecycle_inplace,
    validate_embedding_dimension,
)
from tests._helpers.factories import make_memory


def _hash_vec(text: str, dim: int) -> list[float]:
    raw = hashlib.sha256(text.encode()).digest()
    expanded = raw * (dim // len(raw) + 1)
    return [b / 255.0 for b in expanded[:dim]]


def _vec(*components: float, dim: int = 64) -> list[float]:
    """Build a dim-length vector from the given components, padding with zeros."""
    vec = list(components) + [0.0] * (dim - len(components))
    return vec[:dim]


def _make_memory(
    memory_id: str = "m-1",
    user_id: str = "alice",
    content: str = "hello world",
    embedding_dim: int = 64,
    namespace: str = "default",
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE,
) -> MemoryObject:
    return make_memory(
        memory_id=memory_id,
        user_id=user_id,
        content=content,
        embedding=_hash_vec(content, embedding_dim),
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        source=MemorySource.USER_STATED,
        lifecycle_state=lifecycle_state,
        metadata={},
        embedding_dim=embedding_dim,
        tags=[],
        memory_type=MemoryType.EPISODIC,
        session_id=None,
        namespace=namespace,
        version=1,
    )


class TestValidateEmbeddingDimension:
    def test_empty_results_is_no_op(self) -> None:
        validate_embedding_dimension([], current_dim=64)

    def test_matching_dim_passes(self) -> None:
        m = _make_memory(embedding_dim=64)
        validate_embedding_dimension([m], current_dim=64)

    def test_mismatched_dim_raises(self) -> None:
        m = _make_memory(embedding_dim=128)
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            validate_embedding_dimension([m], current_dim=64)

    def test_none_stored_dim_passes(self) -> None:
        """Some legacy memories may have a None stored dim — accept those."""
        m = _make_memory()
        m.embedding_dim = None
        validate_embedding_dimension([m], current_dim=64)


class TestBuildEntityBoostMaps:
    def test_disabled_returns_none_pair(self) -> None:
        result = build_entity_boost_maps(
            query="alice loves python",
            search_results=[_make_memory()],
            enable_entity_boost=False,
            entity_linker=RegexEntityLinker(),
        )
        assert result == (None, None)

    def test_enabled_uses_cached_metadata(self) -> None:
        linker = RegexEntityLinker()
        m = _make_memory()
        m.metadata["extracted_entities"] = ["python", "rust"]
        result = build_entity_boost_maps(
            query="alice",
            search_results=[m],
            enable_entity_boost=True,
            entity_linker=linker,
        )
        query_entities, memory_map = result
        assert query_entities is not None
        assert memory_map is not None
        assert memory_map[m.memory_id] == {"python", "rust"}

    def test_enabled_extracts_when_unscached(self) -> None:
        linker = RegexEntityLinker()
        m = _make_memory(content="alice likes python")
        # No cached metadata
        result = build_entity_boost_maps(
            query="alice",
            search_results=[m],
            enable_entity_boost=True,
            entity_linker=linker,
        )
        query_entities, memory_map = result
        assert query_entities is not None
        assert m.memory_id in memory_map


class TestUpdateLifecycleInplace:
    def test_active_remains_active(self) -> None:
        m = _make_memory(lifecycle_state=LifecycleState.ACTIVE)
        store = _InMemoryStore()
        update_lifecycle_inplace(
            results=[m], decay_threshold_hours=24.0, store=store, metrics=None
        )
        assert m.lifecycle_state == LifecycleState.ACTIVE
        assert store.updates == []  # no transitions

    def test_inplace_mutation_bumps_accessed_at(self) -> None:
        m = _make_memory()
        before = m.last_accessed_at
        store = _InMemoryStore()
        update_lifecycle_inplace(
            results=[m], decay_threshold_hours=24.0, store=store, metrics=None
        )
        assert m.last_accessed_at >= before

    def test_metrics_incremented_on_transition(self) -> None:
        m = _make_memory(lifecycle_state=LifecycleState.DECAYING)
        # Real-world behaviour: the function bumps `last_accessed_at`
        # to `now` *before* evaluating lifecycle, so a memory that
        # was DECAYING gets re-evaluated to ACTIVE on next access.
        # This is a deliberate design choice: re-accessing a memory
        # should make it active again.
        store = _InMemoryStore()
        metrics = _MetricsSpy()
        update_lifecycle_inplace(
            results=[m],
            decay_threshold_hours=1.0,
            store=store,
            metrics=metrics,
        )
        # The DECAYING → ACTIVE transition fires.
        assert len(store.updates) == 1
        assert metrics.transitions == 1
        # And the new state on the stored object is ACTIVE.
        assert store.updates[0].lifecycle_state == LifecycleState.ACTIVE

    def test_no_transition_when_state_already_correct(self) -> None:
        m = _make_memory(lifecycle_state=LifecycleState.ACTIVE)
        # If a memory is ACTIVE and accessed now, evaluate_lifecycle
        # returns ACTIVE — no transition.
        store = _InMemoryStore()
        metrics = _MetricsSpy()
        update_lifecycle_inplace(
            results=[m],
            decay_threshold_hours=1.0,
            store=store,
            metrics=metrics,
        )
        assert store.updates == []
        assert metrics.transitions == 0


class TestAdaptiveFeedback:
    def test_none_retriever_is_noop(self) -> None:
        adaptive_feedback(None, "alice", "query")  # no exception

    def test_exception_is_swallowed(self) -> None:
        class BadRetriever:
            def analyze_query(self, q: str) -> Any:
                raise RuntimeError("nope")

            def record_feedback(self, u: str, q: str, p: Any) -> None:
                pass

        adaptive_feedback(BadRetriever(), "alice", "query")  # no exception

    def test_happy_path_calls_both(self) -> None:
        spy = _AdaptiveSpy()
        adaptive_feedback(spy, "alice", "query")
        assert spy.analyze_count == 1
        assert spy.record_count == 1


class TestSearchAndFilterStorage:
    def test_no_metadata_filter_returns_raw_results(self) -> None:
        m1 = _make_memory(memory_id="m-1")
        m2 = _make_memory(memory_id="m-2")
        store = _SearchStore([m1, m2])
        results = search_and_filter_storage(
            store=store,
            user_id="alice",
            query_embedding=[1.0, 0.0],
            top_k=5,
            lifecycle_filter=[LifecycleState.ACTIVE],
            namespace="default",
            session_id=None,
            fetch_multiplier=3,
            metadata_filter=None,
        )
        assert len(results) == 2
        assert store.search_calls[0]["top_k"] == 15  # 5 * 3

    def test_metadata_filter_excludes_non_matching(self) -> None:
        m1 = _make_memory(memory_id="m-1")
        m1.metadata["source"] = "form"
        m2 = _make_memory(memory_id="m-2")
        m2.metadata["source"] = "chat"
        store = _SearchStore([m1, m2])
        results = search_and_filter_storage(
            store=store,
            user_id="alice",
            query_embedding=[1.0, 0.0],
            top_k=5,
            lifecycle_filter=[LifecycleState.ACTIVE],
            namespace="default",
            session_id=None,
            fetch_multiplier=3,
            metadata_filter={"source": "form"},
        )
        assert len(results) == 1
        assert results[0].memory_id == "m-1"

    def test_metadata_filter_all_must_match(self) -> None:
        m1 = _make_memory(memory_id="m-1")
        m1.metadata["source"] = "form"
        m1.metadata["type"] = "user"
        m2 = _make_memory(memory_id="m-2")
        m2.metadata["source"] = "form"
        m2.metadata["type"] = "agent"
        store = _SearchStore([m1, m2])
        results = search_and_filter_storage(
            store=store,
            user_id="alice",
            query_embedding=[1.0, 0.0],
            top_k=5,
            lifecycle_filter=[LifecycleState.ACTIVE],
            namespace="default",
            session_id=None,
            fetch_multiplier=3,
            metadata_filter={"source": "form", "type": "user"},
        )
        assert len(results) == 1
        assert results[0].memory_id == "m-1"


class TestAnnotateMemoryForIngestion:
    def test_no_conflict_no_entity_boost(self) -> None:
        m = _make_memory(content="xyz unique")
        others = [_make_memory(memory_id="m-2", content="completely different")]
        detected, conflicts = annotate_memory_for_ingestion(
            memory=m,
            existing_memories=others,
            conflict_threshold=0.99,
            dedup_threshold=0.5,
            enable_entity_boost=False,
            entity_linker=RegexEntityLinker(),
            metrics=None,
        )
        assert not detected
        assert conflicts == []
        assert "conflict_flagged" not in m.metadata
        assert "extracted_entities" not in m.metadata

    def test_conflict_detected_no_entity_boost(self) -> None:
        m = _make_memory(content="new")
        m.embedding = _vec(1.0, 0.0)
        other = _make_memory(memory_id="m-2", content="existing")
        other.embedding = _vec(0.5, 0.866_025_4)
        detected, conflicts = annotate_memory_for_ingestion(
            memory=m,
            existing_memories=[other],
            conflict_threshold=0.65,
            dedup_threshold=0.85,
            enable_entity_boost=False,
            entity_linker=RegexEntityLinker(),
            metrics=None,
        )
        assert detected
        assert len(conflicts) == 1
        assert m.metadata.get("conflict_flagged") is True
        assert "extracted_entities" not in m.metadata

    def test_entity_boost_enabled_no_conflict(self) -> None:
        # Capitalised words so RegexEntityLinker extracts them
        m = _make_memory(content="Alice likes Python")
        others = [_make_memory(memory_id="m-2", content="completely different")]
        detected, conflicts = annotate_memory_for_ingestion(
            memory=m,
            existing_memories=others,
            conflict_threshold=0.99,
            dedup_threshold=0.5,
            enable_entity_boost=True,
            entity_linker=RegexEntityLinker(),
            metrics=None,
        )
        assert not detected
        assert conflicts == []
        assert "extracted_entities" in m.metadata
        assert "python" in m.metadata["extracted_entities"]

    def test_metrics_incremented_on_conflict(self) -> None:
        m = _make_memory(content="new")
        m.embedding = _vec(1.0, 0.0)
        other = _make_memory(memory_id="m-2", content="existing")
        other.embedding = _vec(0.5, 0.866_025_4)
        metrics = _ConflictMetricsSpy()
        detected, conflicts = annotate_memory_for_ingestion(
            memory=m,
            existing_memories=[other],
            conflict_threshold=0.65,
            dedup_threshold=0.85,
            enable_entity_boost=False,
            entity_linker=RegexEntityLinker(),
            metrics=metrics,
        )
        assert detected
        assert metrics.conflicts == 1


class TestHandleDuplicateResolution:
    def test_resolves_into_canonical(self) -> None:
        m = _make_memory(memory_id="m-new", content="new text")
        canonical = _make_memory(memory_id="m-canon", content="old text")
        store = _InMemoryStore()
        resolved = handle_duplicate_resolution(
            memory=m,
            duplicates=[canonical],
            store=store,
            get_version_store=lambda: _VersionStore(),
            auto_prune_versions=lambda _: None,
            dispatch_webhook=_WebhookSpy(),
            track_operation=_TrackSpy(),
            metrics=None,
            audit_batch=None,
        )
        assert resolved.memory_id == "m-canon"
        assert resolved.content == "new text"
        assert len(store.updates) == 1
        assert store.updates[0].memory_id == "m-canon"

    def test_removes_extra_duplicates(self) -> None:
        m = _make_memory(memory_id="m-new", content="new")
        canon = _make_memory(memory_id="m-canon", content="old")
        extra1 = _make_memory(memory_id="m-extra1", content="old")
        extra2 = _make_memory(memory_id="m-extra2", content="old")
        store = _InMemoryStore()
        handle_duplicate_resolution(
            memory=m,
            duplicates=[canon, extra1, extra2],
            store=store,
            get_version_store=lambda: None,
            auto_prune_versions=lambda _: None,
            dispatch_webhook=lambda *a, **k: None,
            track_operation=lambda *a, **k: None,
            metrics=None,
            audit_batch=None,
        )
        assert set(store.deleted) == {"m-extra1", "m-extra2"}

    def test_dispatches_webhook_and_tracks(self) -> None:
        m = _make_memory(memory_id="m-new", content="new")
        canonical = _make_memory(memory_id="m-canon", content="old")
        store = _InMemoryStore()
        webhooks = _WebhookSpy()
        tracks = _TrackSpy()
        handle_duplicate_resolution(
            memory=m,
            duplicates=[canonical],
            store=store,
            get_version_store=lambda: None,
            auto_prune_versions=lambda _: None,
            dispatch_webhook=webhooks,
            track_operation=tracks,
            metrics=None,
            audit_batch=None,
        )
        assert len(webhooks.calls) == 1
        assert webhooks.calls[0][0][0] == WebhookEventType.UPDATED
        assert len(tracks.calls) == 1
        assert tracks.calls[0][0][0] == "remember"

    def test_metrics_incremented(self) -> None:
        m = _make_memory(memory_id="m-new", content="new")
        canonical = _make_memory(memory_id="m-canon", content="old")
        store = _InMemoryStore()
        metrics = _ConflictMetricsSpy()
        handle_duplicate_resolution(
            memory=m,
            duplicates=[canonical],
            store=store,
            get_version_store=lambda: None,
            auto_prune_versions=lambda _: None,
            dispatch_webhook=lambda *a, **k: None,
            track_operation=lambda *a, **k: None,
            metrics=metrics,
            audit_batch=None,
        )
        assert metrics.duplicates == 1

    def test_records_version_when_store_available(self) -> None:
        m = _make_memory(memory_id="m-new", content="new")
        canonical = _make_memory(memory_id="m-canon", content="old")
        store = _InMemoryStore()
        vs = _VersionStore()
        handle_duplicate_resolution(
            memory=m,
            duplicates=[canonical],
            store=store,
            get_version_store=lambda: vs,
            auto_prune_versions=lambda _: None,
            dispatch_webhook=lambda *a, **k: None,
            track_operation=lambda *a, **k: None,
            metrics=None,
            audit_batch=None,
        )
        assert len(vs.versions) == 1
        assert vs.versions[0][1] == "merge"

    def test_skips_version_on_none_store(self) -> None:
        m = _make_memory(memory_id="m-new", content="new")
        canonical = _make_memory(memory_id="m-canon", content="old")
        store = _InMemoryStore()
        vs = _VersionStore()
        handle_duplicate_resolution(
            memory=m,
            duplicates=[canonical],
            store=store,
            get_version_store=lambda: None,
            auto_prune_versions=lambda _: None,
            dispatch_webhook=lambda *a, **k: None,
            track_operation=lambda *a, **k: None,
            metrics=None,
            audit_batch=None,
        )
        assert len(vs.versions) == 0

    def test_audit_batch_forwarded_to_track_operation(self) -> None:
        m = _make_memory(memory_id="m-new", content="new")
        canonical = _make_memory(memory_id="m-canon", content="old")
        store = _InMemoryStore()
        tracks = _TrackSpy()
        audit_batch: list[dict[str, Any]] = [{"op": "batch"}]
        handle_duplicate_resolution(
            memory=m,
            duplicates=[canonical],
            store=store,
            get_version_store=lambda: None,
            auto_prune_versions=lambda _: None,
            dispatch_webhook=lambda *a, **k: None,
            track_operation=tracks,
            metrics=None,
            audit_batch=audit_batch,
        )
        assert len(tracks.calls) == 1
        # audit_batch is passed as a keyword argument
        assert tracks.calls[0][1].get("audit_batch") is audit_batch


class TestFindDuplicates:
    def test_no_duplicates_when_below_threshold(self) -> None:
        m = _make_memory(content="xyz unique")
        others = [_make_memory(memory_id="m-2", content="completely different")]
        dups = dedup.find_duplicates(m, others, threshold=0.99)
        assert dups == []

    def test_finds_exact_duplicate(self) -> None:
        m = _make_memory(content="identical text")
        other = _make_memory(memory_id="m-2", content="identical text")
        dups = dedup.find_duplicates(m, [other], threshold=0.99)
        assert len(dups) == 1
        assert dups[0].memory_id == "m-2"


class TestFindConflicts:
    def test_no_conflicts_when_below_threshold(self) -> None:
        m = _make_memory(content="xyz unique")
        others = [_make_memory(memory_id="m-2", content="completely different")]
        conflicts = dedup.find_conflicts(m, others, conflict_threshold=0.99, dedup_threshold=0.5)
        assert conflicts == []

    def test_finds_single_conflict_in_range(self) -> None:
        """A memory with normalized similarity 0.75 (cos_sim 0.5) is a conflict."""
        m = _make_memory(content="new")
        m.embedding = _vec(1.0, 0.0)
        other = _make_memory(memory_id="m-2", content="existing")
        other.embedding = _vec(0.5, 0.866_025_4)
        conflicts = dedup.find_conflicts(m, [other], conflict_threshold=0.65, dedup_threshold=0.85)
        assert len(conflicts) == 1
        assert conflicts[0].memory_id == "m-2"

    def test_finds_multiple_conflicts(self) -> None:
        m = _make_memory(content="new")
        m.embedding = _vec(1.0, 0.0)
        c1 = _make_memory(memory_id="m-c1", content="c1")
        c1.embedding = _vec(0.5, 0.866_025_4)  # normalized ~0.75
        c2 = _make_memory(memory_id="m-c2", content="c2")
        c2.embedding = _vec(0.6, 0.8)  # normalized ~0.8
        conflicts = dedup.find_conflicts(m, [c1, c2], conflict_threshold=0.65, dedup_threshold=0.85)
        assert len(conflicts) == 2
        assert {c.memory_id for c in conflicts} == {"m-c1", "m-c2"}

    def test_excludes_exact_duplicate(self) -> None:
        """Identical embeddings (normalized similarity = 1.0) are duplicates, not conflicts."""
        m = _make_memory(content="new")
        m.embedding = _vec(1.0, 0.0)
        dup = _make_memory(memory_id="m-dup", content="dup")
        dup.embedding = _vec(1.0, 0.0)
        conflicts = dedup.find_conflicts(m, [dup], conflict_threshold=0.65, dedup_threshold=0.85)
        assert conflicts == []

    def test_excludes_unrelated_below_threshold(self) -> None:
        """Orthogonal embeddings (normalized similarity = 0.5) are below conflict_threshold."""
        m = _make_memory(content="new")
        m.embedding = _vec(1.0, 0.0)
        unrelated = _make_memory(memory_id="m-unrelated", content="unrelated")
        unrelated.embedding = _vec(0.0, 1.0)
        conflicts = dedup.find_conflicts(m, [unrelated], conflict_threshold=0.65, dedup_threshold=0.85)  # noqa: E501
        assert conflicts == []

    def test_empty_existing_returns_empty(self) -> None:
        m = _make_memory(content="new")
        m.embedding = _vec(1.0, 0.0)
        conflicts = dedup.find_conflicts(m, [], conflict_threshold=0.65, dedup_threshold=0.85)
        assert conflicts == []

    def test_none_embedding_returns_empty(self) -> None:
        m = _make_memory(content="new")
        m.embedding = None
        other = _make_memory(memory_id="m-2", content="existing")
        other.embedding = _vec(0.5, 0.866_025_4)
        conflicts = dedup.find_conflicts(m, [other], conflict_threshold=0.65, dedup_threshold=0.85)
        assert conflicts == []

    def test_other_none_embedding_skipped(self) -> None:
        m = _make_memory(content="new")
        m.embedding = _vec(1.0, 0.0)
        other = _make_memory(memory_id="m-2", content="existing")
        other.embedding = None
        conflicts = dedup.find_conflicts(m, [other], conflict_threshold=0.65, dedup_threshold=0.85)
        assert conflicts == []

    def test_boundary_at_conflict_threshold_is_excluded(self) -> None:
        """Normalized similarity exactly equal to conflict_threshold is NOT a conflict (< vs <=)."""
        m = _make_memory(content="new")
        m.embedding = _vec(1.0, 0.0)
        other = _make_memory(memory_id="m-2", content="existing")
        other.embedding = _vec(0.5, 0.0)  # cos_sim = 0.5, normalized = 0.75
        # Use threshold 0.75 so 0.75 is NOT strictly greater
        conflicts = dedup.find_conflicts(m, [other], conflict_threshold=0.75, dedup_threshold=0.85)
        assert conflicts == []

    def test_boundary_at_dedup_threshold_is_excluded(self) -> None:
        """Normalized similarity exactly equal to dedup_threshold is NOT a conflict (< vs <=)."""
        m = _make_memory(content="new")
        m.embedding = _vec(1.0, 0.0)
        other = _make_memory(memory_id="m-2", content="existing")
        other.embedding = _vec(0.5, 0.0)  # cos_sim = 0.5, normalized = 0.75
        # Use dedup_threshold 0.75 so 0.75 is NOT strictly less
        conflicts = dedup.find_conflicts(m, [other], conflict_threshold=0.65, dedup_threshold=0.75)
        assert conflicts == []


class TestResolveDuplicate:
    def test_resolve_preserves_canonical_id(self) -> None:
        m = _make_memory(memory_id="m-new", content="similar text")
        canonical = _make_memory(memory_id="m-canon", content="similar text")
        resolved = dedup.resolve_duplicate(m, canonical)
        assert resolved.memory_id == "m-canon"


# ---------- test helpers ----------


class _InMemoryStore:
    def __init__(self) -> None:
        self.updates: list[MemoryObject] = []
        self.deleted: list[str] = []

    def update(self, memory: MemoryObject) -> None:
        self.updates.append(memory)

    def update_many(self, memories: list[MemoryObject]) -> int:
        self.updates.extend(memories)
        return len(memories)

    def delete_by_id(self, memory_id: str) -> None:
        self.deleted.append(memory_id)


class _SearchStore:
    def __init__(self, memories: list[MemoryObject] | None = None) -> None:
        self.memories = memories or []
        self.search_calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[MemoryObject]:
        self.search_calls.append(kwargs)
        return list(self.memories)


class _MetricsSpy:
    def __init__(self) -> None:
        self.transitions = 0
        outer = self

        class _Counter:
            def inc(self, n: int = 1) -> None:
                outer.transitions += n

        self.lifecycle_transitions = _Counter()


class _ConflictMetricsSpy:
    def __init__(self) -> None:
        self.conflicts = 0
        self.duplicates = 0
        outer = self

        class _Counter:
            def __init__(self, attr: str) -> None:
                self._attr = attr

            def inc(self, n: int = 1) -> None:
                setattr(outer, self._attr, getattr(outer, self._attr) + n)

        self.conflicts_detected = _Counter("conflicts")
        self.duplicates_detected = _Counter("duplicates")


class _WebhookSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class _TrackSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class _VersionStore:
    def __init__(self) -> None:
        self.versions: list[tuple[MemoryObject, str]] = []

    def record_version(self, memory: MemoryObject, *, changed_by: str) -> None:
        self.versions.append((memory, changed_by))


class _AdaptiveSpy:
    def __init__(self) -> None:
        self.analyze_count = 0
        self.record_count = 0

    def analyze_query(self, q: str) -> dict[str, Any]:
        self.analyze_count += 1
        return {"q": q}

    def record_feedback(self, u: str, q: str, p: Any) -> None:
        self.record_count += 1
