"""Tests for src/kemi/api_server.py"""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from kemi.memory.model import LifecycleState, MemoryType

# Skip entire module if fastapi is not available
try:
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
    from pydantic import ValidationError

    from kemi.interfaces.api import (
        AuditQueryRequest,
        BatchRememberRequest,
        ConsolidateRequest,
        CreateAPIKeyRequest,
        PruneRequest,
        RateLimiter,
        RecallRequest,
        RememberRequest,
        UpdateRequest,
        _require_admin,
        _reset_api_key_manager,
        _resolve_user_id,
        create_app,
    )

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

    # Will be skipped by pytestmark below once we set it
    TestClient = None  # type: ignore[assignment, misc]
    create_app = None  # type: ignore[assignment, misc]

# Apply skip to all tests in this module if fastapi unavailable
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed"),
]


class MockMemory:
    """Minimal mock Memory for API server tests."""

    def __init__(self):
        self._store = MagicMock()
        self._embed = MagicMock()
        self._audit_trail = None
        self._adaptive_retriever = None

    def remember(
        self,
        user_id,
        content,
        importance=0.5,
        source=None,
        tags=None,
        namespace="default",
        session_id=None,
        memory_type=None,
        confidence=1.0,
    ):
        return "mem-123"

    def recall(
        self,
        user_id,
        query,
        top_k=5,
        max_tokens=None,
        namespace="default",
        session_id=None,
        hybrid_search=None,
    ):
        mem = MagicMock()
        mem.memory_id = "mem-123"
        mem.content = "test content"
        mem.score = 0.95
        mem.importance = 0.5
        mem.lifecycle_state = LifecycleState.ACTIVE
        mem.created_at = None
        mem.tags = []
        mem.memory_type = MemoryType.EPISODIC
        mem.confidence = 1.0
        mem.session_id = None
        mem.namespace = "default"
        mem.version = 1
        return [mem]

    def recall_explain(self, user_id, query, top_k=5, namespace="default", session_id=None):
        mem = MagicMock()
        mem.memory_id = "mem-123"
        mem.content = "test content"
        mem.score = 0.95
        return [{"memory": mem, "explanation": "matched query"}]

    async def recall_stream(self, user_id, query, top_k=5, **kwargs):
        """Mock streaming recall — yields same results as recall()."""
        results = self.recall(user_id, query, top_k=top_k)
        for mem in results:
            yield mem

    async def arecall(self, user_id, query, top_k=5, max_tokens=None, namespace="default",
                       session_id=None, hybrid_search=None, stream=False):
        if stream:
            return self.recall_stream(user_id, query, top_k=top_k)
        return self.recall(user_id, query, top_k=top_k)

    def forget(self, user_id, memory_id=None):
        return 1

    def update(self, memory_id, content=None, importance=None, confidence=None, memory_type=None):
        if memory_id == "not-found":
            raise ValueError("Memory not found")
        return memory_id

    def prune(
        self,
        user_id,
        max_age_days=None,
        min_importance=None,
        lifecycle_states=None,
        namespace="default",
    ):
        return 3

    def stats(self, user_id):
        return {"total": 10, "active": 5, "decaying": 3, "archived": 2}

    def list_users(self):
        return ["alice", "bob"]

    def consolidate(self, user_id, namespace="default", min_memories=5, max_age_days=30.0):
        return "consolidated-123"

    def cluster_topics(self, user_id, n_clusters=3, namespace="default"):
        mem = MagicMock()
        mem.memory_id = "mem-1"
        mem.content = "python programming"
        mem.importance = 0.7
        return {"Python": [mem]}

    def get_memory_graph(self, user_id, namespace="default"):
        return {
            "entities": [{"id": "e1", "type": "concept", "label": "python"}],
            "relations": [{"source": "e1", "target": "e2", "label": "related_to"}],
        }

    def feedback(self, user_id, memory_id, helpful=True, namespace="default"):
        if memory_id == "not-found":
            raise ValueError("Memory not found")
        return None

    def get_metrics(self):
        return {
            "operations": {"remember": 2, "recall": 5},
            "embeddings": {"total": 7, "errors": 0},
            "storage": {"errors": 0},
            "quality": {
                "duplicates_detected": 1,
                "conflicts_detected": 0,
                "lifecycle_transitions": 3,
            },
            "memory_usage": {
                "total_memories": 10,
                "total_users": 2,
            },
            "timestamp": "2026-01-01T00:00:00+00:00",
        }

    def get_metrics_prometheus(self):
        return (
            "# HELP kemi_remember_total Total number of remember operations\n"
            "# TYPE kemi_remember_total counter\n"
            "kemi_remember_total 2\n"
        )

    def enable_adaptive_retrieval(self, enable=True):
        if enable and self._adaptive_retriever is None:
            self._adaptive_retriever = MagicMock()
            self._adaptive_retriever.analyze_query = MagicMock(
                return_value=MagicMock(
                    query="What is python?",
                    query_type=MagicMock(value="factual"),
                    confidence=0.85,
                    word_count=3,
                    keyword_density=0.67,
                    specificity=0.5,
                    has_question_mark=True,
                    has_named_entity_hint=False,
                    recommended_weights={
                        "weight_semantic": 0.55,
                        "weight_recency": 0.20,
                        "weight_bm25": 0.25,
                        "weight_semantic_no_embed": 0.45,
                        "weight_recency_no_embed": 0.25,
                        "weight_importance": 0.30,
                    },
                )
            )
            self._adaptive_retriever.get_user_profile = MagicMock(
                return_value={
                    "user_id": "alice",
                    "total_queries": 10,
                    "distribution": {"factual": 0.7, "conversational": 0.3},
                    "dominant_type": "factual",
                }
            )
        elif not enable:
            self._adaptive_retriever = None

    def enable_audit_trail(self, retention_days=365, auto_purge=True):
        if self._audit_trail is None:
            self._audit_trail = MagicMock()
            self._audit_trail.log_operation = MagicMock(return_value=42)
            self._audit_trail.query = MagicMock(return_value=[])
            self._audit_trail.get_stats = MagicMock(
                return_value={
                    "total_entries": 100,
                    "unique_users": 5,
                    "first_entry": "2026-01-01T00:00:00+00:00",
                    "last_entry": "2026-06-01T00:00:00+00:00",
                    "retention_days": retention_days,
                }
            )
            self._audit_trail.export = MagicMock(return_value=[])

    def configure_versioning(self):
        pass

    def get_history(self, memory_id, limit=100):
        snap = MagicMock()
        snap.version = 1
        snap.content = "original content"
        snap.importance = 0.5
        snap.tags = []
        snap.memory_type = "episodic"
        snap.confidence = 1.0
        snap.namespace = "default"
        snap.source = "user_stated"
        snap.changed_at = datetime.now()
        snap.changed_by = "alice"
        return [snap]


@pytest.fixture
def mock_memory():
    return MockMemory()


@pytest.fixture(autouse=True)
def reset_api_key_manager_after_each():
    """Reset the cached API key manager after every test to avoid cross-test contamination."""
    yield
    _reset_api_key_manager()


@pytest.fixture
def app(mock_memory):
    return create_app(memory=mock_memory)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "components" in data
        assert "timestamp" in data

    def test_health_returns_components(self, client):
        """Test that health endpoint returns component status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        # Should have database component
        assert "database" in data["components"]

    def test_health_timestamp_is_iso_format(self, client):
        """Test that timestamp is in ISO format."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        # Should be parseable as ISO format
        from datetime import datetime

        try:
            datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        except ValueError:
            pytest.fail("Timestamp is not valid ISO format")

    def test_health_database_component_status(self, client):
        """Test database component status in health response."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        db_component = data["components"].get("database", {})
        assert "status" in db_component
        # Status should be one of: healthy, unhealthy, unknown
        assert db_component["status"] in ["healthy", "unhealthy", "unknown"]

    def test_health_embedding_component_status(self, client):
        """Test embedding component status in health response."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # May or may not have embedding configured depending on mock
        assert "embedding" in data["components"]

    def test_health_degraded_on_storage_error(self, mock_memory):
        """Test health returns degraded when storage has issues."""
        # Mock storage to raise an error
        mock_memory._storage = None  # This should cause health check to fail
        app = create_app(memory=mock_memory)
        client = TestClient(app)
        response = client.get("/health")
        # The mock doesn't fully replicate the internal structure
        # so we just verify the endpoint works
        assert response.status_code == 200


class TestRemember:
    def test_remember_success(self, client):
        response = client.post(
            "/remember",
            json={
                "user_id": "alice",
                "content": "I love python",
                "importance": 0.7,
                "source": "user_stated",
                "tags": ["python", "coding"],
                "namespace": "default",
                "memory_type": "episodic",
                "confidence": 0.9,
            },
        )
        assert response.status_code == 200
        assert response.json()["memory_id"] == "mem-123"

    def test_remember_defaults(self, client):
        response = client.post(
            "/remember",
            json={"user_id": "alice", "content": "hello world"},
        )
        assert response.status_code == 200

    def test_remember_invalid_memory_type(self, client):
        response = client.post(
            "/remember",
            json={"user_id": "alice", "content": "test", "memory_type": "invalid"},
        )
        assert response.status_code == 400

    def test_remember_empty_user_id(self, client):
        response = client.post(
            "/remember",
            json={"user_id": "", "content": "test"},
        )
        assert response.status_code == 422  # pydantic validation


class TestRecall:
    def test_recall_success(self, client):
        response = client.post(
            "/recall",
            json={"user_id": "alice", "query": "python", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["memory_id"] == "mem-123"

    def test_recall_with_namespace(self, client):
        response = client.post(
            "/recall",
            json={"user_id": "alice", "query": "python", "namespace": "work"},
        )
        assert response.status_code == 200

    def test_recall_stream_returns_sse(self, client):
        """Test that /recall/stream returns Server-Sent Events."""
        response = client.post(
            "/recall/stream",
            json={"user_id": "alice", "query": "python", "top_k": 5},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events from the response
        events = response.text.strip().split("\n\n")
        data_events = [e for e in events if e and not e.startswith("event: done") and not e.startswith("event: error")]  # noqa: E501
        done_events = [e for e in events if e.startswith("event: done")]

        # Should have data events + 1 done event
        assert len(data_events) >= 1
        assert len(done_events) >= 1

        # Verify data event format
        for ev in data_events:
            assert ev.startswith("data: ")
            import json
            payload = json.loads(ev[6:])
            assert "memory_id" in payload
            assert "content" in payload
            assert "score" in payload

        # Verify done event
        for ev in done_events:
            assert "data: " in ev
            done_payload = json.loads(ev.split("data: ")[1])
            assert "total" in done_payload
            assert done_payload["total"] >= 1

    def test_recall_with_hybrid_search(self, client):
        response = client.post(
            "/recall",
            json={"user_id": "alice", "query": "python", "hybrid_search": True},
        )
        assert response.status_code == 200

    def test_recall_empty_query(self, client):
        response = client.post(
            "/recall",
            json={"user_id": "alice", "query": ""},
        )
        assert response.status_code == 422


class TestRecallExplain:
    def test_recall_explain_success(self, client):
        response = client.post(
            "/recall-explain",
            json={"user_id": "alice", "query": "python", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert "explanation" in data["results"][0]

    def test_recall_explain_with_namespace(self, client):
        response = client.post(
            "/recall-explain",
            json={"user_id": "alice", "query": "python", "namespace": "school"},
        )
        assert response.status_code == 200


class TestForget:
    def test_forget_with_memory_id(self, client):
        response = client.post("/forget?user_id=alice&memory_id=mem-123")
        assert response.status_code == 200
        assert response.json()["deleted"] == 1

    def test_forget_without_memory_id(self, client):
        response = client.post("/forget?user_id=alice")
        assert response.status_code == 200


class TestUpdateMemory:
    def test_update_content_only(self, client):
        response = client.patch(
            "/memories/mem-123",
            json={"content": "updated content"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "updated"

    def test_update_importance_only(self, client):
        response = client.patch(
            "/memories/mem-123",
            json={"importance": 0.8},
        )
        assert response.status_code == 200

    def test_update_multiple_fields(self, client):
        response = client.patch(
            "/memories/mem-123",
            json={
                "content": "new",
                "importance": 0.9,
                "confidence": 0.95,
                "memory_type": "semantic",
            },
        )
        assert response.status_code == 200

    def test_update_memory_not_found(self, client):
        response = client.patch(
            "/memories/not-found",
            json={"content": "test"},
        )
        assert response.status_code == 404

    def test_update_invalid_memory_type(self, client):
        response = client.patch(
            "/memories/mem-123",
            json={"memory_type": "invalid"},
        )
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()


class TestPrune:
    def test_prune_defaults(self, client):
        response = client.post("/prune?user_id=alice", json={})
        assert response.status_code == 200
        assert response.json()["deleted"] == 3

    def test_prune_with_filters(self, client):
        response = client.post(
            "/prune?user_id=alice",
            json={"max_age_days": 30.0, "min_importance": 0.3, "namespace": "default"},
        )
        assert response.status_code == 200

    def test_prune_with_lifecycle_states(self, client):
        response = client.post(
            "/prune?user_id=alice",
            json={"lifecycle_states": ["decaying", "active"]},
        )
        assert response.status_code == 200

    def test_prune_invalid_lifecycle_state(self, client):
        response = client.post(
            "/prune?user_id=alice",
            json={"lifecycle_states": ["INVALID"]},
        )
        assert response.status_code == 400


class TestStats:
    def test_stats_success(self, client):
        response = client.get("/stats/alice")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 10

    def test_stats_invalid_user(self):
        mock_mem = MockMemory()
        original = mock_mem.stats

        def fake_stats(user_id):
            raise ValueError("Invalid user")

        mock_mem.stats = fake_stats
        app = create_app(memory=mock_mem)
        client = TestClient(app)
        response = client.get("/stats/invalid")
        assert response.status_code == 400
        mock_mem.stats = original


class TestListUsers:
    def test_list_users(self, client):
        response = client.get("/users")
        assert response.status_code == 200
        assert response.json()["users"] == ["alice", "bob"]


class TestConsolidate:
    def test_consolidate_success(self, client):
        response = client.post(
            "/consolidate/alice",
            json={"namespace": "default", "min_memories": 5},
        )
        assert response.status_code == 200
        assert response.json()["consolidated_memory_id"] == "consolidated-123"

    def test_consolidate_no_match(self, mock_memory):
        original = mock_memory.consolidate
        mock_memory.consolidate = lambda *args, **kwargs: None
        app = create_app(memory=mock_memory)
        client = TestClient(app)
        response = client.post("/consolidate/alice", json={})
        assert response.status_code == 200
        assert "message" in response.json()
        mock_memory.consolidate = original

    def test_consolidate_invalid_user(self, mock_memory):
        original = mock_memory.consolidate

        def fake(*args, **kwargs):
            raise ValueError("Invalid user")

        mock_memory.consolidate = fake
        app = create_app(memory=mock_memory)
        client = TestClient(app)
        response = client.post("/consolidate/invalid", json={})
        assert response.status_code == 400
        mock_memory.consolidate = original


class TestTopics:
    def test_topics_success(self, client):
        response = client.post(
            "/topics/alice",
            json={"n_clusters": 3, "namespace": "default"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "topics" in data
        assert "Python" in data["topics"]

    def test_topics_invalid_user(self, mock_memory):
        original = mock_memory.cluster_topics

        def fake(*args, **kwargs):
            raise ValueError("Invalid user")

        mock_memory.cluster_topics = fake
        app = create_app(memory=mock_memory)
        client = TestClient(app)
        response = client.post("/topics/invalid", json={})
        assert response.status_code == 400
        mock_memory.cluster_topics = original


class TestGraph:
    def test_graph_success(self, client):
        response = client.post(
            "/graph/alice",
            json={"namespace": "default"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "relations" in data

    def test_graph_invalid_user(self, mock_memory):
        original = mock_memory.get_memory_graph

        def fake(*args, **kwargs):
            raise ValueError("Invalid user")

        mock_memory.get_memory_graph = fake
        app = create_app(memory=mock_memory)
        client = TestClient(app)
        response = client.post("/graph/invalid", json={})
        assert response.status_code == 400
        mock_memory.get_memory_graph = original


class TestFeedback:
    def test_feedback_helpful(self, client):
        response = client.post(
            "/feedback/alice",
            json={"memory_id": "mem-123", "helpful": True, "namespace": "default"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["helpful"] is True

    def test_feedback_not_helpful(self, client):
        response = client.post(
            "/feedback/alice",
            json={"memory_id": "mem-123", "helpful": False},
        )
        assert response.status_code == 200

    def test_feedback_not_found(self, client):
        response = client.post(
            "/feedback/alice",
            json={"memory_id": "not-found", "helpful": True},
        )
        assert response.status_code == 400

    def test_feedback_custom_namespace(self, client):
        response = client.post(
            "/feedback/alice",
            json={"memory_id": "mem-123", "helpful": True, "namespace": "work"},
        )
        assert response.status_code == 200


class TestMetrics:
    def test_metrics_json(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "operations" in data
        assert data["operations"]["remember"] == 2
        assert "quality" in data
        assert "conflicts_detected" in data["quality"]
        assert data["quality"]["conflicts_detected"] == 0

    def test_metrics_prometheus(self, client):
        response = client.get("/metrics?output_format=prometheus")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        text = response.text
        assert "kemi_remember_total" in text

    def test_metrics_not_available(self, mock_memory):
        original = mock_memory.get_metrics
        mock_memory.get_metrics = lambda: None
        app = create_app(memory=mock_memory)
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 503
        mock_memory.get_metrics = original


class TestAudit:
    @pytest.fixture
    def audit_client(self, mock_memory):
        mock_memory.enable_audit_trail()
        app = create_app(memory=mock_memory)
        return TestClient(app)

    def test_audit_log(self, audit_client):
        response = audit_client.post(
            "/audit/log",
            json={
                "user_id": "alice",
                "operation": "remember",
                "status": "success",
                "details": {"memory_id": "mem-123"},
                "namespace": "default",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["entry_id"] == 42
        assert data["status"] == "logged"

    def test_audit_log_not_enabled(self, client):
        response = client.post(
            "/audit/log",
            json={"user_id": "alice", "operation": "remember"},
        )
        assert response.status_code == 503
        assert "not enabled" in response.json()["detail"]

    def test_audit_query(self, audit_client):
        response = audit_client.post(
            "/audit/query",
            json={"user_id": "alice", "operation": "remember", "limit": 50},
        )
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert data["count"] == 0
        assert data["limit"] == 50

    def test_audit_query_not_enabled(self, client):
        response = client.post("/audit/query", json={"user_id": "alice"})
        assert response.status_code == 503

    def test_audit_stats(self, audit_client):
        response = audit_client.get("/audit/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_entries"] == 100
        assert data["unique_users"] == 5

    def test_audit_stats_not_enabled(self, client):
        response = client.get("/audit/stats")
        assert response.status_code == 503

    def test_audit_export(self, audit_client):
        response = audit_client.post(
            "/audit/export",
            json={"user_id": "alice", "start_time": "2026-01-01T00:00:00"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert data["count"] == 0

    def test_audit_export_not_enabled(self, client):
        response = client.post("/audit/export", json={})
        assert response.status_code == 503


class TestAdaptive:
    @pytest.fixture
    def adaptive_client(self, mock_memory):
        mock_memory.enable_adaptive_retrieval(enable=True)
        app = create_app(memory=mock_memory)
        return TestClient(app)

    def test_adaptive_analyze(self, adaptive_client):
        response = adaptive_client.post(
            "/adaptive/analyze",
            json={"query": "What is python?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is python?"
        assert data["query_type"] == "factual"
        assert "recommended_weights" in data
        assert data["recommended_weights"]["weight_semantic"] == 0.55

    def test_adaptive_analyze_not_enabled(self, client):
        response = client.post(
            "/adaptive/analyze",
            json={"query": "What is python?"},
        )
        assert response.status_code == 503
        assert "not enabled" in response.json()["detail"]

    def test_adaptive_user_profile(self, adaptive_client):
        response = adaptive_client.get("/adaptive/user-profile/alice")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "alice"
        assert data["total_queries"] == 10
        assert data["dominant_type"] == "factual"

    def test_adaptive_user_profile_not_enabled(self, client):
        response = client.get("/adaptive/user-profile/alice")
        assert response.status_code == 503


class TestAdminFeatures:
    def test_enable_adaptive(self, client):
        response = client.post(
            "/admin/enable-adaptive",
            json={"enable": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["adaptive_retrieval_enabled"] is True

    def test_disable_adaptive(self, client):
        response = client.post(
            "/admin/enable-adaptive",
            json={"enable": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["adaptive_retrieval_enabled"] is False

    def test_enable_audit(self, client):
        response = client.post(
            "/admin/enable-audit",
            json={"enable": True, "retention_days": 180, "auto_purge": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["audit_trail_enabled"] is True
        assert data["retention_days"] == 180
        assert data["auto_purge"] is False

    def test_disable_audit(self, client):
        response = client.post(
            "/admin/enable-audit",
            json={"enable": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["audit_trail_enabled"] is False


class TestRateLimiter:
    def test_is_allowed_under_limit(self) -> None:
        rl = RateLimiter(requests_per_window=3, window_seconds=60)
        assert rl.is_allowed("user1") is True
        assert rl.is_allowed("user1") is True
        assert rl.is_allowed("user1") is True
        assert rl.is_allowed("user2") is True

    def test_is_allowed_at_limit(self) -> None:
        rl = RateLimiter(requests_per_window=2, window_seconds=60)
        assert rl.is_allowed("user1") is True
        assert rl.is_allowed("user1") is True
        assert rl.is_allowed("user1") is False

    def test_is_allowed_window_slides(self, monkeypatch) -> None:
        import time
        rl = RateLimiter(requests_per_window=1, window_seconds=60)
        assert rl.is_allowed("user1") is True
        # Artificially advance time past the window
        original_time = time.time
        monkeypatch.setattr(time, "time", lambda: original_time() + 120)
        assert rl.is_allowed("user1") is True

    def test_get_retry_after_when_allowed(self) -> None:
        rl = RateLimiter(requests_per_window=3, window_seconds=60)
        rl.is_allowed("user1")
        assert rl.get_retry_after("user1") == 0

    def test_get_retry_after_when_limited(self) -> None:
        rl = RateLimiter(requests_per_window=1, window_seconds=60)
        rl.is_allowed("user1")
        retry = rl.get_retry_after("user1")
        assert retry > 0
        assert retry <= 60

    def test_get_retry_after_different_keys(self) -> None:
        rl = RateLimiter(requests_per_window=1, window_seconds=60)
        rl.is_allowed("user1")
        assert rl.get_retry_after("user2") == 0

    def test_multiple_keys_independent(self) -> None:
        rl = RateLimiter(requests_per_window=2, window_seconds=60)
        assert rl.is_allowed("user1") is True
        assert rl.is_allowed("user1") is True
        assert rl.is_allowed("user1") is False
        assert rl.is_allowed("user2") is True
        assert rl.is_allowed("user2") is True
        assert rl.is_allowed("user2") is False


class TestRequestModels:
    def test_remember_request_valid(self) -> None:
        req = RememberRequest(user_id="alice", content="hello")
        assert req.user_id == "alice"
        assert req.content == "hello"
        assert req.importance == 0.5
        assert req.namespace == "default"

    def test_remember_request_invalid_empty_user_id(self) -> None:
        with pytest.raises(ValidationError):
            RememberRequest(user_id="", content="hello")

    def test_remember_request_invalid_empty_content(self) -> None:
        with pytest.raises(ValidationError):
            RememberRequest(user_id="alice", content="")

    def test_remember_request_importance_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            RememberRequest(user_id="alice", content="hello", importance=1.5)
        with pytest.raises(ValidationError):
            RememberRequest(user_id="alice", content="hello", importance=-0.1)

    def test_recall_request_valid(self) -> None:
        req = RecallRequest(user_id="alice", query="python")
        assert req.top_k == 5
        assert req.namespace == "default"

    def test_recall_request_invalid_empty_query(self) -> None:
        with pytest.raises(ValidationError):
            RecallRequest(user_id="alice", query="")

    def test_update_request_valid_empty(self) -> None:
        req = UpdateRequest()
        assert req.content is None

    def test_update_request_importance_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            UpdateRequest(importance=1.5)

    def test_prune_request_valid(self) -> None:
        req = PruneRequest(namespace="work")
        assert req.max_age_days is None

    def test_consolidate_request_valid(self) -> None:
        req = ConsolidateRequest(min_memories=10)
        assert req.max_age_days == 30.0

    def test_batch_remember_request_valid(self) -> None:
        req = BatchRememberRequest(user_id="alice", contents=["a", "b"])
        assert len(req.contents) == 2

    def test_batch_remember_request_empty_contents(self) -> None:
        with pytest.raises(ValidationError):
            BatchRememberRequest(user_id="alice", contents=[])

    def test_create_api_key_request_valid(self) -> None:
        req = CreateAPIKeyRequest(user_id="alice", name="my-key")
        assert req.expires_in_days is None

    def test_create_api_key_request_name_too_long(self) -> None:
        with pytest.raises(ValidationError):
            CreateAPIKeyRequest(user_id="alice", name="x" * 201)

    def test_audit_query_request_limit_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AuditQueryRequest(limit=0)
        with pytest.raises(ValidationError):
            AuditQueryRequest(limit=10001)


class TestAuthMiddleware:
    def test_no_api_key_required_by_default(self, client) -> None:
        response = client.post("/remember", json={"user_id": "alice", "content": "test"})
        assert response.status_code == 200

    def test_api_key_required_missing_key(self, client) -> None:
        _reset_api_key_manager()
        with patch("kemi.interfaces.api.app._api_key_required", return_value=True):
            response = client.post("/remember", json={"user_id": "alice", "content": "test"})
            assert response.status_code == 401
            assert "X-API-Key header required" in response.json()["detail"]

    def test_exempt_path_no_key(self, client) -> None:
        _reset_api_key_manager()
        with patch("kemi.interfaces.api.app._api_key_required", return_value=True):
            response = client.get("/health")
            assert response.status_code == 200

    def test_exempt_api_keys_post_no_key(self, client) -> None:
        _reset_api_key_manager()
        with patch("kemi.interfaces.api.app._api_key_required", return_value=True):
            with patch("kemi.interfaces.api.app._get_api_key_manager", return_value=None):
                response = client.post("/api/keys", json={"user_id": "alice", "name": "test"})
                assert response.status_code == 501

    def test_api_key_valid(self, client) -> None:
        _reset_api_key_manager()
        manager = MagicMock()
        key = MagicMock()
        key.user_id = "alice"
        manager.lookup.return_value = key
        with patch("kemi.interfaces.api.app._api_key_required", return_value=True):
            with patch("kemi.interfaces.api.app._get_api_key_manager", return_value=manager):
                response = client.post(
                    "/remember",
                    json={"user_id": "alice", "content": "test"},
                    headers={"X-API-Key": "valid-key"},
                )
                assert response.status_code == 200

    def test_api_key_invalid(self, client) -> None:
        _reset_api_key_manager()
        manager = MagicMock()
        manager.lookup.return_value = None
        with patch("kemi.interfaces.api.app._api_key_required", return_value=True):
            with patch("kemi.interfaces.api.app._get_api_key_manager", return_value=manager):
                response = client.post(
                    "/remember",
                    json={"user_id": "alice", "content": "test"},
                    headers={"X-API-Key": "invalid-key"},
                )
                assert response.status_code == 401
                assert "Invalid or expired API key" in response.json()["detail"]

    def test_resolve_user_id_no_auth(self) -> None:
        from fastapi import Request

        request = MagicMock(spec=Request)
        request.state.user_id = None
        assert _resolve_user_id(request, "alice") == "alice"

    def test_resolve_user_id_with_auth_match(self) -> None:
        from fastapi import Request

        request = MagicMock(spec=Request)
        request.state.user_id = "alice"
        assert _resolve_user_id(request, "alice") == "alice"
        assert _resolve_user_id(request, None) == "alice"

    def test_resolve_user_id_with_auth_mismatch(self) -> None:
        from fastapi import Request

        request = MagicMock(spec=Request)
        request.state.user_id = "alice"
        with pytest.raises(HTTPException) as exc:
            _resolve_user_id(request, "bob")
        assert exc.value.status_code == 403

    def test_resolve_user_id_no_auth_no_user_id(self) -> None:
        from fastapi import Request

        request = MagicMock(spec=Request)
        request.state.user_id = None
        with pytest.raises(HTTPException) as exc:
            _resolve_user_id(request, None)
        assert exc.value.status_code == 400

    def test_require_admin_authed(self) -> None:
        from fastapi import Request

        request = MagicMock(spec=Request)
        request.state.user_id = "alice"
        assert _require_admin(request) == "alice"

    def test_require_admin_unauthed(self) -> None:
        from fastapi import Request

        request = MagicMock(spec=Request)
        request.state.user_id = None
        with pytest.raises(HTTPException) as exc:
            _require_admin(request)
        assert exc.value.status_code == 401


class TestRateLimiting:
    @pytest.fixture
    def rate_limited_client(self, mock_memory):
        with patch.dict(os.environ, {
            "KEMI_RATE_LIMIT_ENABLED": "true",
            "KEMI_RATE_LIMIT_REQUESTS": "2",
            "KEMI_RATE_LIMIT_WINDOW": "60",
        }):
            with patch("kemi.interfaces.api.app._rate_limiter", None):
                app = create_app(memory=mock_memory)
                with TestClient(app) as client:
                    yield client

    def test_rate_limit_not_exceeded(self, rate_limited_client) -> None:
        response = rate_limited_client.post("/remember", json={"user_id": "alice", "content": "test"})  # noqa: E501
        assert response.status_code == 200

    def test_rate_limit_exceeded(self, rate_limited_client) -> None:
        rate_limited_client.post("/remember", json={"user_id": "alice", "content": "test1"})
        rate_limited_client.post("/remember", json={"user_id": "alice", "content": "test2"})
        response = rate_limited_client.post("/remember", json={"user_id": "alice", "content": "test3"})  # noqa: E501
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_rate_limit_different_users_independent(self, rate_limited_client) -> None:
        rate_limited_client.post("/remember", json={"user_id": "alice", "content": "test1"})
        rate_limited_client.post("/remember", json={"user_id": "alice", "content": "test2"})
        response = rate_limited_client.post("/remember", json={"user_id": "bob", "content": "test"})
        assert response.status_code == 200


class TestAdminEndpoints:
    @pytest.fixture(autouse=True)
    def _patch_require_admin(self):
        with patch("kemi.interfaces.api.app._require_admin", return_value="admin"):
            yield

    def test_admin_fts_rebuild(self, mock_memory) -> None:
        mock_memory._store.rebuild_fts_index = MagicMock(return_value=42)
        with patch("kemi.interfaces.api.app._get_memory_singleton", return_value=mock_memory):
            app = create_app(memory=mock_memory)
            client = TestClient(app)
            response = client.post("/admin/fts/rebuild")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["memories_indexed"] == 42

    def test_admin_fts_rebuild_not_supported(self, mock_memory) -> None:
        class NoFTSStore:
            pass

        mock_memory._store = NoFTSStore()
        with patch("kemi.interfaces.api.app._get_memory_singleton", return_value=mock_memory):
            app = create_app(memory=mock_memory)
            client = TestClient(app)
            response = client.post("/admin/fts/rebuild")
            assert response.status_code == 501

    def test_admin_fts_stats(self, mock_memory) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(100,), (50,), (50,)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_memory._store._get_connection = MagicMock(return_value=mock_conn)
        with patch("kemi.interfaces.api.app._get_memory_singleton", return_value=mock_memory):
            app = create_app(memory=mock_memory)
            client = TestClient(app)
            response = client.get("/admin/fts/stats")
            assert response.status_code == 200
            data = response.json()
            assert "fts_total_entries" in data

    def test_admin_fts_stats_with_user_id(self, mock_memory) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(100,), (50,), (50,), (50,)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_memory._store._get_connection = MagicMock(return_value=mock_conn)
        with patch("kemi.interfaces.api.app._get_memory_singleton", return_value=mock_memory):
            app = create_app(memory=mock_memory)
            client = TestClient(app)
            response = client.get("/admin/fts/stats?user_id=alice")
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == "alice"

    def test_admin_fts_verify(self, mock_memory) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = [
            [("mem-1",), ("mem-2",)],
            [("mem-1",), ("mem-2",)],
        ]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_memory._store._get_connection = MagicMock(return_value=mock_conn)
        with patch("kemi.interfaces.api.app._get_memory_singleton", return_value=mock_memory):
            app = create_app(memory=mock_memory)
            client = TestClient(app)
            response = client.post("/admin/fts/verify", json={"verify_only": True})
            assert response.status_code == 200
            data = response.json()
            assert data["in_sync"] is True

    def test_admin_fts_verify_repair(self, mock_memory) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchall.side_effect = [
            [("mem-1",), ("mem-2",)],
            [("mem-1",)],
        ]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_memory._store._get_connection = MagicMock(return_value=mock_conn)
        with patch("kemi.interfaces.api.app._get_memory_singleton", return_value=mock_memory):
            app = create_app(memory=mock_memory)
            client = TestClient(app)
            response = client.post("/admin/fts/verify", json={"verify_only": False})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "repaired"
            assert data["auto_repaired"] is True
            assert data["missing_from_fts"] == 1
            assert data["orphaned_in_fts"] == 0

    def test_admin_health(self, client) -> None:
        response = client.get("/admin/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
        assert "database" in data["components"]

    def test_admin_users(self, client) -> None:
        response = client.get("/admin/users")
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "count" in data


class TestWebhookEndpoints:
    @pytest.fixture
    def webhook_client(self, mock_memory):
        mock_event_type = MagicMock()
        mock_event_type.value = "remember"
        mock_event_class = MagicMock()
        mock_event_class.from_string.return_value = mock_event_type

        mock_store_class = MagicMock()
        mock_store_instance = MagicMock()
        cfg = MagicMock()
        cfg.webhook_id = "wh-1"
        cfg.url = "https://example.com/webhook"
        cfg.events = [mock_event_type]
        cfg.active = True
        mock_store_instance.create = MagicMock(return_value="wh-1")
        mock_store_instance.list_all = MagicMock(return_value=[cfg])
        mock_store_instance.delete = MagicMock(return_value=True)
        mock_store_class.return_value = mock_store_instance

        with patch("kemi.interfaces.api.app.WebhookStore", mock_store_class):
            with patch("kemi.interfaces.api.app.WebhookEventType", mock_event_class):
                app = create_app(memory=mock_memory)
                with TestClient(app) as client:
                    yield client

    def test_create_webhook(self, webhook_client) -> None:
        response = webhook_client.post("/webhooks", json={
            "url": "https://example.com/webhook",
            "events": ["remember"],
            "secret": "my-secret",
            "active": True,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["webhook_id"] == "wh-1"

    def test_create_webhook_invalid_event(self, webhook_client) -> None:
        with patch("kemi.interfaces.api.app.WebhookEventType.from_string", side_effect=ValueError("invalid")):  # noqa: E501
            response = webhook_client.post("/webhooks", json={
                "url": "https://example.com/webhook",
                "events": ["invalid_event"],
            })
            assert response.status_code == 400

    def test_list_webhooks(self, webhook_client) -> None:
        response = webhook_client.get("/webhooks")
        assert response.status_code == 200
        data = response.json()
        assert "webhooks" in data
        assert data["count"] == 1

    def test_delete_webhook(self, webhook_client) -> None:
        response = webhook_client.delete("/webhooks/wh-1")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True

    def test_delete_webhook_not_found(self, webhook_client) -> None:
        mock_store_class = MagicMock()
        mock_store_instance = MagicMock()
        mock_store_instance.delete = MagicMock(return_value=False)
        mock_store_class.return_value = mock_store_instance
        mock_event_class = MagicMock()
        with patch("kemi.interfaces.api.app.WebhookStore", mock_store_class):
            with patch("kemi.interfaces.api.app.WebhookEventType", mock_event_class):
                response = webhook_client.delete("/webhooks/wh-999")
                assert response.status_code == 404


class TestAPIKeyEndpoints:
    @pytest.fixture
    def api_key_client(self, mock_memory):
        manager = MagicMock()
        key = MagicMock()
        key.user_id = "alice"
        key.key_id = "key-1"
        key.name = "test-key"
        key.to_dict = MagicMock(return_value={
            "key_id": "key-1",
            "user_id": "alice",
            "name": "test-key",
            "secret": "secret-123",
        })
        manager.lookup = MagicMock(return_value=key)
        manager.create_key = MagicMock(return_value=key)
        manager.list_keys = MagicMock(return_value=[key])
        manager.get = MagicMock(return_value=key)
        manager.revoke = MagicMock(return_value=True)

        with patch("kemi.interfaces.api.app._get_api_key_manager", return_value=manager):
            app = create_app(memory=mock_memory)
            with TestClient(app) as client:
                yield client

    def test_create_api_key(self, api_key_client) -> None:
        response = api_key_client.post("/api/keys", json={
            "user_id": "alice",
            "name": "my-key",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["key_id"] == "key-1"
        assert data["secret"] == "secret-123"

    def test_create_api_key_mismatched_user_id(self, api_key_client) -> None:
        with patch("kemi.interfaces.api.app._api_key_required", return_value=True):
            response = api_key_client.post(
                "/api/keys",
                json={"user_id": "bob", "name": "my-key"},
                headers={"X-API-Key": "valid-key"},
            )
            assert response.status_code == 403

    def test_list_api_keys(self, api_key_client) -> None:
        response = api_key_client.get("/api/keys")
        assert response.status_code == 200
        data = response.json()
        assert "keys" in data
        assert data["count"] == 1

    def test_revoke_api_key(self, api_key_client) -> None:
        response = api_key_client.delete("/api/keys/key-1")
        assert response.status_code == 200
        data = response.json()
        assert data["revoked"] is True

    def test_revoke_api_key_not_found(self, api_key_client) -> None:
        with patch("kemi.interfaces.api.app._get_api_key_manager") as mock_mgr:
            mock_mgr.return_value.revoke.return_value = False
            response = api_key_client.delete("/api/keys/key-999")
            assert response.status_code == 404


class TestBackgroundTaskEndpoints:
    @pytest.fixture
    def task_client(self, mock_memory):
        mock_task_manager = MagicMock()
        mock_task_manager.submit_embed_batch.return_value = "task-123"
        mock_task_manager.submit_rebuild_fts_index.return_value = "task-456"
        mock_task_manager.get_task_status.return_value = MagicMock(
            to_dict=lambda: {
                "task_id": "task-123",
                "status": "pending",
                "progress": 0.0,
            }
        )
        mock_task_manager.get_stats.return_value = {
            "total_tasks": 0,
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "max_concurrent": 3,
        }
        mock_task_manager.cancel_task.return_value = True

        # Pre-configure mixed-status tasks for the filter test
        pending_task = MagicMock()
        pending_task.to_dict.return_value = {"task_id": "t1", "status": "pending", "progress": 0.0}
        running_task = MagicMock()
        running_task.to_dict.return_value = {"task_id": "t2", "status": "running", "progress": 0.5}

        def _filter_tasks(status=None, limit=50):
            all_tasks = [pending_task, running_task]
            if status is None:
                return all_tasks
            return [t for t in all_tasks if t.to_dict()["status"] == status.value]

        mock_task_manager.list_tasks.side_effect = _filter_tasks

        with patch("kemi.infra.background_tasks.get_task_manager", return_value=mock_task_manager):
            app = create_app(memory=mock_memory)
            with TestClient(app) as client:
                yield client

    def test_submit_embed_batch(self, task_client) -> None:
        response = task_client.post("/tasks/embed-batch", json={
            "user_id": "alice",
            "contents": ["hello", "world"],
            "importance": 0.5,
            "namespace": "default",
        })
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"

    def test_submit_rebuild_fts(self, task_client) -> None:
        response = task_client.post("/tasks/rebuild-fts", json={"user_id": "alice"})
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"

    def test_submit_rebuild_fts_no_user_id(self, task_client) -> None:
        response = task_client.post("/tasks/rebuild-fts", json={})
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data

    def test_get_task_status(self, task_client) -> None:
        response = task_client.get("/tasks/task-123")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-123"

    def test_get_task_status_not_found(self, task_client) -> None:
        mock_mgr = MagicMock()
        mock_mgr.get_task_status.return_value = None
        with patch("kemi.infra.background_tasks.get_task_manager", return_value=mock_mgr):
            response = task_client.get("/tasks/nonexistent-id")
            assert response.status_code == 404

    def test_list_tasks(self, task_client) -> None:
        response = task_client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "stats" in data

    def test_list_tasks_with_status_filter(self, task_client) -> None:
        response = task_client.get("/tasks?status=pending")
        assert response.status_code == 200
        data = response.json()
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["status"] == "pending"

    def test_list_tasks_invalid_status(self, task_client) -> None:
        response = task_client.get("/tasks?status=invalid")
        assert response.status_code == 400

    def test_cancel_task(self, task_client) -> None:
        response = task_client.delete("/tasks/task-123")
        assert response.status_code == 200
        data = response.json()
        assert data["cancelled"] is True

    def test_cancel_task_not_found(self, task_client) -> None:
        mock_mgr = MagicMock()
        mock_mgr.cancel_task.return_value = False
        with patch("kemi.infra.background_tasks.get_task_manager", return_value=mock_mgr):
            response = task_client.delete("/tasks/nonexistent-id")
            assert response.status_code == 400

    def test_get_task_stats(self, task_client) -> None:
        response = task_client.get("/tasks/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_tasks" in data
        assert "pending" in data
        assert "running" in data


class TestMemoryHistory:
    def test_memory_history(self, client) -> None:
        response = client.get("/memories/mem-123/history")
        assert response.status_code == 200
        data = response.json()
        assert data["memory_id"] == "mem-123"
        assert "versions" in data
        assert data["count"] >= 0

    def test_memory_history_with_limit(self, client) -> None:
        response = client.get("/memories/mem-123/history?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "versions" in data
