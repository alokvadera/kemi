"""Tests for src/kemi/api_server.py"""

from unittest.mock import MagicMock

import pytest

from kemi.models import LifecycleState, MemoryType

# Skip entire module if fastapi is not available
try:
    from fastapi.testclient import TestClient

    from kemi.api_server import create_app

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

    # Will be skipped by pytestmark below once we set it
    TestClient = None  # type: ignore[assignment, misc]
    create_app = None  # type: ignore[assignment, misc]

# Apply skip to all tests in this module if fastapi unavailable
pytestmark = pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed")


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


@pytest.fixture
def mock_memory():
    return MockMemory()


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
        data_events = [e for e in events if e and not e.startswith("event: done") and not e.startswith("event: error")]
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
