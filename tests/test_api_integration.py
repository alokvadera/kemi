"""Integration tests for kemi API server with real dependencies.

These tests exercise the new observability, audit, and adaptive retrieval
features through the actual API endpoints using a real SQLite-backed
Memory instance (not mocked).
"""

import pytest

try:
    from fastapi.testclient import TestClient

    from kemi.interfaces.api import create_app

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    TestClient = None  # type: ignore[assignment, misc]
    create_app = None  # type: ignore[assignment, misc]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi not installed"),
]


@pytest.fixture
def integration_client(tmp_path, mock_embedding):
    """Create a TestClient backed by a fresh real SQLite Memory instance."""
    from kemi import Memory
    from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
    from kemi.infra.observability import reset_metrics

    reset_metrics()
    db_path = str(tmp_path / "integration.db")
    adapter = SQLiteStorageAdapter(db_path=db_path)
    mem = Memory(embed=mock_embedding(), store=adapter)
    app = create_app(memory=mem)
    return TestClient(app)


class TestMetricsIntegration:
    """Test /metrics endpoint with real MetricsCollector."""

    def test_metrics_json_structure(self, integration_client):
        response = integration_client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "operations" in data
        assert "embeddings" in data
        assert "storage" in data
        assert "quality" in data
        assert "memory_usage" in data
        assert "timestamp" in data

    def test_metrics_increments_after_remember(self, integration_client):
        before = integration_client.get("/metrics").json()

        integration_client.post(
            "/remember",
            json={"user_id": "alice", "content": "I love python"},
        )

        after = integration_client.get("/metrics").json()
        assert after["operations"]["remember"] > before["operations"]["remember"]
        assert after["embeddings"]["total"] > before["embeddings"]["total"]
        assert after["memory_usage"]["total_memories"] > before["memory_usage"]["total_memories"]

    def test_metrics_prometheus_format(self, integration_client):
        integration_client.post(
            "/remember",
            json={"user_id": "alice", "content": "testing prometheus export"},
        )

        response = integration_client.get("/metrics?output_format=prometheus")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        text = response.text
        assert "kemi_remember_total" in text
        assert "kemi_embed_total" in text
        assert "# TYPE kemi_remember_total counter" in text

    def test_metrics_recall_increments_counter(self, integration_client):
        integration_client.post(
            "/remember",
            json={"user_id": "alice", "content": "something to recall"},
        )

        before = integration_client.get("/metrics").json()
        integration_client.post(
            "/recall",
            json={"user_id": "alice", "query": "something"},
        )
        after = integration_client.get("/metrics").json()
        assert after["operations"]["recall"] > before["operations"]["recall"]


class TestAuditIntegration:
    """Test audit trail endpoints with real AuditTrail + SQLite."""

    def test_audit_enable_and_log(self, integration_client):
        # Enable audit trail
        response = integration_client.post(
            "/admin/enable-audit",
            json={"enable": True, "retention_days": 30, "auto_purge": True},
        )
        assert response.status_code == 200
        assert response.json()["audit_trail_enabled"] is True

        # Log a custom audit entry via API
        response = integration_client.post(
            "/audit/log",
            json={
                "user_id": "alice",
                "operation": "custom_action",
                "status": "success",
                "details": {"key": "value"},
                "namespace": "default",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "logged"
        assert isinstance(data["entry_id"], int)
        assert data["entry_id"] > 0

    def test_audit_stats_after_operations(self, integration_client):
        # Enable audit
        integration_client.post(
            "/admin/enable-audit",
            json={"enable": True},
        )

        # Perform operations that trigger audit logging
        integration_client.post(
            "/remember",
            json={"user_id": "alice", "content": "audit test memory"},
        )

        response = integration_client.get("/audit/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_entries"] >= 1
        assert data["unique_users"] >= 1

    def test_audit_query_returns_entries(self, integration_client):
        # Enable audit
        integration_client.post(
            "/admin/enable-audit",
            json={"enable": True},
        )

        # Trigger a remembered operation
        integration_client.post(
            "/remember",
            json={"user_id": "alice", "content": "query test"},
        )

        response = integration_client.post(
            "/audit/query",
            json={"user_id": "alice", "limit": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert data["count"] >= 1
        assert data["limit"] == 10

        # Verify entry structure
        if data["entries"]:
            entry = data["entries"][0]
            assert "timestamp" in entry
            assert "user_id" in entry
            assert "operation" in entry

    def test_audit_export(self, integration_client):
        # Enable audit
        integration_client.post(
            "/admin/enable-audit",
            json={"enable": True},
        )

        integration_client.post(
            "/remember",
            json={"user_id": "alice", "content": "export test"},
        )

        response = integration_client.post(
            "/audit/export",
            json={"user_id": "alice"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert data["count"] >= 1

    def test_audit_not_enabled_returns_503(self, integration_client):
        # Do not enable audit trail
        response = integration_client.post(
            "/audit/log",
            json={"user_id": "alice", "operation": "remember"},
        )
        assert response.status_code == 503
        assert "not enabled" in response.json()["detail"]

    def test_audit_query_not_enabled_returns_503(self, integration_client):
        response = integration_client.post("/audit/query", json={"user_id": "alice"})
        assert response.status_code == 503

    def test_audit_stats_not_enabled_returns_503(self, integration_client):
        response = integration_client.get("/audit/stats")
        assert response.status_code == 503


class TestAdaptiveIntegration:
    """Test adaptive retrieval endpoints with real AdaptiveRetriever."""

    def test_adaptive_enable_and_analyze(self, integration_client):
        response = integration_client.post(
            "/admin/enable-adaptive",
            json={"enable": True},
        )
        assert response.status_code == 200
        assert response.json()["adaptive_retrieval_enabled"] is True

        response = integration_client.post(
            "/adaptive/analyze",
            json={"query": "What is python?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is python?"
        assert "query_type" in data
        assert "recommended_weights" in data
        assert "confidence" in data
        assert "word_count" in data
        assert "specificity" in data

    def test_adaptive_user_profile_tracks_queries(self, integration_client):
        # Enable adaptive
        integration_client.post(
            "/admin/enable-adaptive",
            json={"enable": True},
        )

        # Remember content so recall has something to return
        integration_client.post(
            "/remember",
            json={"user_id": "alice", "content": "I enjoy cooking Italian food"},
        )

        # Trigger record_feedback via recall (core.py recall() records feedback)
        for query in ["What is python?", "How do I cook pasta?", "Tell me about Rome"]:
            integration_client.post("/recall", json={"user_id": "alice", "query": query})

        response = integration_client.get("/adaptive/user-profile/alice")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "alice"
        assert data["total_queries"] >= 3
        assert "distribution" in data
        assert "dominant_type" in data

    def test_adaptive_not_enabled_returns_503(self, integration_client):
        response = integration_client.post(
            "/adaptive/analyze",
            json={"query": "What is python?"},
        )
        assert response.status_code == 503
        assert "not enabled" in response.json()["detail"]

    def test_adaptive_user_profile_not_enabled_returns_503(self, integration_client):
        response = integration_client.get("/adaptive/user-profile/alice")
        assert response.status_code == 503


class TestAdminIntegration:
    """Test admin feature toggle endpoints."""

    def test_enable_disable_adaptive(self, integration_client):
        response = integration_client.post(
            "/admin/enable-adaptive",
            json={"enable": True},
        )
        assert response.status_code == 200
        assert response.json()["adaptive_retrieval_enabled"] is True

        response = integration_client.post(
            "/admin/enable-adaptive",
            json={"enable": False},
        )
        assert response.status_code == 200
        assert response.json()["adaptive_retrieval_enabled"] is False

        # Verify it's disabled
        response = integration_client.post(
            "/adaptive/analyze",
            json={"query": "test"},
        )
        assert response.status_code == 503

    def test_enable_disable_audit(self, integration_client):
        response = integration_client.post(
            "/admin/enable-audit",
            json={"enable": True, "retention_days": 180, "auto_purge": False},
        )
        assert response.status_code == 200
        assert response.json()["audit_trail_enabled"] is True
        assert response.json()["retention_days"] == 180
        assert response.json()["auto_purge"] is False

        response = integration_client.post(
            "/admin/enable-audit",
            json={"enable": False},
        )
        assert response.status_code == 200
        assert response.json()["audit_trail_enabled"] is False

        # Verify it's disabled
        response = integration_client.post(
            "/audit/log",
            json={"user_id": "alice", "operation": "remember"},
        )
        assert response.status_code == 503


class TestEndToEndWorkflow:
    """End-to-end tests combining all features."""

    def test_full_workflow(self, integration_client):
        # 1. Enable both features
        r = integration_client.post(
            "/admin/enable-audit",
            json={"enable": True},
        )
        assert r.status_code == 200
        r = integration_client.post(
            "/admin/enable-adaptive",
            json={"enable": True},
        )
        assert r.status_code == 200

        # 2. Remember some content
        r1 = integration_client.post(
            "/remember",
            json={"user_id": "alice", "content": "I love Italian food", "tags": ["food"]},
        )
        assert r1.status_code == 200
        mem_id = r1.json()["memory_id"]

        r2_remember = integration_client.post(
            "/remember",
            json={"user_id": "alice", "content": "My favorite color is blue"},
        )
        assert r2_remember.status_code == 200

        # 3. Recall
        r2 = integration_client.post(
            "/recall",
            json={"user_id": "alice", "query": "favorite things"},
        )
        assert r2.status_code == 200
        assert len(r2.json()["results"]) >= 1

        # 4. Check metrics (mock embedding returns identical vectors, so dedup may merge;
        # we assert on counters which always increment regardless of dedup)
        metrics = integration_client.get("/metrics").json()
        assert metrics["operations"]["remember"] >= 2
        assert metrics["operations"]["recall"] >= 1
        # total_memories may be 1 if dedup merged the two identical-embedding memories
        assert metrics["memory_usage"]["total_memories"] >= 1

        # 5. Check audit entries
        audit = integration_client.post(
            "/audit/query",
            json={"user_id": "alice", "limit": 50},
        ).json()
        assert audit["count"] >= 2

        # 6. Analyze query with adaptive
        adaptive = integration_client.post(
            "/adaptive/analyze",
            json={"query": "What are my favorite things?"},
        ).json()
        assert "query_type" in adaptive
        assert "recommended_weights" in adaptive

        # 7. User profile should now have data
        profile = integration_client.get("/adaptive/user-profile/alice").json()
        assert profile["total_queries"] >= 1

        # 8. Update memory
        r_update = integration_client.patch(
            f"/memories/{mem_id}",
            json={"content": "I love Italian and French food"},
        )
        assert r_update.status_code == 200

        # 9. Check metrics reflects update
        metrics_after = integration_client.get("/metrics").json()
        assert metrics_after["operations"]["update"] >= 1

        # 10. Forget a memory
        r_forget = integration_client.post(f"/forget?user_id=alice&memory_id={mem_id}")
        assert r_forget.status_code == 200
        metrics_after_forget = integration_client.get("/metrics").json()
        assert metrics_after_forget["operations"]["forget"] >= 1
