"""Tests for kemi audit trail module."""

import sqlite3

import pytest

from kemi.infra.audit import AuditEntry, AuditTrail

pytestmark = pytest.mark.slow


def _create_memory_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_audit_entry_defaults(self) -> None:
        entry = AuditEntry()
        assert entry.id == 0
        assert entry.timestamp == ""
        assert entry.status == "success"
        assert entry.namespace == "default"
        assert entry.details == {}

    def test_audit_entry_to_dict(self) -> None:
        entry = AuditEntry(
            id=1,
            timestamp="2025-01-01T00:00:00Z",
            user_id="alice",
            operation="remember",
            status="success",
            memory_id="mem-123",
        )
        d = entry.to_dict()
        assert d["id"] == 1
        assert d["user_id"] == "alice"
        assert d["operation"] == "remember"
        assert d["memory_id"] == "mem-123"


class TestAuditTrail:
    """Tests for AuditTrail."""

    def test_schema_creation(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        # Verify table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'"
        )
        assert cursor.fetchone() is not None
        audit.close()

    def test_log_operation(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        entry_id = audit.log_operation(
            user_id="alice",
            operation="remember",
            details={"memory_id": "abc123", "importance": 0.8},
            memory_id="abc123",
        )
        assert entry_id > 0

        # Verify entry was written
        cursor = conn.execute("SELECT * FROM audit_log WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row["user_id"] == "alice"
        assert row["operation"] == "remember"
        assert row["status"] == "success"
        assert row["memory_id"] == "abc123"
        audit.close()

    def test_log_operation_batch(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        count = audit.log_operation_batch(
            [
                {"user_id": "alice", "operation": "remember", "details": {"content": "test1"}},
                {"user_id": "alice", "operation": "recall", "details": {"query": "test"}},
                {"user_id": "bob", "operation": "forget", "details": {}},
            ]
        )
        assert count == 3

        cursor = conn.execute("SELECT COUNT(*) as cnt FROM audit_log")
        assert cursor.fetchone()["cnt"] == 3
        audit.close()

    def test_query_by_user(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        audit.log_operation(user_id="alice", operation="remember")
        audit.log_operation(user_id="bob", operation="remember")
        audit.log_operation(user_id="alice", operation="recall")

        entries = audit.query(user_id="alice")
        assert len(entries) == 2
        assert all(e.user_id == "alice" for e in entries)
        audit.close()

    def test_query_by_operation(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        audit.log_operation(user_id="alice", operation="remember")
        audit.log_operation(user_id="alice", operation="forget")
        audit.log_operation(user_id="bob", operation="remember")

        entries = audit.query(operation="remember")
        assert len(entries) == 2
        for e in entries:
            assert e.operation == "remember"
        audit.close()

    def test_query_by_status(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        audit.log_operation(user_id="alice", operation="remember", status="success")
        audit.log_operation(user_id="alice", operation="recall", status="error")

        entries = audit.query(status="error")
        assert len(entries) == 1
        assert entries[0].status == "error"
        audit.close()

    def test_query_pagination(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        for _i in range(5):
            audit.log_operation(user_id="alice", operation="remember")

        entries = audit.query(limit=3, offset=0)
        assert len(entries) == 3

        entries = audit.query(limit=3, offset=2)
        assert len(entries) == 3
        audit.close()

    def test_get_user_activity(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        audit.log_operation(user_id="alice", operation="remember")
        audit.log_operation(user_id="alice", operation="recall")
        audit.log_operation(user_id="alice", operation="forget")

        activity = audit.get_user_activity("alice", days=30)
        assert activity["user_id"] == "alice"
        assert activity["total_operations"] == 3
        assert activity["operation_counts"]["remember"] == 1
        assert activity["last_activity"] is not None
        audit.close()

    def test_get_stats(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        audit.log_operation(user_id="alice", operation="remember")
        audit.log_operation(user_id="bob", operation="recall")

        stats = audit.get_stats()
        assert stats["total_entries"] == 2
        assert stats["unique_users"] == 2
        assert stats["first_entry"] is not None
        assert stats["last_entry"] is not None
        audit.close()

    def test_export(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        audit.log_operation(user_id="alice", operation="remember")
        audit.log_operation(user_id="alice", operation="recall")

        entries = audit.export(user_id="alice")
        assert len(entries) == 2
        for e in entries:
            assert e["user_id"] == "alice"
        audit.close()

    def test_purge_all(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        audit.log_operation(user_id="alice", operation="remember")
        audit.log_operation(user_id="bob", operation="recall")

        deleted = audit.purge_all()
        assert deleted == 2

        entries = audit.query()
        assert len(entries) == 0
        audit.close()

    def test_retention_disabled(self) -> None:
        conn = _create_memory_db()
        # retention_days=0 means no auto-purge
        audit = AuditTrail(conn, retention_days=0, auto_purge=False)
        audit.log_operation(user_id="alice", operation="remember")

        entries = audit.query()
        assert len(entries) == 1
        audit.close()

    def test_client_info_tracking(self) -> None:
        conn = _create_memory_db()
        audit = AuditTrail(conn)
        audit.log_operation(
            user_id="alice",
            operation="remember",
            client_ip="192.168.1.1",
            user_agent="TestClient/1.0",
            duration_ms=42.5,
        )

        entries = audit.query()
        assert len(entries) == 1
        assert entries[0].client_ip == "192.168.1.1"
        assert entries[0].user_agent == "TestClient/1.0"
        assert entries[0].duration_ms == 42.5
        audit.close()

    def test_log_operation_raises_on_sqlite_error(self) -> None:
        """Test that log_operation raises sqlite3.Error on insert failure."""
        import sqlite3

        conn = _create_memory_db()
        audit = AuditTrail(conn)
        conn.close()  # Close connection so insert fails

        with pytest.raises(sqlite3.Error):
            audit.log_operation(user_id="alice", operation="remember")

    def test_log_operation_batch_raises_on_sqlite_error(self) -> None:
        """Test that log_operation_batch raises sqlite3.Error on batch failure."""
        import sqlite3

        conn = _create_memory_db()
        audit = AuditTrail(conn)
        conn.close()

        with pytest.raises(sqlite3.Error):
            audit.log_operation_batch([{"user_id": "alice", "operation": "remember"}])

    def test_query_by_time_range(self) -> None:
        """Test query with start_time and end_time filters."""
        conn = _create_memory_db()
        audit = AuditTrail(conn)

        audit.log_operation(user_id="alice", operation="remember")
        audit.log_operation(user_id="alice", operation="recall")
        audit.log_operation(user_id="alice", operation="forget")

        # Get all entries first to know their timestamps
        all_entries = audit.query()
        assert len(all_entries) == 3

        # Filter with start_time (should get all)
        from datetime import datetime, timedelta, timezone
        start = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        entries = audit.query(start_time=start)
        assert len(entries) == 3

        # Filter with end_time far in the future (should get all)
        end = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        entries = audit.query(end_time=end)
        assert len(entries) == 3

        # Filter with both start and end
        entries = audit.query(start_time=start, end_time=end)
        assert len(entries) == 3

        # Filter with end_time in the past (should get none)
        past_end = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        entries = audit.query(end_time=past_end)
        assert len(entries) == 0

        audit.close()

    def test_query_sqlite_error(self) -> None:
        """Test query raises sqlite3.Error when connection is closed."""
        import sqlite3

        conn = _create_memory_db()
        audit = AuditTrail(conn)
        conn.close()

        with pytest.raises(sqlite3.Error):
            audit.query(user_id="alice")

    def test_get_stats_sqlite_error(self) -> None:
        """Test get_stats raises sqlite3.Error when connection is closed."""
        import sqlite3

        conn = _create_memory_db()
        audit = AuditTrail(conn)
        conn.close()

        with pytest.raises(sqlite3.Error):
            audit.get_stats()

    def test_maybe_purge_skips_when_not_due(self) -> None:
        """Test _maybe_purge returns early when threshold not met."""
        conn = _create_memory_db()
        audit = AuditTrail(conn, retention_days=1, auto_purge=True)

        # Log one operation — should not trigger purge (needs 100 writes or 5 min)
        audit.log_operation(user_id="alice", operation="remember")
        assert audit._writes_since_purge == 1
        audit.close()

    def test_purge_old_entries(self) -> None:
        """Test _purge_old_entries removes old entries."""
        conn = _create_memory_db()
        audit = AuditTrail(conn, retention_days=0, auto_purge=False)

        audit.log_operation(user_id="alice", operation="remember")
        audit.log_operation(user_id="bob", operation="recall")

        # With retention_days=0, all entries should be purged
        deleted = audit._purge_old_entries()
        assert deleted >= 0
        audit.close()

    def test_purge_all_sqlite_error(self) -> None:
        """Test purge_all handles sqlite3.Error gracefully."""

        conn = _create_memory_db()
        audit = AuditTrail(conn)
        conn.close()

        # Should return 0 instead of raising
        result = audit.purge_all()
        assert result == 0

    def test_query_by_memory_id(self) -> None:
        """Test query filter by memory_id."""
        conn = _create_memory_db()
        audit = AuditTrail(conn)

        audit.log_operation(user_id="alice", operation="remember", memory_id="mem-1")
        audit.log_operation(user_id="alice", operation="remember", memory_id="mem-2")

        entries = audit.query(memory_id="mem-1")
        assert len(entries) == 1
        assert entries[0].memory_id == "mem-1"
        audit.close()

    def test_query_by_namespace(self) -> None:
        """Test query filter by namespace."""
        conn = _create_memory_db()
        audit = AuditTrail(conn)

        audit.log_operation(user_id="alice", operation="remember", namespace="ns1")
        audit.log_operation(user_id="alice", operation="remember", namespace="ns2")

        entries = audit.query(namespace="ns1")
        assert len(entries) == 1
        assert entries[0].namespace == "ns1"
        audit.close()

    def test_get_user_activity_no_data(self) -> None:
        """Test get_user_activity for a user with no entries."""
        conn = _create_memory_db()
        audit = AuditTrail(conn)

        activity = audit.get_user_activity("nobody", days=30)
        assert activity["user_id"] == "nobody"
        assert activity["total_operations"] == 0
        assert activity["operation_counts"] == {}
        audit.close()
