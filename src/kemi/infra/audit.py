"""Audit trail for kemi memory operations.

Provides compliance-grade operation logging with:
- Complete CRUD operation audit trail
- Queryable audit log (by user, operation type, time range)
- Retention policy support
- Export capability for compliance audits

Every memory mutation (remember, forget, update, prune, migrate, etc.)
is logged with timestamp, user ID, operation type, and details.

Stored in a separate SQLite table `audit_log` for clean separation.
Zero external dependencies beyond the existing SQLite adapter.

Usage:
    from kemi.infra.audit import AuditTrail

    audit = AuditTrail(db_connection)
    audit.log_operation("alice", "remember", {"memory_id": "abc123"})
    entries = audit.query(user_id="alice", operation="remember")
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Schema version for future migrations
AUDIT_SCHEMA_VERSION = 1

AUDIT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'success',
    details TEXT NOT NULL DEFAULT '{}',
    memory_id TEXT,
    namespace TEXT DEFAULT 'default',
    client_ip TEXT,
    user_agent TEXT,
    duration_ms REAL,
    schema_version INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_operation ON audit_log(operation);
CREATE INDEX IF NOT EXISTS idx_audit_user_time ON audit_log(user_id, timestamp);
"""


@dataclass
class AuditEntry:
    """A single audit log entry."""

    id: int = 0
    timestamp: str = ""
    user_id: str = ""
    operation: str = ""
    status: str = "success"
    details: dict[str, Any] = field(default_factory=dict)
    memory_id: str | None = None
    namespace: str = "default"
    client_ip: str | None = None
    user_agent: str | None = None
    duration_ms: float | None = None
    schema_version: int = AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "operation": self.operation,
            "status": self.status,
            "details": self.details,
            "memory_id": self.memory_id,
            "namespace": self.namespace,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "duration_ms": self.duration_ms,
        }


class AuditTrail:
    """Audit trail for compliance-grade operation logging.

    Features:
    - Automatic schema creation
    - Batch logging support
    - Retention policy (auto-purge old entries)
    - Query by user, operation, time range, status
    - Export to JSON for compliance audits
    """

    def __init__(
        self,
        db_connection: sqlite3.Connection,
        retention_days: int = 365,
        auto_purge: bool = True,
    ) -> None:
        """Initialize audit trail.

        Args:
            db_connection: SQLite connection to use.
            retention_days: Number of days to keep audit entries (default 365).
            auto_purge: If True, automatically purge old entries on log_operation.
        """
        self._conn = db_connection
        self._retention_days = retention_days
        self._auto_purge = auto_purge

        # Throttle auto-purge: run at most every 5 minutes or every 100 writes
        self._writes_since_purge: int = 0
        self._last_purge_time: float = time.time()
        self._purge_interval_seconds: float = 300.0  # 5 minutes
        self._purge_write_threshold: int = 100

        self._ensure_schema()
        logger.info(
            f"Audit trail initialized (retention: {retention_days}d, auto_purge: {auto_purge})"
        )

    def _ensure_schema(self) -> None:
        """Create audit log table and indexes if they don't exist."""
        self._conn.executescript(AUDIT_SCHEMA_SQL)
        self._conn.commit()

    def log_operation(
        self,
        user_id: str,
        operation: str,
        details: dict[str, Any] | None = None,
        memory_id: str | None = None,
        namespace: str = "default",
        status: str = "success",
        client_ip: str | None = None,
        user_agent: str | None = None,
        duration_ms: float | None = None,
    ) -> int:
        """Log a memory operation to the audit trail.

        Args:
            user_id: User who performed the operation.
            operation: Operation type (remember, recall, forget, update, etc.).
            details: Additional operation details as dict.
            memory_id: ID of the memory involved (if applicable).
            namespace: Memory namespace.
            status: Operation status (success, error, denied).
            client_ip: Client IP address (for API usage).
            user_agent: Client user agent string.
            duration_ms: Operation duration in milliseconds.

        Returns:
            The ID of the new audit entry.
        """
        self._maybe_purge()

        timestamp = datetime.now(timezone.utc).isoformat()
        details_json = json.dumps(details or {}, default=str)

        try:
            cursor = self._conn.execute(
                """INSERT INTO audit_log
                   (timestamp, user_id, operation, status, details, memory_id,
                    namespace, client_ip, user_agent, duration_ms, schema_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    timestamp,
                    user_id,
                    operation,
                    status,
                    details_json,
                    memory_id,
                    namespace,
                    client_ip,
                    user_agent,
                    duration_ms,
                    AUDIT_SCHEMA_VERSION,
                ),
            )
            self._conn.commit()
            entry_id = cursor.lastrowid or 0
            self._writes_since_purge += 1
            logger.debug(f"Audit: {operation} by {user_id} (entry {entry_id})")
            return entry_id
        except sqlite3.Error as e:
            logger.error(f"Failed to write audit entry: {e}")
            raise

    def log_operation_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> int:
        """Log multiple operations atomically.

        Args:
            entries: List of dicts with keys matching log_operation params.

        Returns:
            Number of entries logged.
        """
        self._maybe_purge()

        timestamp = datetime.now(timezone.utc).isoformat()
        count = 0

        try:
            for entry in entries:
                details_json = json.dumps(entry.get("details", {}), default=str)
                self._conn.execute(
                    """INSERT INTO audit_log
                       (timestamp, user_id, operation, status, details, memory_id,
                        namespace, client_ip, user_agent, duration_ms, schema_version)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        timestamp,
                        entry["user_id"],
                        entry["operation"],
                        entry.get("status", "success"),
                        details_json,
                        entry.get("memory_id"),
                        entry.get("namespace", "default"),
                        entry.get("client_ip"),
                        entry.get("user_agent"),
                        entry.get("duration_ms"),
                        AUDIT_SCHEMA_VERSION,
                    ),
                )
                count += 1
            self._conn.commit()
            self._writes_since_purge += count
        except sqlite3.Error as e:
            logger.error(f"Failed to write batch audit entries: {e}")
            self._conn.rollback()
            raise

        return count

    def query(
        self,
        user_id: str | None = None,
        operation: str | None = None,
        status: str | None = None,
        memory_id: str | None = None,
        namespace: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Query audit trail with flexible filters.

        Args:
            user_id: Filter by user ID.
            operation: Filter by operation type.
            status: Filter by status (success, error, denied).
            memory_id: Filter by memory ID.
            namespace: Filter by namespace.
            start_time: ISO timestamp for start of range (inclusive).
            end_time: ISO timestamp for end of range (inclusive).
            limit: Maximum number of entries to return.
            offset: Offset for pagination.

        Returns:
            List of matching AuditEntry objects.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if operation:
            conditions.append("operation = ?")
            params.append(operation)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if memory_id:
            conditions.append("memory_id = ?")
            params.append(memory_id)
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query_sql = (
            f"SELECT id, timestamp, user_id, operation, status, details, "
            f"memory_id, namespace, client_ip, user_agent, duration_ms, "
            f"schema_version "
            f"FROM audit_log WHERE {where_clause} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        try:
            cursor = self._conn.execute(query_sql, params)
            results: list[AuditEntry] = []
            for row in cursor.fetchall():
                try:
                    details = json.loads(row[5])
                except (json.JSONDecodeError, TypeError):
                    details = {}

                results.append(
                    AuditEntry(
                        id=row[0],
                        timestamp=row[1],
                        user_id=row[2],
                        operation=row[3],
                        status=row[4],
                        details=details,
                        memory_id=row[6],
                        namespace=row[7],
                        client_ip=row[8],
                        user_agent=row[9],
                        duration_ms=row[10],
                        schema_version=row[11],
                    )
                )
            return results
        except sqlite3.Error as e:
            logger.error(f"Failed to query audit trail: {e}")
            raise

    def get_user_activity(
        self,
        user_id: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get activity summary for a user.

        Args:
            user_id: User ID to query.
            days: Number of days to look back.

        Returns:
            Dict with operation counts, last activity timestamp, etc.
        """
        from datetime import timedelta

        start_time = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        try:
            cursor = self._conn.execute(
                """SELECT operation, COUNT(*) as count
                   FROM audit_log
                   WHERE user_id = ? AND timestamp >= ?
                   GROUP BY operation""",
                (user_id, start_time),
            )
            operation_counts = {row[0]: row[1] for row in cursor.fetchall()}

            cursor = self._conn.execute(
                """SELECT MAX(timestamp) FROM audit_log WHERE user_id = ?""",
                (user_id,),
            )
            last_activity = cursor.fetchone()[0]

            cursor = self._conn.execute(
                """SELECT COUNT(*) FROM audit_log
                   WHERE user_id = ? AND timestamp >= ?""",
                (user_id, start_time),
            )
            total_operations = cursor.fetchone()[0]

            return {
                "user_id": user_id,
                "period_days": days,
                "total_operations": total_operations,
                "operation_counts": operation_counts,
                "last_activity": last_activity,
            }
        except sqlite3.Error as e:
            logger.error(f"Failed to get user activity: {e}")
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get overall audit trail statistics."""
        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM audit_log")
            total_entries = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(DISTINCT user_id) FROM audit_log")
            unique_users = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM audit_log")
            row = cursor.fetchone()
            first_entry = row[0]
            last_entry = row[1]

            return {
                "total_entries": total_entries,
                "unique_users": unique_users,
                "first_entry": first_entry,
                "last_entry": last_entry,
                "retention_days": self._retention_days,
            }
        except sqlite3.Error as e:
            logger.error(f"Failed to get audit stats: {e}")
            raise

    def export(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Export audit entries as a list of dicts for compliance.

        Args:
            start_time: ISO timestamp filter.
            end_time: ISO timestamp filter.
            user_id: Optional user filter.

        Returns:
            List of dicts suitable for JSON export.
        """
        entries = self.query(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=100000,  # Large limit for exports
        )
        return [e.to_dict() for e in entries]

    def _maybe_purge(self) -> None:
        """Throttled purge: only run if enough time or writes have passed."""
        if not self._auto_purge:
            return
        now = time.time()
        if (
            now - self._last_purge_time < self._purge_interval_seconds
            and self._writes_since_purge < self._purge_write_threshold
        ):
            return
        self._last_purge_time = now
        self._writes_since_purge = 0
        self._purge_old_entries()

    def _purge_old_entries(self) -> int:
        """Remove entries older than retention_days.

        Returns:
            Number of entries purged.
        """
        if self._retention_days <= 0:
            return 0

        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._retention_days)).isoformat()

        try:
            cursor = self._conn.execute(
                "DELETE FROM audit_log WHERE timestamp < ?",
                (cutoff,),
            )
            self._conn.commit()
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Purged {deleted} old audit entries (cutoff: {cutoff})")
            return deleted
        except sqlite3.Error as e:
            logger.error(f"Failed to purge old audit entries: {e}")
            return 0

    def purge_all(self) -> int:
        """Purge all audit entries. Use with caution.

        Returns:
            Number of entries purged.
        """
        try:
            cursor = self._conn.execute("DELETE FROM audit_log")
            self._conn.commit()
            deleted = cursor.rowcount
            logger.warning(f"Purged ALL audit entries: {deleted}")
            return deleted
        except sqlite3.Error as e:
            logger.error(f"Failed to purge all audit entries: {e}")
            return 0

    def close(self) -> None:
        """Close the audit trail (no-op, connection managed externally)."""
        pass
