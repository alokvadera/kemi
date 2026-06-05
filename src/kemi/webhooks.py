"""Webhook callbacks for memory lifecycle events.

Event types: memory.remembered, memory.updated, memory.forgotten,
memory.deleted, memory.conflict, memory.consolidated.

Requires httpx (dev dependency) for HTTP dispatch.
"""

import hashlib
import hmac
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class WebhookEventType(str, Enum):
    """Enum of supported memory lifecycle event types."""

    REMEMBERED = "memory.remembered"
    UPDATED = "memory.updated"
    FORGOTTEN = "memory.forgotten"  # soft delete (lifecycle transition)
    DELETED = "memory.deleted"  # hard delete
    CONFLICT = "memory.conflict"
    CONSOLIDATED = "memory.consolidated"

    @classmethod
    def from_string(cls, value: str) -> "WebhookEventType":
        """Parse a string to an event type, raising ValueError on failure."""
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(e.value for e in cls)
            raise ValueError(f"Invalid event type '{value}'. Valid: {valid}")


# ---------------------------------------------------------------------------
# Retry config
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """Configuration for webhook retry with exponential backoff."""

    max_retries: int = 5
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0

    def delay(self, attempt: int) -> float:
        """Compute delay in seconds for a given attempt (0-indexed)."""
        delay = self.base_delay_seconds * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay_seconds)


# ---------------------------------------------------------------------------
# Webhook config model
# ---------------------------------------------------------------------------


@dataclass
class WebhookConfig:
    """Configuration for a single webhook endpoint."""

    webhook_id: str
    url: str
    events: list[WebhookEventType] = field(default_factory=list)
    secret: str = ""
    active: bool = True
    retry_config: RetryConfig = field(default_factory=RetryConfig)

    def matches_event(self, event: WebhookEventType) -> bool:
        """Return True if this webhook is active and subscribes to *event*."""
        return self.active and event in self.events


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def build_payload(
    event: WebhookEventType,
    memory_id: str,
    user_id: str,
    snapshot: dict[str, Any] | None = None,
    previous_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON-serialisable payload sent to webhook endpoints.

    Args:
        event: The lifecycle event type.
        memory_id: ID of the memory that triggered the event.
        user_id: ID of the memory's owner.
        snapshot: Current snapshot of the memory fields (optional).
        previous_state: Snapshot of the memory *before* the change (only for
            ``memory.updated``).
        extra: Any additional key/value pairs to merge into the payload.
    """
    payload: dict[str, Any] = {
        "event": event.value,
        "memory_id": memory_id,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if snapshot is not None:
        payload["snapshot"] = snapshot
    if previous_state is not None:
        payload["previous_state"] = previous_state
    if extra:
        payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# HMAC signature
# ---------------------------------------------------------------------------


def sign_payload(payload: dict[str, Any], secret: str) -> str:
    """Compute ``X-Kemi-Signature`` header value for a payload.

    Uses HMAC-SHA256 with the webhook's *secret* over the JSON-serialised
    payload (sorted keys, no whitespace).
    """
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    mac = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()


# ---------------------------------------------------------------------------
# Persistent storage
# ---------------------------------------------------------------------------

_WEBHOOKS_DDL = """
CREATE TABLE IF NOT EXISTS webhooks (
    webhook_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    events TEXT NOT NULL,            -- JSON array of event type strings
    secret TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1,
    retry_max INTEGER NOT NULL DEFAULT 5,
    retry_base_delay REAL NOT NULL DEFAULT 1.0,
    retry_max_delay REAL NOT NULL DEFAULT 60.0,
    retry_multiplier REAL NOT NULL DEFAULT 2.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class WebhookStore:
    """Persistent storage for webhook configurations.

    Uses a separate ``webhooks`` table in the same SQLite database as the
    main memory store (or any other SQLite database path).
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute(_WEBHOOKS_DDL)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, cfg: WebhookConfig) -> str:
        """Persist a new webhook config. Returns the webhook_id."""
        if not cfg.webhook_id:
            cfg.webhook_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        events_json = json.dumps([e.value for e in cfg.events])
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO webhooks
                   (webhook_id, url, events, secret, active,
                    retry_max, retry_base_delay, retry_max_delay,
                    retry_multiplier, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?,
                           ?, ?, ?,
                           ?, ?, ?)""",
                (
                    cfg.webhook_id,
                    cfg.url,
                    events_json,
                    cfg.secret,
                    1 if cfg.active else 0,
                    cfg.retry_config.max_retries,
                    cfg.retry_config.base_delay_seconds,
                    cfg.retry_config.max_delay_seconds,
                    cfg.retry_config.backoff_multiplier,
                    now,
                    now,
                ),
            )
        return cfg.webhook_id

    def get(self, webhook_id: str) -> WebhookConfig | None:
        """Retrieve a webhook config by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM webhooks WHERE webhook_id = ?",
                (webhook_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_config(row)

    def list_all(self, active_only: bool = True) -> list[WebhookConfig]:
        """Return all webhook configs, optionally only active ones."""
        with self._get_connection() as conn:
            if active_only:
                rows = conn.execute(
                    "SELECT * FROM webhooks WHERE active = 1 ORDER BY created_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM webhooks ORDER BY created_at"
                ).fetchall()
        return [self._row_to_config(r) for r in rows]

    def list_for_event(self, event: WebhookEventType) -> list[WebhookConfig]:
        """Return active webhooks that subscribe to *event*."""
        all_webhooks = self.list_all(active_only=True)
        return [w for w in all_webhooks if w.matches_event(event)]

    def delete(self, webhook_id: str) -> bool:
        """Delete a webhook config. Returns True if a row was deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM webhooks WHERE webhook_id = ?",
                (webhook_id,),
            )
        return cursor.rowcount > 0

    def update(self, cfg: WebhookConfig) -> bool:
        """Update an existing webhook config. Returns True if updated."""
        now = datetime.now(timezone.utc).isoformat()
        events_json = json.dumps([e.value for e in cfg.events])
        with self._get_connection() as conn:
            cursor = conn.execute(
                """UPDATE webhooks SET
                   url = ?, events = ?, secret = ?, active = ?,
                   retry_max = ?, retry_base_delay = ?, retry_max_delay = ?,
                   retry_multiplier = ?, updated_at = ?
                   WHERE webhook_id = ?""",
                (
                    cfg.url,
                    events_json,
                    cfg.secret,
                    1 if cfg.active else 0,
                    cfg.retry_config.max_retries,
                    cfg.retry_config.base_delay_seconds,
                    cfg.retry_config.max_delay_seconds,
                    cfg.retry_config.backoff_multiplier,
                    now,
                    cfg.webhook_id,
                ),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_config(row: sqlite3.Row) -> WebhookConfig:
        events_str = row["events"]
        try:
            events_list = json.loads(events_str) if events_str else []
        except json.JSONDecodeError:
            events_list = []
        return WebhookConfig(
            webhook_id=row["webhook_id"],
            url=row["url"],
            events=[WebhookEventType(e) for e in events_list],
            secret=row["secret"],
            active=bool(row["active"]),
            retry_config=RetryConfig(
                max_retries=row["retry_max"],
                base_delay_seconds=row["retry_base_delay"],
                max_delay_seconds=row["retry_max_delay"],
                backoff_multiplier=row["retry_multiplier"],
            ),
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class WebhookDispatcher:
    """Dispatches webhook callbacks to registered endpoints.

    Supports both synchronous (blocking) and asynchronous (non-blocking)
    dispatch with automatic retry via exponential backoff.
    """

    def __init__(self, store: WebhookStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Synchronous dispatch
    # ------------------------------------------------------------------

    def dispatch_sync(self, payload: dict[str, Any], event: WebhookEventType) -> list[dict[str, Any]]:
        """Dispatch *payload* synchronously to all subscribers of *event*.

        This blocks until each webhook call completes (including retries).
        Returns a list of result dicts (one per webhook).
        """
        results: list[dict[str, Any]] = []
        for wh in self._store.list_for_event(event):
            result = self._call_with_retry_sync(wh, payload)
            results.append(result)
        return results

    def _call_with_retry_sync(
        self,
        wh: WebhookConfig,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a single webhook synchronously with retry logic."""
        import httpx

        signature = sign_payload(payload, wh.secret)
        headers = {
            "Content-Type": "application/json",
            "X-Kemi-Signature": signature,
            "User-Agent": "kemi-webhook/1.0",
        }
        body = json.dumps(payload)

        last_error: str | None = None
        for attempt in range(wh.retry_config.max_retries):
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(wh.url, content=body, headers=headers)
                if resp.is_success:
                    logger.info(
                        "Webhook %s dispatched to %s (attempt %d)",
                        wh.webhook_id, wh.url, attempt + 1,
                    )
                    return {
                        "webhook_id": wh.webhook_id,
                        "url": wh.url,
                        "status_code": resp.status_code,
                        "success": True,
                    }
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning(
                    "Webhook %s attempt %d failed: %s",
                    wh.webhook_id, attempt + 1, last_error,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Webhook %s attempt %d error: %s",
                    wh.webhook_id, attempt + 1, last_error,
                )

            if attempt < wh.retry_config.max_retries - 1:
                delay = wh.retry_config.delay(attempt)
                logger.info("Retrying webhook %s in %.1fs...", wh.webhook_id, delay)
                time.sleep(delay)

        logger.error(
            "Webhook %s failed after %d retries: %s",
            wh.webhook_id, wh.retry_config.max_retries, last_error,
        )
        return {
            "webhook_id": wh.webhook_id,
            "url": wh.url,
            "error": last_error,
            "success": False,
        }

    # ------------------------------------------------------------------
    # Asynchronous dispatch
    # ------------------------------------------------------------------

    async def dispatch_async(
        self,
        payload: dict[str, Any],
        event: WebhookEventType,
    ) -> list[dict[str, Any]]:
        """Dispatch *payload* asynchronously to all subscribers of *event*.

        Non-blocking — returns immediately after scheduling all HTTP calls.
        Each call runs its retry loop in an async task.
        Returns a list of result dicts (one per webhook).
        """
        import asyncio
        import httpx

        async def _call(wh: WebhookConfig) -> dict[str, Any]:
            signature = sign_payload(payload, wh.secret)
            headers = {
                "Content-Type": "application/json",
                "X-Kemi-Signature": signature,
                "User-Agent": "kemi-webhook/1.0",
            }
            body = json.dumps(payload)

            last_error: str | None = None
            for attempt in range(wh.retry_config.max_retries):
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.post(wh.url, content=body, headers=headers)
                    if resp.is_success:
                        logger.info(
                            "Webhook %s dispatched to %s (attempt %d)",
                            wh.webhook_id, wh.url, attempt + 1,
                        )
                        return {
                            "webhook_id": wh.webhook_id,
                            "url": wh.url,
                            "status_code": resp.status_code,
                            "success": True,
                        }
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    logger.warning(
                        "Webhook %s attempt %d failed: %s",
                        wh.webhook_id, attempt + 1, last_error,
                    )
                except Exception as exc:
                    last_error = str(exc)
                    logger.warning(
                        "Webhook %s attempt %d error: %s",
                        wh.webhook_id, attempt + 1, last_error,
                    )

                if attempt < wh.retry_config.max_retries - 1:
                    delay = wh.retry_config.delay(attempt)
                    logger.info("Retrying webhook %s in %.1fs...", wh.webhook_id, delay)
                    await asyncio.sleep(delay)

            logger.error(
                "Webhook %s failed after %d retries: %s",
                wh.webhook_id, wh.retry_config.max_retries, last_error,
            )
            return {
                "webhook_id": wh.webhook_id,
                "url": wh.url,
                "error": last_error,
                "success": False,
            }

        tasks = [_call(wh) for wh in self._store.list_for_event(event)]
        return await asyncio.gather(*tasks)
