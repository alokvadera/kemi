"""Webhook operations: configure_webhooks, _dispatch_webhook_event.

These free functions are called by the corresponding ``Memory`` methods.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from kemi.webhooks import WebhookDispatcher, WebhookEventType, WebhookStore, build_payload

if TYPE_CHECKING:
    from kemi._memory_impl import Memory

logger = logging.getLogger(__name__)


def configure(memory: "Memory", db_path: str | None) -> None:
    """Enable webhook dispatch for memory lifecycle events."""
    if db_path is None:
        try:
            db_path = memory._store._db_path  # type: ignore[attr-defined]
        except AttributeError:
            logger.warning("Cannot determine database path for webhook store")
            return

    try:
        store = WebhookStore(db_path=db_path)
        memory._webhook_dispatcher = WebhookDispatcher(store=store)
        logger.info("Webhook dispatcher initialized (db: %s)", db_path)
    except (OSError, ValueError) as e:
        logger.warning("Failed to initialise webhook dispatcher: %s", e)


def dispatch(
    memory: "Memory",
    event: WebhookEventType,
    memory_id: str,
    user_id: str,
    snapshot: dict[str, Any] | None = None,
    previous_state: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Dispatch a webhook event if a dispatcher is configured.

    Prefers async dispatch when an event loop is running; falls back to
    synchronous dispatch otherwise (e.g. CLI commands).
    """
    if memory._webhook_dispatcher is None:
        return
    try:
        payload = build_payload(
            event=event,
            memory_id=memory_id,
            user_id=user_id,
            snapshot=snapshot,
            previous_state=previous_state,
            extra=extra or None,
        )
    except (ValueError, TypeError) as e:
        logger.warning("Webhook payload build failed for %s: %s", event.value, e)
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop — fall back to sync dispatch.
        try:
            memory._webhook_dispatcher.dispatch_sync(payload, event)
        except Exception:
            # Broad catch: webhook transports vary (HTTP, CLI subprocess, etc.)
            # Log and continue; webhooks must never break the calling operation.
            logger.warning(
                "Sync webhook dispatch failed for %s", event.value, exc_info=True
            )
        return

    # Running event loop — fire-and-forget.
    try:
        asyncio.ensure_future(
            memory._webhook_dispatcher.dispatch_async(payload, event)
        )
    except Exception:
        # Broad catch: ensure_future can raise if loop is closing, etc.
        logger.warning(
            "Async webhook dispatch failed for %s", event.value, exc_info=True
        )
