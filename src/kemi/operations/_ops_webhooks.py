"""Webhook operations: configure_webhooks, _dispatch_webhook_event.

These free functions are called by the corresponding ``Memory`` methods.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from kemi.infra.webhooks import WebhookDispatcher, WebhookEventType, WebhookStore, build_payload
from kemi.plugins import WebhookDispatcherSink

if TYPE_CHECKING:
    from kemi.memory.service import MemoryService

logger = logging.getLogger(__name__)


def configure(memory: MemoryService, db_path: str | None = None) -> None:
    """Enable webhook dispatch for memory lifecycle events.

    Creates a :class:`kemi.infra.webhooks.WebhookDispatcher` backed by a
    :class:`kemi.infra.webhooks.WebhookStore`, wraps it in a
    :class:`kemi.plugins.WebhookDispatcherSink`, and appends the sink to
    the plugin registry. The legacy ``memory._webhook_dispatcher`` is
    also populated for backward compatibility.
    """
    if db_path is None:
        try:
            db_path = memory._store._db_path
        except AttributeError:
            logger.warning("Cannot determine database path for webhook store")
            return

    try:
        store = WebhookStore(db_path=db_path)
        dispatcher = WebhookDispatcher(store=store)
        memory._webhook_dispatcher = dispatcher
        memory._plugins.webhook_sinks.append(WebhookDispatcherSink(dispatcher))
        logger.info("Webhook dispatcher initialized (db: %s)", db_path)
    except (OSError, ValueError) as e:
        logger.warning("Failed to initialise webhook dispatcher: %s", e)


def dispatch(
    memory: MemoryService,
    event: WebhookEventType,
    memory_id: str,
    user_id: str,
    snapshot: dict[str, Any] | None = None,
    previous_state: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Dispatch a webhook event to every :class:`WebhookSink` in the registry.

    The payload is built once and fanned out to all registered sinks. Each
    sink is responsible for its own transport (HTTP, queue, stdout, etc.)
    and for choosing sync vs. async delivery internally.
    """
    if not memory._plugins.webhook_sinks:
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

    for sink in memory._plugins.webhook_sinks:
        try:
            sink.send(event, payload)
        except Exception:
            # Broad catch: webhook transports vary (HTTP, queue, custom)
            # and must never break the calling operation.
            logger.warning(
                "Webhook sink dispatch failed for %s", event.value, exc_info=True
            )
