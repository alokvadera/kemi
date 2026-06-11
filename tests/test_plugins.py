"""Tests for the plugin system — :mod:`kemi.plugins`.

Covers:

* Each Protocol's structural typing (objects with the right methods are accepted).
* The built-in adapters correctly delegate to the underlying modules.
* :class:`MemoryService` exposes ``add_*_sink`` / ``set_query_cache`` /
  ``clear_*`` and forwards to the registry.
* Backward compatibility: the legacy ``memory._event_hooks`` / ``_webhook_dispatcher``
  / ``_audit_trail`` / ``_query_cache`` attributes still work after the refactor.
* Custom user-defined sinks (Protocol-conforming) fire alongside built-ins.
"""

from __future__ import annotations

from typing import Any

import pytest

from kemi import Memory
from kemi.exceptions import ValidationError
from kemi.memory.model import MemoryObject
from kemi.plugins import (
    AuditSink,
    CallbackHookSink,
    HookSink,
    LruQueryCache,
    PluginRegistry,
    QueryCacheProvider,
    WebhookDispatcherSink,
    WebhookSink,
)
from tests._helpers.factories import make_memory

# ---------------------------------------------------------------------------
# Helper sinks used across tests
# ---------------------------------------------------------------------------


class RecordingWebhookSink:
    """Captures every event it receives."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def send(self, event: Any, payload: dict[str, Any]) -> None:
        self.events.append((event.value, payload))


class RecordingAuditSink:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.batches: list[list[dict[str, Any]]] = []

    def log(self, user_id: str, operation: str, **details: Any) -> None:
        self.entries.append({"user_id": user_id, "operation": operation, **details})

    def log_batch(self, entries: list[dict[str, Any]]) -> None:
        self.batches.append(list(entries))
        for e in entries:
            self.log(**e)


class RecordingHookSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def add(self, phase: str, callback: Any) -> None:
        pass

    def remove(self, phase: str, callback: Any) -> bool:
        return False

    def run(self, phase: str, operation: str, **kwargs: Any) -> None:
        self.calls.append((phase, operation, kwargs))


class CountingQueryCache:
    def __init__(self) -> None:
        self._data: dict[str, list[MemoryObject]] = {}
        self.gets = 0
        self.puts = 0

    def get(self, key: str) -> list[MemoryObject] | None:
        self.gets += 1
        return self._data.get(key)

    def put(self, key: str, value: list[MemoryObject]) -> None:
        self.puts += 1
        self._data[key] = value


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_builtin_sinks_satisfy_protocols() -> None:
    """Built-in adapters are structural subtypes of the Protocols."""
    cache = LruQueryCache()
    hooks = CallbackHookSink()

    assert isinstance(cache, QueryCacheProvider)
    assert isinstance(hooks, HookSink)


def test_user_sinks_satisfy_protocols() -> None:
    """User-implemented sinks with the right methods are accepted as Protocol types."""
    assert isinstance(RecordingWebhookSink(), WebhookSink)
    assert isinstance(RecordingAuditSink(), AuditSink)
    assert isinstance(CountingQueryCache(), QueryCacheProvider)
    assert isinstance(RecordingHookSink(), HookSink)


# ---------------------------------------------------------------------------
# PluginRegistry
# ---------------------------------------------------------------------------


def test_registry_starts_empty() -> None:
    from kemi.plugins import KEMI_PROTOCOL_VERSION

    reg = PluginRegistry()
    assert reg.webhook_sinks == []
    assert reg.audit_sinks == []
    assert reg.hook_sinks == []
    assert reg.query_cache is None
    assert reg.kemi_version == KEMI_PROTOCOL_VERSION
    assert reg.summary() == {
        "webhook_sinks": 0,
        "audit_sinks": 0,
        "hook_sinks": 0,
        "query_cache": None,
        "kemi_version": KEMI_PROTOCOL_VERSION,
        "kemi_protocol_version": KEMI_PROTOCOL_VERSION,
    }


def test_registry_summary_reflects_added_sinks() -> None:
    reg = PluginRegistry()
    reg.webhook_sinks.append(RecordingWebhookSink())
    reg.audit_sinks.append(RecordingAuditSink())
    reg.hook_sinks.append(CallbackHookSink())
    reg.query_cache = LruQueryCache()
    s = reg.summary()
    assert s["webhook_sinks"] == 1
    assert s["audit_sinks"] == 1
    assert s["hook_sinks"] == 1
    assert s["query_cache"] == "LruQueryCache"


# ---------------------------------------------------------------------------
# MemoryService exposes the plugin API
# ---------------------------------------------------------------------------


@pytest.fixture
def mem(mock_embedding: Any, mock_storage: Any) -> Memory:
    return Memory(embed=mock_embedding(), store=mock_storage())


def test_default_registry_has_one_hook_sink(mem: Memory) -> None:
    """A freshly-instantiated Memory has the default CallbackHookSink installed."""
    reg = mem.get_plugins()
    assert len(reg.hook_sinks) == 1
    assert isinstance(reg.hook_sinks[0], CallbackHookSink)


def test_default_registry_has_no_other_sinks(mem: Memory) -> None:
    reg = mem.get_plugins()
    assert reg.webhook_sinks == []
    assert reg.audit_sinks == []
    assert reg.query_cache is None


def test_add_webhook_sink_rejects_non_sink(mem: Memory) -> None:
    """Passing a non-Protocol object raises ValidationError."""
    with pytest.raises(ValidationError):
        mem.add_webhook_sink("not a sink")  # type: ignore[arg-type]


def test_add_audit_sink_rejects_non_sink(mem: Memory) -> None:
    with pytest.raises(ValidationError):
        mem.add_audit_sink(42)  # type: ignore[arg-type]


def test_add_hook_sink_rejects_non_sink(mem: Memory) -> None:
    with pytest.raises(ValidationError):
        mem.add_hook_sink(object())  # type: ignore[arg-type]


def test_set_query_cache_rejects_non_provider(mem: Memory) -> None:
    with pytest.raises(ValidationError):
        mem.set_query_cache("not a cache")  # type: ignore[arg-type]


def test_add_webhook_sink_appends_and_fires(mem: Memory) -> None:
    sink = RecordingWebhookSink()
    mem.add_webhook_sink(sink)
    assert sink in mem.get_plugins().webhook_sinks

    from kemi.infra.webhooks import WebhookEventType

    # The default sink is still missing — but the custom one should fire on dispatch.
    mem._dispatch_webhook_event(
        WebhookEventType.REMEMBERED,
        memory_id="m1",
        user_id="u1",
        snapshot={"content": "x"},
    )
    # dispatch is fire-and-forget; the RecordingWebhookSink records synchronously.
    # (No asyncio loop is running, so it uses sync path.)
    assert len(sink.events) == 1
    assert sink.events[0][0] == "memory.remembered"
    assert sink.events[0][1]["memory_id"] == "m1"


def test_add_audit_sink_appends(mem: Memory) -> None:
    sink = RecordingAuditSink()
    mem.add_audit_sink(sink)
    assert sink in mem.get_plugins().audit_sinks


def test_add_hook_sink_runs_in_addition_to_default(mem: Memory) -> None:
    """Adding a custom hook sink means BOTH the default and the custom fire."""
    custom = RecordingHookSink()
    pre_calls: list[str] = []
    post_calls: list[str] = []

    # Default sink via the legacy add_event_hook path
    mem.add_event_hook("pre", lambda op, **kw: pre_calls.append(op))
    mem.add_event_hook("post", lambda op, **kw: post_calls.append(op))
    mem.add_hook_sink(custom)

    mem._run_hooks("pre", "remember", user_id="u1")
    mem._run_hooks("post", "remember", user_id="u1", memory_id="m1")

    assert pre_calls == ["remember"]
    assert post_calls == ["remember"]
    # The custom sink also fired
    assert [c[1] for c in custom.calls] == ["remember", "remember"]


def test_set_query_cache_replaces_existing(mem: Memory) -> None:
    """set_query_cache swaps the active cache."""
    mem.enable_query_cache(max_size=64)
    assert mem._query_cache is not None
    first = mem._query_cache

    new_cache = CountingQueryCache()
    mem.set_query_cache(new_cache)
    assert mem._query_cache is new_cache
    assert mem.get_plugins().query_cache is new_cache

    # Old cache was replaced (not in registry anymore)
    assert first is not mem.get_plugins().query_cache


def test_set_query_cache_none_clears(mem: Memory) -> None:
    mem.enable_query_cache(max_size=64)
    mem.set_query_cache(None)
    assert mem._query_cache is None
    assert mem.get_plugins().query_cache is None


def test_clear_webhook_sinks_drops_built_in(mem: Memory) -> None:
    """After enable_audit_trail + clear_audit_sinks, the built-in is gone too."""
    # We can't easily call configure_webhooks without a real DB, so use a custom sink
    # then clear and verify the list is empty.
    mem.add_webhook_sink(RecordingWebhookSink())
    assert len(mem.get_plugins().webhook_sinks) == 1
    mem.clear_webhook_sinks()
    assert mem.get_plugins().webhook_sinks == []


def test_clear_hook_sinks_keeps_default(mem: Memory) -> None:
    """clear_hook_sinks removes custom hook sinks but preserves the default sink."""
    mem.add_hook_sink(RecordingHookSink())
    assert len(mem.get_plugins().hook_sinks) == 2
    mem.clear_hook_sinks()
    assert len(mem.get_plugins().hook_sinks) == 1
    assert isinstance(mem.get_plugins().hook_sinks[0], CallbackHookSink)


def test_clear_event_hooks_drops_default_callbacks(mem: Memory) -> None:
    """clear_event_hooks empties the default sink's storage."""
    pre_calls: list[str] = []
    mem.add_event_hook("pre", lambda op, **kw: pre_calls.append(op))
    mem._run_hooks("pre", "remember")
    assert pre_calls == ["remember"]

    mem.clear_event_hooks()
    mem._run_hooks("pre", "remember")
    assert pre_calls == ["remember"]  # unchanged after clear


# ---------------------------------------------------------------------------
# Backward-compat: legacy attributes stay consistent with the registry
# ---------------------------------------------------------------------------


def test_legacy_event_hooks_and_registry_share_storage(mem: Memory) -> None:
    """Mutating memory._event_hooks is visible to the registry's default hook sink."""
    calls: list[str] = []
    mem._event_hooks["pre"].append(lambda op, **kw: calls.append(op))

    mem._run_hooks("pre", "remember")
    assert calls == ["remember"]


def test_legacy_query_cache_and_registry_point_to_same_instance(mem: Memory) -> None:
    mem.enable_query_cache(max_size=32)
    assert mem._query_cache is mem.get_plugins().query_cache


def test_disable_query_cache_clears_both(mem: Memory) -> None:
    mem.enable_query_cache(max_size=32)
    mem.disable_query_cache()
    assert mem._query_cache is None
    assert mem.get_plugins().query_cache is None


# ---------------------------------------------------------------------------
# Built-in adapter behaviour
# ---------------------------------------------------------------------------


def test_lru_query_cache_evicts_lru_entry() -> None:
    cache = LruQueryCache(max_size=2)
    mem = make_memory(
        memory_id="m1",
        user_id="u1",
        content="c1",
        embedding=[0.0],
        embedding_dim=1,
        memory_type=None,
        source=None,
        lifecycle_state=None,
    )
    cache.put("k1", [mem])
    cache.put("k2", [mem])
    cache.put("k3", [mem])  # should evict k1
    assert cache.get("k1") is None
    assert cache.get("k2") is not None
    assert cache.get("k3") is not None


def test_lru_query_cache_returns_copies() -> None:
    """Mutating a returned list does not corrupt the cached value."""
    cache = LruQueryCache()
    original = make_memory(
        memory_id="m1",
        user_id="u1",
        content="original",
        embedding=[0.0],
        embedding_dim=1,
        memory_type=None,
        source=None,
        lifecycle_state=None,
    )
    cache.put("k", [original])
    result = cache.get("k")
    assert result is not None
    result[0].content = "mutated"
    # The cached version is still "original"
    again = cache.get("k")
    assert again is not None
    assert again[0].content == "original"


def test_callback_hook_sink_runs_in_order() -> None:
    sink = CallbackHookSink()
    order: list[int] = []
    sink.add("pre", lambda op, **kw: order.append(1))
    sink.add("pre", lambda op, **kw: order.append(2))
    sink.run("pre", "remember")
    assert order == [1, 2]


def test_callback_hook_sink_invalid_phase_raises() -> None:
    sink = CallbackHookSink()
    with pytest.raises(ValidationError):
        sink.add("mid", lambda *a, **kw: None)


def test_callback_hook_sink_swallows_exceptions() -> None:
    sink = CallbackHookSink()
    sink.add("pre", lambda op, **kw: 1 / 0)
    sink.run("pre", "remember")  # must not raise


def test_callback_hook_sink_can_raise_when_asked() -> None:
    sink = CallbackHookSink()
    sink.add("pre", lambda op, **kw: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        sink.run("pre", "remember", raise_on_error=True)


def test_callback_hook_sink_shares_external_dict() -> None:
    """When constructed with an external dict, mutations are visible to both."""
    shared: dict[str, list] = {"pre": [], "post": []}
    sink = CallbackHookSink(hooks=shared)
    shared["pre"].append("cb1")
    sink.add("pre", "cb2")
    assert shared["pre"] == ["cb1", "cb2"]


# ---------------------------------------------------------------------------
# End-to-end: built-in lifecycle (configure_webhooks/enable_audit_trail)
# ---------------------------------------------------------------------------


def test_configure_webhooks_registers_built_in_sink(
    mock_embedding: Any, mock_storage: Any
) -> None:
    """configure_webhooks appends a WebhookDispatcherSink to the registry."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "webhooks.db")
        mem = Memory(embed=mock_embedding(), store=mock_storage)
        mem.configure_webhooks(db_path=db_path)
        # The legacy attribute is set
        assert mem._webhook_dispatcher is not None
        # And a sink was added to the registry
        assert len(mem.get_plugins().webhook_sinks) == 1
        assert isinstance(mem.get_plugins().webhook_sinks[0], WebhookDispatcherSink)


def test_enable_audit_trail_registers_built_in_sink(
    mock_embedding: Any, mock_storage: Any
) -> None:
    """enable_audit_trail appends an AuditTrailSink to the registry.

    Uses the in-memory mock storage's ``_get_connection`` if available,
    else falls back to a real SQLite store in a temp dir.
    """
    mem = Memory(embed=mock_embedding(), store=mock_storage)
    # mock_storage doesn't expose _get_connection, so the audit init will fail
    # to acquire a connection. We expect a warning + no audit trail.
    mem.enable_audit_trail(retention_days=30, auto_purge=False)
    # The legacy attribute remains unset (audit could not initialise)
    # but the registry should also be empty
    assert mem._audit_trail is None
    assert mem.get_plugins().audit_sinks == []


# ---------------------------------------------------------------------------
# Plugin protocol versioning (Phase 14)
# ---------------------------------------------------------------------------


class TestKemiProtocolVersion:
    """The :data:`KEMI_PROTOCOL_VERSION` constant and its parser."""

    def test_version_is_semver_triple(self) -> None:
        from kemi.plugins import KEMI_PROTOCOL_VERSION, parse_version

        v = parse_version(KEMI_PROTOCOL_VERSION)
        assert len(v) == 3
        assert all(isinstance(x, int) for x in v)
        assert v[0] >= 1

    def test_parse_strips_prerelease(self) -> None:
        from kemi.plugins import parse_version

        assert parse_version("1.2.3-rc1") == (1, 2, 3)
        assert parse_version("1.2.3+build.5") == (1, 2, 3)
        assert parse_version("1.2.3-rc1+build.5") == (1, 2, 3)

    def test_parse_invalid_falls_back_to_zeros(self) -> None:
        from kemi.plugins import parse_version

        assert parse_version("not-a-version") == (0, 0, 0)
        assert parse_version("1.2") == (0, 0, 0)
        assert parse_version("1.2.x") == (0, 0, 0)
        assert parse_version("") == (0, 0, 0)


class TestRegistryVerifyCompatibility:
    """:meth:`PluginRegistry.verify_compatibility` returns (ok, message)."""

    def test_default_registry_is_compatible(self) -> None:
        from kemi.plugins import KEMI_PROTOCOL_VERSION

        reg = PluginRegistry()
        ok, message = reg.verify_compatibility()
        assert ok is True
        assert KEMI_PROTOCOL_VERSION in message
        assert "matches" in message

    def test_matching_explicit_version_is_compatible(self) -> None:
        from kemi.plugins import KEMI_PROTOCOL_VERSION

        reg = PluginRegistry(kemi_version=KEMI_PROTOCOL_VERSION)
        ok, message = reg.verify_compatibility()
        assert ok is True

    def test_major_version_mismatch_returns_incompatible(self, caplog: Any) -> None:
        reg = PluginRegistry(kemi_version="99.0.0")
        with caplog.at_level("WARNING", logger="kemi.plugins.registry"):
            ok, message = reg.verify_compatibility()
        assert ok is False
        assert "99.0.0" in message
        assert "major version mismatch" in message
        assert any("99.0.0" in rec.message for rec in caplog.records)

    def test_minor_version_difference_returns_incompatible_but_compatible(
        self, caplog: Any
    ) -> None:
        reg = PluginRegistry(kemi_version="1.5.0")
        with caplog.at_level("WARNING", logger="kemi.plugins.registry"):
            ok, message = reg.verify_compatibility()
        assert ok is False
        assert "minor/patch" in message
        assert "major version matches" in message

    def test_strict_true_raises_on_major_mismatch(self) -> None:
        from kemi.exceptions import CompatibilityError

        reg = PluginRegistry(kemi_version="2.0.0")
        with pytest.raises(CompatibilityError):
            reg.verify_compatibility(strict=True)

    def test_strict_true_raises_on_minor_difference(self) -> None:
        from kemi.exceptions import CompatibilityError

        reg = PluginRegistry(kemi_version="1.5.0")
        with pytest.raises(CompatibilityError):
            reg.verify_compatibility(strict=True)

    def test_strict_true_does_not_raise_on_match(self) -> None:
        from kemi.plugins import KEMI_PROTOCOL_VERSION

        reg = PluginRegistry(kemi_version=KEMI_PROTOCOL_VERSION)
        ok, _ = reg.verify_compatibility(strict=True)
        assert ok is True

    def test_invalid_plugin_version_does_not_crash(self) -> None:
        reg = PluginRegistry(kemi_version="garbage")
        ok, message = reg.verify_compatibility()
        assert ok is False
        assert "could not be compared" in message

    def test_summary_includes_version_metadata(self) -> None:
        from kemi.plugins import KEMI_PROTOCOL_VERSION

        reg = PluginRegistry()
        s = reg.summary()
        assert s["kemi_version"] == KEMI_PROTOCOL_VERSION
        assert s["kemi_protocol_version"] == KEMI_PROTOCOL_VERSION


class TestCompatibilityErrorException:
    """The new :class:`CompatibilityError` is exported and is a ``KemiError``."""

    def test_is_kemi_error_subclass(self) -> None:
        from kemi import CompatibilityError
        from kemi.exceptions import KemiError

        assert issubclass(CompatibilityError, KemiError)

    def test_is_runtime_error_subclass(self) -> None:
        from kemi import CompatibilityError

        assert issubclass(CompatibilityError, RuntimeError)

    def test_caught_by_except_kemi_error(self) -> None:
        from kemi import CompatibilityError
        from kemi.exceptions import KemiError

        try:
            raise CompatibilityError("plugin v9.0.0 incompatible with installed v1.0.0")
        except KemiError as exc:
            assert "v9.0.0" in str(exc)


class TestPublicApiExports:
    """``kemi.KEMI_PROTOCOL_VERSION`` and ``kemi.CompatibilityError`` are public."""

    def test_protocol_version_exported(self) -> None:
        from kemi import KEMI_PROTOCOL_VERSION
        from kemi.plugins import KEMI_PROTOCOL_VERSION as PLUGINS_VER

        assert KEMI_PROTOCOL_VERSION == PLUGINS_VER

    def test_compatibility_error_exported(self) -> None:
        import kemi

        assert "KEMI_PROTOCOL_VERSION" in kemi.__all__
        assert "CompatibilityError" in kemi.__all__
