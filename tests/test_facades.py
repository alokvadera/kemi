"""Tests for the Phase 8 facade split.

These tests verify:
1. Each facade can be constructed and operated independently.
2. The shared ``_MemoryCore`` state is observable from all facades.
3. The public surface of :class:`MemoryService` is unchanged.
4. Legacy ``_xxx`` attribute access still works (for api_server etc.).
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

from kemi import MemoryConfig
from kemi.exceptions import ValidationError
from kemi.memory.core import _MemoryCore, build_default_store
from kemi.memory.model import MemorySource, MemoryType
from kemi.services import MemoryAdminService, MemoryReadService, MemoryWriteService


@pytest.fixture
def embed(mock_embedding: type) -> Any:
    return mock_embedding()


@pytest.fixture
def store(mock_storage: type) -> Any:
    return mock_storage()


def _mock_embedding() -> Any:
    import sys

    if "tests.conftest" in sys.modules:
        cls = sys.modules["tests.conftest"].MockEmbeddingAdapter
    else:  # pragma: no cover — fall back to inline definition
        import hashlib

        from kemi.adapters.base import EmbeddingAdapter

        class MockEmbeddingAdapter(EmbeddingAdapter):
            def __init__(self) -> None:
                self._dim = 64

            def embed(self, texts):
                return [self._vec(t) for t in texts]

            def embed_single(self, text):
                return self._vec(text)

            def dimension(self):
                return self._dim

            def _vec(self, text):
                raw = hashlib.sha256(text.encode()).digest()
                expanded = raw * (self._dim // len(raw) + 1)
                return [b / 255.0 for b in expanded[: self._dim]]

        cls = MockEmbeddingAdapter
    return cls()


def _mock_storage() -> Any:
    import sys

    if "tests.conftest" in sys.modules:
        cls = sys.modules["tests.conftest"].MockStorageAdapter
    else:  # pragma: no cover — fall back to None (only used in fixture path)
        cls = None
    if cls is None:
        raise RuntimeError("MockStorageAdapter not available outside conftest")
    return cls()


@pytest.fixture
def core(embed: Any, store: Any) -> _MemoryCore:
    return _MemoryCore(
        embed=embed,
        store=store,
        config=MemoryConfig(),
        entity_linker=None,  # type: ignore[arg-type]
        encryption=None,
    )


@pytest.fixture
def read_svc(core: _MemoryCore) -> MemoryReadService:
    return MemoryReadService(core)


@pytest.fixture
def write_svc(core: _MemoryCore) -> MemoryWriteService:
    return MemoryWriteService(core)


@pytest.fixture
def admin_svc(core: _MemoryCore) -> MemoryAdminService:
    return MemoryAdminService(core)


class TestIndependentFacades:
    """Each facade operates on the same core without knowing about the others."""

    def test_write_then_read_via_independent_facades(
        self,
        core: _MemoryCore,
        write_svc: MemoryWriteService,
        read_svc: MemoryReadService,
    ) -> None:
        memory_id = write_svc.remember(
            user_id="alice",
            content="alice loves pickled herring",
            importance=0.7,
        )
        assert isinstance(memory_id, str)

        results = read_svc.recall(user_id="alice", query="herring", top_k=5)
        assert any(m.memory_id == memory_id for m in results)

    def test_admin_configure_visible_from_other_facades(
        self,
        core: _MemoryCore,
        admin_svc: MemoryAdminService,
        read_svc: MemoryReadService,
    ) -> None:
        admin_svc.enable_query_cache(max_size=16)
        assert core._query_cache is not None
        assert core._plugins.query_cache is core._query_cache

    def test_write_svc_uses_core_hooks(
        self,
        core: _MemoryCore,
        write_svc: MemoryWriteService,
    ) -> None:
        fired: list[str] = []
        core._event_hooks["pre"].append(
            lambda operation, **kwargs: fired.append(f"pre:{operation}")
        )
        core._event_hooks["post"].append(
            lambda operation, **kwargs: fired.append(f"post:{operation}")
        )

        write_svc.remember(user_id="bob", content="likes python")
        assert "pre:remember" in fired
        assert "post:remember" in fired

    def test_recall_stream_observable(
        self,
        core: _MemoryCore,
        write_svc: MemoryWriteService,
        read_svc: MemoryReadService,
    ) -> None:
        for content in ("alpha-one", "beta-two", "gamma-three"):
            write_svc.remember(user_id="alice", content=content)

        async def collect() -> list:
            results: list = []
            gen = read_svc.recall_stream(user_id="alice", query="alpha", top_k=3)
            async for m in gen:
                results.append(m)
            return results

        results = asyncio.run(collect())
        assert len(results) >= 1


class TestCoreAsStateContainer:
    """The ``_MemoryCore`` holds all mutable state, facades never duplicate it."""

    def test_core_init_creates_plugins(self, core: _MemoryCore) -> None:
        assert core._plugins is not None
        assert core._plugins.hook_sinks  # default CallbackHookSink present

    def test_facades_share_core_instance(
        self,
        core: _MemoryCore,
        read_svc: MemoryReadService,
        write_svc: MemoryWriteService,
        admin_svc: MemoryAdminService,
    ) -> None:
        assert read_svc._core is core
        assert write_svc._core is core
        assert admin_svc._core is core

    def test_core_attributes_match_legacy_names(self, core: _MemoryCore) -> None:
        # These names are referenced throughout operations/* — if any
        # are renamed without updating the helpers, runtime breaks.
        for name in (
            "_embed",
            "_store",
            "_config",
            "_entity_linker",
            "_metrics",
            "_audit_trail",
            "_adaptive_retriever",
            "_event_hooks",
            "_query_cache",
            "_version_store",
            "_webhook_dispatcher",
            "_plugins",
        ):
            assert hasattr(core, name), f"core missing legacy attr: {name}"


class TestPublicSurfaceUnchanged:
    """The public :class:`MemoryService` API must be byte-for-byte the same."""

    def _public_methods(self) -> set[str]:
        from kemi.memory.service import MemoryService

        return {
            name
            for name, obj in inspect.getmembers(MemoryService, predicate=inspect.isfunction)
            if not name.startswith("_")
        }

    def test_all_expected_public_methods_present(self) -> None:
        expected = {
            # Read
            "recall",
            "recall_many",
            "recall_stream",
            "recall_between",
            "recall_user_profile",
            "recall_session_context",
            "recall_agent_knowledge",
            "recall_explain",
            "recall_since",
            "recall_by_tag",
            "context_block",
            "list_users",
            "stats",
            "get_memory_graph",
            # Write
            "remember",
            "remember_many",
            "update",
            "update_many",
            "forget",
            "forget_many",
            "feedback",
            "backfill_entities",
            "extract_entities",
            "migrate",
            # Admin
            "configure_webhooks",
            "configure_versioning",
            "enable_audit_trail",
            "enable_query_cache",
            "disable_query_cache",
            "enable_adaptive_retrieval",
            "get_metrics",
            "get_metrics_prometheus",
            "add_event_hook",
            "remove_event_hook",
            "get_history",
            "diff_versions",
            "rollback_memory",
            "upgrade",
            "export",
            "import_from",
            "prune",
            "prune_expired",
            "consolidate",
            "cluster_topics",
            "run_maintenance",
            "get_plugins",
            "add_webhook_sink",
            "add_audit_sink",
            "add_hook_sink",
            "set_query_cache",
            "clear_webhook_sinks",
            "clear_audit_sinks",
            "clear_hook_sinks",
            "clear_event_hooks",
        }
        actual = self._public_methods()
        missing = expected - actual
        assert not missing, f"Missing public methods: {missing}"

    def test_async_methods_present(self) -> None:
        from kemi.memory.service import MemoryService

        expected_async = {
            "aremember",
            "aremember_many",
            "aupdate",
            "aupdate_many",
            "aforget",
            "aforget_many",
            "abackfill_entities",
            "arecall",
            "arecall_many",
            "arecall_since",
            "arecall_by_tag",
            "acontext_block",
            "alist_users",
            "astats",
            "aexport",
            "aimport_from",
        }
        for name in expected_async:
            assert hasattr(MemoryService, name), f"Missing async method: {name}"
            coro = inspect.unwrap(getattr(MemoryService, name))
            assert inspect.iscoroutinefunction(coro), f"{name} is not a coroutine"


class TestLegacyAttributeProxy:
    """``MemoryService`` proxies ``_xxx`` attribute access to ``_core``."""

    def test_getattr_proxies_underscore_attrs(self, embed: Any, store: Any) -> None:
        from kemi.memory.service import MemoryService

        mem = MemoryService(embed=embed, store=store)
        # Read through proxy
        assert mem._store is mem._core._store
        assert mem._embed is mem._core._embed
        assert mem._config is mem._core._config
        assert mem._plugins is mem._core._plugins

    def test_setattr_writes_to_core(self, embed: Any, store: Any) -> None:
        from kemi.memory.service import MemoryService

        mem = MemoryService(embed=embed, store=store)
        sentinel = object()
        mem._audit_trail = sentinel
        assert mem._core._audit_trail is sentinel
        # And reading through the proxy returns the same object
        assert mem._audit_trail is sentinel

    def test_public_attrs_not_proxied(self, embed: Any, store: Any) -> None:
        from kemi.memory.service import MemoryService

        mem = MemoryService(embed=embed, store=store)
        # Public methods live on MemoryService itself
        assert callable(mem.recall)
        assert callable(mem.remember)

    def test_unknown_attr_raises(self, embed: Any, store: Any) -> None:
        from kemi.memory.service import MemoryService

        mem = MemoryService(embed=embed, store=store)
        with pytest.raises(AttributeError):
            _ = mem._this_does_not_exist


class TestMemoryServiceClose:
    """Direct unit tests for ``MemoryService.close()``."""

    def test_close_calls_store_close(self, embed: Any, store: Any) -> None:
        from unittest.mock import patch

        from kemi.memory.service import MemoryService

        mem = MemoryService(embed=embed, store=store)
        with patch.object(store, "close") as mock_close:
            mem.close()
        mock_close.assert_called_once()

    def test_close_safe_when_store_has_no_close(self, embed: Any) -> None:
        from kemi.memory.service import MemoryService

        store_without_close = MagicMock(spec=["store"])
        # spec=["store"] means close doesn't exist, so hasattr returns False
        mem = MemoryService(embed=embed, store=store_without_close)
        # Should not raise
        mem.close()

    def test_close_clears_embed_and_webhook_dispatcher(self, embed: Any, store: Any) -> None:
        from kemi.memory.service import MemoryService

        mem = MemoryService(embed=embed, store=store)
        # Set some heavy references
        mem._core._embed = object()
        mem._core._webhook_dispatcher = object()
        mem._core._metrics = object()
        mem._core._query_cache = object()

        mem.close()

        assert mem._core._embed is None
        assert mem._core._webhook_dispatcher is None
        assert mem._core._metrics is None
        assert mem._core._query_cache is None

    def test_close_idempotent(self, embed: Any, store: Any) -> None:
        from unittest.mock import patch

        from kemi.memory.service import MemoryService

        mem = MemoryService(embed=embed, store=store)
        with patch.object(store, "close") as mock_close:
            mem.close()
            mem.close()
        # Should not raise and store.close should be called twice (idempotent on SQLite)
        assert mock_close.call_count == 2

    def test_close_safe_when_core_missing(self, embed: Any) -> None:
        from kemi.memory.service import MemoryService

        mem = MemoryService(embed=embed, store=store)
        # Simulate a partially-initialised object where _core is gone
        object.__setattr__(mem, "_core", None)
        mem.close()
        # Should not raise even though _core is None


class TestFacadeComposition:
    """The three facades together cover the full public API."""

    def test_all_facades_exportable(self) -> None:
        from kemi.services import MemoryAdminService, MemoryReadService, MemoryWriteService

        assert MemoryReadService.__module__ == "kemi.services.read_service"
        assert MemoryWriteService.__module__ == "kemi.services.write_service"
        assert MemoryAdminService.__module__ == "kemi.services.admin_service"

    def test_write_facade_owns_write_methods(self) -> None:
        own = {"remember", "remember_many", "update", "update_many", "forget", "forget_many"}
        for name in own:
            assert hasattr(MemoryWriteService, name), f"Write facade missing {name}"

    def test_read_facade_owns_read_methods(self) -> None:
        own = {"recall", "recall_many", "recall_stream", "stats", "list_users", "context_block"}
        for name in own:
            assert hasattr(MemoryReadService, name), f"Read facade missing {name}"

    def test_admin_facade_owns_admin_methods(self) -> None:
        own = {
            "configure_webhooks",
            "configure_versioning",
            "enable_audit_trail",
            "get_plugins",
            "add_webhook_sink",
            "add_audit_sink",
            "add_hook_sink",
            "set_query_cache",
            "upgrade",
            "export",
            "import_from",
            "prune",
            "consolidate",
            "run_maintenance",
        }
        for name in own:
            assert hasattr(MemoryAdminService, name), f"Admin facade missing {name}"


class TestMemoryCoreBuildHelpers:
    """The static helpers on ``_MemoryCore`` and the module-level ``build_default_store`` work."""

    def test_validate_remember_inputs(self) -> None:
        _MemoryCore.validate_remember_inputs("alice", "hello", 0.5, None)
        with pytest.raises(ValidationError):
            _MemoryCore.validate_remember_inputs("", "hello", 0.5, None)
        with pytest.raises(ValidationError):
            _MemoryCore.validate_remember_inputs("alice", "", 0.5, None)
        with pytest.raises(ValidationError):
            _MemoryCore.validate_remember_inputs("alice", "hello", "0.5", None)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            _MemoryCore.validate_remember_inputs("alice", "hello", 0.5, -1)

    def test_build_memory_object_basic(self) -> None:
        obj = _MemoryCore.build_memory_object(
            user_id="alice",
            content="hi",
            embedding=[0.0] * 8,
            importance=0.4,
            source=MemorySource.USER_STATED,
            metadata=None,
            tags=["a", "b"],
            namespace="default",
            session_id=None,
            memory_type=MemoryType.EPISODIC,
            confidence=0.9,
            agent_id=None,
            run_id=None,
            app_id=None,
            ttl_seconds=None,
        )
        assert obj.importance == 0.4
        assert obj.tags == ["a", "b"]
        assert obj.namespace == "default"
        assert obj.lifecycle_state.value == "active"

    def test_build_memory_object_importance_clamped(self) -> None:
        obj = _MemoryCore.build_memory_object(
            user_id="alice",
            content="hi",
            embedding=[0.0] * 4,
            importance=2.0,
            source=MemorySource.USER_STATED,
            metadata=None,
            tags=None,
            namespace="default",
            session_id=None,
            memory_type=MemoryType.EPISODIC,
            confidence=0.0,
            agent_id=None,
            run_id=None,
            app_id=None,
            ttl_seconds=None,
        )
        assert obj.importance == 1.0
        assert obj.confidence == 0.0

    def test_build_default_store_uses_sqlite(self, tmp_path, monkeypatch, embed: Any) -> None:
        monkeypatch.setenv("KEMI_DB_PATH", str(tmp_path / "x.db"))
        store = build_default_store(embed, encryption=None)
        try:
            assert store is not None
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()


def mock_embedding_factory() -> Any:  # pragma: no cover — backward compat shim
    return _mock_embedding()


def mock_storage_factory() -> Any:  # pragma: no cover — backward compat shim
    return _mock_storage()
