"""Public API surface regression tests (Phase 12).

Pins down every symbol exposed by ``kemi.__init__`` and the public
``kemi.<submodule>`` paths so the layout reorg cannot silently break
downstream users.
"""

from __future__ import annotations

import pytest

PUBLIC_FROM_INIT = {
    "Memory",
    "MemoryService",
    "MemoryConfig",
    "MemoryObject",
    "MemorySource",
    "LifecycleState",
    "MemoryType",
    "PluginRegistry",
    "WebhookSink",
    "AuditSink",
    "QueryCacheProvider",
    "HookSink",
    "WebhookDispatcherSink",
    "AuditTrailSink",
    "LruQueryCache",
    "CallbackHookSink",
    "KEMI_PROTOCOL_VERSION",
    "CompatibilityError",
    "CandidateMemory",
    "extract_memories",
    "remember_from_conversation",
    "LLMMemoryExtractor",
    "RegexMemoryExtractor",
    "OpenAIMemoryExtractor",
    "StaticMemoryExtractor",
    "remember_procedure",
    "recall_procedures",
    "EntityLinker",
    "NoopEntityLinker",
    "RegexEntityLinker",
    "SpacyEntityLinker",
}

CANONICAL_MODULES: dict[str, list[str]] = {
    "kemi.core": ["Memory", "MemoryService"],
    "kemi.exceptions": [
        "KemiError",
        "ConfigurationError",
        "ValidationError",
        "NotFoundError",
        "EmbeddingError",
        "StorageError",
        "MigrationError",
        "IncompatibleSchemaError",
        "EncryptionError",
        "CompatibilityError",
    ],
    "kemi.api_server": [],
    "kemi.mcp_server": [],
    "kemi.cli": [],
}


class TestPublicAPIFromInit:

    def test_all_public_symbols_importable_from_kemi(self) -> None:
        import kemi

        missing = [s for s in PUBLIC_FROM_INIT if not hasattr(kemi, s)]

        assert not missing, f"Missing public symbols: {missing}"


class TestCanonicalModulePaths:

    @pytest.mark.parametrize(
        "module_path,attrs",
        [
            (path, attrs)
            for path, attrs in CANONICAL_MODULES.items()
            if attrs
        ],
    )
    def test_canonical_module_path_works(
        self, module_path: str, attrs: list[str]
    ) -> None:
        import importlib

        mod = importlib.import_module(module_path)
        for attr in attrs:
            assert hasattr(mod, attr), (
                f"Module {module_path} is missing {attr} — "
                f"canonical module is incomplete"
            )


class TestNewModulePathsAlsoWork:

    NEW_PATHS = [
        "kemi.memory.service",
        "kemi.memory.core",
        "kemi.memory.facade",
        "kemi.memory.model",
        "kemi.memory.lifecycle",
        "kemi.memory.scoring",
        "kemi.memory.dedup",
        "kemi.memory.sanitize",
        "kemi.memory.chunker",
        "kemi.memory.versions",
        "kemi.memory.entities",
        "kemi.memory.adaptive",
        "kemi.memory.consolidation",
        "kemi.memory.procedures",
        "kemi.memory.formation",
        "kemi.infra.audit",
        "kemi.infra.background_tasks",
        "kemi.infra.encryption",
        "kemi.infra.webhooks",
        "kemi.infra.api_keys",
        "kemi.infra.observability",
        "kemi.interfaces.api",
        "kemi.interfaces.cli",
        "kemi.nlp.decomposer",
        "kemi.nlp.topics",
        "kemi.nlp.summarizer",
        "kemi.nlp.reranker",
        "kemi.nlp.graph",
    ]

    def test_all_new_paths_importable(self) -> None:
        import importlib

        failed = []
        for path in self.NEW_PATHS:
            try:
                importlib.import_module(path)
            except ImportError as e:
                failed.append((path, str(e)))
        assert not failed, f"New paths not importable: {failed}"
