# Architecture

On-disk layout of the `kemi` library after the Phase 8–14 refactor and the
Phase 15 module reorg described in [phase12_layout.md](phase12_layout.md).

The package is organised by domain. The top-level `src/kemi/` directory
holds the public surface (`__init__.py`, `__main__.py`, `core.py`,
`exceptions.py`) plus thin shims that preserve the historical import paths
for the three server entry points. Everything else is grouped by concern:

| Domain        | Subpackage      | Role                                    |
|---------------|-----------------|-----------------------------------------|
| Memory core   | `memory/`       | Domain models, lifecycle, formation     |
| Persistence   | `adapters/`     | Storage and embedding backend plug-ins  |
| IO            | `operations/`   | Free-function I/O, hooks, metrics       |
| Pipelines     | `pipeline/`     | Ingestion and retrieval state machines  |
| NLP           | `nlp/`          | Reranking, decomposition, summarisation |
| Infrastructure| `infra/`        | Audit, webhooks, encryption, observ.    |
| Plugins       | `plugins/`      | Sink protocols and registry             |
| Services      | `services/`     | Read / write / admin facades            |
| Interfaces    | `interfaces/`   | FastAPI, MCP, CLI                       |
| Integrations  | `integrations/` | Third-party adapters (LangChain)        |

## 1. Top-level files

| File           | Role                                                                 |
|----------------|----------------------------------------------------------------------|
| `__init__.py`  | Public re-exports: `Memory`, `MemoryService`, `MemoryObject`, plugin classes, extractors, entity linkers, procedure helpers. |
| `__main__.py`  | `python -m kemi` entry point; starts the MCP server, falls back to the CLI if the MCP extras are not installed. |
| `core.py`      | Back-compat shim re-exporting `Memory` (`memory.facade`), `MemoryService` (`memory.service`), and `_QueryCache` (`operations._query_cache`). |
| `exceptions.py`| `KemiError` base and the eight subclass exception types. |
| `api_server.py`| One-line re-export of `kemi.interfaces.api.create_app`. |
| `cli.py`       | One-line re-export of `kemi.interfaces.cli.main`. |
| `mcp_server.py`| One-line re-export of `kemi.interfaces.mcp.main`. |
| `py.typed`     | PEP 561 marker; the package ships inline type hints. |

## 2. `memory/` — domain core

Owns the `MemoryObject` model, the read/write/admin faces of the
`MemoryService`, and every helper that operates on a memory in isolation
(scoring, chunking, dedup, lifecycle, sanitisation, formation).

| File              | Role                                                                                  |
|-------------------|---------------------------------------------------------------------------------------|
| `__init__.py`     | Subpackage docstring; no re-exports.                                                 |
| `model.py`        | `MemoryObject`, `MemoryConfig`, enums (`MemorySource`, `MemoryType`, `LifecycleState`), serialisation helpers. |
| `core.py`         | `_MemoryCore` — shared mutable state for the three service facades.                  |
| `service.py`      | `MemoryService` — the public façade. Delegates to the three services.                 |
| `facade.py`       | `Memory` — back-compat subclass of `MemoryService`; emits `DeprecationWarning`.      |
| `lifecycle.py`    | `evaluate_lifecycle`, `transition` — ACTIVE/DECAYING/ARCHIVED/DELETED transitions.    |
| `scoring.py`      | `ScoreConfig`, `cosine_similarity`, `rank_memories` — hybrid search scoring.          |
| `dedup.py`        | `find_duplicates`, `find_conflicts` — sentence-level merge / conflict detection.      |
| `chunker.py`      | `ChunkInfo`, `semantic_chunk` — embedding-based semantic text splitting.              |
| `sanitize.py`     | Prompt-injection detection and sanitisation patterns with audit logging.              |
| `entities.py`     | `EntityLinker` ABC; `NoopEntityLinker`, `RegexEntityLinker`, `SpacyEntityLinker`.     |
| `versions.py`     | `MemoryVersionStore`, `VersionSnapshot`, `DiffResult`, `RollbackResult`, `diff_memories`. |
| `consolidation.py`| Extractive + optional LLM-based summarisation of old episodic memories.               |
| `adaptive.py`     | `AdaptiveRetriever` — query classification and dynamic weight tuning.                |
| `procedures.py`   | `remember_procedure`, `recall_procedures` — procedural-memory helpers.                |
| `formation.py`    | `LLMMemoryExtractor`, `RegexMemoryExtractor`, `OpenAIMemoryExtractor`, `StaticMemoryExtractor`, `remember_from_conversation`, `extract_memories`, `CandidateMemory`. |

## 3. `adapters/` — storage and embedding backends

Pluggable backends that satisfy the ABCs declared in `adapters/base.py`.
All public storage adapters expose the same `StorageAdapter` contract; the
SQLite adapter is the default and ships the schema, FTS, and migrations.

| File                    | Role                                                            |
|-------------------------|-----------------------------------------------------------------|
| `__init__.py`           | Module docstring; no re-exports.                                |
| `base.py`               | `StorageAdapter` and `EmbeddingAdapter` ABCs.                   |
| `embedding/custom.py`   | `CustomEmbedAdapter` — delegate to a user-supplied callable.    |
| `embedding/fastembed.py`| `FastEmbedAdapter` — local BAAI/bge-small-en-v1.5 via fastembed. |
| `embedding/openai.py`   | OpenAI embedding adapter with retry and rate-limit handling.    |
| `storage/sqlite.py`     | `SQLiteStorageAdapter` — default backend; FTS, schema, migrations. |
| `storage/sqlite_vec.py` | Same as SQLite with an HNSW index via `sqlite-vec` for ANN search. |
| `storage/postgres.py`   | PostgreSQL + pgvector adapter using `psycopg_pool`.             |
| `storage/redis.py`      | Redis hash backend with Python-side cosine similarity.          |
| `storage/chroma.py`     | Chroma adapter (cosine similarity).                             |
| `storage/qdrant.py`     | Qdrant adapter (cosine similarity).                             |
| `storage/json.py`       | `JSONStorageAdapter` — file-based, useful for tests and demos.  |
| `storage/custom.py`     | `CustomStorageAdapter` — wrap user-supplied callables.          |

## 4. `operations/` — extracted free functions

Internal helpers that the `Memory` / `MemoryService` classes delegate to.
Split out of the old monolithic `core.py` during Phases 10–11 to keep the
service class slim. Not part of the public API — see
[API_STABILITY.md](API_STABILITY.md).

| File                  | Role                                                                  |
|-----------------------|-----------------------------------------------------------------------|
| `__init__.py`         | Re-exports `_QueryCache` for the legacy `kemi.operations._QueryCache` import path. |
| `_io.py`              | Orchestrator-level CRUD: `update`, `forget`, `context_block`, `stats`, `prune`, `consolidate`, `feedback`, `run_maintenance`, etc. Holds the `MemoryIORuntime` dataclass. |
| `_ops_hooks.py`       | `add_event_hook`, `remove_event_hook`, `_run_hooks`.                  |
| `_ops_metrics.py`     | `latency_tracker`, `enable_metrics`, `disable_metrics`, query-cache enable/disable. |
| `_ops_versioning.py`  | `configure_versioning`, `get_history`, `diff_versions`, `rollback_memory`. |
| `_ops_webhooks.py`    | `configure_webhooks`, `_dispatch_webhook_event`.                      |
| `_query_cache.py`     | `_QueryCache` — thread-safe LRU cache for `recall()` results.         |

## 5. `pipeline/` — ingestion and retrieval state machines

The two pipeline classes that own the candidate- and query-side flows.
The pipelines are decoupled from `Memory` / `MemoryService`; they receive
a context object holding all their dependencies so they can be exercised
in isolation.

| File           | Role                                                              |
|----------------|-------------------------------------------------------------------|
| `__init__.py`  | Empty marker.                                                     |
| `ingestion.py` | `IngestionContext`, `IngestionPipeline` — dedup → conflict → entity → store → webhook → audit. |
| `retrieval.py` | `RetrievalContext`, `RetrievalPipeline` — embed → cache check → search → score → MMR → truncate → lifecycle update → cache write. |
| `_steps.py`    | Pure-function pipeline steps extracted for unit-test isolation.   |

## 6. `nlp/` — natural-language helpers

Optional LLM and statistical helpers used by the pipelines and the
memory utilities. All of these degrade gracefully when the relevant
extras are not installed (the calling code catches `ImportError`).

| File             | Role                                                                |
|------------------|---------------------------------------------------------------------|
| `__init__.py`    | Subpackage docstring.                                               |
| `decomposer.py`  | Query decomposition + Reciprocal Rank Fusion.                       |
| `reranker.py`    | Cross-encoder reranking; stage 2 of a two-stage retrieval pipeline. |
| `summarizer.py`  | `LLMSummarizer` — OpenAI / Anthropic / Ollama / custom callable.    |
| `topics.py`      | Local CPU-only topic clustering (scikit-learn).                     |
| `graph.py`       | Zero-dependency entity / relation extraction.                      |

## 7. `infra/` — cross-cutting infrastructure

Subsystems consumed by the memory core and the interfaces. None of them
depend on `MemoryService` directly.

| File                  | Role                                                              |
|-----------------------|-------------------------------------------------------------------|
| `__init__.py`         | Subpackage docstring.                                             |
| `audit.py`            | Compliance-grade operation log with retention + export.           |
| `background_tasks.py` | `BackgroundTaskManager` for async embedding and FTS rebuilds.     |
| `encryption.py`       | SQLCipher full-DB encryption and Fernet field-level encryption.   |
| `webhooks.py`         | `WebhookDispatcher`, `WebhookEventType`, `build_payload`, signing. |
| `api_keys.py`         | Hashed API-key storage in the SQLite database (schema v8).        |
| `observability.py`    | Prometheus-compatible in-memory metrics.                          |

## 8. `plugins/` — extension contracts

The four Protocol classes and their default implementations. See
[§4 Plugin extension points](#4-plugin-extension-points) for details.

| File          | Role                                                                |
|---------------|---------------------------------------------------------------------|
| `__init__.py` | Re-exports the four Protocols and the built-in sink classes.        |
| `protocols.py`| `WebhookSink`, `AuditSink`, `QueryCacheProvider`, `HookSink` Protocols. |
| `builtin.py`  | Built-in adapters that satisfy the Protocols.                      |
| `registry.py` | `PluginRegistry` — holds the active plugins on a `MemoryService`.   |

## 9. `services/` — read / write / admin facades

The three service objects that the public `MemoryService` class
composes. They share a single `_MemoryCore` so they can be constructed
and tested independently.

| File               | Role                                                          |
|--------------------|---------------------------------------------------------------|
| `__init__.py`      | Re-exports `MemoryReadService`, `MemoryWriteService`, `MemoryAdminService`. |
| `read_service.py`  | `MemoryReadService` — `recall`, `recall_stream`, `stats`, `list_users`, graph queries. |
| `write_service.py` | `MemoryWriteService` — `remember`, `update`, `forget`, `migrate`, `feedback`. |
| `admin_service.py` | `MemoryAdminService` — configuration, maintenance, plugin / version / webhook / audit wiring. |

## 10. `interfaces/` — user-facing entry points

Three independent servers, one per transport. Each lives in its own
subpackage so the corresponding extras install can pull in just the
dependencies it needs.

| File                          | Role                                                          |
|-------------------------------|---------------------------------------------------------------|
| `__init__.py`                 | Subpackage docstring.                                         |
| `api/__init__.py`             | Re-exports `create_app` (FastAPI factory).                    |
| `api/app.py`                  | `create_app` — FastAPI app with rate limiting, API-key auth.   |
| `cli/__init__.py`             | Re-exports the CLI subcommand functions and `main`.           |
| `cli/main.py`                 | `argparse` CLI; one function per subcommand.                  |
| `cli/writer.py`               | `ConsoleWriter`, `JsonWriter`, `SilentWriter` output helpers. |
| `mcp/__init__.py`             | Re-exports `main` (MCP stdio server).                         |
| `mcp/server.py`               | MCP server exposing kemi operations to MCP clients.           |

## 11. `integrations/` — third-party adapters

| File                | Role                                                  |
|---------------------|-------------------------------------------------------|
| `__init__.py`       | Empty marker.                                         |
| `langchain.py`      | `KemiMemory` — LangChain `BaseChatMemory` backend.    |

## 12. Data flow

### 12.1 Ingest path

`MemoryService.add()` (a thin alias for `MemoryWriteService.remember`)
orchestrates the flow. Pre-processing (sanitisation, embedding,
`MemoryObject` construction, pre-hooks) is the facade's responsibility;
the ingestion pipeline owns everything that happens once a candidate
`MemoryObject` exists.

```
caller
  │
  ▼
MemoryService.remember()           (memory/service.py)
  │   validates inputs
  │   runs sanitise (memory/sanitize.py) if requested
  │   embeds content (adapters/embedding/*)
  │   builds MemoryObject (memory/model.py)
  │   fires pre-hooks       (plugins/builtin.py:CallbackHookSink)
  │
  ▼
MemoryWriteService.remember()      (services/write_service.py)
  │   builds IngestionContext
  │
  ▼
IngestionPipeline.ingest()         (pipeline/ingestion.py)
  │   delegates to pure steps in (pipeline/_steps.py):
  │     1. find_duplicates       (memory/dedup.py)
  │     2. find_conflicts        (memory/dedup.py)
  │     3. extract_entities      (memory/entities.py)
  │     4. store (StorageAdapter.add)        ──▶  adapters/storage/*
  │     5. dispatch webhook      (infra/webhooks.py)
  │     6. append audit record   (infra/audit.py)
  │   fires post-hooks      (plugins/builtin.py:CallbackHookSink)
  │
  ▼
returns memory_id to caller
```

### 12.2 Recall path

`MemoryService.search()` (a thin alias for `MemoryReadService.recall`)
handles validation and latency tracking. The retrieval pipeline owns
the rest. Reranking is opt-in: a non-`None` `reranker` argument causes
`nlp/reranker.py` to run after the storage search.

```
caller
  │
  ▼
MemoryService.search() / .recall()  (memory/service.py)
  │   validates inputs
  │   starts latency tracker      (operations/_ops_metrics.py)
  │   resolves defaults
  │
  ▼
MemoryReadService.recall()          (services/read_service.py)
  │   builds RetrievalContext
  │
  ▼
RetrievalPipeline.retrieve()        (pipeline/retrieval.py)
  │   delegates to pure steps in (pipeline/_steps.py):
  │     1. _embed_query        ──▶  adapters/embedding/*
  │     2. _check_cache        ──▶  operations/_query_cache.py
  │     3. fire pre-hook       (plugins/builtin.py:CallbackHookSink)
  │     4. _search_storage     ──▶  adapters/storage/*
  │     5. _validate_embedding_dim
  │     6. _build_entity_maps   (memory/entities.py)
  │     7. _rank / score       (memory/scoring.py, memory/adaptive.py)
  │     8. optional MMR rerank (pipeline/retrieval.py)
  │     9. optional cross-encoder rerank ──▶ nlp/reranker.py
  │    10. _truncate to token budget
  │    11. _update_lifecycle   (memory/lifecycle.py)
  │    12. increment metrics   (infra/observability.py)
  │    13. _cache_results      ──▶  operations/_query_cache.py
  │    14. _adaptive_feedback  (memory/adaptive.py)
  │   fires post-hooks
  │
  ▼
returns list[MemoryObject] to caller
```

## 13. Plugin extension points

The four Protocol classes live in
[`src/kemi/plugins/protocols.py`](../src/kemi/plugins/protocols.py).
Each Protocol is `@runtime_checkable`; any object with matching methods
satisfies the contract structurally.

| Protocol              | Slot on `MemoryService`           | Built-in implementation                    |
|-----------------------|-----------------------------------|--------------------------------------------|
| `WebhookSink`         | `add_webhook_sink()`              | `WebhookDispatcherSink` (infra/webhooks.py) |
| `AuditSink`           | `add_audit_sink()`                | `AuditTrailSink` (infra/audit.py)          |
| `QueryCacheProvider`  | `set_query_cache()` / `disable_query_cache()` | `LruQueryCache` (operations/_query_cache.py) |
| `HookSink`            | `add_hook_sink()`                 | `CallbackHookSink` (plugins/builtin.py)    |

All four are documented in the [API stability tiers](API_STABILITY.md):
the Protocols are **Stable** (no breaking changes without a deprecation
cycle), while the built-in implementations are **Additive** (new methods
allowed; existing surface preserved).

## 14. Exception hierarchy

All custom exceptions descend from `KemiError` in
[`src/kemi/exceptions.py`](../src/kemi/exceptions.py). Several subclasses
also mix in stdlib exception classes so existing `try/except` blocks that
catch `ValueError`, `LookupError`, `OSError`, or `RuntimeError` continue
to work.

```
Exception
└── KemiError                          (message + **context kwargs)
    ├── ConfigurationError             — Memory / adapter misconfigured
    ├── ValidationError                (ValueError, KemiError)  — bad input
    ├── NotFoundError                  (LookupError, KemiError) — record missing
    ├── EmbeddingError                 (RuntimeError, KemiError) — embedder failure
    ├── StorageError                   (OSError, KemiError)     — backend failure
    ├── MigrationError                 (RuntimeError, KemiError) — schema migration / re-embed failure
    ├── IncompatibleSchemaError        (RuntimeError, KemiError) — stored schema version is unsupported
    └── EncryptionError                (RuntimeError, KemiError) — encryption / decryption failure
```

Catch `KemiError` to handle any library error uniformly. Catch a
specific subclass (or a stdlib mixin) for narrower handling.

## 15. Cross-references

- [phase12_layout.md](phase12_layout.md) — detailed per-file mapping
  from the pre-reorg paths to the canonical post-reorg paths; the table
  in §2 of that file is the source of truth for the role labels used
  here.
- [API_STABILITY.md](API_STABILITY.md) — stability tiers (Stable /
  Additive / Experimental / Internal) and the rules that govern
  breaking changes. Modules under `kemi.operations._ops_*` and
  `kemi.core._QueryCache` are classified Internal and may change
  without notice.
- [CHANGELOG.md](../CHANGELOG.md) — release history for the v0.4.x
  refactor series.
