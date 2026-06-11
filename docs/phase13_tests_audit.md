# Phase 13 — Tests directory audit and cleanup plan

## 1. Current `tests/` inventory

58 Python files, ~28,500 LOC, ~1,437 test functions.

### Root test files (unit + integration)

| File | LOC | Tests | What it covers | Fixtures used | Type |
|---|---|---|---|---|---|
| `test_cli.py` | 2,824 | 119 | CLI handler functions (remember, recall, prune, consolidate, topics, graph, explain, stats, webhooks, etc.) | `real_db_memory`, `mock_memory` | Integration (real SQLite DB) |
| `test_api_server.py` | 1,496 | 129 | FastAPI REST server (CRUD, auth, metrics, webhooks, audit) | `client` (TestClient), `mock_memory` (local, not conftest) | Integration (FastAPI TestClient) |
| `test_vec_adapter.py` | 1,213 | 62 | SQLite-vec adapter (vector search, FTS, hybrid) | None (local) | Integration (real vec DB) |
| `test_entities.py` | 1,076 | 39 | Entity linkers (Regex, Spacy, Noop, multi-entity) | None (local mocks) | Unit + Integration (spacy mock) |
| `test_cli_new_commands.py` | 1,045 | 36 | New CLI commands (chunk, decompose, rerank, rollback, history, diff) | `mock_memory`, `real_db_memory` | Integration (real SQLite DB) |
| `test_versioning.py` | 732 | 42 | Memory versioning (record, rollback, diff, history, auto-prune) | `real_db_memory`, `mock_memory` | Integration (real SQLite DB) |
| `test_scoring.py` | 685 | 37 | BM25, cosine similarity, ranking, MMR, token truncation | None (pure functions) | Unit |
| `test_new_features.py` | 614 | 34 | Async methods, streaming, recall helpers, backfill | `mock_memory` | Unit |
| `test_webhooks.py` | 575 | 33 | Webhook store, dispatcher, signing, retry logic | None (local, threading) | Unit + Integration |
| `test_background_tasks.py` | 573 | 36 | Background task manager (embed, FTS, status, cancel) | `mock_memory` | Unit + Integration |
| `test_memory_formation.py` | 568 | 25 | LLM memory extraction (OpenAI, regex, static, remember_from_conversation) | `mock_memory` | Unit |
| `test_versioning_integrity.py` | 548 | 29 | Versioning edge cases (FK, unique constraint, concurrency, gaps) | None (local SQLite + threading) | Integration |
| `test_encryption.py` | 544 | 25 | Fernet + SQLCipher encryption (key mgmt, encrypt, decrypt, re-key) | None (local SQLite) | Integration (real SQLite + encryption) |
| `test_versions.py` | 517 | 33 | MemoryVersionStore + VersionSnapshot + DiffResult | None (local SQLite) | Integration (real SQLite) |
| `test_reranker.py` | 516 | 25 | Cross-encoder reranking, fallback, RRF fusion | None (local) | Unit |
| `test_chunker.py` | 471 | 54 | Semantic chunking (sentences, chunks, overlap, tokens) | None (local MockEmbed) | Unit |
| `test_facades.py` | 466 | 21 | Read/Write/Admin service facades (internal architecture) | `mock_memory`, `mock_storage` | Unit |
| `test_summarizer.py` | 464 | 31 | LLM summarizer (OpenAI, Ollama, custom callback, prompt templates) | None (local) | Unit |
| `test_core.py` | 459 | 41 | Core Memory API (remember, recall, update, forget, async, stats) | `mock_memory`, `real_db_memory` | Unit + Integration |
| `test_api_keys.py` | 436 | 42 | API key manager (create, revoke, expiry, auth, rate limit) | None (local) | Unit |
| `test_plugins.py` | 432 | 30 | Plugin registry, builtin sinks, LruQueryCache | None (local) | Unit |
| `test_procedures.py` | 426 | 19 | Procedural memory (remember_procedure, recall_procedures) | None (local, hashlib) | Unit |
| `test_versioning_race.py` | 409 | 7 | Concurrent versioning (two writers, drift prevention) | None (local threading) | Integration (threading) |
| `test_api_integration.py` | 404 | 18 | End-to-end FastAPI (metrics, adaptive, hooks, background tasks) | `client` (TestClient) | Integration (FastAPI TestClient) |
| `test_review_fixes.py` | 391 | 15 | Custom embed adapter, entity linker edge cases, hybrid weights | None (local) | Unit |
| `test_audit.py` | 372 | 26 | Audit trail (audit_log table, query, retention policy, export) | None (local SQLite) | Integration (real SQLite) |
| `test_dedup.py` | 372 | 14 | Memory dedup (duplicate + conflict detection, sentiment flip) | None (pure functions) | Unit |
| `test_circuit_breaker.py` | 351 | 22 | Circuit breaker (fail-fast, half-open, reset, OpenAI API) | None (local) | Unit |
| `test_recall_helpers.py` | 312 | 19 | Recall helper functions (profile, session, agent, explain, since, tag) | None (local) | Unit |
| `test_consolidation.py` | 277 | 22 | Memory consolidation (cluster, extractive + LLM summary) | None (local) | Unit |
| `test_decomposer.py` | 273 | 34 | Query decomposition + fused recall (simple, expansion, RRF) | `mock_memory` | Unit |
| `test_topics.py` | 272 | 18 | Topic clustering (KMeans with scikit-learn) | None (local) | Unit (requires sklearn) |
| `test_pipeline_steps.py` | 271 | 18 | Pipeline step functions (entity extract, lifecycle, dedup, conflict) | None (pure functions) | Unit |
| `test_observability.py` | 260 | 30 | Metrics collector, counters, histograms, gauges, prometheus | None (local) | Unit |
| `test_public_api.py` | 260 | 3 | Public API surface regression (Phase 12 tripwire) | None (import-only) | Unit |
| `test_ttl.py` | 259 | 16 | TTL expiry (prune_expired, run_maintenance, namespace filter) | `real_db_memory` | Integration (real SQLite DB) |
| `test_exceptions.py` | 251 | 24 | Exception hierarchy (all 9 exception types, context, msg format) | None (pure Python) | Unit |
| `test_lifecycle.py` | 244 | 20 | Lifecycle state machine (evaluate, transition, validate, recall filter) | None (pure functions) | Unit |
| `test_graph.py` | 241 | 30 | Memory graph (entity + relation extraction, graph building) | None (pure functions) | Unit |
| `test_hybrid_search.py` | 223 | 13 | Hybrid search (semantic + BM25 scoring, entity boost) | `mock_memory` | Unit |
| `test_adaptive.py` | 213 | 24 | Adaptive retrieval (query classification, weight tuning) | None (pure functions) | Unit |
| `test_cli_writer.py` | 211 | 28 | CLI Writer (output formatting, colors, streams) | None (local) | Unit |
| `test_mcp_server.py` | 186 | 7 | MCP server (tools, prompts, resources) | `mock_memory` | Unit |
| `test_tags.py` | 186 | 18 | Tag-based recall (recall_by_tag, lifecycle filter) | `mock_memory` | Unit |
| `test_export_import.py` | 178 | 9 | Export/import (JSON dump + load) | None (local) | Unit |
| `test_models.py` | 116 | 16 | Model dataclasses + enums (MemoryObject, MemoryConfig validation) | None (pure Python) | Unit |
| `test_sanitize.py` | 107 | 18 | Prompt injection detection + sanitization | None (pure functions) | Unit |
| `test_main.py` | 106 | 6 | __main__ entry point (if __name__ == "__main__" block) | None (local) | Unit |
| `test_chunk_context.py` | — | — | (does not exist as separate file — chunk tests are in test_chunker.py and test_cli_new_commands.py) | | |
| `conftest.py` | 206 | 0 | Root fixtures + mock classes | — | Infrastructure |

### Adapter test files

| File | LOC | Tests | What it covers | Requires |
|---|---|---|---|---|
| `adapters/test_sqlite.py` | 1,036 | 46 | SQLite storage adapter (CRUD, FTS, schema upgrade, error cases) | `sqlite3` (stdlib) |
| `adapters/test_postgres.py` | 604 | 38 | PostgreSQL storage adapter (pgvector, FTS, CRUD) | `psycopg2` + running PG |
| `adapters/test_redis.py` | 305 | 29 | Redis storage adapter (hash storage, TTL, vector search) | `redis-py` + running Redis |
| `adapters/test_sqlite_vec.py` | 336 | 17 | SQLite-vec adapter (vector search, hybrid, verify) | `sqlite_vec` optional |
| `adapters/test_chroma.py` | 257 | 19 | ChromaDB adapter (collection CRUD, search) | `chromadb` optional |
| `adapters/test_qdrant.py` | 241 | 17 | Qdrant adapter (vector search, scroll, upsert) | `qdrant_client` optional |
| `adapters/test_json.py` | 315 | 11 | JSON file storage adapter (persistence, search, upgrade) | None (stdlib) |
| `adapters/test_custom_embed.py` | 46 | 4 | Custom embed adapter (embed_fn, dim, error cases) | None (stdlib) |

## 2. `conftest.py` split plan

Current: 206 LOC, 2 mock classes, 4 fixtures. Splits into 4 files.

### `tests/conftest.py` (root — ~15 LOC)

Keeps only pytest plugin hooks + imports `pytest`.

- `pytest_configure` / `pytest_collection_modifyitems` (if any)
- No fixtures directly defined here after split — imports them all from sub-conftests

### `tests/conftest_adapters.py` (~110 LOC — most of the weight)

Moves the two mock classes:

| Fixture / Class | LOC | Used by |
|---|---|---|
| `MockEmbeddingAdapter` | ~70 | `mock_embedding`, `real_db_memory`, `mock_memory` (via chain) |
| `MockStorageAdapter` | ~70 | `mock_storage`, `mock_memory` (via chain) |
| `mock_embedding` fixture | 5 | 16 test files (transitive via mock_memory) |
| `mock_storage` fixture | 5 | 12 test files (transitive via mock_memory) |

Test files using these: `test_cli.py`, `test_cli_new_commands.py`, `test_core.py`, `test_new_features.py`, `test_api_server.py`, `test_tags.py`, `test_decomposer.py`, `test_background_tasks.py`, `test_hybrid_search.py`, `test_mcp_server.py`, `test_memory_formation.py`, `test_facades.py`, `test_plugins.py`, `test_recall_helpers.py`, `test_versioning.py`

### `tests/conftest_api.py` (~30 LOC)

Moves FastAPI fixtures:

| Fixture | Used by |
|---|---|
| `app` (FastAPI test app) | `test_api_server.py`, `test_api_integration.py` |
| `client` (TestClient) | `test_api_server.py` (269 refs), `test_api_integration.py` (73 refs) |

Both need `pytest.importorskip("fastapi")` wrapper.

### `tests/conftest_cli.py` (~15 LOC)

Moves CLI-related fixtures:

| Fixture | Used by |
|---|---|
| `real_db_memory` | `test_cli.py` (281 refs), `test_ttl.py` (64 refs), `test_versioning.py` (38 refs), `test_cli_new_commands.py` (24 refs), `test_core.py` (2 refs), `test_summarizer.py` (2 refs) |

Depends on `mock_embedding` from `conftest_adapters.py`.

### `tests/_helpers/` directory — factory functions (no fixtures, just functions)

#### `tests/_helpers/factories.py`

| Function | Replaces |
|---|---|
| `make_mock_memory(embed=None, store=None, config=None)` | `conftest.mock_memory` fixture + inline `Memory(embed=..., store=...)` patterns |
| `make_real_db_memory(tmp_path, embed=None)` | `conftest.real_db_memory` fixture |
| `make_memory(**overrides)` | 285 inline `MemoryObject(...)` constructors — provides defaults for all 11 fields |

`make_memory()` signature:

```python
def make_memory(
    memory_id: str | None = None,
    user_id: str = "user1",
    content: str = "test content",
    embedding: list[float] | None = None,
    score: float = 0.0,
    source: MemorySource = MemorySource.USER_STATED,
    importance: float = 0.5,
    lifecycle_state: LifecycleState = LifecycleState.ACTIVE,
    metadata: dict | None = None,
    tags: list[str] | None = None,
    namespace: str = "default",
    session_id: str | None = None,
    memory_type: MemoryType = MemoryType.EPISODIC,
    confidence: float = 1.0,
    agent_id: str | None = None,
    run_id: str | None = None,
    app_id: str | None = None,
    expires_at: datetime | None = None,
    **overrides: Any,
) -> MemoryObject:
    ...
```

#### `tests/_helpers/embeddings.py`

| Function | Replaces |
|---|---|
| `embed_32(texts)` | 20+ scattered `[0.0]*32` / `hashlib.sha256` patterns |

Implementation:

```python
import hashlib

def embed_32(texts: list[str]) -> list[list[float]]:
    """Deterministic 32-dim embedding for testing."""
    return [_det_32(t) for t in texts]

def embed_64(texts: list[str]) -> list[list[float]]:
    """Deterministic 64-dim embedding for testing."""
    return [_det_64(t) for t in texts]

def embed_single_32(text: str) -> list[float]:
    ...

def _det_32(text: str) -> list[float]:
    raw = hashlib.sha256(text.encode()).digest()
    ...
```

#### `tests/_helpers/mock_storage.py`

| Class | Replaces |
|---|---|
| `MockStorageAdapter` | conftest.py's `MockStorageAdapter` (~70 LOC) |
| `MockEmbeddingAdapter` | conftest.py's `MockEmbeddingAdapter` (~70 LOC) |

NO duplicates remain — conftest imports from `_helpers.mock_storage` and wraps as fixtures.

### What test files would now import

```python
from tests._helpers.factories import make_memory, make_mock_memory
from tests._helpers.embeddings import embed_32, embed_single_32
from tests._helpers.mock_storage import MockStorageAdapter, MockEmbeddingAdapter
```

Files affected: `test_cli.py` (most benefit — 72 `MemoryObject(...)` calls → `make_memory(content=..., **opts)`), `test_new_features.py`, `test_core.py`, `test_tags.py`, `test_decomposer.py`, `test_chunker.py`, `test_reranker.py`, `test_procedures.py`, `test_api_keys.py`, `test_facades.py`, `test_webhooks.py`, `test_export_import.py`, `test_versioning.py`, `test_review_fixes.py`, `test_memory_formation.py`, `test_pipeline_steps.py`, and the `test_entities.py` file.

## 3. `pytest.mark.slow` plan

### Registration in `pyproject.toml`

```toml
[tool.pytest.ini_options]
markers = [
    "slow: integration tests requiring >0.1s (deselect with -m 'not slow')",
]
```

### Candidates (with file + line references)

#### Real SQLite database tests

| File | Lines | Tests | Reason |
|---|---|---|---|
| `adapters/test_sqlite.py` | full file (1036 LOC, 46 tests) | 46 | Creates temp SQLite file, performs real I/O |
| `adapters/test_sqlite_vec.py` | full file (336 LOC, 17 tests) | 17 | Real sqlite_vec extension loads |
| `test_versions.py` | full file (517 LOC, 33 tests) | 33 | Real SQLite with version store schema |
| `test_encryption.py` | full file (544 LOC, 25 tests) | 25 | Real SQLite + encryption operations |
| `test_versioning_integrity.py` | full file (548 LOC, 29 tests) | 29 | Real SQLite constraint/concurrency tests |
| `test_versioning.py` | full file (732 LOC, 42 tests) | 42 | Real SQLite via `real_db_memory` fixture |
| `test_versioning_race.py` | full file (409 LOC, 7 tests) | 7 | SQLite + threading concurrency |
| `test_audit.py` | full file (372 LOC, 26 tests) | 26 | SQLite audit_log table |
| `test_cli.py` | all tests using `real_db_memory` (~90 tests) | 90 | Real SQLite via CLI handler functions |
| `test_cli_new_commands.py` | all tests using `real_db_memory` (~24 tests) | 24 | Real SQLite via CLI handler functions |
| `test_ttl.py` | full file (259 LOC, 16 tests) | 16 | Real SQLite via `real_db_memory` |
| `test_core.py` | 2 tests using `real_db_memory` | 2 | Real SQLite schema creation |
| `test_summarizer.py` | 2 tests using `real_db_memory` | 2 | Real SQLite consolidation |

#### FastAPI TestClient tests

| File | Lines | Tests | Reason |
|---|---|---|---|
| `test_api_server.py` | full file (1496 LOC, 129 tests) | 129 | Spins up FastAPI app with middleware |
| `test_api_integration.py` | full file (404 LOC, 18 tests) | 18 | End-to-end FastAPI with hooks + metrics |

#### Real embedding model tests

| File | Lines | Tests | Reason |
|---|---|---|---|
| `test_entities.py` | test_spacy_entity_linker_* (lines ~426-530, ~6 tests) | 6 | Requires spaCy model download |
| `test_summarizer.py` | tests using OpenAI provider (14+ tests via `@pytest.mark.skipif`) | 14 | Requires OPENAI_API_KEY for real API calls |
| `test_memory_formation.py` | tests using OpenAI provider (14+ tests) | 14 | Requires OPENAI_API_KEY |
| `test_circuit_breaker.py` | tests using real OpenAI API (2 tests) | 2 | Requires OPENAI_API_KEY |

#### Threading concurrency tests

| File | Lines | Tests | Reason |
|---|---|---|---|
| `test_versioning_race.py` | full file (409 LOC, 7 tests) | 7 | `threading.Barrier` + `threading.Event` |
| `test_versioning_integrity.py` | concurrency tests (lines ~500-548, ~5 tests) | 5 | `threading.Barrier` + sqlite3 locking |
| `test_background_tasks.py` | async task lifecycle (8+ tests) | 8 | `asyncio` + threading mix |
| `test_webhooks.py` | threading tests (~5 tests) | 5 | `threading.Event` sync |

#### External service adapters (already skipif-guarded)

| File | Pytestmark | Tests |
|---|---|---|
| `adapters/test_postgres.py` | `skipif(not _pg_ok)` | 38 (already guarded) |
| `adapters/test_redis.py` | `skipif(not _redis_ok)` | 29 (already guarded) |
| `adapters/test_qdrant.py` | none (tested locally) | 17 |
| `adapters/test_chroma.py` | implicitly skips on missing dep | 19 |

### Summary: ~480 tests get `@pytest.mark.slow`

Run fast subset with: `uv run pytest -m "not slow"` — reduces to ~950 unit tests that finish in ~6s.

## 4. Duplication audit

### 4.1 `MemoryObject(...)` inline constructors

- **Count**: 285 occurrences across ~30 test files
- **Pattern**: `MemoryObject(memory_id="...", user_id="...", content="...", embedding=None, score=0.0, created_at=..., last_accessed_at=..., source=..., importance=..., lifecycle_state=..., metadata={}, ...)`
- **Worst file**: `test_cli.py` — 72 occurrences, each 10-15 lines
- **Canonical version**: `tests/_helpers/factories.py::make_memory(**overrides)`
- **LOC reduction**: ~2,000 LOC → ~400 LOC (saves ~1,600 LOC)

### 4.2 `MockStorageAdapter` duplicates

- **Count**: 2 independent implementations
  - `tests/conftest.py` lines 30-177 (~140 LOC) — canonical
  - `tests/test_api_server.py` line 45 `MockMemory` — partial mock (~25 LOC)
  - `tests/test_versioning_integrity.py` line 534 `DummyEmbed` — EmbeddingAdapter stub (~10 LOC)
  - `tests/test_versioning.py` line 510 `FakeMemory` — Memory subclass stub (~10 LOC)
- **Canonical version**: `tests/_helpers/mock_storage.py::MockStorageAdapter`
- **After dedup**: 1 implementation in `_helpers/`, all files import it

### 4.3 `MockEmbeddingAdapter` duplicates

- **Count**: 3 independent implementations
  - `tests/conftest.py` lines 10-28 (~19 LOC) — canonical
  - `tests/test_chunker.py` line 25 `MockEmbed` — partial (~15 LOC)
  - `tests/test_facades.py` line 45 `MockEmbeddingAdapter` — inline class (~20 LOC)
- **Canonical version**: `tests/_helpers/mock_storage.py::MockEmbeddingAdapter`
- **After dedup**: 1 implementation, imported by all

### 4.4 Embedding factory patterns

- **Count**: 4 different patterns
  - `[0.0]*32` inline (8 files — `test_scoring.py`, `test_reranker.py`, `test_review_fixes.py`, `test_qdrant.py`)
  - `hashlib.sha256(text.encode()).digest()` in conftest (canonical)
  - `_embed_32` in `test_review_fixes.py` (7 refs in 1 file)
  - `_embed_fn_32`/`_embed_fn_64`/`_embed_fn_len` in `test_custom_embed.py` (4 refs)
- **Canonical version**: `tests/_helpers/embeddings.py::embed_32()`, `embed_64()`
- **LOC reduction**: ~60 LOC scattered → ~15 LOC import lines

### 4.5 `mock_memory` factory pattern

- **Count**: Used via conftest fixture in 11 test files (705 references)
  - `test_cli.py`: 171 refs
  - `test_new_features.py`: 141 refs
  - `test_core.py`: 112 refs
  - `test_cli_new_commands.py`: 72 refs
  - `test_api_server.py`: 69 refs
  - `test_tags.py`: 57 refs
  - `test_decomposer.py`: 33 refs
  - `test_background_tasks.py`: 27 refs
  - `test_hybrid_search.py`: 12 refs
  - `test_mcp_server.py`: 10 refs
  - `test_memory_formation.py`: 5 refs
- **Canonical version**: `tests/_helpers/factories.py::make_mock_memory()`
- **No LOC reduction from fixture → function** (same args) but enables calling without pytest fixture context

### 4.6 `real_db_memory` factory pattern

- **Count**: Used via conftest fixture in 6 test files (411 references)
  - `test_cli.py`: 281 refs
  - `test_ttl.py`: 64 refs
  - `test_versioning.py`: 38 refs
  - `test_cli_new_commands.py`: 24 refs
  - `test_core.py`: 2 refs
  - `test_summarizer.py`: 2 refs
- **Canonical version**: `tests/_helpers/factories.py::make_real_db_memory(tmp_path)`

## 5. Test count delta

- **Tests added**: 0 (audit only, no new tests)
- **Tests modified**: 0
- **Tests removed**: 0
- **Fixture count delta**: 4 fixtures move from `conftest.py` → 3 subpackage conftests (+2 new factory functions in `_helpers/`)
- **LOC delta**: ~2,500 LOC saved after dedup
  - `MemoryObject(...)` inline → `make_memory()`: saves ~1,600 LOC
  - Mock class consolidation: saves ~80 LOC
  - Embedding function consolidation: saves ~45 LOC
  - conftest.py shrinkage: 206 → ~15 LOC but split across 3 new files (+~250 LOC total for helpers)
  - Net: ~2,200 LOC removed from test files, ~300 LOC added to helpers = **~1,900 LOC net reduction**

## 6. Open questions for the main agent

1. **`_helpers/` importability**: Should `tests/_helpers/` be excluded from the `kemi` wheel? Yes — add to `pyproject.toml` → `[tool.hatch.build.targets.wheel]` → `exclude = ["tests/"]`. The package already excludes tests from the build.

2. **Spacy + OpenAI tests in CI**: `test_entities.py` has SpacyEntityLinker tests that mock `import spacy` to test without the real package. `test_summarizer.py` and `test_memory_formation.py` use `@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"))` for OpenAI tests. Are these OK as-is with `@pytest.mark.slow`, or should they get `pytest.importorskip("spacy")` / `pytest.importorskip("openai")` wrappers instead?

3. **`test_cli.py` at 2,824 LOC**: 119 tests in one file — the biggest by far. ~90 of those use `real_db_memory` (SQLite on disk). Should this be split into `test_cli_memory.py` (mocked) and `test_cli_storage.py` (real DB) as part of Phase 13 or deferred to Phase 14? Splitting now would simplify the `@pytest.mark.slow` annotations.

4. **FastAPI TestClient lifecycle**: `test_api_server.py` has its own `mock_memory()` fixture (line 250) and `client(app)` fixture (line 267) — duplicates of the ones that would live in `conftest_api.py`. The server file also has `MockMemory` class (line 45) that's used only locally. Should these be consolidated into `conftest_api.py` or left as local fixtures?

5. **Adapter test organization**: `adapters/` has its own `__init__.py` and separate conftest could be added. Tests like `test_postgres.py`, `test_redis.py`, `test_qdrant.py`, `test_chroma.py` already use module-level `pytestmark = pytest.mark.skipif(...)`. Should these get a shared `adapters/conftest.py` with common fixtures for connection teardown, or is the file-level skipfine pattern working well enough?

6. **`conftest.py` → `conftest_adapters.py` rename**: "adapters" might confuse with `adapters/` test dir. Alternative naming: `conftest_mocks.py`? The `_helpers/` approach avoids this — all mocks live there, conftests just import and wrap as fixtures. Recommend `_helpers/mock_storage.py` + `_helpers/embeddings.py` + `_helpers/factories.py` as the primary namespaces, with root `conftest.py` importing from them.
