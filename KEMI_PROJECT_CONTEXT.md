# Kemi — Full Project Context

## Project Overview

**kemi** is a Python library for persistent memory in AI agents. Tagline: "Persistent memory for AI agents. Three methods. Zero infra."

- PyPI package: `kemi` (version 0.3.0)
- License: MIT
- Python: 3.10+
- Author: Vadera Alok
- Repo: https://github.com/alokvadera/kemi

Core promise: zero infrastructure (no Docker, no cloud), data stays local by default (SQLite), framework-agnostic, with optional PostgreSQL + pgvector for production scale.

---

## Install Options

```bash
pip install kemi                    # zero deps, bring your own embeddings
pip install "kemi[local]"           # + fastembed (local 384-dim embeddings)
pip install "kemi[openai]"         # + OpenAI embeddings (1536-dim)
pip install "kemi[postgres]"       # + psycopg + pgvector (ANN/FTS/hybrid)
pip install "kemi[all]"            # all extras
```

Other extras: `mcp`, `vec` (sqlite-vec), `redis`, `qdrant`, `chroma`, `langchain`, `encryption`, `sqlcipher`, `dev`.

---

## Core Data Models

### Enums
- `MemorySource`: USER_STATED, AGENT_INFERRED, SYSTEM_GENERATED
- `LifecycleState`: ACTIVE, DECAYING, ARCHIVED, DELETED
- `MemoryType`: EPISODIC (event-based), SEMANTIC (fact-based)

### MemoryObject (dataclass)
Fields: memory_id, user_id, content, embedding (list[float]|None), score, created_at, last_accessed_at, source, importance (0.0-1.0), lifecycle_state, metadata (dict), embedding_dim, tags (list[str]), confidence (0.0-1.0), memory_type, session_id, namespace (default "default"), version (default 1), agent_id, run_id, app_id, expires_at.

### MemoryConfig (dataclass)
- dedup_threshold: 0.85 (cosine similarity above which duplicates are merged)
- conflict_threshold: 0.65 (range 0.65-0.85 flags potential conflicts)
- decay_half_life_hours: 168.0 (7 days)
- decay_threshold_hours: 720.0 (30 days)
- default_importance: 0.5
- default_top_k: 5
- hybrid_search: True
- Scoring weights (must sum to 1.0):
  - hybrid: semantic=0.6, recency=0.25, bm25=0.15
  - no-embedding: semantic=0.5, recency=0.3, importance=0.2
- Summarizer config fields for LLM-powered consolidation

---

## Core API: Memory class (src/kemi/core.py)

The `Memory` class is the main user-facing interface. It accepts:
- `embed`: EmbeddingAdapter (default: FastEmbedAdapter if available)
- `store`: StorageAdapter (default: SQLiteStorageAdapter, auto-detects sqlite-vec if installed)
- `config`: MemoryConfig
- `encryption`: EncryptionConfig (optional field-level Fernet encryption)

### Primary Methods
- `remember(user_id, content, importance=0.5, source=USER_STATED, metadata, tags, namespace, session_id, memory_type, confidence, agent_id, run_id, app_id, ttl_seconds)` → memory_id
- `recall(user_id, query, top_k=5, max_tokens, lifecycle_filter, hybrid_search, namespace, session_id, metadata_filter)` → list[MemoryObject]
- `forget(user_id, memory_id=None)` → count deleted
- `context_block(user_id, query, top_k=5, max_tokens=1500, prefix)` → formatted string for LLM prompts
- `update(memory_id, content, importance, confidence, memory_type, metadata, tags)` → memory_id
- `migrate(user_id, new_embed_fn, batch_size=100)` → re-embeds all memories
- `export(file_path)` / `import_from(file_path)` → JSON backup/restore

### Batch / Async Methods
- `remember_many`, `recall_many`, `update_many`, `forget_many`
- `aremember`, `arecall`, `aforget`, `acontext_block`, `arecall_stream`, `aremember_many`, `arecall_many`, `aupdate_many`, `aforget_many`, `alist_users`, `astats`, `arecall_since`, `aupdate`, `arecall_by_tag`

### Advanced Methods
- `recall_explain(user_id, query)` → memories with score breakdowns (semantic, recency, bm25, importance, final_score)
- `recall_since(user_id, query, hours=24)` → time-bounded recall
- `recall_between(user_id, query, start, end)` → date range recall
- `recall_by_tag(user_id, tag)` → tag-filtered recall
- `recall_stream(user_id, query)` → async generator yielding results progressively with MMR
- `prune(user_id, max_age_days, min_importance, lifecycle_states, namespace)` → auto-delete old/low-importance memories
- `prune_expired(user_id, namespace)` → delete TTL-expired memories
- `run_maintenance(user_id, auto_prune, auto_consolidate, ...)` → one-shot maintenance
- `consolidate(user_id, namespace, min_memories, max_age_days, with_llm_summary)` → cluster old episodic memories into semantic summaries
- `cluster_topics(user_id, n_clusters, namespace)` → KMeans on embeddings (requires scikit-learn)
- `get_memory_graph(user_id, namespace)` / `extract_entities(memory_id)` → entity/relation graph (requires graph module)
- `stats(user_id)` → counts by lifecycle, source, avg importance, tag counts
- `feedback(user_id, memory_id, helpful)` → adjusts importance up/down
- `list_users()` → all user IDs

### Versioning (src/kemi/versions.py)
- `configure_versioning(db_path, max_versions_per_memory=50, auto_prune_versions=True)`
- `get_history(memory_id, limit)` → list[VersionSnapshot]
- `diff_versions(memory_id, from_version, to_version)` → DiffResult
- `rollback_memory(memory_id, target_version)` → RollbackResult
- Auto-prunes old versions when limit exceeded

### Webhooks (src/kemi/webhooks.py)
- `configure_webhooks(db_path)` → enables lifecycle event dispatch
- Events: REMEMBERED, UPDATED, DELETED, CONFLICT, CONSOLIDATED
- Supports async (event loop) and sync (CLI) dispatch
- WebhookStore persists configs in SQLite

### Event Hooks
- `add_event_hook(phase, callback)` / `remove_event_hook(phase, callback)`
- phases: "pre", "post"
- operations: remember, recall, forget, update
- `_run_hooks(phase, operation, raise_on_error, **kwargs)`

### Query Cache
- `enable_query_cache(max_size=128)` → LRU cache for recall results
- `disable_query_cache()`

### Audit Trail
- `enable_audit_trail(retention_days=365, auto_purge=True)`

### Observability
- Metrics collector (embed_total, embed_bytes_total, embed_errors_total, store_errors_total, duplicates_detected, conflicts_detected, lifecycle_transitions, total_memories, remember_many_total)
- `get_metrics()` → dict, `get_metrics_prometheus()` → Prometheus text format

---

## Storage Adapters (src/kemi/adapters/storage/)

All implement `StorageAdapter` abstract base class.

### SQLiteStorageAdapter (src/kemi/adapters/storage/sqlite.py)
- Default storage. Thread-local connections, WAL mode.
- Schema: memories table + schema_version table + memories_fts (FTS5 virtual table) + api_keys table
- CURRENT_VERSION = 8
- Migrations: incremental ALTER TABLE from version 1→8 (tags, confidence, memory_type, session_id, namespace, version, agent_id/run_id/app_id, expires_at, api_keys)
- Embedding stored as BLOB (struct.pack float32 bytes)
- Metadata/tags stored as JSON/comma-separated strings
- Field-level Fernet encryption optional (content, metadata, user_id)
- Indexes: user_id, lifecycle_state, user_lifecycle, tags, namespace, expires_at
- `search()` computes cosine similarity in Python over filtered rows
- `search_by_content()` uses native FTS5 BM25 ranking with fallback to Python BM25
- `rebuild_fts_index(user_id)` rebuilds FTS5 from main table
- `get_api_key_manager()` returns APIKeyManager bound to connection

### JSONStorageAdapter (src/kemi/adapters/storage/json.py)
- Single JSON file. NOT thread-safe.
- Same schema via JSON serialization.
- Good for debugging/inspection.

### PostgresStorageAdapter (src/kemi/adapters/storage/postgres.py)
- PostgreSQL + pgvector. Connection pool via psycopg_pool.ConnectionPool.
- DSN from constructor or PG_DSN env var.
- Schema: memories table with VECTOR(embedding_dim), JSONB metadata, TEXT[] tags
- Indexes: user_id, lifecycle_state, user_lifecycle, namespace, GIN(tags), GIN(to_tsvector('english', content)), ivfflat(embedding vector_cosine_ops)
- schema_version table for migrations (CURRENT_VERSION = 1)
- ANN search via `<=>` cosine distance operator
- FTS via `to_tsvector` / `websearch_to_tsquery` with `ts_rank`
- `search_hybrid()` combines vector_sim + fts_rank + recency with configurable weights
- `store()` uses UPSERT (`ON CONFLICT DO UPDATE`)
- MappingRow factory for dict-like row access in psycopg3

---

## Embedding Adapters (src/kemi/adapters/embedding/)

All implement `EmbeddingAdapter` ABC: embed(texts), embed_single(text), dimension()

### FastEmbedAdapter (src/kemi/adapters/embedding/fastembed.py)
- Default. Model: BAAI/bge-small-en-v1.5 (384-dim)
- Lazy-loads on first use, prints download message to stderr
- Requires: `pip install kemi[local]`

### OpenAIEmbedAdapter (src/kemi/adapters/embedding/openai.py)
- Model: text-embedding-3-small (1536-dim)
- Features: circuit breaker pattern (CLOSED/OPEN/HALF_OPEN), tenacity retry with exponential backoff + jitter, configurable timeout
- Retryable errors: 429, 500, 502, 503, 504, timeouts, connection errors
- `get_circuit_breaker_state()` for monitoring

### CustomEmbedAdapter
- Wraps any user-provided embed function.

---

## Scoring Engine (src/kemi/scoring.py)

### Functions
- `cosine_similarity(a, b)` → [-1, 1]; handles dim mismatch by truncating to min dim; numpy if available, pure Python fallback
- `temporal_recency(last_accessed, half_life_hours=168)` → exponential decay score [0, 1]
- `bm25_score(query, document)` → simple BM25 normalized to [0, 1]
- `bm25_score_corpus(query, document, corpus)` → BM25 with IDF from corpus
- `score_memory(memory, query_embedding, query, hybrid_search, corpus, weights...)` → final relevance score
  - hybrid=True: semantic*0.6 + recency*0.25 + bm25*0.15
  - hybrid=False: semantic*0.5 + recency*0.3 + importance*0.2
- `rank_memories(memories, ...)` → sorts in place by score descending
- `mmr_rerank(memories, query_embedding, top_k, lambda_param=0.7)` → Maximal Marginal Relevance for diversity
- `mmr_rerank_stream(...)` → yields progressively (used by recall_stream)
- `truncate_by_tokens(memories, max_tokens)` → respects token budget, always returns at least one

---

## Deduplication & Conflicts (src/kemi/dedup.py)

- `find_duplicates(new_memory, existing, threshold=0.85)` → cosine similarity > threshold (normalized to [0,1])
- `find_conflicts(new_memory, existing, conflict_threshold=0.65, dedup_threshold=0.85)` → similarity in (0.65, 0.85) range
- `has_sentiment_flip(text_a, text_b)` → detects negation words and sentiment shift pairs (love/hate, like/dislike, etc.)
- `resolve_duplicate(new, existing)` → LATEST_WINS: copies new content into existing memory_id, preserves created_at, updates last_accessed_at

---

## Lifecycle Management (src/kemi/lifecycle.py)

- `evaluate_lifecycle(memory, decay_threshold_hours=720)` → returns state based on last_accessed_at
- `transition(memory, new_state)` → returns new MemoryObject; validates allowed transitions
- Allowed transitions:
  - ACTIVE → DECAYING, DELETED, ARCHIVED
  - DECAYING → ACTIVE, DELETED, ARCHIVED
  - ARCHIVED → terminal
  - DELETED → terminal
- `get_recall_filter()` → [ACTIVE, DECAYING] (excludes ARCHIVED, DELETED)

---

## Sanitization (src/kemi/sanitize.py)

- `is_suspicious(content)` → detects prompt injection patterns (ignore instructions, role override, system/assistant prefixes, INST tokens, markdown instructions)
- `sanitize(content, strict=False)` → replaces suspicious patterns with [SANITIZED]; strict mode also removes role prefixes
- `sanitize_with_rejection(content)` → returns (sanitized, was_suspicious)
- Audit logging via SHA256 hash of content (never logs raw content)

---

## Topic Clustering (src/kemi/topics.py)

- `cluster_memories(store, user_id, n_clusters=3, namespace)` → KMeans on embeddings, returns dict[topic_label, list[MemoryObject]]
- Requires scikit-learn
- `_generate_topic_label()` → TF-like keyword extraction with stopword filtering

---

## API Server (src/kemi/api_server.py)

FastAPI application (`create_app(memory)`).

Endpoints:
- POST /remember, /recall, /recall/stream (SSE), /recall-explain, /forget
- PATCH /memories/{memory_id}
- POST /prune, /consolidate/{user_id}, /topics/{user_id}, /graph/{user_id}, /feedback/{user_id}
- GET /stats/{user_id}, /users, /health
- POST /tasks/embed-batch, /tasks/rebuild-fts, GET /tasks/{task_id}, /tasks, DELETE /tasks/{task_id}, /tasks/stats
- POST /admin/fts/rebuild, GET /admin/fts/stats, POST /admin/fts/verify, GET /admin/health, GET /admin/users
- GET /metrics (json or prometheus)
- POST /audit/log, /audit/query, GET /audit/stats, POST /audit/export
- POST /adaptive/analyze, GET /adaptive/user-profile/{user_id}
- POST /admin/enable-audit, /admin/enable-adaptive
- POST /api/keys (create), GET /api/keys (list), DELETE /api/keys/{key_id} (revoke)
- GET /memories/{memory_id}/history
- POST /webhooks, GET /webhooks, DELETE /webhooks/{webhook_id}

Security:
- Optional API key auth via X-API-Key header (KEMI_API_KEY_REQUIRED env)
- Rate limiting (KEMI_RATE_LIMIT_ENABLED, KEMI_RATE_LIMIT_REQUESTS, KEMI_RATE_LIMIT_WINDOW)
- CORS (KEMI_CORS_ORIGINS), TrustedHostMiddleware (KEMI_TRUSTED_HOSTS)
- Multi-tenant isolation: authed users can only access their own data

---

## MCP Server (src/kemi/mcp_server.py)

Implements Model Context Protocol server for Claude Desktop, Cursor, Continue.

Tools exposed: remember, recall, recall_stream, recall_explain, prune, stats, consolidate, topics, graph, list_users, forget, context_block

Transports:
- stdio (default): `python -m kemi.mcp_server`
- HTTP StreamableHTTP (KEMI_MCP_TRANSPORT=http, requires starlette + uvicorn)

---

## CLI (src/kemi/cli.py)

Commands: list, store, recall, recall-stream, recall-many, forget, forget-many, update, update-many, export, import, stats, list-users, prune, consolidate, topics, graph, explain, decompose, rerank, chunk, webhook (add/list/delete), history, version diff, rollback

Entry point: `kemi` (installed via `[project.scripts]` in pyproject.toml)

---

## Project Structure

```
src/kemi/
  __init__.py          # exports Memory, MemoryConfig, MemoryObject, enums
  core.py              # Memory class (main API)
  models.py            # dataclasses and enums
  scoring.py           # ranking, BM25, cosine, MMR, temporal decay
  dedup.py             # duplicate detection, conflict detection, resolution
  lifecycle.py         # state transitions and evaluation
  sanitize.py          # prompt injection detection
  topics.py            # KMeans clustering
  graph.py             # entity extraction and relation graph
  consolidation.py     # memory summarization / consolidation
  chunker.py           # semantic text chunking
  decomposer.py        # query decomposition + RRF fusion
  reranker.py          # cross-encoder reranking (Fallback + CrossEncoderReranker stubs)
  versions.py          # MemoryVersionStore, versioning, diff, rollback
  webhooks.py          # WebhookDispatcher, WebhookStore, WebhookConfig
  api_keys.py          # API key management for multi-tenant server
  background_tasks.py  # background task manager
  adaptive.py          # adaptive retrieval weights
  audit.py             # audit trail logging
  encryption.py        # Fernet field-level encryption
  observability.py     # metrics collection
  api_server.py        # FastAPI REST server
  mcp_server.py        # MCP tool server
  cli.py               # command-line interface
  adapters/
    base.py            # EmbeddingAdapter, StorageAdapter ABCs
    embedding/         # fastembed, openai, custom
    storage/           # sqlite, sqlite_vec, json, postgres, custom
  integrations/
    langchain.py       # KemiMemory adapter for LangChain
```

Tests: `tests/` — pytest with coverage, pytest-asyncio.

---

## Key Design Decisions

1. **Zero-config defaults**: `Memory()` works immediately with fastembed + SQLite
2. **Adapter pattern**: swappable embedding and storage without changing core logic
3. **Thread safety**: SQLite adapter uses thread-local connections
4. **Privacy-first**: all data local by default, no telemetry
5. **GDPR compliance**: `forget()` deletes all user data
6. **Optional dependencies**: everything beyond core is an extra
7. **Migration system**: schema_version table with incremental migrations per adapter
8. **Hybrid search**: vector + BM25 + recency weighted combination
9. **Lifecycle management**: automatic decay from ACTIVE → DECAYING based on access patterns
10. **Deduplication**: semantic merge prevents duplicate memories
11. **Conflict detection**: flags contradictory memories without merging
12. **Versioning**: opt-in history tracking per memory with rollback
