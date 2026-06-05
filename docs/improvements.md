# kemi — Improvement Roadmap

Current: **v0.3.0** — 9.4k LoC across 18 source files  
Stack: Python 3.10+, SQLite + FastEmbed (default), MIT license  
Core: `Memory.remember/recall/forget/context_block` + semantic dedup, conflict detection, temporal decay, lifecycle (ACTIVE→DECAYING→ARCHIVED→DELETED), hybrid vector+BM25 search, MMR reranking

---

## Already Shipped

| Area | Details |
|------|---------|
| Storage | SQLite (WAL), JSON file, sqlite-vec (ANN), Custom adapter |
| Embeddings | FastEmbed (local, default), OpenAI, Custom adapter |
| Server | FastAPI REST API, MCP server |
| Integrations | LangChain memory adapter |
| CLI | list, recall, forget, export, import, stats, users, update, prune, consolidate, topics, graph, explain |
| Advanced | Adaptive retrieval, audit trail, observability (Prometheus), background tasks, memory graph/entity extraction, topic clustering, consolidation, dedup + conflict detection, lifecycle management, input sanitization, export/import |

---

## Proposed Features

### Tier 1 — High Value, Clearly Missing

#### 1. PostgreSQL + pgvector Storage Adapter
- **File**: `src/kemi/adapters/storage/postgres.py`
- **Dep**: `psycopg[binary]>=3.2`, `pgvector>=0.3`
- **Why**: Enterprises need hosted Postgres; the only missing production-grade backend
- **Design**: Mirror `SQLiteStorageAdapter` API exactly. Connection pool via psycopg. Embeddings stored as `vector(N)` via pgvector. ANN search via `<=>` cosine operator. FTS via `tsvector` + GIN index. Schema versioning via `schema_version` table.
- **Priority**: VERY HIGH

#### 2. Redis / Qdrant / Chroma Storage Adapters
- **Why**: Each unlocks a different use case — Redis for fast ephemeral, Qdrant/Chroma for dedicated vector DB
- **Priority**: HIGH

#### 3. TTL on Memories
- **File**: Add `ttl_seconds: int | None = None` to `remember()` and `MemoryObject`
- **Why**: Critical for session-based agents and temporary context
- **Design**: Background sweeper in `background_tasks.py` transitions expired TTL memories to DELETED
- **Priority**: HIGH

#### 4. Multi-tenancy + API Keys in `api_server.py`
- **Why**: `api_server.py` currently has no auth; unusable for multi-user SaaS
- **Priority**: HIGH

#### 5. Streaming `recall`
- **Why**: Large context assembly — yield top-k incrementally instead of loading all at once
- **Priority**: MEDIUM

#### 6. Auto-Summarization on Consolidate
- **File**: Add to `consolidation.py`
- **Why**: Currently extractive-only; LLM-powered abstractive summarization adds real value
- **Design**: Flag `with_llm_summary=True`, pluggable via callback/LLM adapter
- **Priority**: MEDIUM

#### 7. Encrypted At-Rest Storage
- **Why**: Matches "your data stays yours" — SQLCipher backend or Fernet field-level encryption
- **Priority**: MEDIUM

#### 8. Versioned Memories / History
- **File**: `core.py:get_history(memory_id)`
- **Why**: `version` field exists on `MemoryObject` but no way to inspect history
- **Priority**: MEDIUM

#### 9. Webhooks on Memory Events
- **Why**: `on_remember`, `on_forget`, `on_conflict` callbacks (sync + async)
- **Priority**: MEDIUM

### Tier 2 — Medium Value

| # | Feature | Detail |
|---|---------|--------|
| 10 | Cross-user shared memories | `share(memory_id, to_user)` with ACL queries |
| 11 | Time-window recall | `recall(user, q, since=..., until=...)` — `recall_since` exists but no upper bound |
| 12 | Memory diff / merge | Sync between two Memory instances |
| 13 | Auto-tagging | `remember(..., auto_tag=True)` with pluggable LLM tagger |
| 14 | Cohere / Voyage / HF TEI embeddings | More embedding provider choices |
| 15 | LlamaIndex integration | LangChain exists but no LlamaIndex |
| 16 | Long memory compression | Chunk + link if `len(content) > N` |
| 17 | A/B retrieval testing | Log query→result→feedback, expose `compare_strategies()` |

### Tier 3 — Nice-to-Have

| # | Feature | Detail |
|---|---------|--------|
| 18 | Cross-encoder rerankers | Cohere Rerank, bge-reranker before MMR |
| 19 | Eval harness | `kemi eval --dataset ...` for Recall@k, MRR |
| 20 | PII detector | Auto-redact emails/phones in `sanitize.py` |
| 21 | Visualizer | `kemi graph --user X --html` for memory graph |
| 22 | Tombstone retention | Soft-delete with purge after N days |
| 23 | CLI REPL | `kemi shell` for interactive query/remember |
| 24 | OpenTelemetry exporter | Alongside existing Prometheus |
| 25 | Typed Protocols | `typing.Protocol` for EmbedAdapter / StorageAdapter in `adapters/base.py` |

---

## PostgreSQL + pgvector Adapter Spec

### Files to Create
- `src/kemi/adapters/storage/postgres.py` — Main adapter
- `tests/adapters/test_postgres.py` — Tests (skip if no PG)

### Files to Modify
- `pyproject.toml` — Add `[postgres]` optional dep
- `src/kemi/adapters/storage/__init__.py` — Re-export
- Optionally: `src/kemi/core.py` — Auto-detect connection string or env var

### Schema

```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384),          -- pgvector type, dimension configurable
    embedding_dim INTEGER,
    created_at TIMESTAMPTZ NOT NULL,
    last_accessed_at TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL DEFAULT 'user_stated',
    importance REAL NOT NULL DEFAULT 0.5,
    lifecycle_state TEXT NOT NULL DEFAULT 'active',
    metadata JSONB NOT NULL DEFAULT '{}',
    tags TEXT[] NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 1.0,
    memory_type TEXT NOT NULL DEFAULT 'episodic',
    session_id TEXT,
    namespace TEXT NOT NULL DEFAULT 'default',
    version INTEGER NOT NULL DEFAULT 1
);
```

### Indexes

```sql
CREATE INDEX memories_user_id_idx ON memories(user_id);
CREATE INDEX memories_lifecycle_state_idx ON memories(lifecycle_state);
CREATE INDEX memories_user_lifecycle_idx ON memories(user_id, lifecycle_state);
CREATE INDEX memories_namespace_idx ON memories(namespace);
CREATE INDEX memories_tags_idx ON memories USING GIN(tags);
-- ANN vector index (pgvector)
CREATE INDEX memories_embedding_idx ON memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

### Search Implementation

- **Vector search**: `SELECT * FROM memories ORDER BY embedding <=> :query_vec LIMIT :top_k`
- **FTS search**: `WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query) ORDER BY ts_rank_cd(...)`
- **Hybrid**: Weighted combination of both scores

### Connection Management
- Use `psycopg_pool.ConnectionPool` (thread-safe)
- Accept `dsn: str` in constructor (default from `PG_DSN` env var)
- Pool min/max connections configurable
- Context manager support (`with adapter:`)

### Migration Strategy
- Track version in `schema_version` table (identical to SQLite)
- Migration functions for schema changes
- pgvector extension must be created: `CREATE EXTENSION IF NOT EXISTS vector`

### Python API — mirrors SQLiteStorageAdapter exactly

```python
from kemi.adapters.storage.postgres import PostgresStorageAdapter

# Via env var PG_DSN
adapter = PostgresStorageAdapter()

# Or explicit DSN
adapter = PostgresStorageAdapter(dsn="postgres://user:pass@host:5432/kemi")

# Use it like any other adapter
from kemi import Memory
memory = Memory(store=adapter)
```
