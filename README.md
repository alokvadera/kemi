# kemi

**Persistent memory for AI agents. Three methods. Zero infra.**

[![PyPI version](https://img.shields.io/pypi/v/kemi?color=blue)](https://pypi.org/project/kemi/)
[![Python versions](https://img.shields.io/pypi/pyversions/kemi)](https://pypi.org/project/kemi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/alokvadera/kemi/actions/workflows/ci.yml/badge.svg)](https://github.com/alokvadera/kemi/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](https://github.com/alokvadera/kemi)

```python
from kemi import MemoryService

memory = MemoryService()  # SQLite + local embeddings, no API keys needed

memory.remember("user123", "User prefers dark mode")
memory.remember("user123", "User is vegetarian")

results = memory.recall("user123", "what are the user's preferences?")
# Returns ranked, deduplicated memories

memory.forget("user123")  # GDPR-compliant deletion
```

> **Note:** `from kemi import Memory` still works as a backwards-compatible alias.
> New code should import `MemoryService` directly.

---

## Why kemi?

Every other memory library **either hosts your data on their servers, requires Docker and 4 services to run, or locks you into a specific framework.**

**kemi is different:**

| | |
|---|---|
| **Zero infrastructure** | `pip install kemi`, no Docker, no cloud, no setup |
| **Zero hard dependencies** | Only the core library + SQLite. Optional backends live behind extras (`[chroma]`, `[qdrant]`, `[postgres]`, etc.) |
| **Your data stays yours** | Stored in SQLite on your machine, never leaves |
| **Bring your own embedding** | OpenAI, local models (fastembed), or any function |
| **Framework agnostic** | Works with LangChain, CrewAI, AutoGen, or plain Python |
| **MCP ready** | Use as a memory server for Claude Desktop, Cursor, and Continue |
| **100% free** | MIT license, no paid tiers, no cloud lock-in |

---

## Install

```bash
pip install kemi                    # Core only — zero hard dependencies
pip install "kemi[local]"           # + local embeddings (no API key, ~130MB download)
pip install "kemi[openai]"          # + OpenAI embeddings (1536-dim)
pip install "kemi[postgres]"        # + PostgreSQL + pgvector (ANN, FTS, hybrid search)
pip install "kemi[mcp]"            # + MCP server (Claude Desktop, Cursor, Continue)
pip install "kemi[langchain]"      # + LangChain BaseChatMemory adapter
pip install "kemi[chroma]"         # + ChromaDB vector store
pip install "kemi[qdrant]"         # + Qdrant vector store
pip install "kemi[redis]"          # + Redis vector store
pip install "kemi[all]"            # Everything (heavy)
```

---

## Quick Start

### Zero-config (local embeddings)

```python
from kemi import MemoryService

memory = MemoryService()
memory.remember("user123", "User is vegetarian", importance=0.9)
results = memory.recall("user123", "food preferences")
```

### With OpenAI embeddings

```python
from kemi import MemoryService
from kemi.adapters.embedding.openai import OpenAIEmbedAdapter

memory = MemoryService(embed=OpenAIEmbedAdapter())
memory.remember("user123", "User prefers concise responses")
results = memory.recall("user123", "communication style")
```

### With PostgreSQL + pgvector

```python
from kemi import MemoryService
from kemi.adapters.storage.postgres import PostgresStorageAdapter

store = PostgresStorageAdapter(dsn="postgresql://user:pass@localhost:5432/kemi", embedding_dim=384)
memory = MemoryService(store=store)
memory.remember("user123", "User prefers dark mode")
results = memory.recall("user123", "color theme preferences")
```

### Inject into system prompts

```python
context = memory.context_block("user123", query="user preferences", max_tokens=500)
# Returns formatted string ready to paste into an LLM system prompt
```

### Async (FastAPI)

```python
from fastapi import FastAPI
from kemi import MemoryService

app = FastAPI()
memory = MemoryService()

@app.post("/chat")
async def chat(user_id: str, message: str):
    await memory.aremember(user_id, message)
    context = await memory.acontext_block(user_id, message)
    return {"context": context}
```

### GDPR-compliant deletion

```python
memory.forget("user123")            # Delete all memories for a user
memory.forget("user123", memory_id) # Delete one specific memory
```

---

## Features

| Feature | What it does |
|---|---|
| **Semantic deduplication** | "I'm vegetarian" and "I don't eat meat" are detected as the same memory |
| **Importance-weighted scoring** | Recent, important memories rank higher in search results |
| **Temporal decay** | Memories fade if never recalled -- transitions from ACTIVE to DECAYING |
| **Conflict detection** | Flags contradictory memories ("I love coffee" vs "I hate coffee") |
| **Hybrid search** | Combines semantic (vector) search with keyword (BM25) search |
| **MMR reranking** | Ensures diverse results -- not 5 nearly-identical memories |
| **Lifecycle management** | Automatic state transitions: ACTIVE -> DECAYING -> ARCHIVED -> DELETED |
| **Query decomposition** | Breaks complex queries into sub-queries with Reciprocal Rank Fusion |
| **Entity extraction** | Zero-dependency regex or spaCy-based entity linking |
| **Version history** | Track changes to memories with rollback support |
| **Webhooks** | Dispatch lifecycle events (remembered, updated, deleted, conflict) to HTTP endpoints |
| **Audit trail** | Compliance-grade operation log with retention and export |
| **Plugin system** | Four extension points: WebhookSink, AuditSink, QueryCacheProvider, HookSink |

---

## MCP Server

Any MCP-compatible agent (Claude Desktop, Cursor, Continue) can use kemi as its memory layer:

```bash
pip install "kemi[mcp]"
python -m kemi
```

Claude can then remember facts about you across sessions -- no API keys, no cloud, everything local.

### Exposed tools

`remember`, `recall`, `recall_stream`, `recall_explain`, `forget`, `context_block`, `prune`, `stats`, `consolidate`, `topics`, `graph`, `list_users`

---

## Adapters

| Type | Default | Alternatives |
|---|---|---|
| **Embedding** | fastembed (local, 384-dim) | OpenAI (1536-dim), custom function |
| **Storage** | SQLite (WAL mode) | SQLite-vec (ANN), PostgreSQL + pgvector, Redis, Qdrant, Chroma, JSON file, custom |

---

## Integrations

### LangChain

```python
from kemi import MemoryService
from kemi.integrations.langchain import KemiMemory

memory = MemoryService()
chat_memory = KemiMemory(user_id="alice", memory=memory)
```

### LangGraph / CrewAI / AutoGen

kemi works with any framework. Just use the core `remember` / `recall` / `forget` methods wherever you need persistent memory.

### Export / Import

```python
memory.export("backup.json")       # backup all memories
memory.import_from("backup.json")  # restore from backup
```

---

## CLI

```bash
kemi remember user123 "User prefers dark mode"
kemi recall user123 "preferences"
kemi forget user123
kemi stats user123
kemi export backup.json
kemi import backup.json
```

Use `--json` for machine-readable output or `--quiet` to suppress info messages.

---

## Documentation

| Guide | What you'll learn |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Module map, data flow diagrams, plugin extension points |
| [Quickstart](docs/quickstart.md) | Get running in 5 minutes |
| [Recipes](docs/recipes.md) | Complete working examples |
| [Configuration](docs/configuration.md) | Tuning kemi for your use case |
| [Adapters](docs/adapters.md) | Embeddings, storage, custom implementations |
| [API Stability](docs/API_STABILITY.md) | Stability tiers and deprecation policy |
| [Contributing](CONTRIBUTING.md) | How to contribute, changelog conventions |

---

## Data Privacy

kemi is designed so **your data never leaves your machine**:

- All memories stored in local SQLite at `~/.kemi/memories.db`
- Embeddings computed locally (fastembed) or via your own API key (OpenAI)
- No telemetry, no analytics, no phone-home
- Full GDPR-compliant deletion with `memory.forget()`
- Optional field-level Fernet encryption for content, metadata, and user IDs

---

## Requirements

- Python 3.10+

---

## License

MIT -- free forever, no exceptions.
