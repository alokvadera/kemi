# kemi

**Persistent memory for AI agents. Three methods. Zero infra.**

[![PyPI version](https://img.shields.io/pypi/v/kemi?color=blue)](https://pypi.org/project/kemi/)
[![Python versions](https://img.shields.io/pypi/pyversions/kemi)](https://pypi.org/project/kemi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/alokvadera/kemi/actions/workflows/ci.yml/badge.svg)](https://github.com/alokvadera/kemi/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](https://github.com/alokvadera/kemi)

```python
from kemi import Memory

memory = Memory()  # SQLite + local embeddings, no API keys needed

memory.remember("user123", "User prefers dark mode")
memory.remember("user123", "User is vegetarian")

results = memory.recall("user123", "what are the user's preferences?")
# Returns ranked, deduplicated memories

memory.forget("user123")  # GDPR-compliant deletion
```

---

## Why kemi?

Every other memory library **either hosts your data on their servers, requires Docker and 4 services to run, or locks you into a specific framework.**

**kemi is different:**

- **Zero infrastructure** — `pip install kemi`, no Docker, no cloud, no setup
- **Zero hard dependencies** — `pip install kemi` installs only the core library (no chromadb, no qdrant, no FastAPI). Optional backends live behind `[chroma]`, `[qdrant]`, `[api]`, etc.
- **Your data stays yours** — stored in SQLite on your machine, never leaves
- **Bring your own embedding** — OpenAI, local models (fastembed), or any function
- **Framework agnostic** — works with LangChain, CrewAI, AutoGen, or plain Python
- **MCP ready** — use as a memory server for Claude, Cursor, and Continue
- **100% free** — MIT license, no paid tiers, no cloud lock-in

---

## Install

`pip install kemi` ships **zero hard dependencies** — only the core library plus SQLite (Python stdlib). Pick what you need:

```bash
# Core only — bring your own embedding function
pip install kemi

# With local embeddings (no API key needed, ~130MB model download)
pip install "kemi[local]"

# With OpenAI embeddings
pip install "kemi[openai]"

# With PostgreSQL + pgvector (ANN search, full-text search, hybrid search)
pip install "kemi[postgres]"

# With vector DB backends
pip install "kemi[chroma]"
pip install "kemi[qdrant]"
pip install "kemi[redis]"

# With the MCP server (for Claude Desktop, Cursor, Continue)
pip install "kemi[mcp]"

# With LangChain adapter
pip install "kemi[langchain]"

# All extras (heavy — pulls every backend)
pip install "kemi[all]"
```

**No Docker. No cloud services. No API keys required.**

---

## Quick Start

### Zero-config (local embeddings)

```python
from kemi import Memory

# Uses local SQLite + local embeddings — works immediately
memory = Memory()
memory.remember("user123", "User is vegetarian", importance=0.9)
results = memory.recall("user123", "food preferences")
```

### With OpenAI embeddings

```python
from kemi import Memory
from kemi.adapters.embedding.openai import OpenAIEmbedAdapter

memory = Memory(embed=OpenAIEmbedAdapter())
memory.remember("user123", "User prefers concise responses")
results = memory.recall("user123", "communication style")
```

### With PostgreSQL + pgvector (ANN search, FTS, hybrid)

```bash
# Start PostgreSQL with pgvector
docker compose up -d

# Run with PostgreSQL
PG_DSN="postgresql://postgres:postgres@localhost:5432/kemi_test" python -c "
from kemi import Memory
from kemi.adapters.storage.postgres import PostgresStorageAdapter

store = PostgresStorageAdapter(dsn=PG_DSN, embedding_dim=384)
memory = Memory(store=store)
memory.remember('user123', 'User prefers dark mode')
results = memory.recall('user123', 'color theme preferences')
print(results)
"

### For your system prompt

```python
context = memory.context_block("user123", query="user preferences", max_tokens=500)
# Returns formatted string ready for LLM system prompt injection
```

### Async (FastAPI, asyncio)

```python
from fastapi import FastAPI
from kemi import Memory

app = FastAPI()
memory = Memory()

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

## MCP Server

Any MCP-compatible agent (Claude Desktop, Cursor, Continue) can use kemi as its memory layer:

```bash
pip install "kemi[mcp]"
python -m kemi
```

Claude can then remember facts about you across sessions — no API keys, no cloud, everything local.

---

## How It Works

kemi sits between your AI agent and its storage, handling:

| Feature | What it does |
|---|---|
| **Semantic deduplication** | "I'm vegetarian" and "I don't eat meat" are detected as the same memory |
| **Importance-weighted scoring** | Recent, important memories rank higher in search results |
| **Temporal decay** | Memories fade if never recalled — transitions from ACTIVE → DECAYING |
| **Conflict detection** | Flags contradictory memories ("I love coffee" vs "I hate coffee") |
| **Hybrid search** | Combines semantic (vector) search with keyword (BM25) search |
| **MMR reranking** | Ensures diverse results — not 5 nearly-identical memories |
| **Lifecycle management** | Automatic state transitions through ACTIVE → DECAYING → ARCHIVED → DELETED |

---

## Adapters

| Type | Default | Alternatives |
|---|---|---|
| **Embedding** | fastembed (local, 384-dim) | OpenAI (1536-dim), custom function |
| **Storage** | SQLite (WAL mode) | JSON file, custom backend, PostgreSQL + pgvector |

---

## Integrations

### LangChain

```python
from kemi import Memory
from kemi.integrations.langchain import KemiMemory

memory = Memory()
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

## Documentation

| Guide | What you'll learn |
|---|---|
| [Quickstart](docs/quickstart.md) | Get running in 5 minutes |
| [Recipes](docs/recipes.md) | Complete working examples |
| [Configuration](docs/configuration.md) | Tuning kemi for your use case |
| [Adapters](docs/adapters.md) | Embeddings, storage, custom implementations |
| [CLI](src/kemi/cli.py) | Built-in command-line interface |

---

## Data Privacy

kemi is designed so **your data never leaves your machine**:

- All memories stored in local SQLite at `~/.kemi/memories.db`
- Embeddings computed locally (fastembed) or via your own API key (OpenAI)
- No telemetry, no analytics, no phone-home
- Full GDPR-compliant deletion with `memory.forget()`

---

## Requirements

- Python 3.10+

---

## License

MIT — free forever, no exceptions.
