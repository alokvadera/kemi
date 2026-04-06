# kemi

Persistent memory for AI agents. Three methods. Zero infra.

[![asciicast](https://asciinema.org/a/683480.svg)](https://asciinema.org/a/683480)

```python
from kemi import Memory

memory = Memory()  # SQLite + local embeddings, no API keys needed

memory.remember("user123", "User prefers dark mode")
memory.remember("user123", "User is vegetarian")

results = memory.recall("user123", "what are the user's preferences?")
# Returns ranked, deduplicated memories

memory.forget("user123")  # GDPR-compliant deletion
```

## Install

```bash
# Zero dependencies — SQLite storage, no embedding (bring your own)
pip install kemi

# With local embeddings (no API key needed, ~130MB model download)
pip install kemi[local]

# With OpenAI embeddings
pip install kemi[openai]
```

## Why kemi

Every existing memory library either hosts your data on their servers, requires Docker and 4 services to run, or locks you into a specific framework.

kemi is different:

- **Zero infrastructure** — runs on your machine, single pip install
- **Your data** — never leaves your machine, stored in SQLite by default
- **Bring your own embedding** — OpenAI, local models, or any function
- **Framework agnostic** — works with LangChain, CrewAI, AutoGen, or plain Python
- **100% free** — MIT license, no paid tiers, no cloud lock-in

## Usage

### Zero-config (local embeddings)

```python
from kemi import Memory

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

### Async usage (FastAPI, asyncio)

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

### Inject into system prompt

```python
context = memory.context_block("user123", query="user preferences", max_tokens=500)
# Returns formatted string ready for system prompt injection
```

### GDPR-compliant deletion

```python
memory.forget("user123")               # Delete all memories for user
memory.forget("user123", memory_id)    # Delete one specific memory
```

## How it works

kemi sits between your agent and your storage. It handles:

- **Semantic deduplication** — "I'm vegetarian" and "I don't eat meat" are the same memory
- **Importance-weighted scoring** — recent, important memories rank higher
- **Temporal decay** — memories fade if never recalled
- **Conflict detection** — flags contradictory memories for review
- **Lifecycle management** — active → decaying → archived → deleted

## Adapters

| Type | Default | Available |
|------|---------|-----------|
| Embedding | fastembed (local) | OpenAI, custom |
| Storage | SQLite | JSON, custom |

## Documentation

- [Quickstart](docs/quickstart.md) — get running in 5 minutes
- [Recipes](docs/recipes.md) — complete working examples
- [Configuration](docs/configuration.md) — tuning kemi for your use case
- [Adapters](docs/adapters.md) — embeddings, storage, and custom implementations

## License

MIT — free forever, no exceptions.
