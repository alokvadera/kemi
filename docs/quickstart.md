# Quickstart

Get running with kemi in under 5 minutes.

## Installation

Choose your embedding option:

```bash
# Local embeddings — no API key needed, ~130MB model download
pip install kemi[local]

# Bring your own embeddings — zero dependencies
pip install kemi

# OpenAI embeddings — requires API key
pip install kemi[openai]
```

## Zero-config example

The easiest way to use kemi:

```python
from kemi import Memory

memory = Memory()  # Uses fastembed + SQLite automatically

# Store a memory
memory.remember("user123", "I am vegetarian")

# Retrieve relevant memories
results = memory.recall("user123", "food preferences")
print(results[0].content)
# Output: I am vegetarian

# Delete memories (GDPR-compliant)
memory.forget("user123")
```

## What happens on first run

1. **First call to `Memory()`:**
   - Downloads fastembed model (~130MB) from Hugging Face
   - Model is cached forever after

2. **First `remember()` call:**
   - Creates `kemi.db` in current directory
   - Stores memory with embeddings

3. **Subsequent calls:**
   - Everything runs instantly from cached data

## Where data is stored

By default, kemi stores memories in ~/.kemi/memories.db — your home directory. This means the same memories are available regardless of which directory you run your script from.

```
~/.kemi/
└── memories.db    # all your memories, always in the same place
```

The database contains:
- Memory content and embeddings
- User IDs and metadata
- Importance scores and lifecycle state

## Custom database path

To use a different path:

```python
from kemi import Memory
from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

memory = Memory(
    store=SQLiteStorageAdapter(db_path="/path/to/memories.db")
)
```

Or use a different storage backend entirely:

```python
from kemi.adapters.storage.json import JSONStorageAdapter

memory = Memory(
    store=JSONStorageAdapter(path="/data/memories.json")
)
```