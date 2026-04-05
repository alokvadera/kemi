# Adapters

Customize kemi's embedding and storage backends.

## Embedding Adapters

### FastEmbedAdapter (default)

Local embeddings using BAAI/bge-small-en-v1.5 model.

- **Pros**: No API key needed, runs offline after first download
- **Cons**: ~130MB download on first use, less accurate than OpenAI
- **Dimension**: 384

```python
from kemi import Memory
from kemi.adapters.embedding.fastembed import FastEmbedAdapter

# Use default model
memory = Memory()

# Or specify a different model
memory = Memory(embed=FastEmbedAdapter(model_name="BAAI/bge-small-en-v1.5"))
```

### OpenAIEmbedAdapter

Cloud embeddings using OpenAI's API.

- **Pros**: More accurate, no local model needed
- **Cons**: Requires API key, costs money, needs internet
- **Dimension**: 1536 (text-embedding-3-small), 3072 (text-embedding-3-large)

```python
from kemi import Memory
from kemi.adapters.embedding.openai import OpenAIEmbedAdapter

memory = Memory(embed=OpenAIEmbedAdapter())

# Or with custom model
memory = Memory(embed=OpenAIEmbedAdapter(model_name="text-embedding-3-large"))
```

Requires `OPENAI_API_KEY` environment variable or pass API key directly:

```python
memory = Memory(embed=OpenAIEmbedAdapter(api_key="sk-..."))
```

### CustomEmbedAdapter

Any function that converts text to vectors.

```python
from kemi import Memory
from kemi.adapters.embedding.custom import CustomEmbedAdapter

def my_embedding_function(texts: list[str]) -> list[list[float]]:
    # Your embedding logic here
    return [[0.1] * 384 for _ in texts]

memory = Memory(
    embed=CustomEmbedAdapter(
        embed_fn=my_embedding_function,
        dim=384  # Required: embedding dimension
    )
)
```

Example with Ollama:

```python
import requests
from kemi import Memory
from kemi.adapters.embedding.custom import CustomEmbedAdapter

def ollama_embed(texts: list) -> list:
    return [
        requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": t}
        ).json()["embedding"]
        for t in texts
    ]

memory = Memory(embed=CustomEmbedAdapter(embed_fn=ollama_embed, dim=768))
```

### Implementing your own EmbeddingAdapter

```python
from kemi.adapters.base import EmbeddingAdapter

class MyEmbeddingAdapter(EmbeddingAdapter):
    def __init__(self):
        self._dimension = 512

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Your implementation
        return [self.embed_single(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        # Your implementation
        return [0.0] * self._dimension

    def dimension(self) -> int:
        return self._dimension
```

## Storage Adapters

### SQLiteStorageAdapter (default)

Single-file SQLite database.

- **Pros**: Fast, single file, works well for single-user local apps
- **Cons**: Not suitable for multi-user web apps without additional setup

```python
from kemi import Memory
from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

# Default path (kemi.db in current directory)
memory = Memory()

# Custom path
memory = Memory(
    store=SQLiteStorageAdapter(db_path="/path/to/memories.db")
)
```

### JSONStorageAdapter

Human-readable JSON file.

- **Pros**: Easy to inspect and debug, version-control friendly
- **Cons**: Slower for large datasets, no concurrent write support

```python
from kemi import Memory
from kemi.adapters.storage.json import JSONStorageAdapter

memory = Memory(
    store=JSONStorageAdapter(path="memories.json")
)
```

### Custom storage with functions

Pass custom functions for each operation:

```python
from kemi import Memory
from kemi.adapters.storage.custom import CustomStorageAdapter

storage = CustomStorageAdapter(
    store=lambda memory: None,                    # store implementation
    search=lambda user_id, query_embedding, top_k, lifecycle_filter: [],  # search implementation
    get=lambda memory_id: None,                    # get implementation
    update=lambda memory: None,                   # update implementation
    delete_by_id=lambda memory_id: False,        # delete_by_id implementation
    delete_by_user=lambda user_id: 0,            # delete_by_user implementation
    get_all_by_user=lambda user_id, lifecycle_filter: [],  # get_all_by_user implementation
    count=lambda user_id: 0,                     # count implementation
    upgrade_schema=lambda from_version, to_version: None,
)

memory = Memory(store=storage)
```

### Implementing your own StorageAdapter

```python
from kemi.adapters.base import StorageAdapter
from kemi.models import MemoryObject, LifecycleState
from typing import Optional

class MyStorageAdapter(StorageAdapter):
    def store(self, memory: MemoryObject) -> None:
        # Your implementation
        pass

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        lifecycle_filter: Optional[list[LifecycleState]] = None,
    ) -> list[MemoryObject]:
        # Your implementation
        return []

    def get(self, memory_id: str) -> Optional[MemoryObject]:
        # Your implementation
        return None

    def update(self, memory: MemoryObject) -> None:
        # Your implementation
        pass

    def delete_by_id(self, memory_id: str) -> bool:
        # Your implementation
        return False

    def delete_by_user(self, user_id: str) -> int:
        # Your implementation
        return 0

    def get_all_by_user(
        self,
        user_id: str,
        lifecycle_filter: Optional[list[LifecycleState]] = None,
    ) -> list[MemoryObject]:
        # Your implementation
        return []

    def count(self, user_id: str) -> int:
        # Your implementation
        return 0

    def upgrade_schema(self, from_version: int, to_version: int) -> None:
        # Your implementation
        pass
```

## Mixing adapters

Combine different embedding and storage backends:

```python
from kemi import Memory
from kemi.adapters.embedding.openai import OpenAIEmbedAdapter
from kemi.adapters.storage.sqlite import SQLiteStorageAdapter

memory = Memory(
    embed=OpenAIEmbedAdapter(model_name="text-embedding-3-large"),
    store=SQLiteStorageAdapter(db_path="/data/memories.db"),
)
```

Or local embeddings with JSON storage:

```python
from kemi import Memory
from kemi.adapters.storage.json import JSONStorageAdapter

memory = Memory(
    store=JSONStorageAdapter(path="debug_memories.json")
)
```