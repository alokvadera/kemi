# Memory Formation

Turn raw conversation histories into structured, persistent memories.

## Overview

`kemi.memory.formation` provides a pluggable pipeline that:

1. **Extracts** atomic facts and events from a conversation using an LLM or heuristic extractor.
2. **Embeds** candidate memories in batch for efficiency.
3. **Deduplicates** them against existing memories (and against each other) using kemi's existing cosine-similarity threshold logic.
4. **Persists** the survivors via `Memory.remember_many`.

## Quickstart

```python
from kemi.core import Memory
from kemi.memory.formation import remember_from_conversation, OpenAIMemoryExtractor

# Initialise kemi
mem = Memory()

# A short chat history
conversation = [
    {"role": "user",     "content": "I love hiking in the Alps."},
    {"role": "assistant", "content": "That sounds amazing!"},
    {"role": "user",     "content": "My favourite trail is the Tour du Mont Blanc."},
    {"role": "assistant", "content": "I'll remember that."},
]

# Extract + persist
memory_ids = remember_from_conversation(
    mem,
    conversation,
    user_id="alice",
    extractor=OpenAIMemoryExtractor(model="gpt-4o-mini"),
)
print(f"Created {len(memory_ids)} memories")

# Later, recall them as usual
results = mem.recall(user_id="alice", query="What does Alice like to do?")
for r in results:
    print(f"- {r.content}")
```

## Extractors

### RegexMemoryExtractor

Zero-dependency heuristic extractor.  Matches common patterns such as
"I like …", "My name is …", "I want to …".

```python
from kemi.memory.formation import RegexMemoryExtractor

extractor = RegexMemoryExtractor()
candidates = extractor.extract(conversation, user_id="alice")
```

### OpenAIMemoryExtractor

Uses the OpenAI Chat Completions API with a structured system prompt.

```python
from kemi.memory.formation import OpenAIMemoryExtractor

extractor = OpenAIMemoryExtractor(
    model="gpt-4o-mini",
    api_key="sk-...",      # optional — falls back to OPENAI_API_KEY env var
    base_url=None,         # optional — for OpenAI-compatible proxies
)
```

### StaticMemoryExtractor

Returns a fixed list of candidates.  Useful for deterministic testing
or as a no-op placeholder.

```python
from kemi.memory.formation import CandidateMemory, StaticMemoryExtractor

extractor = StaticMemoryExtractor([
    CandidateMemory(content="Alice likes sushi", importance=0.7, tags=["preference"]),
])
```

### Custom Extractor

Any class implementing the `LLMMemoryExtractor` protocol can be plugged in:

```python
from kemi.memory.formation import CandidateMemory, LLMMemoryExtractor

class MyExtractor:
    def extract(self, conversation, *, user_id, session_id=None):
        # ... your logic ...
        return [CandidateMemory(content="...", importance=0.5)]

memory_ids = remember_from_conversation(
    mem, conversation, user_id="alice", extractor=MyExtractor()
)
```

## Conversation format

Each message is a dictionary with:

| Key       | Type                 | Description                              |
|-----------|----------------------|------------------------------------------|
| `role`    | `str`                | `"user"`, `"assistant"`, or `"system"`   |
| `content` | `str`                | The message text                         |
| `timestamp` | `datetime` (opt)   | When the message was sent                |

## Low-level API

If you need more control, use `extract_memories` directly.  It returns a
list of `MemoryObject` instances that you can inspect or manipulate
before storing.

```python
from kemi.memory.formation import extract_memories

memories = extract_memories(
    conversation,
    user_id="alice",
    embed=mem._embed,
    store=mem._store,
    dedup_threshold=0.85,
)

for m in memories:
    print(f"Candidate: {m.content} (importance={m.importance})")
```

## Deduplication

Both `extract_memories` and `remember_from_conversation` deduplicate
against:

1. **Existing memories** in the store — prevents storing the same fact twice.
2. **Intra-conversation duplicates** — prevents two messages in the same
   conversation from producing identical memories.

The default threshold is **0.85** cosine similarity (same as
`MemoryConfig.dedup_threshold`).  You can override it with the
`dedup_threshold` parameter.
