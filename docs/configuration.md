# Configuration

Tune kemi's behavior with MemoryConfig.

## MemoryConfig fields

### dedup_threshold (default 0.92)

Similarity score above which two memories are considered duplicates (merged into one).

- **Raise to 0.95+**: When you want stricter dedup — only very similar memories merge
- **Lower to 0.85**: When you want looser dedup — more things are considered duplicates

Example — stricter dedup:
```python
memory = Memory(config=MemoryConfig(dedup_threshold=0.95))
```

At extremes:
- 0.99: Almost nothing merges
- 0.80: Almost everything similar merges

### conflict_threshold (default 0.65)

Similarity range (between conflict_threshold and dedup_threshold) where memories are flagged as potentially conflicting.

When similarity is in this range (0.65-0.92 by default), kemi logs a warning about potential conflicts.

This range means the memories are:
- Related enough to be compared
- Different enough to possibly contradict

### decay_half_life_hours (default 168 = 7 days)

Time after which memory importance starts decaying by half.

- **24 hours**: Task-focused agents — short-lived context
- **168 hours (7 days)**: General use — balance
- **720 hours (30 days)**: Long-term personal assistants

```python
# Short decay for task agents
memory = Memory(config=MemoryConfig(decay_half_life_hours=24.0))
```

### decay_threshold_hours (default 720 = 30 days)

Time after which untouched memories transition from ACTIVE to DECAYING state.

Memories in DECAYING state rank lower in recall results.

### default_importance (default 0.5)

Importance score (0.0-1.0) assigned to memories when not explicitly specified.

Higher importance = higher recall ranking.

```python
# Weight memories more heavily
memory = Memory(config=MemoryConfig(default_importance=0.7))
```

### sanitize (default False)

When True, applies input sanitization to memory content.

What it catches:
- Repeated characters (e.g., "hellooooo" → "hello")
- Excessive whitespace

What it misses:
- Profanity
- PII

```python
# Enable sanitization for user-generated content
memory = Memory(config=MemoryConfig(sanitize=True))
```

### default_top_k (default 5)

Number of memories to return by default in recall().

```python
# Return more memories
memory = Memory(config=MemoryConfig(default_top_k=10))
```

### max_tokens_default (default None)

Default token budget for context_block() when max_tokens not specified.

None means no limit — all relevant memories are included.

```python
# Limit context to 1000 tokens
memory = Memory(config=MemoryConfig(max_tokens_default=1000))
```

## Complete examples

### Task-focused agent configuration

For agents that handle short-lived tasks:

```python
from kemi import Memory, MemoryConfig

memory = Memory(config=MemoryConfig(
    decay_half_life_hours=24.0,      # memories fade after 1 day
    dedup_threshold=0.95,             # stricter deduplication
    default_top_k=3,                  # fewer memories per recall
))
```

### Personal assistant configuration

For long-running assistant applications:

```python
from kemi import Memory, MemoryConfig

memory = Memory(config=MemoryConfig(
    decay_half_life_hours=720.0,     # memories persist for 30 days
    default_importance=0.7,           # weight memories higher
    max_tokens_default=1000,         # more context per recall
))
```

### Debug configuration

For development and debugging:

```python
from kemi import Memory, MemoryConfig

memory = Memory(config=MemoryConfig(
    decay_half_life_hours=8760,       # effectively disable decay
    dedup_threshold=0.99,            # disable deduplication for testing
))
```