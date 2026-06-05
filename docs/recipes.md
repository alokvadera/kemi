# Recipes

Complete, copy-pasteable examples for common use cases.

## Recipe 1: Chatbot with persistent memory

A simple chatbot that remembers user preferences across sessions.

```python
# A simple chatbot that remembers user preferences across sessions
# Run this script multiple times — it remembers between runs

from kemi import Memory

memory = Memory()
user_id = "alice"

print("Chat with an AI that remembers you. Type 'quit' to exit.")
print("Try saying: I am vegetarian, I prefer short answers, I live in Mumbai")
print()

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        break
    if not user_input:
        continue

    # Store what the user said
    memory.remember(user_id, user_input)

    # Get relevant context from memory
    context = memory.context_block(user_id, query=user_input, max_tokens=300)

    print(f"\nMemory context:\n{context}")
    print(f"\nTotal memories: {memory._store.count(user_id)}")
    print()
```

Run it multiple times — the bot remembers everything between runs.

## Recipe 2: FastAPI async endpoint

```python
# pip install fastapi uvicorn kemi[local]
# Run with: uvicorn app:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
from kemi import Memory

app = FastAPI()
memory = Memory()

class Message(BaseModel):
    user_id: str
    content: str

@app.post("/remember")
async def remember(msg: Message):
    memory_id = await memory.aremember(msg.user_id, msg.content)
    return {"memory_id": memory_id}

@app.get("/recall/{user_id}")
async def recall(user_id: str, query: str):
    results = await memory.arecall(user_id, query)
    return {"memories": [r.content for r in results]}

@app.get("/context/{user_id}")
async def context(user_id: str, query: str):
    ctx = await memory.acontext_block(user_id, query)
    return {"context": ctx}

@app.delete("/forget/{user_id}")
async def forget(user_id: str):
    count = await memory.aforget(user_id)
    return {"deleted": count}
```

Test it:
```bash
curl -X POST http://localhost:8000/remember -H "Content-Type: application/json" -d '{"user_id": "alice", "content": "I love coffee"}'
curl "http://localhost:8000/recall/alice?query=drinks"
curl "http://localhost:8000/context/alice?query=beverages"
```

## Recipe 3: OpenAI chatbot with memory

```python
# pip install kemi[openai] openai
# Requires OPENAI_API_KEY environment variable

import os
from openai import OpenAI
from kemi import Memory
from kemi.adapters.embedding.openai import OpenAIEmbedAdapter

client = OpenAI()
memory = Memory(embed=OpenAIEmbedAdapter())
user_id = "alice"

def chat(user_message: str) -> str:
    # Store the user message
    memory.remember(user_id, user_message)

    # Get relevant context
    context = memory.context_block(user_id, query=user_message, max_tokens=500)

    # Build system prompt with memory context
    system = "You are a helpful assistant."
    if context:
        system += f"\n\n{context}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
    )
    return response.choices[0].message.content

print(chat("I am vegetarian"))
print(chat("What should I eat for dinner?"))
# Second response will know you're vegetarian
```

The second call automatically knows you're vegetarian from the first message.

## Recipe 4: Custom embedding with Ollama

```python
# pip install kemi requests
# Requires Ollama running locally: https://ollama.ai
# Pull a model first: ollama pull nomic-embed-text

import requests
from kemi import Memory
from kemi.adapters.embedding.custom import CustomEmbedAdapter

def ollama_embed(texts: list) -> list:
    embeddings = []
    for text in texts:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text}
        )
        embeddings.append(response.json()["embedding"])
    return embeddings

memory = Memory(
    embed=CustomEmbedAdapter(
        embed_fn=ollama_embed,
        dim=768  # nomic-embed-text dimension
    )
)

memory.remember("user1", "I prefer Python over JavaScript")
results = memory.recall("user1", "programming language preference")
print(results[0].content)
```

This uses Ollama's local embedding model instead of fastembed or OpenAI.

## Recipe 5: Multi-level recall helpers

kemi provides ergonomic helpers for different *levels* of memory retrieval.

### User profile (semantic facts)

```python
from kemi import Memory
from kemi.models import MemoryType

memory = Memory()

# Store long-lived preferences as SEMANTIC memories
memory.remember("alice", "I am vegetarian", memory_type=MemoryType.SEMANTIC, importance=0.9)
memory.remember("alice", "I live in Mumbai", memory_type=MemoryType.SEMANTIC, importance=0.8)
memory.remember("alice", "I prefer short answers", memory_type=MemoryType.SEMANTIC, importance=0.7)

# Retrieve the user's profile — returns SEMANTIC memories sorted by importance
profile = memory.recall_user_profile("alice", top_k=10)
for mem in profile:
    print(f"  {mem.content} (importance: {mem.importance})")
```

### Session context (episodic events)

```python
from kemi.models import MemoryType

session_id = "sess_2024_06_05"

# Store session-specific events as EPISODIC memories
memory.remember("alice", "User asked about Python asyncio", memory_type=MemoryType.EPISODIC, session_id=session_id)
memory.remember("alice", "User mentioned they use FastAPI", memory_type=MemoryType.EPISODIC, session_id=session_id)

# Retrieve recent session context — returns EPISODIC memories sorted by recency
context = memory.recall_session_context("alice", session_id, top_k=10)
for mem in context:
    print(f"  {mem.content}")
```

### Agent knowledge (agent-scoped memories)

```python
from kemi.models import MemoryType

# Store agent-specific rules or knowledge
memory.remember("alice", "Always greet users by name", memory_type=MemoryType.SEMANTIC, agent_id="support_bot", importance=0.9)
memory.remember("bob", "Escalate billing issues to tier-2", memory_type=MemoryType.SEMANTIC, agent_id="support_bot", importance=0.8)

# Retrieve all knowledge for a specific agent across all users
knowledge = memory.recall_agent_knowledge("support_bot", top_k=20)
for mem in knowledge:
    print(f"  {mem.content} (user: {mem.user_id})")
```

## Recipe 6: Procedural memory (how-to workflows)

Use **procedural** memory for reusable step-by-step instructions, SOPs, or
agent playbooks. It is distinct from *episodic* (past events) and *semantic*
(facts/preferences).

| Memory type | Use when | Example |
|-------------|----------|---------|
| **EPISODIC** | Remembering something that happened | "User asked about Python asyncio" |
| **SEMANTIC** | Remembering a fact or preference | "User is vegetarian" |
| **PROCEDURAL** | Remembering a reusable workflow | "How to reset a password" |

### Store a procedure

```python
from kemi import Memory, remember_procedure

memory = Memory()

remember_procedure(
    memory,
    user_id="alice",
    name="password_reset",
    steps=[
        "Ask the user for their registered email address",
        "Send a one-time reset link to the verified email",
        "Confirm the reset was initiated and provide ETA",
    ],
    metadata={"team": "support", "priority": "high"},
    importance=0.9,
)
```

### Recall a procedure

```python
from kemi import recall_procedures

results = recall_procedures(
    memory,
    "how do I reset a password?",
    user_id="alice",
    top_k=3,
)

for proc in results:
    print(proc.content)
```

`recall_procedures` performs a semantic search and returns only memories whose
`memory_type` is `PROCEDURAL`, so episodic or semantic matches are automatically
filtered out.

## Recipe 7: Entity-aware retrieval

When a query mentions specific entities (names, dates, places), kemi can boost
memories that contain the same entities. This is useful when a user asks about
a specific person, product, or event and you want matching memories to rank
higher even if their raw semantic similarity is close.

### Enable entity boost

```python
from kemi import Memory, MemoryConfig

config = MemoryConfig(
    enable_entity_boost=True,
    entity_boost_weight=0.15,  # 0.0–1.0; higher = stronger boost
)

memory = Memory(config=config)

memory.remember("alice", "Alice visited Paris in June 2024")
memory.remember("alice", "Bob moved to Berlin last year")

# "Alice" and "Paris" are extracted from the query and matched against memory content
results = memory.recall("alice", "What did Alice do in Paris?")
# The first memory (which mentions Alice and Paris) is ranked higher
```

### Custom entity linker

```python
from kemi import Memory, EntityLinker

class ProductLinker(EntityLinker):
    def extract(self, text: str) -> set[str]:
        # Simple regex for product SKUs like PROD-12345
        import re
        return set(re.findall(r"PROD-\d+", text))

memory = Memory(entity_linker=ProductLinker())
```

### spaCy NER entity linker (more accurate)

For production use, swap the default regex linker for spaCy’s trained NER
pipeline. It recognises people, organisations, locations, dates, products, and
more with far higher accuracy than regex heuristics.

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

```python
from kemi import Memory, SpacyEntityLinker

# Use spaCy NER for entity extraction
memory = Memory(entity_linker=SpacyEntityLinker())

# Or restrict to specific entity types only
memory = Memory(
    entity_linker=SpacyEntityLinker(
        model="en_core_web_sm",
        allowed_labels={"PERSON", "ORG", "GPE", "PRODUCT"},
    )
)
```

Note: spaCy models are ~10–50 MB and load on first instantiation. Re-use the same
`SpacyEntityLinker()` across multiple `Memory()` instances or cache it at
application startup to avoid reloading.

### Inspect entity scores

```python
explained = memory.recall_explain("alice", "Alice in Paris", top_k=2)
for item in explained:
    print(item["memory"].content)
    print("  entity_score:", item["explanation"]["entity_score"])
    print("  final_score: ", item["explanation"]["final_score"])
```

When `enable_entity_boost=True`, `recall_explain` includes `entity_score` and
an `entity` weight in the explanation dict. The boost is computed as a Jaccard
overlap between query entities and memory entities, then multiplied by
`entity_boost_weight` and added to the final score.

## Recipe 8: LangChain integration

```python
# pip install kemi[langchain] langchain langchain-openai

from kemi import Memory
from kemi.integrations.langchain import KemiMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize kemi memory
memory = Memory()

# Create LangChain memory adapter
chat_memory = KemiMemory(user_id="alice", memory=memory)

# Build prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_functions_agent(
    ChatOpenAI(model="gpt-4o-mini"),
    prompt,
    tools=[],
)

# Create executor with memory
executor = AgentExecutor(
    agent=agent,
    tools=[],
    memory=chat_memory,
)

# Run conversations - memory persists automatically
executor.invoke({"input": "My name is Alice"})
executor.invoke({"input": "What's my name?"})  # Agent knows your name from memory
```

The `KemiMemory` class automatically stores every human message to kemi and retrieves relevant context for each new conversation turn.