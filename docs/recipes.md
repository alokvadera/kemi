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

## Recipe 5: LangChain integration

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