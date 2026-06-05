"""MCP server for kemi persistent memory."""

import logging
import os
import sys
import uuid
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from kemi import Memory
from kemi.adapters.embedding.fastembed import FastEmbedAdapter
from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from kemi.models import MemoryType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: StreamableHTTP transport for clients that support HTTP streaming
# ---------------------------------------------------------------------------
try:
    from mcp.server.streamable_http import StreamableHTTP as _StreamableHTTP
    from starlette.applications import Starlette as _Starlette
    from starlette.requests import Request as _Request
    from starlette.responses import Response as _Response
    from starlette.routing import Route as _Route

    _HAS_STREAMABLE_HTTP = True
except ImportError:
    _HAS_STREAMABLE_HTTP = False


def _get_config() -> dict[str, Any]:
    """Read configuration from environment variables."""
    return {
        "db_path": os.environ.get("KEMI_DB_PATH", os.path.expanduser("~/.kemi/memories.db")),
        "model": os.environ.get("KEMI_MODEL", "BAAI/bge-small-en-v1.5"),
        "top_k": int(os.environ.get("KEMI_TOP_K", "5")),
    }


def _print_config(config: dict[str, Any]) -> None:
    """Print active configuration to stderr."""
    print(
        f"[kemi MCP] Config: db={config['db_path']}, "
        f"model={config['model']}, top_k={config['top_k']}",
        file=sys.stderr,
    )


class KemiMCPServer:
    """MCP server exposing kemi Memory tools."""

    def __init__(self) -> None:
        config = _get_config()
        _print_config(config)

        db_path = config["db_path"]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        store = SQLiteStorageAdapter(db_path=db_path)
        embed = FastEmbedAdapter(model_name=config["model"])

        self.memory = Memory(embed=embed, store=store)
        self.server = Server("kemi")
        self._top_k = config["top_k"]

        @self.server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="remember",
                    description="Store a memory for a user. Merges duplicates automatically.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Unique identifier for the user",
                            },
                            "content": {
                                "type": "string",
                                "description": "The memory content to store",
                            },
                            "importance": {
                                "type": "number",
                                "description": "Importance 0.0-1.0",
                                "default": 0.5,
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional tags",
                            },
                            "namespace": {
                                "type": "string",
                                "description": "Memory namespace",
                                "default": "default",
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Optional session ID",
                            },
                            "memory_type": {
                                "type": "string",
                                "description": "episodic or semantic",
                                "default": "episodic",
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence 0.0-1.0",
                                "default": 1.0,
                            },
                        },
                        "required": ["user_id", "content"],
                    },
                ),
                Tool(
                    name="recall",
                    description="Search memories for a user by query.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Unique identifier for the user",
                            },
                            "query": {"type": "string", "description": "Search query"},
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results",
                                "default": 5,
                            },
                            "namespace": {
                                "type": "string",
                                "description": "Memory namespace",
                                "default": "default",
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Optional session ID",
                            },
                            "hybrid_search": {
                                "type": "boolean",
                                "description": "Use hybrid search",
                                "default": True,
                            },
                        },
                        "required": ["user_id", "query"],
                    },
                ),
                Tool(
                    name="recall_stream",
                    description="Stream recall results progressively. "
                    "Each result is sent as a progress notification "
                    "as it becomes available, then all results are "
                    "returned in full at the end. Compatible with "
                    "MCP clients that support progress notifications "
                    "and StreamableHTTP transport.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Unique identifier for the user",
                            },
                            "query": {"type": "string", "description": "Search query"},
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results",
                                "default": 5,
                            },
                            "namespace": {
                                "type": "string",
                                "description": "Memory namespace",
                                "default": "default",
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Optional session ID",
                            },
                            "hybrid_search": {
                                "type": "boolean",
                                "description": "Use hybrid search",
                                "default": True,
                            },
                        },
                        "required": ["user_id", "query"],
                    },
                ),
                Tool(
                    name="recall_explain",
                    description="Search memories with detailed score explanations.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 5},
                            "namespace": {"type": "string", "default": "default"},
                        },
                        "required": ["user_id", "query"],
                    },
                ),
                Tool(
                    name="prune",
                    description="Auto-prune old or low-importance memories.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "max_age_days": {"type": "number"},
                            "min_importance": {"type": "number"},
                            "namespace": {"type": "string", "default": "default"},
                        },
                        "required": ["user_id"],
                    },
                ),
                Tool(
                    name="stats",
                    description="Get memory statistics for a user.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                        },
                        "required": ["user_id"],
                    },
                ),
                Tool(
                    name="consolidate",
                    description="Consolidate old episodic memories into semantic summaries.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "namespace": {"type": "string", "default": "default"},
                            "min_memories": {"type": "integer", "default": 5},
                            "max_age_days": {"type": "number", "default": 30.0},
                        },
                        "required": ["user_id"],
                    },
                ),
                Tool(
                    name="topics",
                    description="Cluster memories into topic groups.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "n_clusters": {"type": "integer", "default": 3},
                            "namespace": {"type": "string", "default": "default"},
                        },
                        "required": ["user_id"],
                    },
                ),
                Tool(
                    name="graph",
                    description="Build a memory graph of entities and relations.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "namespace": {"type": "string", "default": "default"},
                        },
                        "required": ["user_id"],
                    },
                ),
                Tool(
                    name="list_users",
                    description="List all users with memories.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="forget",
                    description="Delete memories for a user. GDPR-compliant.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Unique identifier for the user",
                            },
                            "memory_id": {
                                "type": "string",
                                "description": "Optional specific memory ID to delete",
                            },
                        },
                        "required": ["user_id"],
                    },
                ),
                Tool(
                    name="context_block",
                    description="Get formatted context block for system prompt injection.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Unique identifier for the user",
                            },
                            "query": {"type": "string", "description": "Search query"},
                            "top_k": {
                                "type": "integer",
                                "description": "Number of memories",
                                "default": 5,
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Token budget",
                                "default": 1500,
                            },
                        },
                        "required": ["user_id", "query"],
                    },
                ),
            ]

        @self.server.call_tool()  # type: ignore[untyped-decorator]
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            if name == "remember":
                mtype_str = arguments.get("memory_type", "episodic")
                try:
                    mtype = MemoryType(mtype_str)
                except ValueError:
                    mtype = MemoryType.EPISODIC

                result = self.memory.remember(
                    user_id=arguments["user_id"],
                    content=arguments["content"],
                    importance=arguments.get("importance", 0.5),
                    tags=arguments.get("tags"),
                    namespace=arguments.get("namespace", "default"),
                    session_id=arguments.get("session_id"),
                    memory_type=mtype,
                    confidence=arguments.get("confidence", 1.0),
                )
                return [TextContent(type="text", text=f"Stored memory: {result}")]

            elif name == "recall":
                results = self.memory.recall(
                    user_id=arguments["user_id"],
                    query=arguments["query"],
                    top_k=arguments.get("top_k", self._top_k) or self._top_k,
                    namespace=arguments.get("namespace", "default"),
                    session_id=arguments.get("session_id"),
                    hybrid_search=arguments.get("hybrid_search", True),
                )
                if not results:
                    return [TextContent(type="text", text="No memories found")]
                output = "\n".join([f"- {r.content}" for r in results])
                return [TextContent(type="text", text=output)]

            elif name == "recall_stream":
                return await self._handle_recall_stream(arguments)

            elif name == "recall_explain":
                explained = self.memory.recall_explain(
                    user_id=arguments["user_id"],
                    query=arguments["query"],
                    top_k=arguments.get("top_k", 5),
                    namespace=arguments.get("namespace", "default"),
                )
                if not explained:
                    return [TextContent(type="text", text="No memories found")]
                lines = []
                for item in explained:
                    mem = item["memory"]
                    exp = item["explanation"]
                    lines.append(f"- {mem.content} (score: {exp['final_score']})")
                    bm25 = exp.get("bm25_score", "n/a")
                    lines.append(
                        f"  semantic={exp['semantic_score']} "
                        f"recency={exp['recency_score']} bm25={bm25}"
                    )
                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "prune":
                deleted = self.memory.prune(
                    user_id=arguments["user_id"],
                    max_age_days=arguments.get("max_age_days"),
                    min_importance=arguments.get("min_importance"),
                    namespace=arguments.get("namespace", "default"),
                )
                return [TextContent(type="text", text=f"Pruned {deleted} memories")]

            elif name == "stats":
                stats_data = self.memory.stats(arguments["user_id"])
                output = (
                    f"Total: {stats_data['total']}\n"
                    f"By lifecycle: {stats_data['by_lifecycle']}\n"
                    f"Avg importance: {stats_data['avg_importance']:.2f}"
                )
                return [TextContent(type="text", text=output)]

            elif name == "consolidate":
                mid = self.memory.consolidate(
                    user_id=arguments["user_id"],
                    namespace=arguments.get("namespace", "default"),
                    min_memories=arguments.get("min_memories", 5),
                    max_age_days=arguments.get("max_age_days", 30.0),
                )
                if mid:
                    return [TextContent(type="text", text=f"Consolidated into memory: {mid}")]
                return [TextContent(type="text", text="No consolidation needed")]

            elif name == "topics":
                clusters = self.memory.cluster_topics(
                    user_id=arguments["user_id"],
                    n_clusters=arguments.get("n_clusters", 3),
                    namespace=arguments.get("namespace", "default"),
                )
                if not clusters:
                    return [TextContent(type="text", text="No topics found")]
                lines = []
                for label, mems in clusters.items():
                    lines.append(f"- {label}: {len(mems)} memories")
                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "graph":
                graph_data = self.memory.get_memory_graph(
                    user_id=arguments["user_id"],
                    namespace=arguments.get("namespace", "default"),
                )
                entities = [e["text"] for e in graph_data.get("entities", [])[:10]]
                relations = [
                    f"{r['subject']} -{r['predicate']}-> {r['object']}"
                    for r in graph_data.get("relations", [])[:10]
                ]
                lines = ["Entities:", ", ".join(entities) if entities else "None"]
                lines.append("Relations:")
                lines.extend(relations if relations else ["None"])
                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "list_users":
                users = self.memory.list_users()
                return [
                    TextContent(type="text", text=f"Users: {', '.join(users) if users else 'None'}")
                ]

            elif name == "forget":
                count = self.memory.forget(
                    user_id=arguments["user_id"],
                    memory_id=arguments.get("memory_id"),
                )
                return [TextContent(type="text", text=f"Deleted {count} memory(ies)")]

            elif name == "context_block":
                result = self.memory.context_block(
                    user_id=arguments["user_id"],
                    query=arguments["query"],
                    top_k=arguments.get("top_k", 5),
                    max_tokens=arguments.get("max_tokens", 1500),
                )
                return [TextContent(type="text", text=result or "No context found")]

            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _handle_recall_stream(
        self, arguments: dict[str, Any]
    ) -> list[TextContent]:
        """Handle recall_stream tool call with progress notifications.

        Each memory is yielded progressively via MCP progress notifications
        as it becomes available. After all results are collected, the full
        list is returned as the tool output.

        Compatible with MCP clients that support progress notifications
        and StreamableHTTP transport.
        """
        top_k = arguments.get("top_k", self._top_k) or self._top_k
        user_id = arguments["user_id"]
        query = arguments["query"]

        # Unique token per request to avoid collisions across concurrent calls
        progress_token = str(uuid.uuid4())

        results: list[str] = []
        count = 0
        try:
            ctx = self.server.request_context()
        except LookupError:
            ctx = None

        async for memory in self.memory.recall_stream(
            user_id=user_id,
            query=query,
            top_k=top_k,
            namespace=arguments.get("namespace", "default"),
            session_id=arguments.get("session_id"),
            hybrid_search=arguments.get("hybrid_search", True),
        ):
            line = f"- {memory.content}"
            results.append(line)
            count += 1

            # Send progress notification for each yielded memory
            if ctx is not None:
                try:
                    await ctx.session.send_progress_notification(
                        progress_token=progress_token,
                        progress=float(count),
                        total=float(top_k),
                        message=line,
                    )
                except Exception:
                    logger.debug(
                        "Failed to send progress notification for recall_stream",
                        exc_info=True,
                    )

        if not results:
            return [TextContent(type="text", text="No memories found")]
        return [TextContent(type="text", text="\n".join(results))]

    async def run(self) -> None:
        """Run the MCP server with stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )

    def create_app(self) -> Any | None:
        """Create a Starlette ASGI app for StreamableHTTP transport.

        Returns None if StreamableHTTP dependencies are not installed.
        Callers can serve this with ``uvicorn`` or another ASGI server:

            uvicorn kemi.mcp_server:app

        The resulting endpoint accepts POST ``/message`` with
        ``Content-Type: application/json`` and returns
        ``text/event-stream`` for streaming tool results.
        """
        if not _HAS_STREAMABLE_HTTP:
            return None

        init_opts = self.server.create_initialization_options()

        async def handle_message(request: _Request) -> _Response:
            streamable = _StreamableHTTP(self.server, request, init_opts)
            return await streamable.handle_message(request)

        app = _Starlette(
            routes=[
                _Route("/message", handle_message, methods=["POST"]),
            ],
        )
        return app


app: Any | None = None


def _create_app() -> Any | None:
    """Create the global app instance for StreamableHTTP.

    Exposed at module level so ASGI servers (uvicorn, etc.) can do:
        uvicorn kemi.mcp_server:app
    """
    global app
    if app is None:
        server = KemiMCPServer()
        app = server.create_app()
    return app


async def main() -> None:
    """Entry point for python -m kemi.mcp_server.

    Starts the MCP server with stdio transport by default.
    Set ``KEMI_MCP_TRANSPORT=http`` to use StreamableHTTP
    instead (requires starlette + uvicorn).
    """
    transport = os.environ.get("KEMI_MCP_TRANSPORT", "stdio")

    if transport == "http":
        if not _HAS_STREAMABLE_HTTP:
            print(
                "StreamableHTTP transport requires starlette. "
                "Install with: pip install 'kemi[mcp]' starlette",
                file=sys.stderr,
            )
            sys.exit(1)
        import uvicorn

        host = os.environ.get("KEMI_MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("KEMI_MCP_PORT", "10888"))
        print(
            f"[kemi MCP] Starting StreamableHTTP on {host}:{port}",
            file=sys.stderr,
        )
        mcp_app = _create_app()
        await uvicorn.run(mcp_app, host=host, port=port, log_level="info")  # type: ignore[arg-type]
        return

    # Default: stdio transport
    server = KemiMCPServer()
    await server.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
