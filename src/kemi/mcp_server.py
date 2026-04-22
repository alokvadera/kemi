"""MCP server for kemi persistent memory."""

import os
import sys

from kemi import Memory
from kemi.adapters.embedding.fastembed import FastEmbedAdapter
from kemi.adapters.storage.sqlite import SQLiteStorageAdapter
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


def _get_config() -> dict:
    """Read configuration from environment variables."""
    return {
        "db_path": os.environ.get("KEMI_DB_PATH", os.path.expanduser("~/.kemi/memories.db")),
        "model": os.environ.get("KEMI_MODEL", "BAAI/bge-small-en-v1.5"),
        "top_k": int(os.environ.get("KEMI_TOP_K", "5")),
    }


def _print_config(config: dict) -> None:
    """Print active configuration to stderr."""
    print(
        f"[kemi MCP] Config: db={config['db_path']}, "
        f"model={config['model']}, top_k={config['top_k']}",
        file=sys.stderr,
    )


class KemiMCPServer:
    """MCP server exposing kemi Memory tools."""

    def __init__(self):
        config = _get_config()
        _print_config(config)

        db_path = config["db_path"]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        store = SQLiteStorageAdapter(db_path=db_path)
        embed = FastEmbedAdapter(model_name=config["model"])

        self.memory = Memory(embed=embed, store=store)
        self.server = Server("kemi")
        self._top_k = config["top_k"]

        @self.server.list_tools()
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
                        },
                        "required": ["user_id", "query"],
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

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "remember":
                result = self.memory.remember(
                    user_id=arguments["user_id"],
                    content=arguments["content"],
                    importance=arguments.get("importance", 0.5),
                )
                return [TextContent(type="text", text=f"Stored memory: {result}")]

            elif name == "recall":
                results = self.memory.recall(
                    user_id=arguments["user_id"],
                    query=arguments["query"],
                    top_k=arguments.get("top_k", self._top_k),
                )
                if not results:
                    return [TextContent(type="text", text="No memories found")]
                output = "\n".join([f"- {r.content}" for r in results])
                return [TextContent(type="text", text=output)]

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

    async def run(self):
        """Run the MCP server with stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main():
    """Entry point for python -m kemi.mcp_server"""
    server = KemiMCPServer()
    await server.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
