"""MCP (Model Context Protocol) server entry point.

The implementation lives in :mod:`kemi.interfaces.mcp.server`; this
module re-exports ``main`` for callers that want to start the
server programmatically.
"""

from kemi.interfaces.mcp.server import main

__all__ = ["main"]

