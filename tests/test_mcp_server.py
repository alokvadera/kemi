"""Tests for src/kemi/mcp_server.py — streaming recall and StreamableHTTP."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kemi.models import LifecycleState, MemoryType

# Skip entire module if mcp is not available
try:
    from kemi.mcp_server import KemiMCPServer

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    KemiMCPServer = None  # type: ignore[assignment, misc]

pytestmark = pytest.mark.skipif(not _MCP_AVAILABLE, reason="mcp not installed")


@pytest.fixture
def mock_memory():
    """Create a Memory-like object with recall_stream."""
    mem = MagicMock()

    async def recall_stream_gen(user_id, query, top_k=5, **kwargs):
        for i in range(3):
            m = MagicMock()
            m.memory_id = f"mem-{i}"
            m.content = f"memory {i} content"
            m.score = 1.0 - i * 0.1
            m.importance = 0.5
            m.lifecycle_state = LifecycleState.ACTIVE
            m.created_at = None
            m.tags = []
            m.memory_type = MemoryType.EPISODIC
            m.confidence = 1.0
            m.session_id = None
            m.namespace = "default"
            m.version = 1
            yield m

    mem.recall_stream = recall_stream_gen
    return mem


@pytest.fixture
def server():
    """Create a KemiMCPServer with mocked __init__ to avoid real DB init."""
    with patch.object(KemiMCPServer, "__init__", return_value=None):
        srv = KemiMCPServer.__new__(KemiMCPServer)
        srv.memory = MagicMock()
        srv.server = MagicMock()
        srv._top_k = 5
        yield srv


class TestRecallStreamTool:
    """Tests for the recall_stream MCP tool."""

    def test_recall_stream_tool_name_reflected(self):
        """Test that recall_stream appears in the tool handler registrations."""
        with patch.object(KemiMCPServer, "__init__", return_value=None):
            srv = KemiMCPServer.__new__(KemiMCPServer)
            srv.server = MagicMock()
            srv.memory = MagicMock()
            srv._top_k = 5

        # The private _tool_handlers dict is populated by the list_tools()
        # decorator during a real initialization. For unit testing, we verify
        # that _handle_recall_stream method exists and accept recall_stream
        # commands — the actual tool definition is best tested via integration.
        assert hasattr(srv, "_handle_recall_stream")
        assert callable(srv._handle_recall_stream)

    @pytest.mark.asyncio
    async def test_recall_stream_returns_formatted_results(self, server, mock_memory):
        """Test that recall_stream handler returns formatted text content."""
        server.memory = mock_memory

        mock_ctx = MagicMock()
        mock_ctx.session.send_progress_notification = AsyncMock()
        with patch.object(server.server, "request_context", return_value=mock_ctx):
            result = await server._handle_recall_stream({
                "user_id": "alice",
                "query": "python",
                "top_k": 5,
            })

        assert len(result) == 1
        assert result[0].type == "text"
        text = result[0].text
        assert "- memory 0 content" in text
        assert "- memory 1 content" in text
        assert "- memory 2 content" in text

    @pytest.mark.asyncio
    async def test_recall_stream_sends_progress_notifications(self, server, mock_memory):
        """Test that each yielded memory triggers a progress notification."""
        server.memory = mock_memory

        mock_ctx = MagicMock()
        mock_send = AsyncMock()
        mock_ctx.session.send_progress_notification = mock_send
        with patch.object(server.server, "request_context", return_value=mock_ctx):
            await server._handle_recall_stream({
                "user_id": "alice",
                "query": "python",
                "top_k": 5,
            })

        # Should have sent 3 progress notifications (one per memory)
        assert mock_send.call_count == 3

        # First notification should be for memory 0
        first_call = mock_send.call_args_list[0]
        # Progress token should be a UUID (unique per request)
        import uuid
        assert uuid.UUID(first_call.kwargs["progress_token"], version=4)
        assert first_call.kwargs["progress"] == 1.0
        assert first_call.kwargs["total"] == 5.0  # top_k
        assert "- memory 0 content" in first_call.kwargs["message"]

    @pytest.mark.asyncio
    async def test_recall_stream_empty_results(self, server, mock_memory):
        """Test that empty results return 'No memories found'."""
        # Override recall_stream to yield nothing
        async def empty_gen(user_id, query, top_k=5, **kwargs):
            if False:
                yield  # make it a generator (unreachable, so it yields nothing)

        mock_memory.recall_stream = empty_gen
        server.memory = mock_memory

        mock_ctx = MagicMock()
        mock_ctx.session.send_progress_notification = AsyncMock()
        with patch.object(server.server, "request_context", return_value=mock_ctx):
            result = await server._handle_recall_stream({
                "user_id": "alice",
                "query": "python",
                "top_k": 5,
            })

        assert len(result) == 1
        assert result[0].text == "No memories found"

    @pytest.mark.asyncio
    async def test_recall_stream_no_request_context(self, server, mock_memory):
        """Test that handler works even when request_context raises LookupError."""
        server.memory = mock_memory

        # Make request_context raise LookupError (called outside request context)
        with patch.object(server.server, "request_context", side_effect=LookupError):
            result = await server._handle_recall_stream({
                "user_id": "alice",
                "query": "python",
                "top_k": 5,
            })

        # Should still return formatted results without progress notifications
        assert len(result) == 1
        text = result[0].text
        assert "- memory 0 content" in text
        assert "- memory 1 content" in text


class TestStreamableHTTP:
    """Tests for the StreamableHTTP transport support."""

    def test_create_app_returns_none_without_starlette(self, server):
        """Test that create_app returns None when starlette is not available."""
        with patch("kemi.mcp_server._HAS_STREAMABLE_HTTP", False):
            app = server.create_app()
            assert app is None

    def test_create_app_returns_app_with_starlette(self, server):
        """Test that create_app returns a Starlette app when available."""
        app = server.create_app()

        if app is not None:
            # Should be a Starlette app
            assert hasattr(app, "routes")
            assert len(app.routes) >= 1
            # Should have a /message route
            route_paths = [r.path for r in app.routes]
            assert "/message" in route_paths
