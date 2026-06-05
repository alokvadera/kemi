"""Tests for __main__.py — entry point for python -m kemi.

Covers both the MCP server path and the CLI fallback path.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch


class TestMainMCPPath:
    """Tests for the MCP server code path."""

    def test_main_calls_mcp_server(self):
        """When mcp_server is importable, main() calls asyncio.run(mcp_main())."""
        mock_mcp_main = MagicMock()
        mock_mod = MagicMock()
        mock_mod.main = mock_mcp_main

        with patch.dict(sys.modules, {"kemi.mcp_server": mock_mod}):
            with patch("asyncio.run") as mock_run:
                import kemi.__main__ as mod

                importlib.reload(mod)
                mod.main()

        # __main__ does: asyncio.run(mcp_main()) — calls mcp_main, passes result
        mock_mcp_main.assert_called_once()
        mock_run.assert_called_once_with(mock_mcp_main())

    def test_main_prints_starting_message(self):
        """MCP path prints 'Starting kemi MCP server...' to stderr."""
        mock_mod = MagicMock()

        with patch.dict(sys.modules, {"kemi.mcp_server": mock_mod}):
            with patch("asyncio.run"):
                import kemi.__main__ as mod

                importlib.reload(mod)
                with patch("builtins.print") as mock_print:
                    mod.main()

        mock_print.assert_any_call("Starting kemi MCP server...", file=sys.stderr)


class TestMainCLIFallback:
    """Tests for the CLI fallback code path."""

    def test_main_falls_back_to_cli_on_import_error(self):
        """When mcp_server is not importable, main() falls back to cli."""
        mock_cli_main = MagicMock()

        # Make kemi.mcp_server raise ImportError when imported
        with patch.dict(sys.modules, {"kemi.mcp_server": None}):
            with patch("kemi.cli.main", mock_cli_main):
                import kemi.__main__ as mod

                importlib.reload(mod)
                with patch.object(sys, "argv", ["kemi", "list", "test"]):
                    mod.main()

        mock_cli_main.assert_called_once()

    def test_fallback_prints_fallback_message(self):
        """CLI fallback prints a message about MCP not being available."""
        with patch.dict(sys.modules, {"kemi.mcp_server": None}):
            with patch("kemi.cli.main"):
                import kemi.__main__ as mod

                importlib.reload(mod)
                with patch("builtins.print") as mock_print:
                    with patch.object(sys, "argv", ["kemi"]):
                        mod.main()

        mock_print.assert_any_call(
            "MCP server not available (install with: pip install 'kemi[mcp]'). "
            "Falling back to CLI.",
            file=sys.stderr,
        )

    def test_fallback_sets_argv0(self):
        """CLI fallback sets sys.argv[0] to 'kemi'."""
        with patch.dict(sys.modules, {"kemi.mcp_server": None}):
            with patch("kemi.cli.main"):
                import kemi.__main__ as mod

                importlib.reload(mod)
                with patch.object(sys, "argv", ["something_else", "list"]):
                    mod.main()

                    assert sys.argv[0] == "kemi"


class TestMainIfNameBlock:
    """Tests for the if __name__ == '__main__' block."""

    def test_main_is_callable(self):
        """main() can be called directly."""
        mock_mcp_main = MagicMock()
        with patch.dict(sys.modules, {"kemi.mcp_server": MagicMock(main=mock_mcp_main)}):
            with patch("asyncio.run"):
                import kemi.__main__ as mod

                importlib.reload(mod)
                # Should not raise
                mod.main()
