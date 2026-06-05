"""Entry point for python -m kemi."""

import sys


def main() -> None:
    try:
        import asyncio

        from kemi.mcp_server import main as mcp_main

        print("Starting kemi MCP server...", file=sys.stderr)
        asyncio.run(mcp_main())
    except ImportError:
        print(
            "MCP server not available (install with: pip install 'kemi[mcp]'). "
            "Falling back to CLI.",
            file=sys.stderr,
        )
        from kemi.cli import main as cli_main

        sys.argv[0] = "kemi"
        cli_main()


if __name__ == "__main__":
    main()
