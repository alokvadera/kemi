"""Entry point for python -m kemi.mcp_server"""

import asyncio

from kemi.mcp_server import main

if __name__ == "__main__":
    asyncio.run(main())
