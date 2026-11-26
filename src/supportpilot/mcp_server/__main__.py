"""Run the order and policy-search tools as a standalone stdio MCP server (for MCP Inspector / debugging).

Identity still comes from request ``_meta["supportpilot/user_id"]``; calls without it are refused.

    uv run python -m supportpilot.mcp_server
"""

import anyio

from supportpilot.config import get_settings
from supportpilot.db import create_pool
from supportpilot.kb import RetrievalConfig, Retriever
from supportpilot.mcp_server.server import build_mcp_server


async def _run() -> None:
    settings = get_settings()
    pool = create_pool(settings.database_url)
    await pool.open(wait=True)
    try:
        retriever = Retriever(pool, RetrievalConfig.from_settings(settings))
        await build_mcp_server(pool, retriever).run_stdio_async()
    finally:
        await pool.close()


if __name__ == "__main__":
    anyio.run(_run)
