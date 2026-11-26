"""Postgres connection helpers."""

from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

SQL_DIR = Path(__file__).resolve().parents[2] / "db" / "init"

_DROP_ALL = """
DROP TABLE IF EXISTS tickets, idempotency_keys, order_events, orders, customers CASCADE;
DROP SEQUENCE IF EXISTS ticket_seq;
"""


def create_pool(conninfo: str, *, min_size: int = 1, max_size: int = 10) -> AsyncConnectionPool:
    """Create (but do not open) an async pool. Open it with ``await pool.open()``."""
    return AsyncConnectionPool(
        conninfo,
        min_size=min_size,
        max_size=max_size,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )


async def reset_database(conninfo: str) -> None:
    """Drop the application tables and recreate them with seed data."""
    schema = (SQL_DIR / "01_schema.sql").read_text()
    seed = (SQL_DIR / "02_seed.sql").read_text()
    async with await psycopg.AsyncConnection.connect(conninfo, autocommit=True) as conn:
        await conn.execute(_DROP_ALL)
        await conn.execute(schema)
        await conn.execute(seed)
