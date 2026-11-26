"""Rebuild the dev database (tables + seed data) from db/init/*.sql.

uv run python scripts/reset_db.py
"""

import asyncio

from supportpilot.config import get_settings
from supportpilot.db import reset_database

if __name__ == "__main__":
    url = get_settings().database_url
    asyncio.run(reset_database(url))
    print(f"Database reset: {url.rsplit('@', 1)[-1]}")
