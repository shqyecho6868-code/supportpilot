"""Chunk, contextualize, embed and index the policy documents in kb/policies.

uv run python scripts/ingest_kb.py                 # dev database
uv run python scripts/ingest_kb.py --database-url postgresql://.../supportpilot_test
uv run python scripts/ingest_kb.py --offline       # only reuse kb/contexts.json, never call the API

Chunk contexts are cached in kb/contexts.json; only new or changed chunks call the fast model.
"""

import argparse
import asyncio
import logging

from supportpilot.config import get_settings
from supportpilot.kb.contextualize import Contextualizer
from supportpilot.kb.ingest import ingest


async def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    ctx = Contextualizer(settings.fast_model, settings.anthropic_api_key, allow_generate=not args.offline)
    result = await ingest(args.database_url, ctx, settings.embedding_model)
    s = ctx.stats
    print(f"Ingested {result} into {args.database_url.rsplit('@', 1)[-1]}")
    print(
        f"Contexts: {s.generated} generated, {s.reused} reused from kb/contexts.json; "
        f"prompt cache read={s.cache_read_tokens} write={s.cache_write_tokens} "
        f"uncached={s.uncached_input_tokens} tokens"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
