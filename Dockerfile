# SupportPilot API image. Built by `docker compose up` (see README).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models \
    MEM0_TELEMETRY=False

WORKDIR /app

# Dependencies first (cached layer); runtime only, no dev / eval groups.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY src ./src
COPY db ./db
COPY kb ./kb
COPY scripts ./scripts
RUN uv sync --frozen --no-default-groups

EXPOSE 8000
# Index the knowledge base (offline: chunk contexts come from kb/contexts.json), then serve.
CMD ["sh", "-c", "uv run --no-sync python scripts/ingest_kb.py --offline && exec uv run --no-sync supportpilot"]
