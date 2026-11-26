"""Contextual Retrieval: give every chunk a short blurb that situates it in its document.

The blurb is prepended to the chunk before embedding and BM25 indexing, so a chunk
such as "## Limits — Each item can be exchanged once" also carries "exchanges policy".

Generation uses the fast model with prompt caching: the whole document sits in a
cached system block, and only the chunk changes between calls. Results are stored
in ``kb/contexts.json`` keyed by chunk id and a hash of (document, chunk), so
re-ingesting unchanged documents is offline, free and deterministic.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from supportpilot.kb.documents import KB_DIR, Chunk, Document

log = logging.getLogger(__name__)

CONTEXTS_FILE = KB_DIR / "contexts.json"

_INSTRUCTIONS = (
    "You write short retrieval context for chunks of an online store's customer-support policy "
    "documents. Given the whole document and one chunk of it, write one or two sentences that "
    "situate the chunk within the document, naming the document and the topic, so the chunk can "
    "be found by search. Answer with the context only, no preamble."
)


def _hash(doc: Document, chunk: Chunk) -> str:
    return hashlib.sha256(f"{doc.text}\x00{chunk.content}".encode()).hexdigest()[:16]


@dataclass
class ContextStats:
    generated: int = 0
    reused: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    uncached_input_tokens: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)


class Contextualizer:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        cache_file: Path = CONTEXTS_FILE,
        allow_generate: bool = True,
    ):
        self._model = model
        self._api_key = api_key
        self._file = cache_file
        self._allow_generate = allow_generate
        self._cache: dict[str, dict[str, str]] = (
            json.loads(cache_file.read_text()) if cache_file.exists() else {}
        )
        self.stats = ContextStats()
        self._client: anthropic.AsyncAnthropic | None = None

    async def context_for(self, doc: Document, chunk: Chunk) -> str:
        digest = _hash(doc, chunk)
        cached = self._cache.get(chunk.id)
        if cached and cached["hash"] == digest:
            self.stats.reused += 1
            return cached["context"]
        if not self._allow_generate:
            raise RuntimeError(
                f"No stored context for {chunk.id}; run `uv run python scripts/ingest_kb.py` with an API key."
            )
        context = await self._generate(doc, chunk)
        self._cache[chunk.id] = {"hash": digest, "model": self._model, "context": context}
        self.stats.generated += 1
        return context

    async def _generate(self, doc: Document, chunk: Chunk) -> str:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=[
                {"type": "text", "text": _INSTRUCTIONS},
                # The document is identical for every chunk of it: cache it.
                {
                    "type": "text",
                    "text": f'<document title="{doc.title}">\n{doc.text}\n</document>',
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[{"role": "user", "content": f"<chunk>\n{chunk.content}\n</chunk>"}],
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
        )
        usage = response.usage
        self.stats.cache_read_tokens += usage.cache_read_input_tokens or 0
        self.stats.cache_write_tokens += usage.cache_creation_input_tokens or 0
        self.stats.uncached_input_tokens += usage.input_tokens
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "refusal" or not text:
            log.warning("no context generated for %s (stop_reason=%s)", chunk.id, response.stop_reason)
            return ""
        return text

    def save(self) -> None:
        self._file.write_text(json.dumps(self._cache, indent=2, sort_keys=True) + "\n")
