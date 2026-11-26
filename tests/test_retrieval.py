"""Knowledge-base retrieval: chunking, every pipeline configuration, and the MCP tool."""

import json

import pytest
from mcp import Client

from supportpilot.graph.builder import cited_sources
from supportpilot.kb import RetrievalConfig, Retriever
from supportpilot.kb.documents import chunk_document, load_documents
from supportpilot.mcp_server import USER_ID_META_KEY, build_mcp_server
from tests.conftest import ALICE

CONFIGS = {
    "vector": RetrievalConfig(bm25=False, rerank=False, contextual=False),
    "bm25": RetrievalConfig(vector=False, rerank=False, contextual=False),
    "hybrid": RetrievalConfig(rerank=False, contextual=False),
    "hybrid_rerank": RetrievalConfig(contextual=False),
    "full": RetrievalConfig(),
    "fts_fallback": RetrievalConfig(bm25_backend="postgres_fts"),
}


def test_documents_chunk_by_section_with_stable_ids():
    docs = load_documents()
    assert {d.doc_id for d in docs} == {
        "returns", "refunds", "shipping", "exchanges", "warranty", "payment", "membership", "privacy"
    }  # fmt: skip
    chunks = [c for d in docs for c in chunk_document(d)]
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.content.startswith(f"## {c.section}") for c in chunks)
    assert "Returns Policy § Electronics" in {c.source for c in chunks}


@pytest.mark.parametrize("name", list(CONFIGS))
async def test_every_configuration_finds_the_exchange_window(pool, name):
    hits = await Retriever(pool).search(
        "Can I exchange a sweater for another size after 40 days?", CONFIGS[name]
    )
    assert 0 < len(hits) <= 5
    assert "exchanges#exchange-window" in [h.chunk_id for h in hits]


async def test_contextual_retrieval_uses_the_contextualized_text(pool):
    hits = await Retriever(pool).search("refund timing", RetrievalConfig())
    assert all(h.context for h in hits), "every chunk should carry a generated context"


async def test_at_least_one_retriever_is_required(pool):
    with pytest.raises(ValueError):
        await Retriever(pool).search("refunds", RetrievalConfig(vector=False, bm25=False))


async def test_search_policy_tool_returns_citable_passages(pool):
    async with Client(build_mcp_server(pool)) as client:
        result = await client.call_tool(
            "search_policy", {"query": "Do gift cards expire?", "top_k": 3}, meta={USER_ID_META_KEY: ALICE}
        )
    assert not result.is_error
    data = json.loads(result.content[0].text)
    assert len(data["results"]) == 3
    assert data["results"][0]["source"] == "Payment Policy § Gift cards"


async def test_search_policy_requires_identity(pool):
    async with Client(build_mcp_server(pool)) as client:
        result = await client.call_tool("search_policy", {"query": "returns"})
    assert result.is_error


def test_only_citations_of_retrieved_passages_count():
    hits = [{"source": "Returns Policy § Electronics"}]
    reply = "15 days [Returns Policy § Electronics]. Also [Made Up § Thing] and [not a citation]."
    assert cited_sources(reply, hits) == ["Returns Policy § Electronics"]
