"""Local embedding and reranking models (fastembed / ONNX, CPU).

Models are loaded lazily and shared per process: the first call downloads them
into the fastembed cache, later calls are offline.
"""

import asyncio
import threading
from functools import lru_cache

_lock = threading.Lock()


@lru_cache(maxsize=4)
def _embedder(name: str):
    from fastembed import TextEmbedding

    with _lock:
        return TextEmbedding(name)


@lru_cache(maxsize=4)
def _reranker(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    with _lock:
        return TextCrossEncoder(name)


def embed_passages(model: str, texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _embedder(model).passage_embed(texts)]


def embed_query(model: str, text: str) -> list[float]:
    return next(iter(_embedder(model).query_embed(text))).tolist()


def rerank_scores(model: str, query: str, texts: list[str]) -> list[float]:
    return [float(s) for s in _reranker(model).rerank(query, texts)]


async def aembed_query(model: str, text: str) -> list[float]:
    return await asyncio.to_thread(embed_query, model, text)


async def arerank_scores(model: str, query: str, texts: list[str]) -> list[float]:
    return await asyncio.to_thread(rerank_scores, model, query, texts)
