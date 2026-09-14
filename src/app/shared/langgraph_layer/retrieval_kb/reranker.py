"""Cross-encoder reranking adapter for legal retrieval."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

import httpx

from app.config import get_settings
from app.utils import logger

if TYPE_CHECKING:
    from .state import RetrievedChunk


class Reranker(Protocol):
    """Minimum reranking behaviour consumed by the retrieval graph.

    The keyword-only limit keeps graph and service callers on one contract.
    """

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]: ...


class HostedReranker:
    """Provider reranking over HTTP, satisfying `Reranker` with no local model.

    Speaks the Cohere-compatible rerank wire shape (`POST {endpoint}` with
    `{model, query, documents, top_n}`); the client is injectable so tests run
    without a network. Any provider failure — including a missing key — degrades
    to the fused order truncated to `limit`, never raising.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if not self._api_key:
            logger.bind(operation="hosted_rerank").warning(
                "hosted_reranker_key_absent_degrading_to_fused_order"
            )
            return chunks[:limit]
        try:
            return await self._rerank_remote(query, chunks, limit=limit)
        except Exception as exc:  # noqa: BLE001 — degradation path by contract
            exc.add_note(f"model={self._model}, operation=hosted_rerank")
            logger.bind(model=self._model, operation="hosted_rerank", error=str(exc)).warning(
                "hosted_rerank_failed"
            )
            return chunks[:limit]

    async def _rerank_remote(
        self, query: str, chunks: list[RetrievedChunk], *, limit: int
    ) -> list[RetrievedChunk]:
        documents = [f"{chunk.preamble}\n\n{chunk.chunk_text}" for chunk in chunks]
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            response = await client.post(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "query": query,
                    "documents": documents,
                    "top_n": limit,
                },
            )
            response.raise_for_status()
            return self._order_by_provider(chunks, response.json(), limit=limit)
        finally:
            if owns_client:
                await client.aclose()

    @staticmethod
    def _order_by_provider(
        chunks: list[RetrievedChunk], payload: object, *, limit: int
    ) -> list[RetrievedChunk]:
        if not isinstance(payload, dict):
            message = f"Unexpected rerank response shape: {type(payload).__name__}"
            raise TypeError(message)
        results = payload.get("results")
        if not isinstance(results, list):
            message = "Rerank response carries no result list"
            raise TypeError(message)
        ordered: list[RetrievedChunk] = []
        for entry in results[:limit]:
            if not isinstance(entry, dict):
                continue
            index = entry.get("index")
            if not isinstance(index, int) or not 0 <= index < len(chunks):
                continue
            score = entry.get("relevance_score", 0.0)
            ordered.append(
                chunks[index].model_copy(update={"score": float(score)})
            )
        if not ordered:
            message = "Rerank response ordered nothing"
            raise ValueError(message)
        return ordered


@lru_cache(maxsize=1)
def get_shared_reranker() -> HostedReranker:
    """Return the process-shared hosted reranker configuration."""
    settings = get_settings()
    return HostedReranker(
        api_key=settings.RERANKER_API_KEY.get_secret_value(),
        model=settings.RERANKER_MODEL,
        endpoint=settings.RERANKER_ENDPOINT,
        timeout_seconds=settings.RERANKER_TIMEOUT_SECONDS,
    )


def get_configured_reranker() -> Reranker:
    """Resolve the single hosted reranking implementation."""
    return get_shared_reranker()
