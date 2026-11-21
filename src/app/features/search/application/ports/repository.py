"""Application-owned persistence interface for the search slice."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import Any

    from app.features.search.domain.errors import SearchResult
    from app.features.search.domain.models import SearchChunkUpsert, SearchDocument
    from app.retrieval.fusion import RankedResultRow


class SearchTransaction(Protocol):
    """The single caller-owned transaction capability job staging needs."""

    execute: Callable[..., Awaitable[Any]]


class SearchRepository(Protocol):
    """Persistence behavior required by search application use cases."""

    @property
    def transaction(self) -> SearchTransaction: ...

    async def get_document_by_content_hash(
        self, *, tenant_id: str, content_hash: str
    ) -> SearchResult[SearchDocument | None]: ...

    async def get_document_by_id(
        self, *, tenant_id: str, document_id: str
    ) -> SearchResult[SearchDocument | None]: ...

    async def create_document(
        self,
        *,
        tenant_id: str,
        title: str,
        source_uri: str | None,
        content_hash: str,
        doc_metadata: dict[str, object],
    ) -> SearchResult[SearchDocument]: ...

    async def set_document_status(self, *, document_id: str, status: str) -> SearchResult[None]: ...

    async def rollback(self) -> None: ...

    async def upsert_chunks(self, chunks: Sequence[SearchChunkUpsert]) -> SearchResult[None]: ...

    async def analyze_chunks(self) -> SearchResult[None]: ...

    async def bm25_search(
        self,
        *,
        tenant_id: str,
        query: str,
        candidate_limit: int,
        metadata_filter: dict[str, object],
    ) -> SearchResult[list[RankedResultRow]]: ...

    async def vector_search(
        self,
        *,
        tenant_id: str,
        embedding: list[float],
        candidate_limit: int,
        metadata_filter: dict[str, object],
    ) -> SearchResult[list[RankedResultRow]]: ...

    async def trigram_search(
        self,
        *,
        tenant_id: str,
        query: str,
        candidate_limit: int,
        metadata_filter: dict[str, object],
    ) -> SearchResult[list[RankedResultRow]]: ...

    async def fetch_chunks_by_ids(
        self, *, tenant_id: str, chunk_ids: Sequence[str]
    ) -> SearchResult[dict[str, dict[str, Any]]]: ...
