"""Application-owned embedding interface for search."""

from typing import Protocol


class SearchEmbeddingQuotaExhaustedError(Exception):
    """The provider's current embedding quota cannot serve this job."""


class SearchEmbeddings(Protocol):
    """The two embedding operations search needs, without provider types."""

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
