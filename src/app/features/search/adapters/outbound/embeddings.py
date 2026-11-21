"""Provider adapter for the search embedding port."""

from typing import override

from app.ai.embeddings import (
    EmbeddingQuotaExhaustedError,
    EmbeddingTaskType,
    embed_text,
    embed_texts,
)
from app.features.search.application.ports import (
    SearchEmbeddingQuotaExhaustedError,
    SearchEmbeddings,
)


class ProviderSearchEmbeddings(SearchEmbeddings):
    """Bind search's two semantic operations to the configured provider."""

    @override
    async def embed_query(self, text: str) -> list[float]:
        return await embed_text(text, task_type=EmbeddingTaskType.QUERY)

    @override
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return await embed_texts(texts, task_type=EmbeddingTaskType.DOCUMENT)
        except EmbeddingQuotaExhaustedError as exc:
            raise SearchEmbeddingQuotaExhaustedError(exc.detail) from exc
