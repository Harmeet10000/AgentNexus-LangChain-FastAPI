"""The Docling ingestion path delegates one batch to the shared embedder."""

from unittest.mock import AsyncMock

import pytest

from app.shared.langchain_layer.embeddings import EmbeddingTaskType
from app.shared.rag.docling import ingest_v2
from app.shared.rag.docling.models import Chunk


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            document_id="doc-1",
            chunk_index=0,
            preamble="Agreement / Indemnity",
            content="Supplier shall indemnify Buyer.",
        ),
        Chunk(document_id="doc-1", chunk_index=1, content="Bare second clause."),
    ]


async def test_docling_uses_one_shared_document_embedding_batch(monkeypatch) -> None:
    shared = AsyncMock(return_value=[[0.1], [0.2]])
    monkeypatch.setattr(ingest_v2, "embed_texts", shared)

    embedded = await ingest_v2.embed_document_chunks(_chunks())

    shared.assert_awaited_once_with(
        ["Agreement / Indemnity\n\nSupplier shall indemnify Buyer.", "Bare second clause."],
        task_type=EmbeddingTaskType.DOCUMENT,
    )
    assert [chunk.embedding for chunk in embedded] == [[0.1], [0.2]]


async def test_provider_failure_is_not_replaced_with_vectors(monkeypatch) -> None:
    failure = RuntimeError("provider unavailable")
    monkeypatch.setattr(ingest_v2, "embed_texts", AsyncMock(side_effect=failure))

    with pytest.raises(RuntimeError) as exc_info:
        await ingest_v2.embed_document_chunks(_chunks())

    assert exc_info.value is failure
