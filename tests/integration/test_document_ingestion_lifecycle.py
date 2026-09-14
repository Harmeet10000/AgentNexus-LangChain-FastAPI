"""Live upload-to-compiled-graph acceptance check for graph lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from returns.result import Success
from sqlalchemy import delete, func, select

from app.config import get_settings
from app.connections import init_db
from app.features.documents import service as document_service
from app.features.documents.ingestion_graph import build_document_ingestion_graph
from app.features.documents.model import UnifiedChunk, UnifiedDocument
from app.features.documents.repository import DocumentRepository
from app.features.documents.service import process_document_ingestion, run_document_ingestion_task
from app.shared.services.storage import StorageService

if TYPE_CHECKING:
    from typing import Any

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]


async def test_uploaded_fixture_reaches_chunk_rows_through_the_compiled_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, session_local = await init_db()
    document_id = ""
    run_id = uuid4().hex
    user_id = f"graph-lifecycle-{run_id}"
    object_store = MagicMock(spec=StorageService)
    object_store.get_object = AsyncMock(
        return_value=Success(b"Graph lifecycle fixture.\n\nThis paragraph becomes a searchable chunk.")
    )
    llm = FakeListChatModel(responses=["unused"])
    width = get_settings().EMBEDDING_DIMENSION

    async def embed_texts(texts: list[str], **_kwargs: object) -> list[list[float]]:
        return [[0.0] * width for _ in texts]

    monkeypatch.setattr(document_service, "embed_texts", embed_texts)
    graph = build_document_ingestion_graph(
        object_store=cast("Any", object_store),
        graphiti=None,
        ingest_document_fn=process_document_ingestion,
        llm=cast("Any", llm),
    )

    try:
        async with session_local() as session:
            repo = DocumentRepository(session)
            created = await repo.create_document(
                user_id=user_id,
                title="fixture.md",
                source_uri=None,
                object_uri="s3://test/fixture.txt",
                content_hash=run_id,
                document_kind="generic",
                status="received",
                jurisdiction=None,
                contract_type=None,
                parties=[],
                metadata_={},
            )
            document = created.unwrap()
            document_id = str(document.id)
            await session.commit()

        result = await run_document_ingestion_task(
            document_id=document_id,
            user_id=user_id,
            filename="fixture.md",
            content_type="text/markdown",
            object_uri="s3://test/fixture.txt",
            graph=graph,
            session_local=session_local,
        )

        async with session_local() as session:
            chunk_count = await session.scalar(
                select(func.count()).select_from(UnifiedChunk).where(
                    UnifiedChunk.document_id == document_id
                )
            )
            stored_document = await session.scalar(
                select(UnifiedDocument).where(UnifiedDocument.id == document_id)
            )
        assert result["status"] in {"completed", "completed_with_warnings"}
        assert chunk_count is not None
        assert chunk_count > 0
        assert stored_document is not None
        assert stored_document.structural_tree
        assert stored_document.extraction_incomplete is True
    finally:
        if document_id:
            async with session_local() as session, session.begin():
                await session.execute(delete(UnifiedDocument).where(UnifiedDocument.id == document_id))
        await engine.dispose()
