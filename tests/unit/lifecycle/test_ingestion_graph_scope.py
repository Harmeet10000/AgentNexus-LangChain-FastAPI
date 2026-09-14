"""The compiled ingestion graph keeps job-scoped objects out of graph state."""

from __future__ import annotations

import inspect
from typing import get_type_hints
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.documents.ingestion_graph import (
    DocumentIngestionState,
    _make_ingest_document_node,
    build_document_ingestion_graph,
)
from app.features.documents.repository import DocumentRepository

pytestmark = pytest.mark.unit


def test_compiled_graph_captures_no_repository_or_session() -> None:
    object_store = MagicMock()
    graphiti = MagicMock()
    llm = MagicMock()
    ingest = AsyncMock()

    node = _make_ingest_document_node(
        object_store=object_store,
        graphiti=graphiti,
        ingest_document_fn=ingest,
        llm=llm,
    )
    graph = build_document_ingestion_graph(
        object_store=object_store,
        graphiti=graphiti,
        ingest_document_fn=ingest,
        llm=llm,
    )

    captured = inspect.getclosurevars(node).nonlocals.values()
    assert not any(isinstance(value, (DocumentRepository, AsyncSession)) for value in captured)
    assert "repo" not in get_type_hints(DocumentIngestionState)
    assert "repository" not in get_type_hints(DocumentIngestionState)
    assert "session" not in get_type_hints(DocumentIngestionState)
    assert graph is not None
