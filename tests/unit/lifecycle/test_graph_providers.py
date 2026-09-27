"""Process-scoped graph provider wiring tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.features.documents import ingestion_graph
from app.lifecycle.graphs import provide_document_ingestion_graph
from app.shared.langchain_layer import models

pytestmark = pytest.mark.unit


def test_document_graph_provider_uses_the_canonical_model_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = object()
    compiled_graph = object()
    model_factory = MagicMock(return_value=llm)
    graph_factory = MagicMock(return_value=compiled_graph)
    monkeypatch.setattr(models, "_build_chat_model", model_factory)
    monkeypatch.setattr(ingestion_graph, "build_document_ingestion_graph", graph_factory)
    settings = SimpleNamespace(GEMINI_FLASH_MODEL="gemini-flash")

    result = provide_document_ingestion_graph(
        settings=settings,
        object_store=object(),
        graphiti=None,
        ingest_document_fn=MagicMock(),
    )

    assert result is compiled_graph
    model_factory.assert_called_once_with(
        model_name="gemini-flash",
        temperature=0.1,
        implementation="generic",
    )
    assert graph_factory.call_args.kwargs["llm"] is llm
