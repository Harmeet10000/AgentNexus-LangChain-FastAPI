"""The legacy compatibility path maps its state keys explicitly.

``IngestionService`` used to hand LangGraph a dict with six undeclared keys;
LangGraph dropped them and the endpoint reported success over an empty
ingestion. These tests pin the two truthful outcomes: every key is mapped onto a
declared channel or invocation config, and an unplaceable key fails loudly,
naming itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from returns.result import Failure, Success

from app.features.documents.ingestion.legacy_errors import (
    IngestionGraphError,
    IngestionValidationError,
)
from app.features.documents.ingestion.legacy_service import IngestionService
from app.features.documents.ingestion.workflow import DocumentIngestionState
from app.utils.storage import StorageValidationError

if TYPE_CHECKING:
    from typing import Any

    from app.utils.storage import StorageService

pytestmark = pytest.mark.unit


class _CapturingGraph:
    def __init__(self) -> None:
        self.state: dict[str, object] | None = None
        self.config: dict[str, Any] | None = None

    async def ainvoke(
        self, state: dict[str, object], config: dict[str, Any] | None = None
    ) -> dict[str, object]:
        self.state = state
        self.config = config
        return {"stored_entity_ids": [], "stored_clause_ids": [], "stored_relationship_ids": []}


class _RecordingObjectStore:
    def __init__(self, *, uri: str = "s3://bucket/documents/user-1/doc/hash.pdf") -> None:
        self.uri = uri
        self.calls: list[dict[str, object]] = []

    async def put_object(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str],
    ):
        self.calls.append(
            {"key": key, "data": data, "content_type": content_type, "metadata": metadata}
        )
        return Success(self.uri)


class _FailingObjectStore:
    async def put_object(self, **_kwargs: object):
        return Failure(StorageValidationError(message="object storage rejected the upload"))


async def test_the_legacy_path_places_every_key_and_none_are_undeclared() -> None:
    graph = _CapturingGraph()
    store = _RecordingObjectStore()
    service = IngestionService(graph, object_store=cast("StorageService", store))

    result = await service.ingest_document(
        raw_bytes=b"clause text",
        user_id="user-1",
        filename="contract.pdf",
        source="contract.pdf",
        content_type="application/pdf",
        document_type="legal_contract",
        jurisdiction="India",
        doc_id="doc-1",
    )

    assert isinstance(result, Success)
    assert graph.state is not None
    assert set(graph.state) <= set(DocumentIngestionState.__annotations__)
    assert graph.state["document_id"] == "doc-1"
    assert graph.state["document_kind"] == "legal_contract"
    assert graph.state["tenant_id"] == "personal:user-1"
    assert graph.state["object_uri"] == store.uri
    assert "raw_bytes" not in graph.state
    assert store.calls
    assert store.calls[0]["data"] == b"clause text"
    assert graph.config is not None
    assert set(graph.config["configurable"]) == {"thread_id", "source", "jurisdiction"}


async def test_an_unroutable_raw_bytes_payload_fails_loudly_naming_the_key() -> None:
    graph = _CapturingGraph()
    service = IngestionService(graph)  # no object store: bytes cannot be placed

    result = await service.ingest_document(
        raw_bytes=b"clause text",
        user_id="user-1",
        filename="contract.pdf",
        source="contract.pdf",
        doc_id="doc-1",
    )

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, IngestionValidationError)
    assert error.details is not None
    assert error.details["unmapped_keys"] == ["raw_bytes"]
    assert graph.state is None, "an unmapped key must never reach ainvoke"


async def test_a_storage_failure_is_a_typed_infrastructure_failure() -> None:
    graph = _CapturingGraph()
    service = IngestionService(graph, object_store=cast("StorageService", _FailingObjectStore()))

    result = await service.ingest_document(
        raw_bytes=b"clause text",
        user_id="user-1",
        filename="contract.pdf",
        source="contract.pdf",
        doc_id="doc-1",
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), IngestionGraphError)
    assert graph.state is None
