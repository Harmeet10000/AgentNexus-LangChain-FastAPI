"""Contract tests for the retained, deprecated ingestion upload route."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from fakeredis.aioredis import FakeRedis
from fastapi import FastAPI, Response
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from returns.result import Success
from starlette.datastructures import Headers, UploadFile

from app.config import get_settings
from app.features.documents.api.http import dependencies as document_dependencies
from app.features.documents.api.http import legacy_ingest_router as ingestion_router_module
from app.features.documents.application.commands.upload_document import (
    DocumentUploadResponse as CanonicalUploadResponse,
)
from app.features.documents.ingestion.legacy_dto import DocumentUploadResponse

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit


def _legacy_route() -> APIRoute:
    return next(
        route
        for route in ingestion_router_module.router.routes
        if isinstance(route, APIRoute) and route.path == "/ingestion/documents/upload"
    )


def test_legacy_upload_is_marked_deprecated_in_openapi_metadata() -> None:
    route = _legacy_route()

    assert route.deprecated is True
    assert route.summary == "Deprecated: upload a legal document"
    assert "/documents/upload" in (route.description or "")


def test_legacy_usage_telemetry_is_migration_addressable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BoundLogger:
        def __init__(self) -> None:
            self.fields: dict[str, object] | None = None
            self.message: str | None = None

        def bind(self, **fields: object) -> _BoundLogger:
            self.fields = fields
            return self

        def info(self, message: str) -> None:
            self.message = message

    logger = _BoundLogger()
    monkeypatch.setattr(ingestion_router_module, "logger", logger)

    ingestion_router_module._log_legacy_usage(
        user_id="user-1",
        filename="contract.pdf",
        sunset="2027-06-15",
    )

    assert logger.message == "Deprecated ingestion upload endpoint used"
    assert logger.fields == {
        "event": "legacy_ingestion_upload_deprecated",
        "endpoint": "/ingestion/documents/upload",
        "successor_endpoint": "/documents/upload",
        "user_id": "user-1",
        "filename": "contract.pdf",
        "sunset": "2027-06-15",
        "migration_action": "migrate_to_documents_upload",
    }


async def test_legacy_upload_advertises_successor_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeIngestionService:
        def __init__(self, *, ingestion_graph: object) -> None:
            del ingestion_graph

        async def ingest_document(self, **_kwargs: object):
            return Success(
                DocumentUploadResponse(
                    doc_id="doc-1",
                    status="completed",
                    entity_count=0,
                    clause_count=1,
                    relationship_count=0,
                    dropped_entity_count=0,
                )
            )

    monkeypatch.setattr(ingestion_router_module, "IngestionService", _FakeIngestionService)
    upload = UploadFile(
        file=BytesIO(b"clause text"),
        filename="contract.txt",
        headers=Headers({"content-type": "text/plain"}),
    )
    response = Response()

    await ingestion_router_module.upload_document(
        upload,
        object(),
        "user-1",
        response,
    )

    assert response.headers["deprecation"] == "true"
    assert response.headers["sunset"] == get_settings().API_SUNSET_DATE
    assert response.headers["link"] == (
        '</documents/upload>; rel="successor-version"; title="Canonical document upload"'
    )


async def test_legacy_upload_reports_a_typed_failure_when_bytes_cannot_be_placed() -> None:
    """The compatibility branch no longer reports success over an empty ingestion.

    The route builds ``IngestionService`` without an object store, so the legacy
    state's ``raw_bytes`` cannot be routed by reference. The truthful outcome is
    a typed validation failure naming the key — and the deprecation headers stay
    exactly where they were.
    """
    upload = UploadFile(
        file=BytesIO(b"clause text"),
        filename="contract.txt",
        headers=Headers({"content-type": "text/plain"}),
    )
    response = Response()

    result = await ingestion_router_module.upload_document(
        upload,
        object(),
        "user-1",
        response,
    )

    assert result.success is False
    assert response.status_code == 422
    assert response.headers["deprecation"] == "true"
    assert response.headers["sunset"] == get_settings().API_SUNSET_DATE
    assert response.headers["link"].startswith("</documents/upload>")


async def test_legacy_upload_keeps_deprecation_headers_on_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnexpectedIngestionService:
        def __init__(self, *, ingestion_graph: object) -> None:
            del ingestion_graph
            message = "invalid uploads must not reach the ingestion service"
            raise AssertionError(message)

    monkeypatch.setattr(ingestion_router_module, "IngestionService", _UnexpectedIngestionService)
    upload = UploadFile(
        file=BytesIO(b"not a supported document"),
        filename="contract.exe",
        headers=Headers({"content-type": "application/octet-stream"}),
    )
    response = Response()

    result = await ingestion_router_module.upload_document(
        upload,
        object(),
        "user-1",
        response,
    )

    assert response.status_code == 422
    assert response.headers["deprecation"] == "true"
    assert response.headers["link"].startswith("</documents/upload>")
    assert result.success is False


# --------------------------------------------------------------------------------------
# F2: the canonical command-service branch must survive FastAPI response serialization
# --------------------------------------------------------------------------------------


class _FakeAuthorization:
    """Minimal stand-in for the canonical tenant-aware authorization context."""

    user_id = "user-legacy"
    tenant_id = "personal:user-legacy"


class _FakeCommandService:
    """Queues like the canonical service and answers before extraction runs.

    The real `DocumentCommandService.upload_document` returns the canonical
    `dto.DocumentUploadResponse`, which carries none of the four legacy
    extraction counts — the exact shape that killed serialization (F2).
    """

    def __init__(self) -> None:
        self.persisted = False

    async def upload_document(self, **_kwargs: object) -> Success[CanonicalUploadResponse]:
        self.persisted = True
        return Success(
            CanonicalUploadResponse(doc_id="doc-legacy-1", status="queued", duplicate=False)
        )


@pytest.fixture
def legacy_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The legacy router mounted on a live FastAPI app with faked dependencies.

    The defect was a `ResponseValidationError` — FastAPI validating the endpoint's declared
    response model AFTER the handler ran and AFTER the document persisted. Direct function
    calls (the tests above) skip exactly that validation, so this test must drive the full
    stack. `app.state.redis` gets a fakeredis client so the upload rate limiter runs for real.
    """
    service = _FakeCommandService()
    app = FastAPI()
    app.state.redis = FakeRedis()
    app.include_router(ingestion_router_module.router)
    app.dependency_overrides[document_dependencies.get_document_command_service] = lambda: service
    app.dependency_overrides[document_dependencies.get_document_authorization] = _FakeAuthorization
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_canonical_upload_via_legacy_route_survives_response_serialization(
    legacy_client: TestClient,
) -> None:
    """F2 regression: 201 + deprecation headers, no ResponseValidationError after persist.

    The smoke run showed the document row created and the object written to R2, then a 500
    from `ResponseValidationError` because the declared response model demanded
    `entity_count/clause_count/relationship_count/dropped_entity_count` that the canonical
    command service never supplies. The counts are optional with deprecation defaults now, so
    serialization succeeds and the promised headers finally reach the client.
    """
    response = legacy_client.post(
        "/ingestion/documents/upload",
        files={"file": ("contract.txt", b"clause text", "text/plain")},
    )

    assert response.status_code == 201
    assert response.headers["deprecation"] == "true"
    assert response.headers["sunset"] == get_settings().API_SUNSET_DATE
    assert response.headers["link"].startswith("</documents/upload>")

    body = response.json()
    assert body["success"] is True
    assert body["data"]["doc_id"] == "doc-legacy-1"
    assert body["data"]["status"] == "queued"
    # Legacy counts are honestly absent on the queued-document path, not fabricated.
    assert body["data"]["entity_count"] is None
    assert body["data"]["clause_count"] is None
