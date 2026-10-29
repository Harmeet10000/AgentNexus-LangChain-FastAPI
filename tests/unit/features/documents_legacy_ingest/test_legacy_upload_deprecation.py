"""Contract tests for the retained, deprecated ingestion upload route."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import Response
from fastapi.routing import APIRoute
from returns.result import Success
from starlette.datastructures import Headers, UploadFile

from app.config import get_settings
from app.features.documents.api.http import legacy_ingest_router as ingestion_router_module
from app.features.documents.ingestion.legacy_dto import DocumentUploadResponse

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
