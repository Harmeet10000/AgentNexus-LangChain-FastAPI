"""F10 regression: version identity is non-null from receive time onward.

The 2026-11-15 post-fix smoke found ``GET /documents/{id}/status`` reporting
``document_version: null, pipeline_version: null`` at ``received`` — the status
handler never surfaced the version fields, and the receive write persisted no
version identity for a chunk-less document. Now:

- the receive write persists the same revision the queued ingestion job
  carries (``document_version`` 1, ``CANONICAL_PIPELINE_VERSION``) into the
  document metadata;
- the status rollup coalesces the chunk provenance with that receive-time
  metadata, so ``received`` documents report non-null versions while ingested
  documents still report the chunk-level truth.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from returns.result import Success
from sqlalchemy.dialects import postgresql

from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)
from app.features.documents.application.commands.upload_document import (
    DocumentCommandService,
    DocumentStatusResponse,
)
from app.features.documents.domain.constants import CANONICAL_PIPELINE_VERSION

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

pytestmark = pytest.mark.unit


class _StatementCapturedError(AssertionError):
    """Sentinel stopping fetch_status right after its statement is compiled."""


class _FakeStatusRepository:
    """fetch_status stand-in returning the receive-time rollup record."""

    def __init__(self, record: Mapping[str, Any]) -> None:
        self._record = dict(record)

    async def fetch_status(
        self, *, user_id: str, document_id: str, tenant_id: str
    ) -> Success[dict[str, Any] | None]:
        del user_id, document_id, tenant_id
        return Success(self._record)


def _received_record() -> dict[str, Any]:
    """What the coalesced rollup yields for a just-received document."""
    return {
        "document_id": "9c2f3c2e-0000-4000-8000-000000000001",
        "status": "received",
        "object_uri": "s3://bucket/doc.txt",
        "title": "contract.txt",
        "document_kind": "generic",
        "chunk_count": 0,
        "verified_chunk_count": 0,
        "warnings": [],
        "document_metadata": {
            "content_type": "text/plain",
            "filename": "contract.txt",
            "document_version": 1,
            "pipeline_version": CANONICAL_PIPELINE_VERSION,
        },
        # The coalesce fallback (metadata) — no chunks exist yet.
        "document_version": 1,
        "pipeline_version": CANONICAL_PIPELINE_VERSION,
        "extraction_incomplete": False,
    }


@pytest.mark.asyncio
async def test_received_document_status_reports_non_null_versions() -> None:
    service = DocumentCommandService(
        repo=_FakeStatusRepository(_received_record()),  # type: ignore[arg-type]
        object_store=None,
    )

    result = await service._get_status(
        user_id="user-1", document_id="9c2f3c2e-0000-4000-8000-000000000001", tenant_id="tenant-1"
    )

    assert isinstance(result, Success)
    response = result.unwrap()
    assert isinstance(response, DocumentStatusResponse)
    assert response.document_version == 1
    assert response.pipeline_version == CANONICAL_PIPELINE_VERSION


def test_fetch_status_coalesces_chunk_version_with_receive_metadata() -> None:
    """The status rollup falls back to the receive write when no chunks exist."""
    repository = DocumentRepository.__new__(DocumentRepository)
    # fetch_status builds the statement first; capture it without a DB session
    # by stubbing the only session use — execute — right after compilation.
    captured: list[object] = []

    class _CaptureSession:
        async def execute(self, statement: object) -> None:
            captured.append(statement)
            message = "statement captured; no execution expected"
            raise _StatementCapturedError(message)

    repository.session = _CaptureSession()  # type: ignore[assignment]

    with pytest.raises(_StatementCapturedError) as excinfo:
        asyncio.run(
            repository.fetch_status(
                user_id="user-1",
                document_id="9c2f3c2e-0000-4000-8000-000000000001",
                tenant_id="tenant-1",
            )
        )
    assert "statement captured" in str(excinfo.value)
    assert captured, "fetch_status must build its statement before touching the session"

    compiled = captured[0].compile(dialect=postgresql.dialect())  # type: ignore[attr-defined]
    sql = str(compiled)
    assert "document_version" in sql
    assert "pipeline_version" in sql
    # The receive-time fallback reads the persisted metadata keys (they are
    # bound parameters in the rendered SQL).
    params = compiled.params
    assert "document_version" in params.values()
    assert "pipeline_version" in params.values()
