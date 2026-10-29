"""V1 evidence-handle authorization and lifecycle tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from returns.result import Success

from app.features.documents.api.http.dto import EvidenceSearchRequest
from app.features.documents.application.evidence import service as evidence_module
from app.features.documents.application.evidence.service import (
    EvidenceHandle,
    EvidenceService,
    InMemoryEvidenceHandleStore,
)
from app.retrieval.fusion import RankedChunk

if TYPE_CHECKING:
    from typing import Any


DOC_A = "00000000-0000-0000-0000-000000000001"
DOC_B = "00000000-0000-0000-0000-000000000002"


class _Repo:
    def __init__(self) -> None:
        self.row: dict[str, object] = {
            "chunk_id": "00000000-0000-0000-0000-000000000011",
            "document_id": DOC_A,
            "title": "contract.pdf",
            "content": "A" * 1200,
            "preamble": "",
            "document_version": 1,
            "chunk_index": 0,
            "chunk_kind": "legal_contract",
            "clause_type": "termination",
            "page_no": 4,
            "chunk_metadata": {"pipeline_version": "p1"},
            "document_metadata": {},
            "quality_warnings": [],
            "graphiti_verified": True,
        }
        self.status: dict[str, object] = {
            "document_id": DOC_A,
            "status": "completed",
            "chunk_count": 1,
            "verified_chunk_count": 1,
            "document_version": 1,
            "document_metadata": {"pipeline_version": "p1"},
            "warnings": [],
            "extraction_incomplete": False,
        }

    async def fetch_chunks_by_ids(self, chunk_ids: list[str], **_: object) -> Any:
        if self.row["chunk_id"] not in chunk_ids:
            return Success({})
        return Success({str(self.row["chunk_id"]): dict(self.row)})

    async def fetch_status(self, **_: object) -> Any:
        return Success(dict(self.status))


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> tuple[EvidenceService, _Repo]:
    repo = _Repo()

    async def _embed(*_: object, **__: object) -> list[float]:
        return [0.0]

    async def _retrieve(**_: object) -> Any:
        return Success(
            (
                [RankedChunk(chunk_id=str(repo.row["chunk_id"]), score=0.9, rank=1)],
                {str(repo.row["chunk_id"]): dict(repo.row)},
            )
        )

    monkeypatch.setattr(evidence_module, "embed_text", _embed)
    monkeypatch.setattr(evidence_module, "retrieve_fused", _retrieve)
    return EvidenceService(repo=repo, handle_store=InMemoryEvidenceHandleStore()), repo


async def _search(service: EvidenceService) -> str:
    result = await service.search_evidence(
        principal_id="alice",
        tenant_id="tenant-a",
        allowed_doc_ids=(DOC_A,),
        thread_id="thread-a",
        run_id="run-a",
        request=EvidenceSearchRequest(query="termination"),
    )
    assert result.status == "ok"
    assert len(result.items) == 1
    assert len(result.items[0].snippet) <= 800
    return result.items[0].evidence.evidence_id


@pytest.mark.asyncio
async def test_valid_read_returns_exact_content_and_page(
    service: tuple[EvidenceService, _Repo],
) -> None:
    evidence, _ = service
    evidence_id = await _search(evidence)
    result = await evidence.read_evidence_by_id(
        evidence_id=evidence_id,
        principal_id="alice",
        tenant_id="tenant-a",
        allowed_doc_ids=(DOC_A,),
        thread_id="thread-a",
        run_id="run-a",
    )
    assert result.status == "ok"
    assert result.items[0].content == "A" * 1200
    assert result.items[0].evidence.page_no == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("principal_id", "tenant_id", "thread_id", "allowed_doc_ids"),
    [
        ("bob", "tenant-a", "thread-a", (DOC_A,)),
        ("alice", "tenant-b", "thread-a", (DOC_A,)),
        ("alice", "tenant-a", "thread-b", (DOC_A,)),
        ("alice", "tenant-a", "thread-a", (DOC_B,)),
    ],
)
async def test_handle_cannot_cross_scope(
    service: tuple[EvidenceService, _Repo],
    principal_id: str,
    tenant_id: str,
    thread_id: str,
    allowed_doc_ids: tuple[str, ...],
) -> None:
    evidence, _ = service
    evidence_id = await _search(evidence)
    result = await evidence.read_evidence_by_id(
        evidence_id=evidence_id,
        principal_id=principal_id,
        tenant_id=tenant_id,
        allowed_doc_ids=allowed_doc_ids,
        thread_id=thread_id,
        run_id="run-a",
    )
    assert result.status == "unauthorized"


@pytest.mark.asyncio
async def test_stale_version_is_rejected(service: tuple[EvidenceService, _Repo]) -> None:
    evidence, repo = service
    evidence_id = await _search(evidence)
    repo.row["document_version"] = 2
    result = await evidence.read_evidence_by_id(
        evidence_id=evidence_id,
        principal_id="alice",
        tenant_id="tenant-a",
        allowed_doc_ids=(DOC_A,),
        thread_id="thread-a",
        run_id="run-a",
    )
    assert result.status == "stale_version"


@pytest.mark.asyncio
async def test_expired_handle_is_rejected(service: tuple[EvidenceService, _Repo]) -> None:
    evidence, _ = service
    evidence_id = await _search(evidence)
    handle = await evidence.handle_store.resolve(evidence_id)
    assert isinstance(handle, EvidenceHandle)
    expired = handle.model_copy(update={"expires_at": datetime.now(tz=UTC) - timedelta(seconds=1)})
    await evidence.handle_store.issue(expired, ttl_seconds=1)
    result = await evidence.read_evidence_by_id(
        evidence_id=evidence_id,
        principal_id="alice",
        tenant_id="tenant-a",
        allowed_doc_ids=(DOC_A,),
        thread_id="thread-a",
        run_id="run-a",
    )
    assert result.status == "expired"


@pytest.mark.asyncio
async def test_forged_handle_is_rejected(service: tuple[EvidenceService, _Repo]) -> None:
    evidence, _ = service
    result = await evidence.read_evidence_by_id(
        evidence_id="evh_forged",
        principal_id="alice",
        tenant_id="tenant-a",
        allowed_doc_ids=(DOC_A,),
        thread_id="thread-a",
        run_id="run-a",
    )
    assert result.status == "invalid_handle"


@pytest.mark.asyncio
async def test_ingestion_state_reports_readiness(service: tuple[EvidenceService, _Repo]) -> None:
    evidence, repo = service
    result = await evidence.get_ingestion_state(
        principal_id="alice",
        tenant_id="tenant-a",
        allowed_doc_ids=(DOC_A,),
    )
    assert result.status == "ok"
    assert result.documents[0].state == "ready"
    repo.status["status"] = "processing"
    result = await evidence.get_ingestion_state(
        principal_id="alice",
        tenant_id="tenant-a",
        allowed_doc_ids=(DOC_A,),
    )
    assert result.status == "index_not_ready"


@pytest.mark.asyncio
async def test_search_distinguishes_not_ready_from_no_evidence(
    service: tuple[EvidenceService, _Repo],
) -> None:
    evidence, repo = service
    repo.status["status"] = "processing"

    result = await evidence.search_evidence(
        principal_id="alice",
        tenant_id="tenant-a",
        allowed_doc_ids=(DOC_A,),
        thread_id="thread-a",
        run_id="run-a",
        request=EvidenceSearchRequest(query="termination"),
    )

    assert result.status == "index_not_ready"
    assert result.coverage.documents_not_ready == 1
