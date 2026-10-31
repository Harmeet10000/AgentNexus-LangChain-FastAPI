"""Ingest orchestration: sha256 dedupe, typed-job enqueue, fail-closed seams.

The service must never read ``tenant_id`` from the request body: it takes the
authorization-derived tenant and writes it into both the document row and the
queued payload. Every stub here records the tenant it was called with so the
scoping claims are assertions, not narrative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fakeredis.aioredis import FakeRedis
from returns.result import Failure, Success

from app.features.search.adapters.persistence.models import SearchDocument
from app.features.search.api.http.dto import SearchIngestRequest
from app.features.search.application.service import SearchService
from app.features.search.domain.errors import (
    SearchConflictError,
    SearchInfrastructureError,
    SearchNotFoundError,
)
from app.shared.jobs import (
    InMemoryDirectAdapter,
    InMemoryTransactionalAdapter,
    JobBus,
)
from app.shared.jobs.names import SEARCH_INGEST

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from app.features.search.api.http.dto import HybridSearchRequest

TENANT = "tenant-1"
OTHER_TENANT = "tenant-2"


class _StubRepository:
    """Duck-typed SearchRepository double recording tenant scoping per call."""

    def __init__(
        self,
        *,
        existing_document: SearchDocument | None = None,
        create_failure: SearchConflictError | None = None,
        lookup_failure: SearchInfrastructureError | None = None,
    ):
        self.existing_document = existing_document
        self.create_failure = create_failure
        self.lookup_failure = lookup_failure
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((method, kwargs))

    async def get_document_by_content_hash(self, *, tenant_id: str, content_hash: str) -> Any:
        self._record("get_document_by_content_hash", {"tenant_id": tenant_id})
        if self.lookup_failure is not None:
            return Failure(self.lookup_failure)
        return Success(self.existing_document)

    async def create_document(self, **kwargs: Any) -> Any:
        self._record("create_document", dict(kwargs))
        if self.create_failure is not None:
            return Failure(self.create_failure)
        document = SearchDocument(
            id=_uuid("00000000-0000-0000-0000-0000000000aa"),
            tenant_id=kwargs["tenant_id"],
            title=kwargs["title"],
            source_uri=kwargs["source_uri"],
            content_hash=kwargs["content_hash"],
            doc_metadata=kwargs["doc_metadata"],
        )
        return Success(document)

    async def get_document_by_id(self, *, tenant_id: str, document_id: str) -> Any:
        self._record("get_document_by_id", {"tenant_id": tenant_id})
        if self.lookup_failure is not None:
            return Failure(self.lookup_failure)
        return Success(None)

    async def bm25_search(self, **kwargs: Any) -> Any:
        self._record("bm25_search", dict(kwargs))
        return Success(_rows(["chunk-bm25"]))

    async def vector_search(self, **kwargs: Any) -> Any:
        self._record("vector_search", dict(kwargs))
        return Success(_rows(["chunk-vector"]))

    async def trigram_search(self, **kwargs: Any) -> Any:
        self._record("trigram_search", dict(kwargs))
        return Success(_rows([]))

    async def fetch_chunks_by_ids(self, *, tenant_id: str, chunk_ids: Sequence[str]) -> Any:
        self._record("fetch_chunks_by_ids", {"tenant_id": tenant_id})
        return Success(
            {
                chunk_id: {
                    "chunk_id": chunk_id,
                    "document_id": "00000000-0000-0000-0000-0000000000bb",
                    "chunk_index": 0,
                    "content": f"body of {chunk_id}",
                    "chunk_metadata": {},
                    "title": "hydrated title",
                }
                for chunk_id in chunk_ids
            }
        )

    async def upsert_chunks(self, rows: list[dict[str, Any]]) -> Any:
        self._record("upsert_chunks", {"rows": rows})
        return Success(None)

    async def analyze_chunks(self) -> Any:
        self._record("analyze_chunks", {})
        return Success(None)

    async def set_document_status(self, *, document_id: str, status: str) -> Any:
        self._record("set_document_status", {"document_id": document_id, "status": status})
        return Success(None)


def _uuid(seed: str) -> Any:
    from uuid import UUID

    return UUID(seed)


def _rows(chunk_ids: list[str]) -> list[Any]:
    from app.retrieval.fusion import RankedResultRow

    return [
        RankedResultRow(chunk_id=chunk_id, score=1.0, rank=rank)
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


def _service(
    repo: _StubRepository,
    *,
    job_bus: JobBus | None = None,
    redis: Any = None,
) -> SearchService:
    return SearchService(
        repo=repo,  # type: ignore[arg-type]
        redis=redis,
        job_bus=job_bus,
        embed_query=_stub_embed_query,
    )


async def _stub_embed_query(text: str, *, task_type: Any) -> list[float]:
    return [0.1, 0.2, 0.3]


def _ingest_payload(**overrides: str) -> SearchIngestRequest:
    values: dict[str, Any] = {
        "title": "Site page",
        "content": "arbitrary site content",
    }
    values.update(overrides)
    return SearchIngestRequest(**values)


@pytest.mark.asyncio
async def test_duplicate_content_short_circuits_without_enqueue() -> None:
    existing = SearchDocument(id=_uuid("00000000-0000-0000-0000-0000000000aa"))
    repo = _StubRepository(existing_document=existing)
    direct = InMemoryDirectAdapter()
    service = _service(repo, job_bus=JobBus(direct, InMemoryTransactionalAdapter()))

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Success)
    response = result.unwrap()
    assert response.duplicate is True
    assert response.task_id is None
    assert response.status == "completed"
    assert direct.published == []
    # Dedupe lookup itself must be tenant-scoped.
    assert repo.calls[0] == (
        "get_document_by_content_hash",
        {"tenant_id": TENANT},
    )


@pytest.mark.asyncio
async def test_fresh_ingest_enqueues_typed_job_scoped_to_the_caller_tenant() -> None:
    repo = _StubRepository()
    direct = InMemoryDirectAdapter()
    service = _service(repo, job_bus=JobBus(direct, InMemoryTransactionalAdapter()))

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Success)
    response = result.unwrap()
    assert response.duplicate is False
    assert response.status == "queued"
    assert response.task_id == "broker-1"

    assert len(direct.published) == 1
    job = direct.published[0]
    assert job.task_name == SEARCH_INGEST
    payload = job.wire_payload()
    assert payload["tenant_id"] == TENANT
    assert payload["document_id"] == response.document_id
    assert payload["title"] == "Site page"
    # The document row also carries the caller's tenant.
    create_call = next(call for name, call in repo.calls if name == "create_document")
    assert create_call["tenant_id"] == TENANT


@pytest.mark.asyncio
async def test_dedupe_is_per_tenant_for_identical_content() -> None:
    """The same content ingested by a second tenant must not dedupe to the first."""
    repo = _StubRepository()
    direct = InMemoryDirectAdapter()
    service = _service(repo, job_bus=JobBus(direct, InMemoryTransactionalAdapter()))

    first = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())
    second = await service.ingest_document(tenant_id=OTHER_TENANT, payload=_ingest_payload())

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert first.unwrap().duplicate is False
    assert second.unwrap().duplicate is False
    second_lookup = next(
        call
        for name, call in repo.calls
        if name == "get_document_by_content_hash" and call["tenant_id"] == OTHER_TENANT
    )
    assert second_lookup["tenant_id"] == OTHER_TENANT


@pytest.mark.asyncio
async def test_missing_job_bus_fails_closed() -> None:
    """A document row without a queued pipeline is a pending orphan — an outage."""
    repo = _StubRepository()
    service = _service(repo, job_bus=None)

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, SearchInfrastructureError)


@pytest.mark.asyncio
async def test_enqueue_failure_is_a_typed_infrastructure_failure() -> None:
    repo = _StubRepository()
    direct = InMemoryDirectAdapter(failure=RuntimeError("broker down"))
    service = _service(repo, job_bus=JobBus(direct, InMemoryTransactionalAdapter()))

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, SearchInfrastructureError)
    assert error.details is not None
    assert "document_id" in error.details


@pytest.mark.asyncio
async def test_database_failure_propagates_as_failure() -> None:
    repo = _StubRepository(
        create_failure=SearchConflictError(message="constraint violation", source="test")
    )
    service = _service(
        repo, job_bus=JobBus(InMemoryDirectAdapter(), InMemoryTransactionalAdapter())
    )

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SearchConflictError)


# --- get_ingest_status: fail closed across tenants -------------------------


class _FakeAsyncResult:
    def __init__(self, *, state: str, result: object):
        self.state = state
        self.result = result


@pytest.mark.asyncio
async def test_ingest_status_of_another_tenant_reads_as_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task created by tenant-1 is invisible to tenant-2 — same as missing."""
    from app.connections import celery_app

    repo = _StubRepository()
    monkeypatch.setattr(
        celery_app,
        "AsyncResult",
        lambda _task_id: _FakeAsyncResult(
            state="SUCCESS",
            result={"status": "completed", "tenant_id": TENANT, "document_id": "doc-1"},
        ),
    )
    service = _service(repo)

    result = await service.get_ingest_status(tenant_id=OTHER_TENANT, task_id="task-1")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SearchNotFoundError)


@pytest.mark.asyncio
async def test_own_tenant_sees_its_ingest_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.connections import celery_app

    repo = _StubRepository()
    monkeypatch.setattr(
        celery_app,
        "AsyncResult",
        lambda _task_id: _FakeAsyncResult(
            state="SUCCESS",
            result={"status": "completed", "tenant_id": TENANT, "document_id": "doc-1"},
        ),
    )
    service = _service(repo)

    result = await service.get_ingest_status(tenant_id=TENANT, task_id="task-1")

    assert isinstance(result, Success)
    status = result.unwrap()
    assert status.task_id == "task-1"
    assert status.status == "SUCCESS"


@pytest.mark.asyncio
async def test_failure_state_surfaces_the_error_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.connections import celery_app

    repo = _StubRepository()
    monkeypatch.setattr(
        celery_app,
        "AsyncResult",
        lambda _task_id: _FakeAsyncResult(
            state="FAILURE",
            result={"tenant_id": TENANT, "detail": "boom"},
        ),
    )
    service = _service(repo)

    result = await service.get_ingest_status(tenant_id=TENANT, task_id="task-1")

    assert isinstance(result, Success)
    status = result.unwrap()
    assert status.error is not None
    assert "boom" in status.error


# --- hybrid search: fusion, cache, tenant scoping --------------------------


@pytest.mark.asyncio
async def test_hybrid_search_fuses_and_hydrates() -> None:
    repo = _StubRepository()
    repo.fetch_results = {}  # type: ignore[attr-defined]
    direct = InMemoryDirectAdapter()
    service = _service(
        repo,
        job_bus=JobBus(direct, InMemoryTransactionalAdapter()),
        redis=FakeRedis(decode_responses=True),
    )

    payload: HybridSearchRequest = _hybrid_payload()
    result = await service.hybrid_search(tenant_id=TENANT, payload=payload)

    assert isinstance(result, Success)
    response = result.unwrap()
    assert response.cache_hit is False
    # chunk-bm25 (rank 1 in bm25 only) vs chunk-vector (rank 1 in vector only):
    # symmetric, tie-broken by id — chunk-bm25 < chunk-vector.
    assert [item.chunk_id for item in response.items] == ["chunk-bm25", "chunk-vector"]
    assert response.items[0].rank == 1
    # Every branch and the hydration fetch were tenant-scoped.
    for name in ("bm25_search", "vector_search", "trigram_search", "fetch_chunks_by_ids"):
        call = next(call for call_name, call in repo.calls if call_name == name)
        assert call["tenant_id"] == TENANT


def _hybrid_payload(**overrides: Any) -> Any:
    from app.features.search.api.http.dto import HybridSearchRequest

    values: dict[str, Any] = {"query": "site query", "limit": 10, "candidate_limit": 50}
    values.update(overrides)
    return HybridSearchRequest(**values)


@pytest.mark.asyncio
async def test_hybrid_search_caches_per_tenant() -> None:
    repo = _StubRepository()
    redis = FakeRedis(decode_responses=True)
    service = _service(repo, job_bus=None, redis=redis)

    first = await service.hybrid_search(tenant_id=TENANT, payload=_hybrid_payload())
    second = await service.hybrid_search(tenant_id=TENANT, payload=_hybrid_payload())
    other = await service.hybrid_search(tenant_id=OTHER_TENANT, payload=_hybrid_payload())

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert isinstance(other, Success)
    assert first.unwrap().cache_hit is False
    assert second.unwrap().cache_hit is True
    # The identical query from another tenant must not read tenant-1's cache.
    assert other.unwrap().cache_hit is False


@pytest.mark.asyncio
async def test_bypass_cache_skips_reads_and_writes() -> None:
    repo = _StubRepository()
    redis = FakeRedis(decode_responses=True)
    service = _service(repo, job_bus=None, redis=redis)

    first = await service.hybrid_search(
        tenant_id=TENANT, payload=_hybrid_payload(bypass_cache=True)
    )
    second = await service.hybrid_search(
        tenant_id=TENANT, payload=_hybrid_payload(bypass_cache=True)
    )

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert first.unwrap().cache_hit is False
    assert second.unwrap().cache_hit is False


@pytest.mark.asyncio
async def test_a_broken_branch_fails_the_request_and_names_it() -> None:
    class _BrokenBranchRepository(_StubRepository):
        async def vector_search(self, **kwargs: Any) -> Any:
            return Failure(SearchInfrastructureError(message="diskann exploded", source="test"))

    repo = _BrokenBranchRepository()
    service = _service(repo, job_bus=None, redis=FakeRedis(decode_responses=True))

    result = await service.hybrid_search(tenant_id=TENANT, payload=_hybrid_payload())

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, SearchInfrastructureError)
    assert error.message.startswith("vector retrieval branch failed")
