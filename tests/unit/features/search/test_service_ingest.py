"""Ingest orchestration: sha256 dedupe, atomic staging, ownership, fail-closed seams.

The service must never read ``tenant_id`` from the request body: it takes the
authorization-derived tenant and writes it into the document row, the staged
job, and the task-ownership record. Every stub here records the tenant it was
called with so the scoping claims are assertions, not narrative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import orjson
import pytest
from fakeredis.aioredis import FakeRedis
from returns.result import Failure, Success

from app.features.search.adapters.persistence.models import SearchDocument
from app.features.search.api.http.dto import SearchIngestRequest
from app.features.search.application import service as search_service_module
from app.features.search.application.service import SearchService
from app.features.search.domain.errors import (
    SearchConflictError,
    SearchInfrastructureError,
    SearchNotFoundError,
)
from app.shared.jobs import JobBus
from app.shared.jobs.names import SEARCH_INGEST

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from app.features.search.api.http.dto import HybridSearchRequest
    from app.shared.jobs import AggregateRef
    from app.shared.jobs.bus import TypedJob
    from app.shared.jobs.payloads import CeleryTaskPayload

TENANT = "tenant-1"
OTHER_TENANT = "tenant-2"
DOC_ID = "00000000-0000-0000-0000-0000000000aa"


class _FakeSession:
    """Records executed statements (the outbox payload stamp) for assertions."""

    def __init__(self) -> None:
        self.executed: list[tuple[object, dict[str, object] | None]] = []

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        params = kwargs.get("params")
        if params is None and len(args) > 1:
            params = args[1]
        self.executed.append((args[0] if args else None, params))
        return None


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
        self.rollbacks = 0
        # The service binds transactional staging to the repo's session; the
        # staging guard checks for an async `execute` capability.
        self.session = _FakeSession()

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((method, kwargs))

    async def rollback(self) -> None:
        self.rollbacks += 1

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
            id=_uuid(DOC_ID),
            tenant_id=kwargs["tenant_id"],
            title=kwargs["title"],
            source_uri=kwargs["source_uri"],
            content_hash=kwargs["content_hash"],
            doc_metadata=kwargs["doc_metadata"],
        )
        return Success(document)

    async def get_document_by_id(self, *, tenant_id: str, document_id: str) -> Any:
        self._record("get_document_by_id", {"tenant_id": tenant_id, "document_id": document_id})
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


class _RecordingStagingAdapter:
    """Transactional staging double that accepts any transaction handle.

    The production adapter (``PostgresOutboxAdapter``) stages onto the
    caller-owned session; this double records the staged jobs so tests can
    assert on the wire payload without a database.
    """

    def __init__(self, *, failure: Exception | None = None):
        self.failure = failure
        self.staged: list[tuple[TypedJob[CeleryTaskPayload], AggregateRef]] = []
        self._next_id = 0

    async def stage(
        self,
        _tx: object,
        job: TypedJob[CeleryTaskPayload],
        *,
        aggregate: AggregateRef,
    ) -> str:
        if self.failure is not None:
            raise self.failure
        self.staged.append((job, aggregate))
        self._next_id += 1
        return f"event-{self._next_id}"


class _SetFailingRedis:
    """Redis double whose ``set`` fails (ownership write cannot be recorded)."""

    def __init__(self, inner: Any):
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def set(self, *_args: Any, **_kwargs: Any) -> Any:
        message = "cache down"
        raise RuntimeError(message)


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


def _job_bus(staging: _RecordingStagingAdapter) -> JobBus:
    from app.shared.jobs import InMemoryDirectAdapter

    return JobBus(InMemoryDirectAdapter(), staging)  # type: ignore[arg-type]


async def _stub_embed_query(text: str, *, task_type: Any) -> list[float]:
    return [0.1, 0.2, 0.3]


def _ingest_payload(**overrides: str) -> SearchIngestRequest:
    values: dict[str, Any] = {
        "title": "Site page",
        "content": "arbitrary site content",
    }
    values.update(overrides)
    return SearchIngestRequest(**values)


def _owner_key(task_id: str) -> str:
    return f"search:task_owner:{task_id}"


def _celery_id_key(task_id: str) -> str:
    return f"search:task_celery_id:{task_id}"


# --- ingest: dedupe, staging, ownership -------------------------------------


@pytest.mark.asyncio
async def test_completed_document_short_circuits_without_staging() -> None:
    existing = SearchDocument(id=_uuid(DOC_ID), status="completed")
    repo = _StubRepository(existing_document=existing)
    staging = _RecordingStagingAdapter()
    service = _service(repo, job_bus=_job_bus(staging))

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Success)
    response = result.unwrap()
    assert response.duplicate is True
    assert response.task_id is None
    assert response.status == "completed"
    assert staging.staged == []
    # Dedupe lookup itself must be tenant-scoped.
    assert repo.calls[0] == (
        "get_document_by_content_hash",
        {"tenant_id": TENANT},
    )


@pytest.mark.asyncio
async def test_fresh_ingest_stages_typed_job_atomically_with_the_document() -> None:
    repo = _StubRepository()
    staging = _RecordingStagingAdapter()
    redis = FakeRedis(decode_responses=True)
    service = _service(repo, job_bus=_job_bus(staging), redis=redis)

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Success)
    response = result.unwrap()
    assert response.duplicate is False
    assert response.status == "queued"
    assert response.task_id == "event-1"

    # The job went through the transactional staging seam (outbox), bound to
    # the request session — never a direct broker publish.
    assert len(staging.staged) == 1
    job, aggregate = staging.staged[0]
    assert job.task_name == SEARCH_INGEST
    assert aggregate.aggregate_type == "search_document"
    assert aggregate.aggregate_id == response.document_id
    payload = job.wire_payload()
    assert payload["tenant_id"] == TENANT
    assert payload["document_id"] == response.document_id
    assert payload["title"] == "Site page"
    # The document row also carries the caller's tenant.
    create_call = next(call for name, call in repo.calls if name == "create_document")
    assert create_call["tenant_id"] == TENANT
    # The ownership record for GET /ingest/{task_id} was written.
    owner_raw = await redis.get(_owner_key("event-1"))
    assert owner_raw is not None
    owner = orjson.loads(owner_raw)
    assert owner == {"tenant_id": TENANT, "document_id": response.document_id}
    # The staged payload was stamped with its own event id, so the worker can
    # bind its real Celery task id back to the id the client polls.
    stamp_params = next(
        params for _statement, params in repo.session.executed if params and "event_id" in params
    )
    assert stamp_params is not None
    assert stamp_params["event_id"] == "event-1"


@pytest.mark.asyncio
async def test_pending_row_is_re_enqueued_not_reported_done() -> None:
    """A stored row whose pipeline never completed must be re-driven."""
    existing = SearchDocument(id=_uuid(DOC_ID), status="pending")
    repo = _StubRepository(existing_document=existing)
    staging = _RecordingStagingAdapter()
    service = _service(repo, job_bus=_job_bus(staging), redis=FakeRedis(decode_responses=True))

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Success)
    response = result.unwrap()
    assert response.duplicate is False
    assert response.status == "queued"
    assert response.document_id == DOC_ID
    # No second row: the staged job reuses the existing document identity.
    assert not any(name == "create_document" for name, _ in repo.calls)
    job, aggregate = staging.staged[0]
    assert aggregate.aggregate_id == DOC_ID
    assert job.wire_payload()["document_id"] == DOC_ID


@pytest.mark.asyncio
async def test_dedupe_is_per_tenant_for_identical_content() -> None:
    """The same content ingested by a second tenant must not dedupe to the first."""
    repo = _StubRepository()
    staging = _RecordingStagingAdapter()
    service = _service(repo, job_bus=_job_bus(staging), redis=FakeRedis(decode_responses=True))

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
async def test_missing_job_bus_fails_closed_before_any_write() -> None:
    """A document row without a queueable pipeline is a pending orphan — an outage."""
    repo = _StubRepository()
    service = _service(repo, job_bus=None)

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SearchInfrastructureError)
    # Failed before the document row was created — nothing to strand.
    assert not any(name == "create_document" for name, _ in repo.calls)


@pytest.mark.asyncio
async def test_staging_failure_rolls_back_and_fails_typed() -> None:
    """A staged-but-failed ingest must unwind the flushed document row."""
    repo = _StubRepository()
    staging = _RecordingStagingAdapter(failure=RuntimeError("broker down"))
    service = _service(repo, job_bus=_job_bus(staging), redis=FakeRedis(decode_responses=True))

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, SearchInfrastructureError)
    assert error.details is not None
    assert "document_id" in error.details
    # Without this rollback, get_postgres_db would commit the pending row and
    # every later submission would report it as an in-flight duplicate.
    assert repo.rollbacks == 1


@pytest.mark.asyncio
async def test_failed_ownership_record_rolls_back_and_fails_closed() -> None:
    """Without the ownership record the tenant could never poll — fail closed."""
    repo = _StubRepository()
    staging = _RecordingStagingAdapter()
    service = _service(repo, job_bus=_job_bus(staging), redis=_SetFailingRedis(FakeRedis()))

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SearchInfrastructureError)
    assert repo.rollbacks == 1


@pytest.mark.asyncio
async def test_database_failure_propagates_as_failure() -> None:
    repo = _StubRepository(
        create_failure=SearchConflictError(message="constraint violation", source="test")
    )
    service = _service(
        repo, job_bus=_job_bus(_RecordingStagingAdapter()), redis=FakeRedis(decode_responses=True)
    )

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SearchConflictError)


# --- get_ingest_status: ownership, states, fail closed ----------------------


class _FakeAsyncResult:
    def __init__(self, *, state: str, result: object):
        self.state = state
        self.result = result


def _seed_owner(monkeypatch: pytest.MonkeyPatch, state: str, result: object) -> None:
    from app.connections import celery_app

    monkeypatch.setattr(
        celery_app,
        "AsyncResult",
        lambda _task_id: _FakeAsyncResult(state=state, result=result),
    )


@pytest.mark.asyncio
async def test_queued_task_is_authorized_by_the_ownership_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task no worker has run yet is visible to its owner — via the record."""
    _seed_owner(monkeypatch, state="PENDING", result=None)
    repo = _StubRepository()
    redis = FakeRedis(decode_responses=True)
    await redis.set(
        _owner_key("task-1"),
        orjson.dumps({"tenant_id": TENANT, "document_id": DOC_ID}),
    )
    service = _service(repo, redis=redis)

    result = await service.get_ingest_status(tenant_id=TENANT, task_id="task-1")

    assert isinstance(result, Success)
    status = result.unwrap()
    # Celery's PENDING means "never seen"; the ownership record proves the
    # task is staged, so the reportable state is QUEUED.
    assert status.status == "QUEUED"
    assert status.document_id == DOC_ID


@pytest.mark.asyncio
async def test_queued_task_of_another_tenant_reads_as_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_owner(monkeypatch, state="PENDING", result=None)
    repo = _StubRepository()
    redis = FakeRedis(decode_responses=True)
    await redis.set(
        _owner_key("task-1"),
        orjson.dumps({"tenant_id": TENANT, "document_id": DOC_ID}),
    )
    service = _service(repo, redis=redis)

    result = await service.get_ingest_status(tenant_id=OTHER_TENANT, task_id="task-1")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SearchNotFoundError)


@pytest.mark.asyncio
async def test_task_without_ownership_evidence_reads_as_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PENDING with no owner record and no payload tenant is someone else's task."""
    _seed_owner(monkeypatch, state="PENDING", result=None)
    repo = _StubRepository()
    service = _service(repo, redis=FakeRedis(decode_responses=True))

    result = await service.get_ingest_status(tenant_id=TENANT, task_id="task-1")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SearchNotFoundError)


@pytest.mark.asyncio
async def test_completed_task_is_authorized_by_the_result_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a worker runs, the result payload carries the tenant as a fallback."""
    _seed_owner(
        monkeypatch,
        state="SUCCESS",
        result={"status": "completed", "tenant_id": TENANT, "document_id": DOC_ID},
    )
    repo = _StubRepository()
    service = _service(repo, redis=FakeRedis(decode_responses=True))

    result = await service.get_ingest_status(tenant_id=TENANT, task_id="task-1")

    assert isinstance(result, Success)
    status = result.unwrap()
    assert status.status == "SUCCESS"
    assert status.document_id == DOC_ID


@pytest.mark.asyncio
async def test_foreign_payload_tenant_reads_as_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_owner(
        monkeypatch,
        state="SUCCESS",
        result={"status": "completed", "tenant_id": TENANT, "document_id": DOC_ID},
    )
    repo = _StubRepository()
    service = _service(repo, redis=FakeRedis(decode_responses=True))

    result = await service.get_ingest_status(tenant_id=OTHER_TENANT, task_id="task-1")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), SearchNotFoundError)


@pytest.mark.asyncio
async def test_failure_state_surfaces_the_exception_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The FAILURE result is an exception instance, and its message is the error."""
    _seed_owner(monkeypatch, state="FAILURE", result=RuntimeError("embedding quota exceeded"))
    repo = _StubRepository()
    redis = FakeRedis(decode_responses=True)
    await redis.set(
        _owner_key("task-1"),
        orjson.dumps({"tenant_id": TENANT, "document_id": DOC_ID}),
    )
    service = _service(repo, redis=redis)

    result = await service.get_ingest_status(tenant_id=TENANT, task_id="task-1")

    assert isinstance(result, Success)
    status = result.unwrap()
    assert status.status == "FAILURE"
    assert status.error is not None
    assert "embedding quota exceeded" in status.error


@pytest.mark.asyncio
async def test_worker_recorded_celery_id_is_polled_instead_of_the_event_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the worker records the mapping, status reflects the real state."""
    polled: list[str] = []

    class _RecordingAsyncResult:
        def __init__(self, task_id: str):
            polled.append(task_id)
            self.state = "SUCCESS"
            self.result = {"status": "completed", "tenant_id": TENANT, "document_id": DOC_ID}

    from app.connections import celery_app

    monkeypatch.setattr(celery_app, "AsyncResult", _RecordingAsyncResult)
    repo = _StubRepository()
    redis = FakeRedis(decode_responses=True)
    await redis.set(
        _owner_key("event-1"),
        orjson.dumps({"tenant_id": TENANT, "document_id": DOC_ID}),
    )
    await redis.set(_celery_id_key("event-1"), "celery-abc")
    service = _service(repo, redis=redis)

    result = await service.get_ingest_status(tenant_id=TENANT, task_id="event-1")

    assert isinstance(result, Success)
    assert status_task_id_polled(polled) == "celery-abc"
    assert result.unwrap().status == "SUCCESS"


def status_task_id_polled(polled: list[str]) -> str:
    assert len(polled) == 1
    return polled[0]


@pytest.mark.asyncio
async def test_durable_document_status_beats_a_pending_broker_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed document column is the terminal truth the DTO prefers."""
    _seed_owner(monkeypatch, state="PENDING", result=None)
    repo = _StubRepository()
    redis = FakeRedis(decode_responses=True)
    await redis.set(
        _owner_key("task-1"),
        orjson.dumps({"tenant_id": TENANT, "document_id": DOC_ID}),
    )

    # The tenant-scoped document row resolves and reports completion.
    async def _completed_document(*, tenant_id: str, document_id: str) -> Any:
        repo._record("get_document_by_id", {"tenant_id": tenant_id, "document_id": document_id})
        return Success(SearchDocument(id=_uuid(DOC_ID), status="completed"))

    repo.get_document_by_id = _completed_document  # type: ignore[method-assign]
    service = _service(repo, redis=redis)

    result = await service.get_ingest_status(tenant_id=TENANT, task_id="task-1")

    assert isinstance(result, Success)
    assert result.unwrap().status == "SUCCESS"


# --- hybrid search: fusion, cache, width guard, tenant scoping --------------


@pytest.mark.asyncio
async def test_hybrid_search_fuses_and_hydrates() -> None:
    repo = _StubRepository()
    service = _service(repo, job_bus=None, redis=FakeRedis(decode_responses=True))

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
async def test_branches_run_sequentially_on_the_shared_session() -> None:
    """The three branches must not interleave on one request-scoped AsyncSession."""
    repo = _StubRepository()
    service = _service(repo, job_bus=None, redis=FakeRedis(decode_responses=True))

    result = await service.hybrid_search(tenant_id=TENANT, payload=_hybrid_payload())

    assert isinstance(result, Success)
    branch_order = [
        name for name, _ in repo.calls if name in {"bm25_search", "vector_search", "trigram_search"}
    ]
    assert branch_order == ["bm25_search", "vector_search", "trigram_search"]


@pytest.mark.asyncio
async def test_width_mismatch_fails_closed_before_querying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A column/setting width disagreement ranks nothing — refuse before embedding."""
    repo = _StubRepository()
    service = _service(repo, job_bus=None, redis=FakeRedis(decode_responses=True))
    monkeypatch.setattr(search_service_module, "SEARCH_EMBEDDING_DIM", 123)

    result = await service.hybrid_search(tenant_id=TENANT, payload=_hybrid_payload())

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, SearchInfrastructureError)
    assert "search_chunks.embedding" in error.message
    assert repo.calls == []


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


@pytest.mark.asyncio
async def test_ingest_stages_through_the_transactional_seam_not_direct_publish() -> None:
    """Direct publication would break document+job atomicity; it must never run."""
    repo = _StubRepository()
    bus = JobBus(_ExplodingDirectAdapter(), _RecordingStagingAdapter())  # type: ignore[arg-type]
    service = _service(repo, job_bus=bus, redis=FakeRedis(decode_responses=True))

    result = await service.ingest_document(tenant_id=TENANT, payload=_ingest_payload())

    # The transactional adapter was used (its staging recorded, receipt is the
    # outbox event id), so the direct path was never touched.
    assert isinstance(result, Success)


class _ExplodingDirectAdapter:
    async def publish(self, _job: Any, **_kwargs: Any) -> str:
        message = "direct publish must not be used by the search service"
        raise AssertionError(message)
