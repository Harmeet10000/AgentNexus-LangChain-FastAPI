"""Policy-gated, tenant-scoped, idempotent document deletion (harden 7.2).

Pins the D1 machinery contract:

- the gate is FIRST: while `DELETION_POLICY_APPROVED` is closed (fail-closed
  default), every delete attempt returns the typed
  `deletion_policy_not_approved` refusal and observes/removes nothing;
- with the gate force-opened (as the policy-approval wiring will do), every
  derived-artifact kind is removed — Graphiti episodes, the original object,
  the Postgres chunk/document rows, the Redis cache namespace, and the
  idempotency records — while the append-only access audit and the tombstone
  are retained;
- the operation is idempotent through the tombstone: a retry answers
  `already_deleted` with zero further removals;
- unknown and cross-tenant ids refuse typed with zero removals;
- deletion is not a model-callable operation at any gate state;
- the Postgres removal statements carry the mandatory tenant bind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from returns.result import Failure, Success
from sqlalchemy.dialects import postgresql

from app.config.settings import Settings
from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentDeletionInventory,
    DocumentRepository,
)
from app.features.documents.application.access_audit import (
    InMemoryDocumentAccessAuditRecorder,
)
from app.features.documents.application.commands.delete_document import (
    DELETION_HARD_EVENT,
    DeletionDocumentRecord,
    DeletionPolicyGate,
    DocumentDeletionService,
    deletion_group_ids,
)
from app.features.documents.domain.errors import (
    DeletionPolicyNotApprovedError,
    DocumentGraphWriteError,
    DocumentNotFoundError,
)
from app.retrieval.graphiti.grouping import build_graphiti_group_id

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from app.features.documents.application.commands.delete_document import DeletionTombstone

TENANT = "personal:alice"
OTHER_TENANT = "personal:bob"
DOCUMENT_ID = "3f6d3b1e-0000-4000-8000-000000000001"
ACTOR = "user-alice"
OBJECT_URI = "s3://legal-documents/personal:alice/user-alice/3f6d3b1e/hash.pdf"


class _FakeRepository:
    """Records the deletion repository calls; serves one live document."""

    def __init__(self, *, known_document: bool = True) -> None:
        self.known_document = known_document
        self.load_calls: list[str] = []
        self.inventory_calls: list[str] = []
        self.artifact_delete_calls: list[tuple[str, str]] = []

    async def load_document_for_deletion(self, *, tenant_id: str, document_id: str):
        self.load_calls.append(f"{tenant_id}:{document_id}")
        if not self.known_document or tenant_id != TENANT or document_id != DOCUMENT_ID:
            return Failure(
                DocumentNotFoundError(
                    message="Document not found",
                    details={"tenant_id": tenant_id, "document_id": document_id},
                    source="fake",
                )
            )
        return Success(
            DeletionDocumentRecord(
                document_id=DOCUMENT_ID,
                tenant_id=tenant_id,
                object_uri=OBJECT_URI,
                user_id=ACTOR,
            )
        )

    async def collect_deletion_inventory(self, *, tenant_id: str, document_id: str):
        self.inventory_calls.append(f"{tenant_id}:{document_id}")
        return Success(
            DocumentDeletionInventory(
                chunk_count=7,
                min_document_version=1,
                max_document_version=2,
                graphiti_episode_ids=("episode-1", "episode-2"),
                graphiti_partitions=((1, None), (2, 3)),
            )
        )

    async def delete_document_artifacts(self, *, tenant_id: str, document_id: str):
        self.artifact_delete_calls.append((tenant_id, document_id))
        return Success(7)


class _FakeTombstones:
    def __init__(self, *, pre_existing: bool = False) -> None:
        self.pre_existing = pre_existing
        self.recorded: list[DeletionTombstone] = []

    async def record(self, tombstone: DeletionTombstone):
        self.recorded.append(tombstone)
        return Success(None)

    async def exists(self, *, tenant_id: str, document_id: str) -> bool:
        return self.pre_existing or bool(self.recorded)


class _RecordingRemover:
    def __init__(self, *, removed: int = 1, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._removed = removed
        self._fail = fail

    async def delete_document_episodes(
        self,
        *,
        tenant_id: str,
        document_id: str,
        episode_ids: Sequence[str],
        group_ids: Sequence[str],
    ):
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "episode_ids": tuple(episode_ids),
                "group_ids": tuple(group_ids),
            }
        )
        if self._fail:
            return Failure(
                DocumentGraphWriteError(
                    message="episode removal failed",
                    details={"document_id": document_id},
                    source="fake",
                )
            )
        return Success(self._removed)


class _FakeOriginalRemover:
    def __init__(self) -> None:
        self.uris: list[str] = []

    async def delete_original(self, *, object_uri: str):
        self.uris.append(object_uri)
        return Success(1)


class _FakeCachePurger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def purge_document_caches(self, *, tenant_id: str, document_id: str):
        self.calls.append((tenant_id, document_id))
        return Success(12)


class _FakeIdempotencyPurger:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def purge_idempotency_records(
        self, *, tenant_id: str, document_id: str, document_user_ids: Sequence[str]
    ):
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "document_user_ids": tuple(document_user_ids),
            }
        )
        return Success(3)


def _service(
    *,
    approved: bool,
    repository: _FakeRepository | None = None,
    tombstones: _FakeTombstones | None = None,
    access_audit: InMemoryDocumentAccessAuditRecorder | None = None,
) -> tuple[
    DocumentDeletionService,
    _FakeRepository,
    _FakeTombstones,
    _RecordingRemover,
    _FakeOriginalRemover,
    _FakeCachePurger,
    _FakeIdempotencyPurger,
]:
    repository = repository or _FakeRepository()
    tombstones = tombstones or _FakeTombstones()
    episodes = _RecordingRemover(removed=2)
    originals = _FakeOriginalRemover()
    caches = _FakeCachePurger()
    idempotency = _FakeIdempotencyPurger()
    service = DocumentDeletionService(
        policy_gate=DeletionPolicyGate(
            approved=approved,
            soft_window_days=30,
            tombstone_retention_days=90,
        ),
        repository=repository,  # type: ignore[arg-type] — structural fake
        tombstones=tombstones,
        original_remover=originals,
        episode_remover=episodes,
        cache_purger=caches,
        idempotency_purger=idempotency,
        access_audit=access_audit,
    )
    return service, repository, tombstones, episodes, originals, caches, idempotency


# ---------------------------------------------------------------------------
# Gate blocks (typed refusal, zero artifacts removed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closed_gate_refuses_typed_and_removes_nothing() -> None:
    """The fail-closed default answers deletion_policy_not_approved first."""
    service, repository, tombstones, episodes, originals, caches, idempotency = _service(
        approved=False,
        tombstones=_FakeTombstones(pre_existing=True),
    )

    result = await service.delete_document(
        tenant_id=TENANT, document_id=DOCUMENT_ID, actor_id=ACTOR
    )

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, DeletionPolicyNotApprovedError)
    assert str(error.code) == "deletion_policy_not_approved"
    assert error.kind.value == "authorization"
    # The gate observed NOTHING: not even a document load or a tombstone check.
    assert repository.load_calls == []
    assert repository.inventory_calls == []
    assert tombstones.recorded == []
    assert episodes.calls == []
    assert originals.uris == []
    assert caches.calls == []
    assert idempotency.calls == []


def test_gate_refusal_is_typed_before_any_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unapproved policy leaves the gate closed, windows named constants.

    The owner approved the policy constants (DELETION_POLICY_APPROVED=true in the
    dev env since 2026-11-14), so the default-settings branch pins the CLOSED
    state explicitly via the env override — the gate's fail-closed default is
    the contract under test, not whatever the local env happens to say.
    """
    monkeypatch.setenv("DELETION_POLICY_APPROVED", "false")
    gate = DeletionPolicyGate.from_settings(Settings())
    assert gate.approved is False
    refusal = gate.refusal()
    assert refusal is not None
    assert str(refusal.code) == "deletion_policy_not_approved"
    assert refusal.details is not None
    assert refusal.details["soft_window_days"] == 30
    assert refusal.details["tombstone_retention_days"] == 90


# ---------------------------------------------------------------------------
# Gate force-opened: every derived-artifact kind removed, audit+tombstone kept
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opened_gate_removes_every_derived_artifact_and_retains_audit_tombstone() -> None:
    repository = _FakeRepository()
    audit = InMemoryDocumentAccessAuditRecorder()
    tombstones = _FakeTombstones()
    service, _, tombstones, episodes, originals, caches, idempotency = _service(
        approved=True,
        repository=repository,
        tombstones=tombstones,
        access_audit=audit,
    )

    result = await service.delete_document(
        tenant_id=TENANT, document_id=DOCUMENT_ID, actor_id=ACTOR
    )

    assert isinstance(result, Success)
    outcome = result.unwrap()
    assert outcome.outcome == "deleted"
    assert outcome.removed == {
        "graphiti_episodes": 2,
        "originals": 1,
        "chunk_rows": 7,
        "cache_keys": 12,
        "idempotency_records": 3,
    }
    # Every derived-artifact kind was removed.
    assert episodes.calls, "Graphiti episodes were not removed"
    assert episodes.calls[0]["episode_ids"] == ("episode-1", "episode-2")
    expected_groups = (
        build_graphiti_group_id(
            tenant_id=TENANT, document_id=DOCUMENT_ID, document_version=1, contract_index=None
        ),
        build_graphiti_group_id(
            tenant_id=TENANT, document_id=DOCUMENT_ID, document_version=2, contract_index=3
        ),
    )
    assert episodes.calls[0]["group_ids"] == expected_groups
    assert originals.uris == [OBJECT_URI]
    assert repository.artifact_delete_calls == [(TENANT, DOCUMENT_ID)]
    assert caches.calls == [(TENANT, DOCUMENT_ID)]
    assert idempotency.calls[0]["document_user_ids"] == (ACTOR,)
    # Audit + tombstone are the retained records.
    assert len(tombstones.recorded) == 1
    tombstone = tombstones.recorded[0]
    assert tombstone.event_type == DELETION_HARD_EVENT
    assert tombstone.actor_id == ACTOR
    assert tombstone.version_range == (1, 2)
    assert tombstone.retention_days == 90
    assert len(audit.events) == 1
    deletion_event = audit.events[0]
    assert deletion_event.operation == "delete"
    assert deletion_event.outcome == "deleted"
    assert deletion_event.document_ids == (DOCUMENT_ID,)


@pytest.mark.asyncio
async def test_removal_paths_never_touch_the_access_audit() -> None:
    """The append-only audit survives deletion: no path rewrites or drops it."""
    audit = InMemoryDocumentAccessAuditRecorder()
    await audit.append(_existing_read_event())
    service, _, _, _, _, _, _ = _service(approved=True, access_audit=audit)

    result = await service.delete_document(
        tenant_id=TENANT, document_id=DOCUMENT_ID, actor_id=ACTOR
    )

    assert isinstance(result, Success)
    # The pre-existing read event is still first and intact; deletion only
    # appended its own non-content record.
    assert [event.operation for event in audit.events] == ["read", "delete"]


def _existing_read_event():
    from app.features.documents.application.access_audit import DocumentAccessAuditEvent

    return DocumentAccessAuditEvent(
        tenant_id=TENANT,
        principal_id=ACTOR,
        operation="read",
        requested_document_ids=(DOCUMENT_ID,),
        document_ids=(DOCUMENT_ID,),
        policy="evidence_read",
        outcome="ok",
    )


# ---------------------------------------------------------------------------
# Mid-sequence typed failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mid_sequence_failure_stops_the_sequence_typed() -> None:
    """A failing removal path returns its typed error; later paths never run."""
    repository = _FakeRepository()
    tombstones = _FakeTombstones()
    episodes = _RecordingRemover(fail=True)
    originals = _FakeOriginalRemover()
    caches = _FakeCachePurger()
    idempotency = _FakeIdempotencyPurger()
    service = DocumentDeletionService(
        policy_gate=DeletionPolicyGate(
            approved=True, soft_window_days=30, tombstone_retention_days=90
        ),
        repository=repository,  # type: ignore[arg-type] — structural fake
        tombstones=tombstones,
        original_remover=originals,
        episode_remover=episodes,
        cache_purger=caches,
        idempotency_purger=idempotency,
    )

    result = await service.delete_document(
        tenant_id=TENANT, document_id=DOCUMENT_ID, actor_id=ACTOR
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), DocumentGraphWriteError)
    # The failing step ran, but nothing after it did.
    assert len(episodes.calls) == 1
    assert originals.uris == []
    assert repository.artifact_delete_calls == []
    assert caches.calls == []
    assert idempotency.calls == []
    assert tombstones.recorded == []


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_after_deletion_is_idempotent_with_zero_further_removals() -> None:
    """A tombstoned document answers already_deleted and removes nothing."""
    repository = _FakeRepository()
    tombstones = _FakeTombstones()
    service, _, tombstones, episodes, originals, caches, idempotency = _service(
        approved=True,
        repository=repository,
        tombstones=tombstones,
    )

    first = await service.delete_document(tenant_id=TENANT, document_id=DOCUMENT_ID, actor_id=ACTOR)
    assert isinstance(first, Success)
    assert first.unwrap().outcome == "deleted"

    second = await service.delete_document(
        tenant_id=TENANT, document_id=DOCUMENT_ID, actor_id=ACTOR
    )
    assert isinstance(second, Success)
    outcome = second.unwrap()
    assert outcome.outcome == "already_deleted"
    assert outcome.removed == {}
    # No further removal of any kind.
    assert len(episodes.calls) == 1
    assert len(originals.uris) == 1
    assert len(repository.artifact_delete_calls) == 1
    assert len(caches.calls) == 1
    assert len(idempotency.calls) == 1
    assert len(tombstones.recorded) == 1


@pytest.mark.asyncio
async def test_tombstone_check_precedes_the_document_load() -> None:
    """A tombstone without a document row (hard purge already done) still
    answers idempotently instead of failing as not-found."""
    repository = _FakeRepository(known_document=False)
    tombstones = _FakeTombstones(pre_existing=True)
    service, _, _, episodes, _, _, _ = _service(
        approved=True,
        repository=repository,
        tombstones=tombstones,
    )

    result = await service.delete_document(
        tenant_id=TENANT, document_id=DOCUMENT_ID, actor_id=ACTOR
    )

    assert isinstance(result, Success)
    assert result.unwrap().outcome == "already_deleted"
    assert repository.load_calls == []
    assert episodes.calls == []


# ---------------------------------------------------------------------------
# Tenant scope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_tenant_or_unknown_document_refuses_typed_with_zero_removals() -> None:
    """An id that is not the calling tenant's document removes nothing."""
    repository = _FakeRepository()
    service, _, tombstones, episodes, originals, caches, idempotency = _service(
        approved=True,
        repository=repository,
    )

    for tenant in (OTHER_TENANT, TENANT):
        result = await service.delete_document(
            tenant_id=tenant,
            document_id=DOCUMENT_ID
            if tenant == OTHER_TENANT
            else "00000000-0000-4000-8000-000000000099",
            actor_id=ACTOR,
        )
        assert isinstance(result, Failure)
        assert isinstance(result.failure(), DocumentNotFoundError)
    assert episodes.calls == []
    assert originals.uris == []
    assert caches.calls == []
    assert idempotency.calls == []
    assert tombstones.recorded == []


# ---------------------------------------------------------------------------
# Not a model-callable operation
# ---------------------------------------------------------------------------


def test_delete_is_not_model_callable_at_any_gate_state() -> None:
    """No tool surface exposes deletion (spec: not model-callable)."""
    from app.features.documents.application.tools import bundle as bundle_module
    from app.features.documents.application.tools import registry as registry_module

    bundle_fields = set(bundle_module.AgentToolBundle.model_fields)
    registry_names = set(registry_module.BASE_REQUIRED_ROLE_TOOLS) | set(
        registry_module.TRANSITIONAL_TOOLS
    )
    for names in registry_module.BASE_REQUIRED_ROLE_TOOLS.values():
        registry_names |= set(names)
    for names in registry_module.TRANSITIONAL_TOOLS.values():
        registry_names |= set(names)
    for names in registry_module.TRAVERSAL_SEATS.values():
        registry_names |= set(names)
    registry_names |= set(registry_module.EVIDENCE_SEAM_TOOLS)
    registry_names |= set(registry_module.EVIDENCE_READ_TOOLS)
    registry_names |= set(registry_module.GRAPHITI_WRITE_TOOLS)

    assert not any("delete" in name for name in bundle_fields)
    assert not any("delete" in name for name in registry_names)


# ---------------------------------------------------------------------------
# Postgres removal statements carry the mandatory tenant bind
# ---------------------------------------------------------------------------


class _DeletionCaptureSession:
    """Records statements; scripts the inventory aggregate + partition rows."""

    def __init__(
        self,
        *,
        inventory_row: dict[str, Any],
        partition_rows: list[dict[str, Any]],
    ) -> None:
        self.statements: list[Any] = []
        self._inventory_row = inventory_row
        self._partition_rows = partition_rows
        self.flushed = 0

    async def execute(self, statement: Any) -> Any:
        self.statements.append(statement)
        if len(self.statements) == 1:
            inventory_row = self._inventory_row

            class _One:
                rowcount = 0

                def __init__(self) -> None:
                    self._row = inventory_row

                def mappings(self) -> _One:
                    return self

                def one(self) -> dict[str, Any]:
                    return self._row

            return _One()
        partition_rows = self._partition_rows

        class _All:
            rowcount = 0

            def __init__(self) -> None:
                self._rows = partition_rows

            def mappings(self) -> _All:
                return self

            def all(self) -> list[dict[str, Any]]:
                return self._rows

        return _All()

    async def flush(self) -> None:
        self.flushed += 1


def _bound_values(statement: Any) -> list[Any]:
    compiled = statement.compile(dialect=postgresql.dialect())
    return list(compiled.params.values())


def _rendered(statement: Any) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


@pytest.mark.asyncio
async def test_inventory_collection_binds_tenant_and_document() -> None:
    session = _DeletionCaptureSession(
        inventory_row={
            "chunk_count": 2,
            "min_version": 1,
            "max_version": 1,
            "episode_ids": ["episode-1"],
        },
        partition_rows=[{"document_version": 1, "contract_index": None}],
    )
    repo = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repo.collect_deletion_inventory(tenant_id=TENANT, document_id=DOCUMENT_ID)

    assert isinstance(result, Success)
    inventory = result.unwrap()
    assert inventory.chunk_count == 2
    assert inventory.graphiti_episode_ids == ("episode-1",)
    assert inventory.graphiti_partitions == ((1, None),)
    assert len(session.statements) == 2
    for statement in session.statements:
        values = _bound_values(statement)
        assert TENANT in values
        assert DOCUMENT_ID in values
        assert OTHER_TENANT not in _rendered(statement)


@pytest.mark.asyncio
async def test_artifact_delete_statements_bind_tenant_and_document() -> None:
    session = _DeletionCaptureSession(
        inventory_row={
            "chunk_count": 0,
            "min_version": None,
            "max_version": None,
            "episode_ids": None,
        },
        partition_rows=[],
    )
    repo = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repo.delete_document_artifacts(tenant_id=TENANT, document_id=DOCUMENT_ID)

    assert isinstance(result, Success)
    assert result.unwrap() == 0
    assert session.flushed == 1
    # Both the chunks delete and the documents delete bind the tenant.
    assert len(session.statements) == 2
    for statement in session.statements:
        values = _bound_values(statement)
        assert TENANT in values, "removal statement forgot the tenant bind"
        assert DOCUMENT_ID in values
        rendered = _rendered(statement)
        assert OTHER_TENANT not in rendered


def test_deletion_group_ids_skip_legacy_partitions() -> None:
    """Version-0 (pre-canonical) partitions never produce group ids."""
    ids = deletion_group_ids(
        tenant_id=TENANT,
        document_id=DOCUMENT_ID,
        partitions=((0, None), (1, None), (2, 3)),
    )
    assert ids == (
        build_graphiti_group_id(
            tenant_id=TENANT, document_id=DOCUMENT_ID, document_version=1, contract_index=None
        ),
        build_graphiti_group_id(
            tenant_id=TENANT, document_id=DOCUMENT_ID, document_version=2, contract_index=3
        ),
    )


# ---------------------------------------------------------------------------
# Security: the idempotency purge must cover the tenant identity the READ
# tools persist under (`user_id=authz.tenant_id`), not just the document owner.
# ---------------------------------------------------------------------------


class _FakeDeleteConnection:
    """Applies the purger's DELETE against an in-memory row list."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows
        self.rowcount = 0
        self.statements: list[tuple[str, dict[str, object]]] = []

    async def execute(self, statement: Any, parameters: dict[str, object] | None = None) -> Any:
        rendered = str(statement)
        self.statements.append((rendered, dict(parameters or {})))
        params = parameters or {}
        user_ids = {str(value) for value in (params.get("user_ids") or [])}
        tenant_id = str(params.get("tenant_id") or "")
        before = len(self._rows)
        self._rows[:] = [
            row
            for row in self._rows
            if row["user_id"] not in user_ids and row["user_id"] != tenant_id
        ]
        self.rowcount = before - len(self._rows)
        return self


class _FakeDeleteEngine:
    """Minimal `AsyncEngine.begin()` shape serving the fake connection."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._connection = _FakeDeleteConnection(rows)

    def begin(self) -> Any:
        outer = self

        class _Ctx:
            async def __aenter__(self) -> _FakeDeleteConnection:
                return outer._connection

            async def __aexit__(self, *_exc: object) -> None:
                return None

        return _Ctx()


@pytest.mark.asyncio
async def test_idempotency_purge_removes_results_saved_under_the_tenant_identity() -> None:
    """Deleted document text must not survive a purge through a tenant-keyed row.

    The legal read tools persist their tool results with `user_id=authz.tenant_id`
    (`personal:alice`), not `document.user_id` (`user-alice`). A purge keyed only on the
    owner left those rows alive, and `IdempotencyGuard.get` reloads from PostgreSQL —
    deleted text came back after the delete. The purge now covers both identities, and the
    Redis namespace clear removes the hot-path key the guard reads first.
    """
    from fakeredis.aioredis import FakeRedis

    from app.features.documents.application.commands.delete_adapters import (
        RedisIdempotencyRecordPurger,
    )
    from app.resilience.idempotency import IdempotencyGuard, ToolResult

    rows = [
        {"user_id": ACTOR, "key": "owner-row"},
        {"user_id": TENANT, "key": "tenant-identity-row"},
    ]
    engine = _FakeDeleteEngine(rows)
    redis = FakeRedis()
    guard = IdempotencyGuard(redis=redis, db_engine=engine)  # type: ignore[arg-type]

    key = IdempotencyGuard.make_key(
        step_id="step-1",
        structural={"doc_id": DOCUMENT_ID},
        content=None,
        user_id=TENANT,
        tool_name="search_legal_precedents",
        tenant_id=TENANT,
        allowed_doc_ids=(DOCUMENT_ID,),
    )
    await guard.set(
        key,
        ToolResult.ok({"excerpt": "clause text of the deleted document"}),
        tool_name="search_legal_precedents",
        user_id=TENANT,
        thread_id="thread-1",
        step_id="step-1",
    )
    assert await guard.get(key) is not None, "setup failed: guard must serve before the delete"

    purger = RedisIdempotencyRecordPurger(db_engine=engine, redis=redis)
    result = await purger.purge_idempotency_records(
        tenant_id=TENANT,
        document_id=DOCUMENT_ID,
        document_user_ids=(ACTOR,),
    )

    assert isinstance(result, Success)
    # Both the owner-scoped row and the tenant-identity row are gone, plus the Redis key.
    assert rows == []
    assert await guard.get(key) is None, "deleted text survived the purge through the cache"

    delete_statements = [
        rendered
        for rendered, _params in engine._connection.statements
        if "DELETE FROM tool_executions" in rendered
    ]
    assert delete_statements, "the purge never issued its DELETE"
    assert ":tenant_id" in delete_statements[0], (
        "purge statement no longer constrains the tenant identity"
    )


@pytest.mark.asyncio
async def test_idempotency_purge_redis_clear_hits_guard_reads() -> None:
    """The Redis half alone must already break the hot-path read of a deleted result."""
    from fakeredis.aioredis import FakeRedis

    from app.features.documents.application.commands.delete_adapters import (
        RedisIdempotencyRecordPurger,
    )
    from app.resilience.idempotency import IdempotencyGuard, ToolResult

    rows: list[dict[str, str]] = []
    engine = _FakeDeleteEngine(rows)
    redis = FakeRedis()
    guard = IdempotencyGuard(redis=redis, db_engine=engine)  # type: ignore[arg-type]

    key = IdempotencyGuard.make_key(
        step_id="step-2",
        structural={"doc_id": DOCUMENT_ID},
        content=None,
        user_id=TENANT,
        tool_name="query_knowledge_graph",
        tenant_id=TENANT,
    )
    await guard.set(
        key,
        ToolResult.ok({"nodes": ["deleted clause"]}),
        tool_name="query_knowledge_graph",
        user_id=TENANT,
        thread_id="thread-1",
        step_id="step-2",
    )

    purger = RedisIdempotencyRecordPurger(db_engine=engine, redis=redis)
    result = await purger.purge_idempotency_records(
        tenant_id=TENANT,
        document_id=DOCUMENT_ID,
        document_user_ids=(ACTOR,),
    )

    assert isinstance(result, Success)
    assert await redis.get(f"idempotency:v2:{key}") is None
