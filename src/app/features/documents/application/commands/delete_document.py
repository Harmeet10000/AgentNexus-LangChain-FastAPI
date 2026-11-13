"""Policy-gated, tenant-scoped, idempotent document deletion (D1, harden 7.2).

The machinery behind the deletion policy drafted in
`openspec/changes/harden-agent-saul-evidence-pipeline/notes/deletion-policy-draft.md`
and required by the document-delete-and-access-audit spec delta.

Gate FIRST: until `DELETION_POLICY_APPROVED` is set (fail-closed default
`False`, alongside the L1 legal/security sign-off), every delete attempt
returns the typed `deletion_policy_not_approved` refusal and performs zero
removals — "keep delete disabled" means gated, not unbuilt. Once the gate is
opened by policy approval, the removal sequence executes in an auditable order:

1. Graphiti clause-episode nodes within the document's partitions (episode ids
   are collected from the chunk rows BEFORE those rows are deleted — they are
   unrecoverable afterwards);
2. the original artifact in object storage (the R2/S3 `object_uri`);
3. the Postgres derived rows (chunks, then the document row — both statements
   carry the mandatory tenant predicate, so a cross-tenant deletion is not
   expressible);
4. the Redis search/answer caches (a regenerable namespace; the policy
   explicitly tolerates cache expiry as the backstop);
5. the `IdempotencyGuard` records (Postgres `tool_executions` rows scoped by
   the document's users plus the Redis idempotency namespace — both
   TTL-bounded and regenerable).

Exactly two records survive: the append-only non-content access audit (no
removal path touches it) and the tombstone written at the end of the sequence.
Tombstones make the operation idempotent: a document whose tombstone exists
answers `already_deleted` with zero further removals, whichever of the document
row or the tombstone a retry sees first.

This is NOT a model-callable operation: no tool in the Agent Saul registry
exposes deletion (`application/tools/registry.py`), at any gate state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Success

from app.features.documents.application.access_audit import (
    DocumentAccessAuditEvent,
    DocumentAccessAuditRecorder,
)
from app.features.documents.domain.errors import (
    DeletionPolicyNotApprovedError,
    DocumentNotFoundError,
)
from app.retrieval.graphiti.grouping import build_graphiti_group_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.config.settings import Settings
    from app.features.documents.adapters.persistence.postgres.document_repository import (
        DocumentDeletionInventory,
        DocumentRepository,
    )
    from app.features.documents.domain.errors import DocumentResult

DELETION_SOFT_EVENT = "soft_delete"
DELETION_HARD_EVENT = "hard_purge"
DELETION_RECOVERED_EVENT = "recovered"

#: The audit action the production tombstone adapter writes. Append-only rows
#: in `audit_logs` are the tombstone store until policy approval adds a
#: dedicated table; the retention windows are configuration, not literals.
DELETION_TOMBSTONE_ACTION = "document.deleted"

#: The access-audit operation label for deletion attempts (no content fields).
DELETION_ACCESS_AUDIT_OPERATION = "delete"

_REMOVED_ORIGINALS = "originals"
_REMOVED_EPISODES = "graphiti_episodes"
_REMOVED_CHUNK_ROWS = "chunk_rows"
_REMOVED_CACHE_KEYS = "cache_keys"
_REMOVED_IDEMPOTENCY_RECORDS = "idempotency_records"


class DeletionEventCode(StrEnum):
    """The deletion event vocabulary a tombstone record carries."""

    SOFT_DELETE = DELETION_SOFT_EVENT
    HARD_PURGE = DELETION_HARD_EVENT
    RECOVERED = DELETION_RECOVERED_EVENT


def deletion_group_ids(
    *,
    tenant_id: str,
    document_id: str,
    partitions: Sequence[tuple[int, int | None]],
) -> tuple[str, ...]:
    """Translate the deletion inventory's partitions into Graphiti group ids.

    Versions below 1 predate the canonical partitioning (no canonical group id
    was ever written for them), and a partition the writer's own validation
    would have rejected cannot exist in the graph — both are skipped instead of
    failing the deletion.
    """
    group_ids: list[str] = []
    for version, contract_index in partitions:
        if version < 1:
            continue
        try:
            group_ids.append(
                build_graphiti_group_id(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    document_version=version,
                    contract_index=contract_index,
                )
            )
        except ValueError:
            continue
    return tuple(dict.fromkeys(group_ids))


@dataclass(frozen=True, slots=True)
class DeletionPolicyGate:
    """The D1 policy gate — the named-constant settings decision.

    Checked before anything else in the sequence, so a closed gate never
    observes — let alone removes — a single artifact. The two windows are
    REQUIRES-LEGAL-CONFIRMATION (L1, task 7.3).
    """

    approved: bool
    soft_window_days: int
    tombstone_retention_days: int

    @classmethod
    def from_settings(cls, settings: Settings) -> DeletionPolicyGate:
        return cls(
            approved=settings.DELETION_POLICY_APPROVED,
            soft_window_days=settings.DELETION_SOFT_WINDOW_DAYS,
            tombstone_retention_days=settings.AUDIT_TOMBSTONE_RETENTION_DAYS,
        )

    def soft_window(self) -> timedelta:
        return timedelta(days=self.soft_window_days)

    def tombstone_retention(self) -> timedelta:
        return timedelta(days=self.tombstone_retention_days)

    def refusal(self) -> DeletionPolicyNotApprovedError | None:
        """The typed gate refusal while the policy is unapproved, else None.

        The refusal is typed (`deletion_policy_not_approved`) — never a silent
        no-op, never a 500 (draft Decision 5; spec "Delete is blocked before
        policy approval"). Result-based like every documents feature error:
        callers return `Failure(refusal())` rather than raising, because the
        typed error models are pydantic values, not exceptions.
        """
        if self.approved:
            return None
        return DeletionPolicyNotApprovedError(
            message="Document deletion is disabled until the deletion policy "
            "and its retention windows are approved",
            details={
                "policy": "deletion-policy-draft",
                "soft_window_days": self.soft_window_days,
                "tombstone_retention_days": self.tombstone_retention_days,
            },
            source="document_deletion_policy_gate",
        )


class DeletionDocumentRecord(BaseModel):
    """The content-free document identity the removal sequence needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    tenant_id: str
    object_uri: str
    user_id: str


class DeletionTombstone(BaseModel):
    """The only deletion record retained; it contains no document content.

    Spec fields: document id, tenant id, deletion event type, timestamp,
    authenticated actor, and the document's version range at deletion.
    Retained for `AUDIT_TOMBSTONE_RETENTION_DAYS` after the purge.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    tenant_id: str
    event_type: str
    actor_id: str
    version_range: tuple[int, int]
    retention_days: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class DocumentDeletionOutcome(BaseModel):
    """The idempotent result of a delete attempt: what was removed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    tenant_id: str
    outcome: Literal["deleted", "already_deleted"]
    actor_id: str
    removed: dict[str, int]


class OriginalObjectRemover(Protocol):
    """Removes the original artifact behind an object-store URI.

    Adapters MUST translate backend failures into a `Failure` wrapping a
    `FeatureError` from `app.features.documents.domain.errors` (storage
    failures map to `DocumentStorageError`) so the deletion sequence fails
    typed.
    """

    async def delete_original(self, *, object_uri: str) -> DocumentResult[int]: ...


class GraphitiEpisodeRemover(Protocol):
    """Removes clause-episode nodes/edges for the document's partitions.

    Adapters MUST translate backend failures into a `Failure` wrapping
    `DocumentGraphWriteError`. The service passes both the episode ids
    harvested from the (soon-to-be-deleted) chunk rows and the canonical group
    ids of every ``(document_version, contract_index)`` partition the document
    occupies.
    """

    async def delete_document_episodes(
        self,
        *,
        tenant_id: str,
        document_id: str,
        episode_ids: Sequence[str],
        group_ids: Sequence[str],
    ) -> DocumentResult[int]: ...


class RetrievalCachePurger(Protocol):
    """Purges the regenerable search/answer cache namespace.

    Cache keys are content hashes (not per-document keys), so the adapter
    purges the whole `documents:*` namespace — tenant-safe (no cached value
    survives its key cross-tenant) and complete for the deleted document.
    """

    async def purge_document_caches(
        self, *, tenant_id: str, document_id: str
    ) -> DocumentResult[int]: ...


class IdempotencyRecordPurger(Protocol):
    """Purges `IdempotencyGuard` records reachable from the document's users.

    The Postgres `tool_executions` rows are digest-keyed with a `user_id`
    column, so the adapter deletes by user scope plus the Redis idempotency
    namespace; both stores are TTL-bounded and regenerable.
    """

    async def purge_idempotency_records(
        self,
        *,
        tenant_id: str,
        document_id: str,
        document_user_ids: Sequence[str],
    ) -> DocumentResult[int]: ...


class DeletionTombstoneStore(Protocol):
    """Records and reads deletion tombstones (append-only)."""

    async def record(self, tombstone: DeletionTombstone) -> DocumentResult[None]: ...

    async def exists(self, *, tenant_id: str, document_id: str) -> bool: ...


class DocumentDeletionRepository(Protocol):
    """The tenant-bound repository surface the deletion sequence needs."""

    async def load_document_for_deletion(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> DocumentResult[DeletionDocumentRecord]: ...

    async def collect_deletion_inventory(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> DocumentResult[DocumentDeletionInventory]: ...

    async def delete_document_artifacts(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> DocumentResult[int]: ...


class DocumentDeletionService:
    """Executes the gated, idempotent deletion sequence for one document."""

    def __init__(
        self,
        *,
        policy_gate: DeletionPolicyGate,
        repository: DocumentDeletionRepository,
        tombstones: DeletionTombstoneStore,
        original_remover: OriginalObjectRemover,
        episode_remover: GraphitiEpisodeRemover,
        cache_purger: RetrievalCachePurger,
        idempotency_purger: IdempotencyRecordPurger,
        access_audit: DocumentAccessAuditRecorder | None = None,
    ) -> None:
        self._gate = policy_gate
        self._repository = repository
        self._tombstones = tombstones
        self._original_remover = original_remover
        self._episode_remover = episode_remover
        self._cache_purger = cache_purger
        self._idempotency_purger = idempotency_purger
        self._access_audit = access_audit

    async def delete_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        actor_id: str,
    ) -> DocumentResult[DocumentDeletionOutcome]:
        """Run the deletion sequence, gate first.

        The order of operations is load-bearing:
        1. the policy gate — a closed gate observes and removes nothing;
        2. the tombstone check — a previously deleted document answers
           idempotently with zero further removals;
        3. the tenant-scoped load — an unknown or cross-tenant id is a typed
           not-found, again with zero removals;
        4. inventory collection, then the removal paths, then the tombstone.
        """
        # 1. The gate is FIRST: nothing below runs while the policy is closed.
        gate_refusal = self._gate.refusal()
        if gate_refusal is not None:
            return Failure(gate_refusal)

        # 2. Idempotency: a tombstoned document is already handled.
        if await self._tombstones.exists(tenant_id=tenant_id, document_id=document_id):
            outcome = DocumentDeletionOutcome(
                document_id=document_id,
                tenant_id=tenant_id,
                outcome="already_deleted",
                actor_id=actor_id,
                removed={},
            )
            await self._record_access_audit(outcome)
            return Success(outcome)

        # 3. Tenant-scoped load: unknown and cross-tenant ids never remove.
        document_result = await self._repository.load_document_for_deletion(
            tenant_id=tenant_id,
            document_id=document_id,
        )
        if isinstance(document_result, Failure):
            return document_result
        document = document_result.unwrap()

        inventory_result = await self._repository.collect_deletion_inventory(
            tenant_id=tenant_id,
            document_id=document_id,
        )
        if isinstance(inventory_result, Failure):
            return inventory_result
        inventory = inventory_result.unwrap()

        # 4. The removal sequence. Episode ids were collected BEFORE the chunk
        #    rows they are recorded on are deleted.
        group_ids = deletion_group_ids(
            tenant_id=tenant_id,
            document_id=document_id,
            partitions=inventory.graphiti_partitions,
        )
        removed: dict[str, int] = {}
        episodes_result = await self._episode_remover.delete_document_episodes(
            tenant_id=tenant_id,
            document_id=document_id,
            episode_ids=inventory.graphiti_episode_ids,
            group_ids=group_ids,
        )
        if isinstance(episodes_result, Failure):
            return episodes_result
        removed[_REMOVED_EPISODES] = episodes_result.unwrap()

        originals_result = await self._original_remover.delete_original(
            object_uri=document.object_uri,
        )
        if isinstance(originals_result, Failure):
            return originals_result
        removed[_REMOVED_ORIGINALS] = originals_result.unwrap()

        chunk_rows_result = await self._repository.delete_document_artifacts(
            tenant_id=tenant_id,
            document_id=document_id,
        )
        if isinstance(chunk_rows_result, Failure):
            return chunk_rows_result
        removed[_REMOVED_CHUNK_ROWS] = chunk_rows_result.unwrap()

        caches_result = await self._cache_purger.purge_document_caches(
            tenant_id=tenant_id,
            document_id=document_id,
        )
        if isinstance(caches_result, Failure):
            return caches_result
        removed[_REMOVED_CACHE_KEYS] = caches_result.unwrap()

        idempotency_result = await self._idempotency_purger.purge_idempotency_records(
            tenant_id=tenant_id,
            document_id=document_id,
            document_user_ids=(document.user_id,),
        )
        if isinstance(idempotency_result, Failure):
            return idempotency_result
        removed[_REMOVED_IDEMPOTENCY_RECORDS] = idempotency_result.unwrap()

        # 5. The tombstone is the only deletion record retained.
        tombstone_result = await self._tombstones.record(
            DeletionTombstone(
                document_id=document_id,
                tenant_id=tenant_id,
                event_type=DELETION_HARD_EVENT,
                actor_id=actor_id,
                version_range=(
                    inventory.min_document_version,
                    inventory.max_document_version,
                ),
                retention_days=self._gate.tombstone_retention_days,
            )
        )
        if isinstance(tombstone_result, Failure):
            return tombstone_result

        outcome = DocumentDeletionOutcome(
            document_id=document_id,
            tenant_id=tenant_id,
            outcome="deleted",
            actor_id=actor_id,
            removed=removed,
        )
        await self._record_access_audit(outcome)
        return Success(outcome)

    async def _record_access_audit(self, outcome: DocumentDeletionOutcome) -> None:
        """Append the non-content audit record; never blocks the outcome.

        The append-only access audit is one of the two records a deletion MUST
        retain; no removal path touches it. A recorder failure is logged and
        swallowed by the recorder implementation, not here.
        """
        if self._access_audit is None:
            return
        await self._access_audit.append(
            DocumentAccessAuditEvent(
                tenant_id=outcome.tenant_id,
                principal_id=outcome.actor_id,
                operation=DELETION_ACCESS_AUDIT_OPERATION,
                requested_document_ids=(outcome.document_id,),
                document_ids=(outcome.document_id,),
                policy="deletion_policy_d1",
                outcome=outcome.outcome,
            )
        )


class DocumentDeletionRepositoryAdapter:
    """Maps the concrete `DocumentRepository` onto the deletion protocol."""

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def load_document_for_deletion(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> DocumentResult[DeletionDocumentRecord]:
        result = await self._repository.get_document_by_id(
            user_id=tenant_id,
            document_id=document_id,
            tenant_id=tenant_id,
        )
        if isinstance(result, Failure):
            return result
        document = result.unwrap()
        if document is None:
            return Failure(
                DocumentNotFoundError(
                    message="Document not found for the given tenant and document ID",
                    details={"tenant_id": tenant_id, "document_id": document_id},
                    source="document_deletion_repository_adapter",
                )
            )
        return Success(
            DeletionDocumentRecord(
                document_id=str(document.id),
                tenant_id=document.tenant_id,
                object_uri=document.object_uri,
                user_id=document.user_id,
            )
        )

    async def collect_deletion_inventory(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> DocumentResult[DocumentDeletionInventory]:
        return await self._repository.collect_deletion_inventory(
            tenant_id=tenant_id,
            document_id=document_id,
        )

    async def delete_document_artifacts(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> DocumentResult[int]:
        return await self._repository.delete_document_artifacts(
            tenant_id=tenant_id,
            document_id=document_id,
        )
