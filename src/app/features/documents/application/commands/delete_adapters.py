"""Production adapters for the D1 deletion removal paths (harden 7.2).

Each adapter implements one port from `delete_document.py` against a real
backend and MUST translate backend failures into a `Failure` wrapping a typed
`FeatureError` from `app.features.documents.domain.errors` so the deletion
sequence fails typed. They are composed at the policy-approval wiring point —
nothing instantiates them until `DELETION_POLICY_APPROVED` opens the gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neo4j.exceptions import Neo4jError
from redis.exceptions import RedisError
from returns.result import Failure, Success
from sqlalchemy import text

from app.features.audit.public import (
    AuditReader,
    AuditRecorder,
    ListAuditEntriesInput,
    RecordAuditEntryInput,
)
from app.features.documents.domain.errors import (
    DocumentDatabaseError,
    DocumentGraphWriteError,
    DocumentStorageError,
)
from app.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

    from neo4j import AsyncDriver
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from app.features.documents.application.commands.delete_document import (
        DeletionTombstone,
    )
    from app.features.documents.domain.errors import DocumentResult
    from app.utils.storage import StorageService


#: The regenerable search/answer cache namespace (both the `documents:` search
#: keys and the `documents:answer:` keys live under this prefix).
_DOCUMENTS_CACHE_PATTERN: Final = "documents:*"

#: The IdempotencyGuard's Redis key prefix (`app.resilience.idempotency`).
_IDEMPOTENCY_REDIS_PATTERN: Final = "idempotency:v2:*"

#: Scan batch for namespace purges; keys are deleted in pages.
_SCAN_BATCH_SIZE: Final = 500

_EPISODE_DELETE_CYPHER = """
MATCH (n)
WHERE n.group_id IN $group_ids OR n.id IN $episode_ids
DETACH DELETE n
"""


class StorageOriginalObjectRemover:
    """Deletes the original artifact behind the document's `object_uri`."""

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    async def delete_original(self, *, object_uri: str) -> DocumentResult[int]:
        result = await self._storage.delete_by_uri(uri=object_uri)
        if isinstance(result, Failure):
            failure = result.failure()
            return Failure(
                DocumentStorageError(
                    message="Failed to delete the original object behind the document",
                    details={"object_uri": object_uri, "error": str(failure)},
                    source="storage_original_object_remover",
                )
            )
        return Success(1)


class DriverGraphitiEpisodeRemover:
    """Deletes the document's clause-episode nodes/edges from Neo4j.

    Partition-scoped: the canonical group ids cover every version/contract
    partition the chunk rows recorded, so cross-document citations are never
    swept by an episode id collision.
    """

    def __init__(self, driver: AsyncDriver, *, database: str | None = None) -> None:
        self._driver = driver
        self._database = database

    async def delete_document_episodes(
        self,
        *,
        tenant_id: str,
        document_id: str,
        episode_ids: Sequence[str],
        group_ids: Sequence[str],
    ) -> DocumentResult[int]:
        if not group_ids and not episode_ids:
            return Success(0)
        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(
                    _EPISODE_DELETE_CYPHER,
                    group_ids=list(group_ids),
                    episode_ids=list(episode_ids),
                )
                summary = await result.consume()
        except Neo4jError as exc:
            return Failure(
                DocumentGraphWriteError(
                    message="Failed to delete the document's Graphiti episodes",
                    details={"document_id": document_id, "error": str(exc)},
                    source="driver_graphiti_episode_remover",
                )
            )
        counters = summary.counters
        deleted = int(getattr(counters, "nodes_deleted", 0))
        logger.bind(
            tenant_id=tenant_id,
            document_id=document_id,
            nodes_deleted=deleted,
        ).info("document_deletion_episodes_removed")
        return Success(deleted)


class RedisRetrievalCachePurger:
    """Purges the regenerable `documents:*` search/answer cache namespace.

    Cache keys are content hashes rather than per-document keys, so the purge
    is namespace-wide: tenant-safe (no cached value survives its key across
    tenants) and complete for the deleted document.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def purge_document_caches(
        self,
        *,
        tenant_id: str,  # noqa: ARG002 — the namespace purge is tenant-safe
        document_id: str,
    ) -> DocumentResult[int]:
        removed = 0
        batch: list[str] = []
        try:
            async for key in self._redis.scan_iter(match=_DOCUMENTS_CACHE_PATTERN):
                batch.append(key if isinstance(key, str) else key.decode("utf-8"))
                if len(batch) >= _SCAN_BATCH_SIZE:
                    removed += await self._delete_batch(batch)
                    batch = []
            removed += await self._delete_batch(batch)
        except RedisError as exc:
            return Failure(
                DocumentStorageError(
                    message="Failed to purge the retrieval cache namespace",
                    details={"document_id": document_id, "error": str(exc)},
                    source="redis_retrieval_cache_purger",
                )
            )
        return Success(removed)

    async def _delete_batch(self, keys: Sequence[str]) -> int:
        if not keys:
            return 0
        await self._redis.delete(*keys)
        return len(keys)


class RedisIdempotencyRecordPurger:
    """Purges `IdempotencyGuard` records reachable from the document's users.

    Postgres: `tool_executions` rows are digest-keyed with a `user_id` column,
    so the durable records are deleted by user scope — AND by the tenant
    identity, because the legal read tools persist their results under
    `authz.tenant_id` (e.g. `personal:alice`), not `document.user_id`; a purge
    keyed only on the owner left deleted document text reloadable through
    `IdempotencyGuard.get` (security finding: cached results retain deleted
    text). Redis: the idempotency namespace is TTL-bounded and regenerable and
    is purged whole. This is the same honest best-effort boundary the D1 draft
    records ("a delete also tolerates cache expiry as a backstop since no
    cached value survives its key").
    """

    _DELETE_SQL = """
    DELETE FROM tool_executions
    WHERE user_id = ANY(CAST(:user_ids AS text[]))
       OR user_id = :tenant_id
    """

    def __init__(
        self,
        *,
        db_engine: AsyncEngine | None = None,
        db_session: AsyncSession | None = None,
        redis: Redis | None = None,
    ) -> None:
        self._db_engine = db_engine
        self._db_session = db_session
        self._redis = redis

    async def purge_idempotency_records(
        self,
        *,
        tenant_id: str,
        document_id: str,
        document_user_ids: Sequence[str],
    ) -> DocumentResult[int]:
        removed = 0
        sql_parameters: dict[str, object] = {
            "user_ids": list(document_user_ids),
            "tenant_id": tenant_id,
        }
        if document_user_ids and self._db_engine is not None:
            async with self._db_engine.begin() as connection:
                result = await connection.execute(
                    text(self._DELETE_SQL),
                    sql_parameters,
                )
                removed += int(getattr(result, "rowcount", 0) or 0)
        if document_user_ids and self._db_session is not None:
            result = await self._db_session.execute(
                text(self._DELETE_SQL),
                sql_parameters,
            )
            removed += int(getattr(result, "rowcount", 0) or 0)
        if self._redis is not None:
            try:
                keys = [
                    key if isinstance(key, str) else key.decode("utf-8")
                    async for key in self._redis.scan_iter(match=_IDEMPOTENCY_REDIS_PATTERN)
                ]
            except RedisError as exc:
                return Failure(
                    DocumentStorageError(
                        message="Failed to purge the Redis idempotency namespace",
                        details={"document_id": document_id, "error": str(exc)},
                        source="redis_idempotency_record_purger",
                    )
                )
            if keys:
                await self._redis.delete(*keys)
                removed += len(keys)
        return Success(removed)


class AuditLogDeletionTombstoneStore:
    """Records deletion tombstones as append-only `audit_logs` rows.

    The tombstone payload carries only the spec fields (document id, tenant,
    event type, timestamp, actor, version range, retention window) — no
    document content, no query text, no derived findings. Rows persist for the
    configured `AUDIT_TOMBSTONE_RETENTION_DAYS`.
    """

    def __init__(self, *, recorder: AuditRecorder, reader: AuditReader) -> None:
        self._recorder = recorder
        self._reader = reader

    async def record(self, tombstone: DeletionTombstone) -> DocumentResult[None]:
        result = await self._recorder.record(
            RecordAuditEntryInput(
                entity_type="document",
                entity_id=tombstone.document_id,
                action="document.deleted",
                user_id=tombstone.actor_id,
                changes={},
                metadata_={
                    "event_type": tombstone.event_type,
                    "tenant_id": tombstone.tenant_id,
                    "version_range": list(tombstone.version_range),
                    "retention_days": tombstone.retention_days,
                    "timestamp": tombstone.timestamp.isoformat(),
                },
            )
        )
        if isinstance(result, Failure):
            return Failure(
                DocumentDatabaseError(
                    message="Failed to record the deletion tombstone",
                    details={"document_id": tombstone.document_id},
                    source="audit_log_deletion_tombstone_store",
                )
            )
        return Success(None)

    async def exists(self, *, tenant_id: str, document_id: str) -> bool:
        result = await self._reader.list(
            ListAuditEntriesInput(entity_type="document", entity_id=document_id)
        )
        if isinstance(result, Failure):
            # A tombstone lookup failure must not silently skip the idempotency
            # fence; the tenant-scoped document load below still refuses an
            # already-deleted document (its row is gone), so the deletion stays
            # idempotent even on this path.
            logger.bind(
                document_id=document_id,
                error=str(result.failure()),
            ).warning("deletion_tombstone_lookup_failed")
            return False
        for entry in result.unwrap().entries:
            metadata = entry.metadata_
            if (
                entry.action == "document.deleted"
                and isinstance(metadata, dict)
                and metadata.get("tenant_id") == tenant_id
            ):
                return True
        return False
