"""The repository's first agent-memory call surface (band F group 6).

One service, one failure idiom, one partition-name builder. The library's
full-rebuild operation is structurally unreachable from here (Trap3), and there
is deliberately no destructive-cleanup operation: the permanent memory graph
shares a Neo4j instance with the document entity graph, so wiping there would
destroy another library's data.
"""

from __future__ import annotations

import re
from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TC003 — used in method bodies at runtime
    Protocol,
)

import cognee

from app.utils import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import LiteralString

    from neo4j import Query

#: Tenant and kind identifiers may only contain these characters. The separator
#: below cannot therefore appear inside either component, which is what makes
#: two different tenants unable to produce the same partition name.
_VALID_ID = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
_ID_SEPARATOR = "::"


class AgentMemoryError(RuntimeError):
    """Base class for agent-memory caller errors."""


class ConversationIdentityRequiredError(AgentMemoryError):
    """A write needing a conversation identity arrived without one."""


class PartitionIdentityInvalidError(AgentMemoryError):
    """A tenant or kind identifier failed validation — never defaulted."""


class ConsolidationPreconditionError(AgentMemoryError):
    """Consolidation refused: its graph preconditions (APOC/GDS) are absent.

    The underlying rebuild fails without raising, so silence would read as
    success. Refusing loudly is the observable behaviour this change ships.
    """


def memory_partition(*, tenant_id: str, kind: str, prefix: str) -> str:
    """Construct a memory partition name — the only tenant boundary we have.

    With backend access control unavailable (NG6), this name is what keeps two
    tenants' memories apart, so both components are validated rather than
    interpolated, and a failing identity raises instead of being defaulted.
    """
    for label, value in (("tenant", tenant_id), ("kind", kind)):
        if not _VALID_ID.match(value) or _ID_SEPARATOR in value:
            msg = f"invalid memory partition {label}: {value!r}"
            raise PartitionIdentityInvalidError(msg)
    return f"{prefix}{_ID_SEPARATOR}{tenant_id}{_ID_SEPARATOR}{kind}"


def _full_dump(value: Any) -> Any:
    """Serialise completely, recursing into nested models.

    A shallow ``dict(r)`` leaves nested models as objects, which type-checks and
    then fails at serialisation time.
    """
    if hasattr(value, "model_dump"):
        return {key: _full_dump(item) for key, item in value.model_dump().items()}
    if isinstance(value, list):
        return [_full_dump(item) for item in value]
    return value


def _origin_of(dumped: dict[str, Any]) -> str | None:
    """Preserve the field distinguishing a conversation hit from a graph hit."""
    for key in ("search_result_type", "origin", "source", "retrieval_type"):
        if key in dumped:
            return str(dumped[key])
    return None


class ProceduresQueryDriver(Protocol):
    """Minimal query surface the consolidation precondition probe needs.

    Declared structurally so this module never imports the driver package at
    runtime: the probe needs exactly one method, with the driver's own
    parameter shape so a real driver satisfies the protocol.
    """

    async def execute_query(self, query_: LiteralString | Query, **kwargs: Any) -> Any: ...


def make_neo4j_procedures_probe(
    driver: ProceduresQueryDriver | None,
) -> Callable[[], Awaitable[bool]]:
    """Build the consolidation precondition probe over a Neo4j driver.

    A probe failure answers False — consolidation then refuses loudly with
    :data:`ConsolidationPreconditionError` — so an unreachable graph or a
    graph without the required procedures can never read as a success.
    """

    async def probe() -> bool:
        if driver is None:
            return False
        try:
            # The same graph-procedure precondition the health surface reports
            # as ``graphProceduresAvailable``. Inlined as a literal: the driver
            # types the query as ``LiteralString``, which a variable does not
            # satisfy.
            records, _, _ = await driver.execute_query(
                "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc.' "
                "OR name STARTS WITH 'gds.' RETURN count(name) AS n"
            )
        except Exception:  # noqa: BLE001 — any probe failure means "precondition absent"
            return False
        if not records:
            return False
        first = records[0]
        # neo4j Record supports mapping access; a test double may not.
        count = first.get("n", 0) if hasattr(first, "get") else getattr(first, "n", 0)
        return bool(count > 0)

    return probe


class AgentMemoryService:
    """Conversation-scoped memory writes, typed writes, recall, consolidation."""

    def __init__(
        self,
        *,
        partition_prefix: str,
        pending_sessions: set[str] | None = None,
        remember_fn: Callable[..., Awaitable[Any]] | None = None,
        recall_fn: Callable[..., Awaitable[Sequence[Any]]] | None = None,
        improve_fn: Callable[..., Awaitable[Any]] | None = None,
        procedures_probe: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        self._prefix = partition_prefix
        self._remember = remember_fn or cognee.remember
        self._recall = recall_fn or cognee.recall
        self._improve = improve_fn or cognee.improve
        self._procedures_probe = procedures_probe
        # Sessions written since the last consolidation — improve() needs them to
        # bridge conversation cache into the permanent graph.
        self._pending_sessions: set[str] = (
            pending_sessions if pending_sessions is not None else set()
        )

    async def _graph_procedures_available(self) -> bool:
        if self._procedures_probe is not None:
            return await self._procedures_probe()
        return False

    async def store_report(
        self,
        *,
        report_text: str,
        conversation_id: str,
        tenant_id: str,
        dataset_kind: str = "reports",
    ) -> None:
        """Store an approved report in conversation scope, enrichment deferred.

        Self-improvement stays disabled: enabled, the library bridges session
        data into the permanent graph via a detached background task started
        inside the caller's event loop — exactly what must not happen on a
        request path. Consolidation belongs to the scheduled job alone.
        """
        partition = memory_partition(tenant_id=tenant_id, kind=dataset_kind, prefix=self._prefix)
        try:
            await self._remember(
                report_text,
                dataset_name=partition,
                session_id=conversation_id,
                self_improvement=False,
                run_in_background=False,
            )
        except Exception as exc:
            exc.add_note(f"operation=store_report, partition={partition}")
            logger.bind(partition=partition, error=str(exc)).warning("agent_memory_store_failed")
            raise

    async def store_typed_entry(
        self,
        *,
        entry_text: str,
        entry_kind: str,
        conversation_id: str,
        tenant_id: str,
    ) -> None:
        """Store a trace / QA / feedback entry, rejected before the library.

        The library raises ``session_id is required for typed memory entries``
        deep inside itself; that is a caller error here, not a memory-store
        failure, so it is surfaced before the call happens at all.
        """
        if not conversation_id:
            msg = f"a {entry_kind} memory entry requires a conversation identity"
            raise ConversationIdentityRequiredError(msg)
        partition = memory_partition(tenant_id=tenant_id, kind="session", prefix=self._prefix)
        try:
            await self._remember(
                entry_text,
                dataset_name=partition,
                session_id=conversation_id,
                self_improvement=False,
                run_in_background=False,
            )
        except Exception as exc:
            exc.add_note(f"operation=store_typed_entry, kind={entry_kind}")
            logger.bind(kind=entry_kind, error=str(exc)).warning("agent_memory_store_failed")
            raise

    async def recall(
        self,
        *,
        query_text: str,
        tenant_id: str,
        conversation_id: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query memory, returning fully serialisable mappings with origin kept."""
        partition = memory_partition(tenant_id=tenant_id, kind="reports", prefix=self._prefix)
        try:
            if conversation_id:
                # Conversation scope: cognee reads its session cache. Datasets are
                # deliberately NOT passed — pre-consolidation the dataset does not
                # exist yet, and name resolution would 404 over the cached answer.
                results = await self._recall(
                    query_text=query_text,
                    session_id=conversation_id,
                    top_k=top_k,
                )
            else:
                results = await self._recall(
                    query_text=query_text,
                    datasets=[partition],
                    top_k=top_k,
                )
        except Exception as exc:
            exc.add_note(f"operation=recall, partition={partition}")
            raise
        dumped = [_full_dump(result) for result in (results or [])]
        for item in dumped:
            item.setdefault("origin", _origin_of(item))
        logger.bind(partition=partition, result_count=len(dumped)).debug("agent_memory_recalled")
        return dumped

    async def consolidate(self, *, tenant_ids: Sequence[str]) -> dict[str, int]:
        """Consolidate conversation memory into the permanent graph.

        The ONLY caller of enrichment in the repository. Refuses with a named
        precondition failure when the required graph procedures are absent —
        the underlying rebuild fails silently, so refusing is the only honest
        behaviour available. Reports what it consolidated so growth is
        observable, even though nothing bounds it here.
        """
        if not await self._graph_procedures_available():
            msg = (
                "consolidation precondition failed: the target graph does not expose "
                "the procedures the rebuild requires (APOC/GDS)"
            )
            raise ConsolidationPreconditionError(msg)
        consolidated = 0
        for tenant_id in tenant_ids:
            partition = memory_partition(tenant_id=tenant_id, kind="reports", prefix=self._prefix)
            await self._improve(
                dataset=partition,
                session_ids=list(self._pending_sessions),
            )
            consolidated += 1
        return {
            "conversations_consolidated": consolidated,
            "sessions_bridged": len(self._pending_sessions),
        }
