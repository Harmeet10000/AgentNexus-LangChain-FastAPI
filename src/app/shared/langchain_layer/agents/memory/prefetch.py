"""Band F group 8: the read seam — speculative by construction (D17, NG10).

Nothing here is wired into the mounted graph: the prefetch node exists as a
node factory with no caller inside ``build_saul_graph``, and the deeper-retrieval
operation is a service method awaiting change 3's tool registration. Node
reachability is NOT claimed by this module; its proofs are import-, type- and
unit-level only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.shared.langchain_layer.agents.memory.agent_memory_service import (
    AgentMemoryError,
    memory_partition,
)
from app.utils import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from app.shared.langchain_layer.agents.memory.agent_memory_service import (
        AgentMemoryService,
    )

#: The only tasks whose reasoning benefits from a bounded knowledge-graph
#: supplement. ``obligation_chain`` keeps eligibility deliberately — dropping it
#: would be a silent regression smuggled in under a relocation (B5, Decision 10).
_SUPPLEMENT_ELIGIBLE_TASKS = frozenset({"risk_analysis", "obligation_chain", "compliance"})

#: Roles allowed to call the deeper-retrieval operation (8.4).
_DEEPER_RETRIEVAL_ROLES = frozenset({"risk_analysis", "compliance"})


def make_prefetch_memory_node(
    memory_service: AgentMemoryService,
    graphiti_search: Callable[..., Awaitable[str]] | None = None,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Agent memory first; a bounded knowledge-graph supplement for eligible tasks.

    Fail-open by contract (8.3): a recall failure leaves the run continuing on
    current-run context plus any supplement already obtained.
    """

    async def prefetch_memory_node(state: dict[str, Any]) -> dict[str, Any]:
        task = str(state.get("task", ""))
        tenant_id = str(state.get("user_id", ""))
        conversation_id = state.get("thread_id")
        query = str((state.get("working_memory") or {}).get("clarified_intent", state.get("user_query", "")))

        memory_context = ""
        try:
            results = await memory_service.recall(
                query_text=query,
                tenant_id=tenant_id,
                conversation_id=str(conversation_id) if conversation_id else None,
            )
            memory_context = "\n".join(str(item.get("text", "")) for item in results[:5])
        except Exception as exc:  # noqa: BLE001 — fail-open read path (8.3)
            exc.add_note("node=prefetch_memory")
            logger.bind(error=str(exc)).warning("agent_memory_prefetch_failed")

        supplement = ""
        if task in _SUPPLEMENT_ELIGIBLE_TASKS and graphiti_search is not None:
            try:
                supplement = await graphiti_search(
                    query=query,
                    user_id=tenant_id,
                    doc_id=str(state.get("doc_id", "")),
                    task=task,
                )
            except Exception as exc:  # noqa: BLE001 — fail-open read path (8.3)
                exc.add_note("node=prefetch_memory, stage=supplement")
                logger.bind(error=str(exc)).warning("graphiti_supplement_failed")

        working_memory = {**(state.get("working_memory") or {})}
        if memory_context or supplement:
            working_memory["prefetched_context"] = "\n".join(
                part for part in (memory_context, supplement) if part
            )
        return {"working_memory": working_memory}

    return prefetch_memory_node


async def deeper_retrieval(
    service: AgentMemoryService,
    *,
    role: str,
    tenant_id: str,
    query_text: str,
    top_k: int = 25,
) -> list[dict[str, Any]]:
    """Deeper retrieval over the caller's OWN partition — restricted.

    Available to the risk-analysis and compliance roles only. Any other role —
    or a call without a usable partition identity — is refused with a named
    reason rather than answered with an empty list, so a refusal can never be
    mistaken for "nothing in memory".
    """
    if role not in _DEEPER_RETRIEVAL_ROLES:
        msg = f"deeper retrieval is not available to role {role!r}"
        raise AgentMemoryError(msg)
    if not tenant_id:
        msg = "deeper retrieval requires a partition identity"
        raise AgentMemoryError(msg)

    partition = memory_partition(tenant_id=tenant_id, kind="reports", prefix=service._prefix)
    try:
        results = await service._recall(
            query_text=query_text,
            datasets=[partition],
            top_k=top_k,
        )
    except Exception as exc:
        exc.add_note(f"operation=deeper_retrieval, partition={partition}")
        raise
    return [_full_result_dump(result) for result in (results or [])]


def _full_result_dump(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return dict(result)


def eligible_for_supplement(task: str) -> bool:
    """Public predicate mirroring the gate, for tests and callers."""
    return task in _SUPPLEMENT_ELIGIBLE_TASKS


__all__ = [
    "_DEEPER_RETRIEVAL_ROLES",
    "_SUPPLEMENT_ELIGIBLE_TASKS",
    "deeper_retrieval",
    "eligible_for_supplement",
    "make_prefetch_memory_node",
]
