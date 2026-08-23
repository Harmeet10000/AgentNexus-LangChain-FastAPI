"""Band F group 8: the read seam — prefetch node and deeper-retrieval restriction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from app.shared.langchain_layer.agents.memory.agent_memory_service import (
    AgentMemoryError,
    AgentMemoryService,
)
from app.shared.langchain_layer.agents.memory.prefetch import (
    deeper_retrieval,
    eligible_for_supplement,
    make_prefetch_memory_node,
)

if TYPE_CHECKING:
    from typing import Any


def _service(recall: Any = None) -> AgentMemoryService:
    async def _recall(*_args: Any, **_kwargs: Any) -> list[Any]:
        if recall is not None:
            return recall()
        return []

    return AgentMemoryService(partition_prefix="legal", recall_fn=_recall)


# --- 8.2 supplement eligibility ---


@pytest.mark.parametrize("task", ["risk_analysis", "obligation_chain", "compliance"])
async def test_eligible_tasks_fetch_a_supplement(task: str) -> None:
    fetched: list[str] = []

    async def search(**kwargs: Any) -> str:
        fetched.append(kwargs["task"])
        return "[90%] precedent"

    node = make_prefetch_memory_node(_service(), graphiti_search=search)
    result = await node({"task": task, "user_id": "acme", "user_query": "q", "working_memory": {}})
    assert fetched == [task]
    assert "[90%] precedent" in result["working_memory"]["prefetched_context"]


@pytest.mark.parametrize("task", ["summarize", "clarify"])
async def test_ineligible_tasks_fetch_no_supplement_and_proceed_on_memory_alone(
    task: str,
) -> None:
    async def search(**_kwargs: Any) -> str:
        msg = "supplement must not be fetched for ineligible tasks"
        raise AssertionError(msg)

    node = make_prefetch_memory_node(_service(), graphiti_search=search)
    result = await node({"task": task, "user_id": "acme", "user_query": "q", "working_memory": {}})
    assert "prefetched_context" not in result["working_memory"]


def test_the_gate_predicate_agrees_with_the_node() -> None:
    assert eligible_for_supplement("obligation_chain")
    assert not eligible_for_supplement("finalization")


# --- 8.3 fail-open ---


async def test_a_recall_failure_fails_open() -> None:
    class _BrokenService:
        async def recall(self, **_kw: Any) -> list[dict[str, Any]]:
            msg = "recall down"
            raise RuntimeError(msg)

    node = make_prefetch_memory_node(_BrokenService())  # type: ignore[arg-type]
    result = await node(
        {"task": "risk_analysis", "user_id": "acme", "user_query": "q", "working_memory": {}}
    )
    # The node returned; the run continues with no prefetched context.
    assert isinstance(result, dict)


# --- 8.4 deeper retrieval restrictions ---


async def test_deeper_retrieval_permits_risk_analysis_role() -> None:
    service = _service()
    results = await deeper_retrieval(
        service, role="risk_analysis", tenant_id="acme", query_text="q"
    )
    assert results == []


async def test_deeper_retrieval_permits_compliance_role() -> None:
    service = _service()
    results = await deeper_retrieval(service, role="compliance", tenant_id="acme", query_text="q")
    assert results == []


async def test_deeper_retrieval_refuses_the_orchestrating_role_with_a_named_reason() -> None:
    service = _service()
    with pytest.raises(AgentMemoryError, match="not available to role"):
        await deeper_retrieval(service, role="orchestrator", tenant_id="acme", query_text="q")


async def test_deeper_retrieval_refuses_a_missing_partition_identity() -> None:
    service = _service()
    with pytest.raises(AgentMemoryError, match="partition identity"):
        await deeper_retrieval(service, role="compliance", tenant_id="", query_text="q")
