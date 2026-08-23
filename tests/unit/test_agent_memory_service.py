"""Band F group 6: the agent-memory service contract.

The library is faked at the function boundary; what these tests pin is what the
service promises: validated partition identity, conversation-scoped writes with
enrichment deferred to the scheduled job, caller errors raised before the
library, fully serialisable recall, and a consolidation that refuses loudly when
its graph preconditions are absent.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.shared.langchain_layer.agents.memory.agent_memory_service import (
    AgentMemoryService,
    ConsolidationPreconditionError,
    ConversationIdentityRequiredError,
    PartitionIdentityInvalidError,
    memory_partition,
)


class _Recorder:
    def __init__(self, *, results: list[Any] | None = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.results = results or []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.results.pop(0) if self.results else None


def _service(
    *,
    remember: _Recorder | None = None,
    recall: _Recorder | None = None,
    improve: _Recorder | None = None,
    procedures: bool = True,
) -> AgentMemoryService:
    async def probe() -> bool:
        return procedures

    return AgentMemoryService(
        partition_prefix="legal",
        remember_fn=remember or _Recorder(),
        recall_fn=recall or _Recorder(),
        improve_fn=improve or _Recorder(),
        procedures_probe=probe,
    )


# --- 6.1 partition identity ---


def test_two_tenants_never_produce_the_same_partition() -> None:
    names = {memory_partition(tenant_id=t, kind="reports", prefix="legal") for t in ("a", "b", "c")}
    assert len(names) == 3


@pytest.mark.parametrize("tenant", ["", "UPPER", "with space", "with::sep", "x" * 100])
def test_an_invalid_identity_raises_rather_than_defaulting(tenant: str) -> None:
    with pytest.raises(PartitionIdentityInvalidError):
        memory_partition(tenant_id=tenant, kind="reports", prefix="legal")


# --- 6.2 conversation-scoped report write ---


async def test_report_write_is_conversation_scoped_with_self_improvement_disabled() -> None:
    recorder = _Recorder()
    service = _service(remember=recorder)
    await service.store_report(report_text="r", conversation_id="conv-1", tenant_id="acme")
    (call_args, call_kwargs), *_ = recorder.calls
    assert call_args == ("r",)
    assert call_kwargs["session_id"] == "conv-1"
    assert call_kwargs["self_improvement"] is False


async def test_report_write_starts_no_background_task() -> None:
    service = _service()
    before = asyncio.all_tasks()
    await service.store_report(report_text="r", conversation_id="conv-1", tenant_id="acme")
    assert asyncio.all_tasks() - before == set()


# --- 6.3 typed writes rejected before the library ---


async def test_a_typed_write_without_conversation_identity_never_reaches_the_library() -> None:
    remember = _Recorder()
    service = _service(remember=remember)
    with pytest.raises(ConversationIdentityRequiredError):
        await service.store_typed_entry(
            entry_text="t", entry_kind="feedback", conversation_id="", tenant_id="acme"
        )
    assert remember.calls == []


async def test_a_typed_write_reaches_the_library_conversation_scoped() -> None:
    remember = _Recorder()
    service = _service(remember=remember)
    await service.store_typed_entry(
        entry_text="t", entry_kind="qa", conversation_id="conv-2", tenant_id="acme"
    )
    ((_, call_kwargs), *_ ) = remember.calls
    assert call_kwargs["session_id"] == "conv-2"
    assert call_kwargs["self_improvement"] is False


# --- 6.4 recall serialisability ---


class _NestedModel:
    """Stands in for a pydantic model nested inside a recall result."""

    def model_dump(self) -> dict[str, Any]:
        return {"score": 0.5}


async def test_recall_results_round_trip_through_json_with_origin_kept() -> None:
    class _Result:
        def model_dump(self) -> dict[str, Any]:
            return {"text": "hit", "search_result_type": "CHUNK_COMPLETION", "meta": _NestedModel()}

    recall = _Recorder(results=[[_Result()]])
    service = _service(recall=recall)
    results = await service.recall(query_text="q", tenant_id="acme")
    for item in results:
        json.dumps(item)  # must not raise
    assert results[0]["origin"] == "CHUNK_COMPLETION"


async def test_recall_passes_the_caller_partition_only() -> None:
    recall = _Recorder()
    service = _service(recall=recall)
    await service.recall(query_text="q", tenant_id="acme")
    ((_, call_kwargs), *_ ) = recall.calls
    assert call_kwargs["datasets"] == ["legal::acme::reports"]


# --- 6.5/6.6 consolidation ---


async def test_cognify_is_never_called_and_improve_has_one_caller() -> None:
    import inspect

    import app.shared.langchain_layer.agents.memory.agent_memory_service as module

    source = inspect.getsource(module)
    assert "cognify(" not in source, "the full-rebuild call must be unreachable"
    assert source.count("self._improve(") == 1


async def test_consolidation_reports_what_it_consolidated() -> None:
    improve = _Recorder()
    service = _service(improve=improve, procedures=True)
    report = await service.consolidate(tenant_ids=["acme", "other"])
    assert report == {"conversations_consolidated": 2}
    assert len(improve.calls) == 2


async def test_consolidation_refuses_when_graph_procedures_are_absent() -> None:
    improve = _Recorder()
    service = _service(improve=improve, procedures=False)
    with pytest.raises(ConsolidationPreconditionError):
        await service.consolidate(tenant_ids=["acme"])
    assert improve.calls == []


# --- 6.7 failure idiom ---


async def test_a_store_failure_surfaces_as_the_chosen_idiom_not_an_empty_list() -> None:
    async def failing(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("store down")

    service = _service(remember=_Recorder())
    service._remember = failing
    with pytest.raises(RuntimeError, match="store down") as exc_info:
        await service.store_report(report_text="r", conversation_id="c", tenant_id="acme")
    notes = getattr(exc_info.value, "__notes__", [])
    assert any("store_report" in note for note in notes)


# --- 6.8 prune guard ---


def test_prune_does_not_exist_on_the_service_or_in_source() -> None:
    import inspect

    from app.shared.langchain_layer.agents.memory.agent_memory_service import AgentMemoryService

    assert not hasattr(AgentMemoryService, "prune")
    assert "prune" not in inspect.getsource(AgentMemoryService)
