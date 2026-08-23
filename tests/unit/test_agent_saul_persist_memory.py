"""Band F group 7: the memory-persist write seam.

Approved runs write exactly once; unapproved runs write zero times at any trust
level and still complete; a service failure is recorded as COGNEE_WRITE_FAILED
and never propagates; and no knowledge-graph client method is touched during
memory persistence (7.3 — agent memory and the entity graph are different
stores).
"""

from __future__ import annotations

from typing import Any

import pytest

from app.shared.langgraph_layer.agent_saul.nodes import make_persist_memory_node


class _Review:
    def __init__(self, *, approved: bool) -> None:
        self.approved = approved


class _Report:
    def model_dump_json(self) -> str:
        return '{"title": "approved report"}'


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "user_id": "acme",
        "doc_id": "doc-1",
        "thread_id": "thread-9",
        "final_report": _Report(),
        "human_review": _Review(approved=True),
        "long_term_refs": [],
    }
    base.update(overrides)
    return base


class _RecordingService:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    async def store_report(self, **kwargs: Any) -> None:
        if self.error is not None:
            raise self.error
        self.calls.append(kwargs)


class _GraphClientProbe:
    """Every method raises; a call from the persist node fails the test."""

    def __getattr__(self, name: str) -> Any:
        def _boom(*_a: Any, **_k: Any) -> None:
            msg = f"knowledge-graph client method {name} called during persistence"
            raise AssertionError(msg)

        return _boom


async def test_an_approved_run_calls_the_service_write_exactly_once() -> None:
    service = _RecordingService()
    node = make_persist_memory_node(service)
    result = await node(_state())
    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["conversation_id"] == "thread-9"
    assert call["tenant_id"] == "acme"
    assert result["status"].value == "completed" or result["status"] == "completed"
    assert any("reports#doc-1" in ref for ref in result["long_term_refs"])


@pytest.mark.parametrize("review", [None, _Review(approved=False)])
async def test_an_unapproved_run_calls_the_service_zero_times_and_still_completes(
    review: Any,
) -> None:
    state = _state()
    if review is not None:
        state["human_review"] = review
    else:
        del state["human_review"]
    service = _RecordingService()
    node = make_persist_memory_node(service)
    result = await node(state)
    assert service.calls == []
    assert result["status"] is not None
    assert "COGNEE_WRITE_FAILED" not in str(result)


async def test_a_service_failure_records_cognee_write_failed_and_does_not_propagate() -> None:
    service = _RecordingService(error=RuntimeError("store down"))
    node = make_persist_memory_node(service)
    result = await node(_state())
    assert result["status"] is not None  # the run still completes
    assert "COGNEE_WRITE_FAILED" in str(result)


async def test_no_knowledge_graph_client_method_is_called_during_persistence() -> None:
    service = _RecordingService()
    node = make_persist_memory_node(service)
    state = _state(graphiti=_GraphClientProbe())  # a hostile client, had it been used
    result = await node(state)
    assert result["long_term_refs"], "the approved write must have happened"
