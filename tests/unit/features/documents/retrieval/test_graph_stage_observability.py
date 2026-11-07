"""Per-stage observability of the compiled retrieval graph (tasks 1.6 / 1.7).

A two-iteration run including the cap-exhaustion escape must show: a
stage-usage entry per executed stage with its counts in the result state
(no imports from ``features/agent_saul``), a span per executed stage with its
duration, and a structured grade decision-log event per verdict recording the
outcome, the iteration, and the escaped-to-fallback flag.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

import pytest
from loguru import logger as loguru_logger
from returns.result import Success

import app.features.documents.application.queries.documents as documents_module
from app.features.documents.retrieval.graph import (
    RetrievalGraphContext,
    build_retrieval_graph,
)
from app.features.documents.retrieval.graph.state import (
    ContextGrade,
    GeneratedAnswer,
    QueryPlan,
    SourceCriteria,
)

if TYPE_CHECKING:
    from typing import Any

_DOC_ID = "doc-a"
_TENANT_A = "tenant-a"
_QUERY = "termination rights India"

_STAGE_USAGE_STAGE_NAMES = {
    "plan",
    "source_identifier",
    "retrieve_fusion",
    "rerank",
    "grade",
    "generate",
    "edge_grounding",
}


class _ScriptedLoopLLM:
    """Structured double: always insufficient grades, echoing answers."""

    def __init__(self) -> None:
        self.analyzer_prompts: list[str] = []

    def with_structured_output(self, schema: type) -> Any:
        return _LoopHandler(schema, self)

    async def _invoke(self, schema: type, messages: list[Any]) -> Any:
        prompt = str(messages[-1].content)
        if schema is QueryPlan:
            self.analyzer_prompts.append(prompt)
            return QueryPlan(rewritten_query=_QUERY)
        if schema is SourceCriteria:
            return SourceCriteria()
        if schema is ContextGrade:
            return ContextGrade(
                sufficient=False,
                missing_aspects=["termination clause"],
                rewrite_suggestion=_QUERY,
            )
        if schema is GeneratedAnswer:
            return GeneratedAnswer(answer="x", citations=[], confidence="uncertain")
        raise AssertionError(f"unexpected structured schema {schema}")  # noqa: EM102


class _LoopHandler:
    def __init__(self, schema: type, llm: _ScriptedLoopLLM) -> None:
        self._schema = schema
        self._llm = llm

    async def ainvoke(self, messages: list[Any]) -> Any:
        return await self._llm._invoke(self._schema, messages)


class _RecordingRepo:
    async def bm25_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([{"chunk_id": "chunk-1", "score": 2.0}])

    async def vector_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([])

    async def trigram_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([])

    async def fetch_chunks_by_ids(
        self, chunk_ids: list[str], **_kwargs: object
    ) -> Success[dict[str, dict[str, Any]]]:
        return Success(
            {
                chunk_id: {
                    "document_id": _DOC_ID,
                    "title": "t",
                    "content": "content",
                    "preamble": "",
                    "document_version": 1,
                    "contract_index": 0,
                    "pipeline_version": "canonical",
                    "chunk_index": 0,
                    "chunk_kind": "legal_contract",
                    "clause_type": "payment",
                    "chunk_metadata": {},
                    "quality_warnings": [],
                    "graphiti_verified": False,
                    "graph_indexed_at": None,
                }
                for chunk_id in chunk_ids
            }
        )

    async def resolve_scope_versions(
        self, *, tenant_id: str, document_ids: tuple[str, ...]
    ) -> tuple[tuple[str, int, str | None], ...]:
        return ((_DOC_ID, 1, None),) if document_ids else ()


class _PassThroughReranker:
    async def rerank(self, _query: str, chunks: list[Any], *, limit: int) -> list[Any]:
        return chunks[:limit]


async def _fake_embed(*_args: object, **_kwargs: object) -> list[float]:
    return [0.1, 0.2, 0.3]


@pytest.fixture
def _wired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(documents_module, "embed_text", _fake_embed)
    import app.features.documents.retrieval.graph.nodes as graph_nodes

    monkeypatch.setattr(graph_nodes, "embed_text", _fake_embed)


def _captured_events() -> tuple[list[dict[str, Any]], ExitStack]:
    events: list[dict[str, Any]] = []
    stack = ExitStack()

    def _sink(message: Any) -> None:
        record = message.record
        if record["message"] in {"retrieval_stage_span", "grade_decision"}:
            events.append({"event": record["message"], **dict(record["extra"])})

    sink_id = loguru_logger.add(_sink, level="DEBUG", format="{message}")
    stack.callback(loguru_logger.remove, sink_id)
    return events, stack


@pytest.mark.usefixtures("_wired")
async def test_two_iteration_run_reports_stage_usage_spans_and_decision_events() -> None:
    """Cap-exhaustion escape: stage usage, spans, and grade decisions all land."""
    events, stack = _captured_events()
    with stack:
        graph = build_retrieval_graph()
        result = await graph.ainvoke(
            context=RetrievalGraphContext(
                llm=_ScriptedLoopLLM(),  # ty: ignore[invalid-argument-type]
                repo=_RecordingRepo(),  # ty: ignore[invalid-argument-type]
                reranker=_PassThroughReranker(),
            ),
            input={
                "user_id": "user-1",
                "tenant_id": _TENANT_A,
                "allowed_doc_ids": [_DOC_ID],
                "document_versions": [(_DOC_ID, 1, None)],
                "document_contracts": [(_DOC_ID, 1, (0,), False)],
                "scope_unavailable_documents": [],
                "scope_bound": True,
                "query": _QUERY,
                "doc_ids_filter": [],
                "contract_index": None,
                "jurisdiction": None,
                "contract_type": None,
                "clause_type": None,
                "require_graphiti_verified": False,
                "messages": [],
                "iteration_count": 0,
            },
        )

    # The loop ran two grade iterations and escaped to the fallback.
    assert result["iteration_count"] == 2
    assert result["generated_answer"].citations == []

    # Task 1.6: a stage-usage entry per executed stage, aggregated by stage.
    entries: dict[str, int] = {}
    usage: dict[str, int] = {}
    for entry in result["stage_usage"]:
        assert entry.stage in _STAGE_USAGE_STAGE_NAMES
        entries[entry.stage] = entries.get(entry.stage, 0) + 1
        usage[entry.stage] = usage.get(entry.stage, 0) + entry.calls
    # Every executed stage appears — two iterations ran through the loop.
    assert entries == {
        "plan": 2,
        "source_identifier": 2,
        "retrieve_fusion": 2,
        "rerank": 2,
        "grade": 2,
        "edge_grounding": 2,
        "generate": 1,
    }
    # The counts are the billable operations: the retry's source-identifier
    # widening spends no LLM call, the fail-closed edge boundary spends no
    # edge read, and the cap-exhaustion fallback spends no generation call.
    assert usage == {
        "plan": 2,
        "source_identifier": 1,
        "retrieve_fusion": 2,
        "rerank": 2,
        "grade": 2,
        "edge_grounding": 0,
        "generate": 0,
    }

    # Task 1.7: one span per executed stage, each with a duration.
    spans = [event for event in events if event["event"] == "retrieval_stage_span"]
    span_stages: dict[str, int] = {}
    for span in spans:
        assert isinstance(span["duration_ms"], int | float)
        span_stages[span["stage"]] = span_stages.get(span["stage"], 0) + 1
    assert span_stages == {
        "plan": 2,
        "source_identifier": 2,
        "retrieve_fusion": 2,
        "rerank": 2,
        "grade": 2,
    }

    # Task 1.7: a structured decision event per grade verdict — outcome,
    # iteration, and the cap-exhaustion escape flag.
    decisions = [event for event in events if event["event"] == "grade_decision"]
    assert [(d["outcome"], d["iteration"], d["escaped_to_fallback"]) for d in decisions] == [
        ("insufficient", 1, False),
        ("insufficient", 2, True),
    ]


@pytest.mark.usefixtures("_wired")
async def test_sufficient_run_has_no_escape_and_no_fallback() -> None:
    class _SufficientLLM(_ScriptedLoopLLM):
        async def _invoke(self, schema: type, messages: list[Any]) -> Any:
            if schema is ContextGrade:
                return ContextGrade(sufficient=True, missing_aspects=[])
            return await super()._invoke(schema, messages)

    events, stack = _captured_events()
    with stack:
        graph = build_retrieval_graph()
        context = RetrievalGraphContext(
            llm=_SufficientLLM(),  # ty: ignore[invalid-argument-type]
            repo=_RecordingRepo(),  # ty: ignore[invalid-argument-type]
            reranker=_PassThroughReranker(),
        )
        result = await graph.ainvoke(
            {
                "user_id": "user-1",
                "tenant_id": _TENANT_A,
                "allowed_doc_ids": [_DOC_ID],
                "document_versions": [(_DOC_ID, 1, None)],
                "document_contracts": [(_DOC_ID, 1, (0,), False)],
                "scope_unavailable_documents": [],
                "scope_bound": True,
                "query": _QUERY,
                "doc_ids_filter": [],
                "contract_index": None,
                "jurisdiction": None,
                "contract_type": None,
                "clause_type": None,
                "require_graphiti_verified": False,
                "messages": [],
                "iteration_count": 0,
            },
            context=context,
        )

    decisions = [event for event in events if event["event"] == "grade_decision"]
    assert [(d["outcome"], d["iteration"], d["escaped_to_fallback"]) for d in decisions] == [
        ("sufficient", 1, False)
    ]
    # One attempt: the generation stage actually spent one LLM call.
    usage: dict[str, int] = {}
    for entry in result["stage_usage"]:
        usage[entry.stage] = usage.get(entry.stage, 0) + entry.calls
    assert usage["generate"] == 1
    assert usage["grade"] == 1


@pytest.mark.usefixtures("_wired")
async def test_stage_usage_survives_without_agent_saul_imports() -> None:
    """The stage-usage surface never imports from features/agent_saul."""
    import ast
    import importlib
    from pathlib import Path

    graph_module = importlib.import_module("app.features.documents.retrieval.graph")
    graph_package = Path(graph_module.__file__ or "").parent
    sources = sorted(graph_package.glob("*.py"))
    assert sources
    for source_file in sources:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "agent_saul" not in alias.name, (
                        f"divergent behaviour: stage usage must not import agent_saul "
                        f"({source_file.name}: {alias.name})"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert "agent_saul" not in node.module, (
                    f"divergent behaviour: stage usage must not import agent_saul "
                    f"({source_file.name}: {node.module})"
                )
