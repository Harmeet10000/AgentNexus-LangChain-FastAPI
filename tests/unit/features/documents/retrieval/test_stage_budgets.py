"""Per-stage budget enforcement and the standing degradation alert (G6 tasks 4.1/4.2).

Over-budget exhaustion produces the typed degraded outcome naming the stage
(never an unbounded run, never a silent overrun), the standing alert fires on
sustained degradations attributable per stage and document class — earlier
than and distinct from the harden change's pilot-kill criteria — and with no
tracker in the invocation context nothing changes (the parity suites' contract).
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

import pytest
from loguru import logger as loguru_logger

from app.ai.validation.pipeline import BudgetLedger
from app.features.documents.application.retrieval.graph import (
    RetrievalGraphContext,
    build_retrieval_graph,
)
from app.features.documents.application.retrieval.graph.budgets import (
    DEFAULT_STAGE_BUDGETS,
    STAGE_GENERATE,
    STAGE_GRADE,
    STAGE_POST_PROCESS,
    StageBudget,
    StageBudgetTracker,
    build_stage_budget_tracker,
    reset_degradation_window,
)
from app.features.documents.application.retrieval.graph.state import (
    ContextGrade,
    GeneratedAnswer,
    QueryPlan,
    SourceCriteria,
    StageUsage,
)

if TYPE_CHECKING:
    from typing import Any

_DOC_ID = "doc-a"
_TENANT_A = "tenant-a"
_QUERY = "termination rights India"
_FALLBACK = (
    "I do not have enough grounded document context to answer this reliably. "
    "Please narrow the question or ingest the relevant document sections."
)


class _ScriptedLLM:
    """Structured double recording which stages actually spent calls."""

    def __init__(self) -> None:
        self.generated: bool = False

    def with_structured_output(self, schema: type) -> Any:
        return _Handler(schema, self)


class _Handler:
    def __init__(self, schema: type, llm: _ScriptedLLM) -> None:
        self._schema = schema
        self._llm = llm

    async def ainvoke(self, messages: list[Any]) -> Any:
        if self._schema is QueryPlan:
            return QueryPlan(rewritten_query=_QUERY)
        if self._schema is SourceCriteria:
            return SourceCriteria()
        if self._schema is ContextGrade:
            return ContextGrade(sufficient=True, missing_aspects=[])
        if self._schema is GeneratedAnswer:
            self._llm.generated = True
            return GeneratedAnswer(answer="grounded answer", citations=[], confidence="high")
        raise AssertionError(f"unexpected structured schema {self._schema}")  # noqa: EM102


class _RecordingRepo:
    async def bm25_search(self, **_kwargs: object) -> Any:
        from returns.result import Success

        return Success([{"chunk_id": "chunk-1", "score": 2.0}])

    async def vector_search(self, **_kwargs: object) -> Any:
        from returns.result import Success

        return Success([])

    async def trigram_search(self, **_kwargs: object) -> Any:
        from returns.result import Success

        return Success([])

    async def fetch_chunks_by_ids(self, chunk_ids: list[str], **_kwargs: object) -> Any:
        from returns.result import Success

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


class _PassThroughReranker:
    async def rerank(self, _query: str, chunks: list[Any], *, limit: int) -> list[Any]:
        return chunks[:limit]


async def _fake_embed(*_args: object, **_kwargs: object) -> list[float]:
    return [0.1, 0.2, 0.3]


@pytest.fixture(autouse=True)
def _clean_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.features.documents.application.retrieval.graph.nodes.embed_text", _fake_embed
    )
    reset_degradation_window()
    yield
    reset_degradation_window()


def _tracker(**overrides: StageBudget) -> StageBudgetTracker:
    budgets = dict(DEFAULT_STAGE_BUDGETS)
    budgets.update(overrides)
    return StageBudgetTracker(budgets=budgets, alert_threshold=3, alert_window_seconds=300)


def _input() -> dict[str, Any]:
    return {
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
    }


def _captured_events() -> tuple[list[dict[str, Any]], ExitStack]:
    events: list[dict[str, Any]] = []
    stack = ExitStack()

    def _sink(message: Any) -> None:
        record = message.record
        if record["message"] in {
            "stage_budget_degradation",
            "retrieval_budget_degradation_alert",
        }:
            events.append({"event": record["message"], **dict(record["extra"])})

    sink_id = loguru_logger.add(_sink, level="DEBUG", format="{message}")
    stack.callback(loguru_logger.remove, sink_id)
    return events, stack


async def test_over_budget_generate_stage_degrades_honestly() -> None:
    """Spec: an over-budget stage returns the typed degraded outcome, unspent."""
    llm = _ScriptedLLM()
    tracker = _tracker(**{STAGE_GENERATE: StageBudget(latency_ms=180_000, max_calls=0)})
    graph = build_retrieval_graph()
    result = await graph.ainvoke(
        _input(),
        context=RetrievalGraphContext(
            llm=llm,  # ty: ignore[invalid-argument-type]
            repo=_RecordingRepo(),  # ty: ignore[invalid-argument-type]
            reranker=_PassThroughReranker(),
            budgets=tracker,
        ),
    )

    assert llm.generated is False  # the unbounded call never happened
    assert result["generated_answer"].answer == _FALLBACK
    assert result["stage_usage"][-1].stage == STAGE_GENERATE
    assert result["stage_usage"][-1].calls == 0
    budget_warnings = [
        warning
        for warning in result["quality_warnings"]
        if warning.code == "STAGE_BUDGET_EXHAUSTED"
    ]
    assert [(w.stage, w.severity) for w in budget_warnings] == [(STAGE_GENERATE, "warning")]
    assert "calls" in budget_warnings[0].message
    assert STAGE_GENERATE in budget_warnings[0].message


async def test_latency_breach_names_the_stage_but_does_not_break_the_ask() -> None:
    tracker = _tracker(plan=StageBudget(latency_ms=0, max_calls=8))
    graph = build_retrieval_graph()
    result = await graph.ainvoke(
        _input(),
        context=RetrievalGraphContext(
            llm=_ScriptedLLM(),  # ty: ignore[invalid-argument-type]
            repo=_RecordingRepo(),  # ty: ignore[invalid-argument-type]
            reranker=_PassThroughReranker(),
            budgets=tracker,
        ),
    )

    budget_warnings = [
        warning
        for warning in result["quality_warnings"]
        if warning.code == "STAGE_BUDGET_EXHAUSTED" and warning.stage == "plan"
    ]
    assert len(budget_warnings) == 1
    assert "latency_ms" in budget_warnings[0].message
    # The ask itself completed normally — the latency breach degrades honestly
    # without discarding the run.
    assert result["generated_answer"].answer == "grounded answer"


async def test_degradation_event_is_attributable_per_stage_and_document_class() -> None:
    events, stack = _captured_events()
    with stack:
        tracker = _tracker(**{STAGE_GENERATE: StageBudget(latency_ms=180_000, max_calls=0)})
        graph = build_retrieval_graph()
        await graph.ainvoke(
            _input(),
            context=RetrievalGraphContext(
                llm=_ScriptedLLM(),  # ty: ignore[invalid-argument-type]
                repo=_RecordingRepo(),  # ty: ignore[invalid-argument-type]
                reranker=_PassThroughReranker(),
                budgets=tracker,
            ),
        )

    degradations = [event for event in events if event["event"] == "stage_budget_degradation"]
    assert degradations
    assert degradations[0]["stage"] == STAGE_GENERATE
    assert degradations[0]["document_class"] == "legal_contract"  # from the hydrated row
    assert degradations[0]["budget_kind"] == "calls"


async def test_standing_alert_fires_on_sustained_degradation() -> None:
    tracker = _tracker(
        **{
            STAGE_GENERATE: StageBudget(latency_ms=180_000, max_calls=0),
            "plan": StageBudget(latency_ms=60_000, max_calls=0),
        }
    )
    events, stack = _captured_events()
    with stack:
        # threshold=3: the first sustained period fires the alert naming its
        # stages and classes, then the window resets (one alert per period).
        for _ in range(3):
            tracker.try_charge(STAGE_GENERATE, document_class="legal_contract")
        alerts = [
            event for event in events if event["event"] == "retrieval_budget_degradation_alert"
        ]
        assert len(alerts) == 1
        assert alerts[0]["alert"] == "standing_degradation"
        assert alerts[0]["stages"] == {STAGE_GENERATE: 3}
        assert alerts[0]["document_classes"] == {"legal_contract": 3}

        reset_degradation_window()
        events.clear()
        # A fresh sustained period (a different stage/class) alerts again.
        for _ in range(3):
            tracker.try_charge("plan", document_class="statutes")
        alerts = [
            event for event in events if event["event"] == "retrieval_budget_degradation_alert"
        ]
        assert len(alerts) == 1
        assert alerts[0]["stages"] == {"plan": 3}
        assert alerts[0]["document_classes"] == {"statutes": 3}


async def test_single_degradation_below_threshold_raises_no_alert() -> None:
    events, stack = _captured_events()
    with stack:
        tracker = _tracker(grade=StageBudget(latency_ms=0, max_calls=8))
        tracker.check_latency("grade", 5_000, document_class="contracts")
    assert not [event for event in events if event["event"] == "retrieval_budget_degradation_alert"]
    assert [event for event in events if event["event"] == "stage_budget_degradation"]


async def test_ledger_adapter_matches_the_validation_budget_port() -> None:
    tracker = _tracker(**{STAGE_GENERATE: StageBudget(latency_ms=180_000, max_calls=1)})
    ledger = tracker.ledger_for(STAGE_GENERATE)
    assert isinstance(ledger, BudgetLedger)  # the validation pipeline's port
    assert ledger.charge() is True
    assert ledger.charge() is False  # exhausted is an expected answer, not an error


def test_default_budgets_are_generous_and_enabled_by_default() -> None:
    from app.config.settings import Settings

    assert Settings().DOCUMENTS_STAGE_BUDGETS_ENABLED is True
    tracker = build_stage_budget_tracker()
    assert tracker is not None
    # Generous: a healthy two-iteration ask never comes near these.
    assert tracker.budget_for(STAGE_GENERATE).max_calls >= 8
    assert tracker.budget_for(STAGE_GENERATE).latency_ms >= 180_000


class _EmptyRepo:
    """The three fused legs return no rows at all."""

    async def bm25_search(self, **_kwargs: object) -> Any:
        from returns.result import Success

        return Success([])

    async def vector_search(self, **_kwargs: object) -> Any:
        from returns.result import Success

        return Success([])

    async def trigram_search(self, **_kwargs: object) -> Any:
        from returns.result import Success

        return Success([])

    async def fetch_chunks_by_ids(self, chunk_ids: list[str], **_kwargs: object) -> Any:
        from returns.result import Success

        return Success({})


class _RecordingRedis:
    """Minimal async redis double recording the envelopes it stores."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def setex(self, key: str, _ttl: int, value: str) -> None:
        self.values[key] = value


class _TransientGenerateLLM:
    """Plan/grading succeed; generation fails transiently `failures` times.

    Counts the provider-visible generation attempts so the per-attempt charge
    is observable as an exact provider-call count.
    """

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.generate_calls = 0
        self.generated = False

    def with_structured_output(self, schema: type) -> Any:
        return _TransientHandler(schema, self)


class _TransientHandler:
    def __init__(self, schema: type, llm: _TransientGenerateLLM) -> None:
        self._schema = schema
        self._llm = llm

    async def ainvoke(self, messages: list[Any]) -> Any:
        if self._schema is QueryPlan:
            return QueryPlan(rewritten_query=_QUERY)
        if self._schema is SourceCriteria:
            return SourceCriteria()
        if self._schema is ContextGrade:
            return ContextGrade(sufficient=True, missing_aspects=[])
        if self._schema is GeneratedAnswer:
            self._llm.generate_calls += 1
            if self._llm.generate_calls <= self._llm.failures:
                message = "provider timeout"
                raise TimeoutError(message)
            self._llm.generated = True
            return GeneratedAnswer(answer="grounded answer", citations=[], confidence="high")
        raise AssertionError(f"unexpected structured schema {self._schema}")  # noqa: EM102


def _fast_retry_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the retry boundary's backoff so budget tests stay fast."""
    import app.ai.graph.retry as retry_module

    monkeypatch.setattr(retry_module, "_INITIAL_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(retry_module, "_JITTER_SECONDS", 0.0)


def _budget_warnings(result: dict[str, Any]) -> list[Any]:
    return [
        warning
        for warning in result["quality_warnings"]
        if warning.code == "STAGE_BUDGET_EXHAUSTED"
    ]


async def test_grade_budget_exhaustion_takes_the_no_evidence_fallback() -> None:
    """Spec: a refused grading charge never marks unchecked evidence sufficient.

    The exhausted grade is the typed insufficiency (the same escape an
    exhausted grade loop takes): generation never runs, and the ask returns
    the fallback answer naming the grade stage's budget.
    """
    llm = _ScriptedLLM()
    tracker = _tracker(**{STAGE_GRADE: StageBudget(latency_ms=60_000, max_calls=0)})
    graph = build_retrieval_graph()
    result = await graph.ainvoke(
        _input(),
        context=RetrievalGraphContext(
            llm=llm,  # ty: ignore[invalid-argument-type]
            repo=_RecordingRepo(),  # ty: ignore[invalid-argument-type]
            reranker=_PassThroughReranker(),
            budgets=tracker,
        ),
    )

    assert llm.generated is False  # generation never ran over ungraded evidence
    assert result["generated_answer"].answer == _FALLBACK
    assert result["generated_answer"].confidence == "uncertain"
    budget_warnings = _budget_warnings(result)
    assert [(w.stage, w.severity) for w in budget_warnings] == [(STAGE_GRADE, "warning")]


async def test_empty_retrieval_with_exhausted_grade_never_caches_a_generated_answer() -> None:
    """Spec: empty retrieval + exhausted grading reaches the fallback, not the
    generator — only the fallback envelope (if anything) enters the cache."""
    llm = _ScriptedLLM()
    redis = _RecordingRedis()
    tracker = _tracker(**{STAGE_GRADE: StageBudget(latency_ms=60_000, max_calls=0)})
    graph = build_retrieval_graph()
    result = await graph.ainvoke(
        _input(),
        context=RetrievalGraphContext(
            llm=llm,  # ty: ignore[invalid-argument-type]
            repo=_EmptyRepo(),  # ty: ignore[invalid-argument-type]
            reranker=_PassThroughReranker(),
            budgets=tracker,
            redis=redis,  # ty: ignore[invalid-argument-type]
        ),
    )

    assert llm.generated is False
    assert result["generated_answer"].answer == _FALLBACK
    from app.features.documents.application.retrieval.graph.state import CachedAskEnvelope

    cached = [CachedAskEnvelope.model_validate_json(value) for value in redis.values.values()]
    assert all(envelope.answer == _FALLBACK and envelope.citations == [] for envelope in cached), (
        "an unchecked answer reached the cache"
    )


async def test_generate_retry_refusal_blocks_the_second_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec: generate.max_calls=1 — a transient failure gets no second provider
    call through; the retry is refused like any other over-budget call."""
    _fast_retry_waits(monkeypatch)
    llm = _TransientGenerateLLM(failures=1)
    tracker = _tracker(**{STAGE_GENERATE: StageBudget(latency_ms=180_000, max_calls=1)})
    graph = build_retrieval_graph()
    result = await graph.ainvoke(
        _input(),
        context=RetrievalGraphContext(
            llm=llm,  # ty: ignore[invalid-argument-type]
            repo=_RecordingRepo(),  # ty: ignore[invalid-argument-type]
            reranker=_PassThroughReranker(),
            budgets=tracker,
        ),
    )

    assert llm.generate_calls == 1  # the retry attempt never reached the provider
    assert llm.generated is False
    assert result["generated_answer"].answer == _FALLBACK
    budget_warnings = _budget_warnings(result)
    assert [(w.stage, w.severity) for w in budget_warnings] == [(STAGE_GENERATE, "warning")]


async def test_generate_retry_performs_exactly_the_budgeted_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec: budget for 2 calls and 3 retry attempts performs exactly 2 — the
    third attempt's charge is refused and the ask degrades to the fallback."""
    _fast_retry_waits(monkeypatch)
    llm = _TransientGenerateLLM(failures=2)
    tracker = _tracker(**{STAGE_GENERATE: StageBudget(latency_ms=180_000, max_calls=2)})
    graph = build_retrieval_graph()
    result = await graph.ainvoke(
        _input(),
        context=RetrievalGraphContext(
            llm=llm,  # ty: ignore[invalid-argument-type]
            repo=_RecordingRepo(),  # ty: ignore[invalid-argument-type]
            reranker=_PassThroughReranker(),
            budgets=tracker,
        ),
    )

    assert llm.generate_calls == 2
    assert llm.generated is False  # the third (refused) attempt never ran
    assert result["generated_answer"].answer == _FALLBACK
    budget_warnings = _budget_warnings(result)
    assert [(w.stage, w.severity) for w in budget_warnings] == [(STAGE_GENERATE, "warning")]


async def test_over_budget_post_process_passes_fused_order_through_unspent() -> None:
    """A7: the advertised post_process budget is now charged like every stage.

    A refused charge must NOT silently trim context with no trace: the fused
    order passes through untouched, the violation names the stage, and the
    usage delta records the unspent zero.
    """
    tracker = _tracker(**{STAGE_POST_PROCESS: StageBudget(latency_ms=30_000, max_calls=0)})
    graph = build_retrieval_graph()
    result = await graph.ainvoke(
        _input(),
        context=RetrievalGraphContext(
            llm=_ScriptedLLM(),  # ty: ignore[invalid-argument-type]
            repo=_RecordingRepo(),  # ty: ignore[invalid-argument-type]
            reranker=_PassThroughReranker(),
            budgets=tracker,
        ),
    )

    usage = [entry for entry in result["stage_usage"] if entry.stage == STAGE_POST_PROCESS]
    assert usage == [StageUsage(stage=STAGE_POST_PROCESS, calls=0)]
    budget_warnings = [
        warning
        for warning in result["quality_warnings"]
        if warning.code == "STAGE_BUDGET_EXHAUSTED"
    ]
    assert any(w.stage == STAGE_POST_PROCESS for w in budget_warnings)
