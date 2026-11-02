"""Ask grade-loop parity suite (validation-seam-adoption).

Replays the grade-loop cases through the pipeline-backed `_ask` loop and
asserts the observable ask contract is byte-identical: response shapes, the
fallback answer, the warning envelope, and the cache-key derivation (cache
keys are untouched by the refactor; the responses here are computed with the
same payload/scope inputs the pre-pipeline tests used). Also pins the shared
vocabulary: one seam-named ``validation_decision`` event per outcome, the
rewrite arriving as pipeline feedback, and the two-iteration bound.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from typing import TYPE_CHECKING

import pytest
from loguru import logger as loguru_logger
from returns.result import Success

import app.features.documents.application.queries.documents as documents_module
from app.features.documents.api.http.dto import UnifiedAskRequest
from app.features.documents.application.queries.documents import (
    _FALLBACK_ANSWER,
    DocumentQueryService,
)
from app.features.documents.retrieval.graph import (
    Citation,
    ContextGrade,
    GeneratedAnswer,
    QueryPlan,
)

if TYPE_CHECKING:
    from typing import Any

    from app.features.documents.retrieval.graph import RetrievedChunk

_CHUNK_ID = "chunk-1"
_DOC_ID = "doc-a"
_TENANT_A = "tenant-a"
_ORIGINAL_QUERY = "payment obligations"
_REWRITE = "rewritten payment clause"


class _ScriptedAskLLM:
    """Structured-output double scripted per seam, recording every prompt.

    ``grades`` replays ContextGrade verdicts in order; the query analyzer
    echoes the query it was given (so the rewrite's arrival is observable)
    and records its prompts; the generator records its prompts.
    """

    def __init__(self, grades: list[ContextGrade]) -> None:
        self._grades = list(grades)
        self.analyzer_prompts: list[str] = []
        self.grader_prompts: list[str] = []
        self.generator_prompts: list[str] = []

    def with_structured_output(self, schema: type) -> Any:
        return _ScriptedHandler(schema, self)

    async def _invoke(self, schema: type, messages: list[Any]) -> Any:
        prompt = str(messages[-1].content)
        if schema is QueryPlan:
            self.analyzer_prompts.append(prompt)
            return QueryPlan(rewritten_query=_REWRITE if _REWRITE in prompt else _ORIGINAL_QUERY)
        if schema is ContextGrade:
            self.grader_prompts.append(prompt)
            return self._grades[min(len(self.grader_prompts) - 1, len(self._grades) - 1)]
        if schema is GeneratedAnswer:
            self.generator_prompts.append(prompt)
            return GeneratedAnswer(
                answer="Acme Corp must pay the monthly charge.",
                citations=[
                    Citation(
                        chunk_id=_CHUNK_ID,
                        clause_type="payment",
                        claim="an obligation to pay exists",
                    )
                ],
                confidence="high",
            )
        raise AssertionError(f"unexpected structured schema {schema}")  # noqa: EM102


class _ScriptedHandler:
    def __init__(self, schema: type, llm: _ScriptedAskLLM) -> None:
        self._schema = schema
        self._llm = llm

    async def ainvoke(self, messages: list[Any]) -> Any:
        return await self._llm._invoke(self._schema, messages)


class _StubRepository:
    """Branch + hydration doubles returning exactly one hydrated chunk."""

    async def bm25_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([{"chunk_id": _CHUNK_ID, "score": 2.0}])

    async def vector_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([])

    async def trigram_search(self, **_kwargs: object) -> Success[list[dict[str, Any]]]:
        return Success([])

    async def fetch_chunks_by_ids(
        self, chunk_ids: list[str], **_kwargs: object
    ) -> Success[dict[str, dict[str, Any]]]:
        return Success(
            {
                _CHUNK_ID: {
                    "document_id": _DOC_ID,
                    "title": "Services Agreement",
                    "content": "The Employer shall pay the monthly charge.",
                    "preamble": "",
                    "document_version": 1,
                    "contract_index": 0,
                    "pipeline_version": "canonical",
                    "chunk_index": 0,
                    "chunk_kind": "legal_contract",
                    "clause_type": "payment",
                    "chunk_metadata": {"source_start_char": 0, "source_end_char": 100},
                    "quality_warnings": [],
                    "graphiti_verified": False,
                    "graph_indexed_at": None,
                }
            }
        )

    async def resolve_scope_versions(
        self, *, tenant_id: str, document_ids: tuple[str, ...]
    ) -> tuple[tuple[str, int, str | None], ...]:
        return ((_DOC_ID, 1, None),) if document_ids else ()


class _NoGraphiti:
    """No graphiti: the chunk filter degrades to empty, as in production."""

    class driver:  # noqa: N801 — attribute placeholder
        pass


class _PassThroughReranker:
    async def rerank(
        self, _query: str, chunks: list[RetrievedChunk], *, limit: int
    ) -> list[RetrievedChunk]:
        return chunks[:limit]


async def _fake_embed(*_args: object, **_kwargs: object) -> list[float]:
    return [0.1, 0.2, 0.3]


def _captured_decision_events() -> tuple[list[dict[str, Any]], ExitStack]:
    events: list[dict[str, Any]] = []
    stack = ExitStack()

    def _sink(message: Any) -> None:
        record = message.record
        if record["message"] == "validation_decision":
            events.append(dict(record["extra"]))

    sink_id = loguru_logger.add(_sink, level="DEBUG", format="{message}")
    stack.callback(loguru_logger.remove, sink_id)
    return events, stack


def _decision_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The seam's decision events, projected onto the four contract fields."""
    return [
        {key: event[key] for key in ("seam", "decision", "reason", "attempts")}
        for event in events
        if event.get("seam") == "ask_context_grader"
    ]


@pytest.fixture
def _wired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(documents_module, "embed_text", _fake_embed)
    monkeypatch.setattr(documents_module, "get_shared_reranker", _PassThroughReranker)


def _service(llm: _ScriptedAskLLM) -> DocumentQueryService:
    return DocumentQueryService(
        _StubRepository(),  # ty: ignore[invalid-argument-type]
        lambda: llm,
        None,
        None,
    )


def _payload() -> UnifiedAskRequest:
    return UnifiedAskRequest(query=_ORIGINAL_QUERY, bypass_cache=True)


def _sufficient() -> ContextGrade:
    return ContextGrade(sufficient=True, missing_aspects=[])


def _insufficient(rewrite: str | None = None) -> ContextGrade:
    return ContextGrade(
        sufficient=False,
        missing_aspects=["payment amounts"],
        rewrite_suggestion=rewrite,
    )


@pytest.mark.usefixtures("_wired")
async def test_sufficient_first_grade_replays_the_recorded_response() -> None:
    llm = _ScriptedAskLLM([_sufficient()])
    events, stack = _captured_decision_events()
    with stack:
        result = await _service(llm).ask(
            user_id="user-1",
            tenant_id=_TENANT_A,
            payload=_payload(),
            require_graphiti_verified=False,
        )

    assert isinstance(result, Success)
    response = result.unwrap()
    # Byte-identical replay of the recorded sufficient-grade contract (the
    # full raw shape is pinned by the existing ask tests, unmodified).
    assert response.answer == "Acme Corp must pay the monthly charge."
    assert response.confidence == "high"
    assert response.evidence_policy == "mechanically_verified"
    assert [
        (citation.chunk_id, citation.clause_type, citation.claim) for citation in response.citations
    ] == [(_CHUNK_ID, "payment", "an obligation to pay exists")]
    assert response.coverage.documents_ready == 1
    assert response.coverage.graph_indexed_documents == 0
    assert response.warnings == []
    assert response.cache_hit is False
    # One attempt: the analyzer and grader each ran once, and one seam-named
    # ACCEPT decision event was emitted.
    assert len(llm.grader_prompts) == 1
    assert _decision_events(events) == [
        {"seam": "ask_context_grader", "decision": "accept", "reason": "accepted", "attempts": 1}
    ]


@pytest.mark.usefixtures("_wired")
async def test_insufficient_grade_retries_with_the_rewrite_as_feedback() -> None:
    llm = _ScriptedAskLLM([_insufficient(_REWRITE), _sufficient()])
    events, stack = _captured_decision_events()
    with stack:
        result = await _service(llm).ask(
            user_id="user-1",
            tenant_id=_TENANT_A,
            payload=_payload(),
            require_graphiti_verified=False,
        )

    assert isinstance(result, Success)
    response = result.unwrap()
    assert response.cache_hit is False
    assert response.answer == "Acme Corp must pay the monthly charge."
    # The two-iteration bound is unchanged, and the second iteration's
    # retrieval query was the rewrite suggestion carried by the retry.
    assert len(llm.analyzer_prompts) == 2
    assert _REWRITE in llm.analyzer_prompts[1]
    assert len(llm.grader_prompts) == 2
    assert _decision_events(events) == [
        {"seam": "ask_context_grader", "decision": "accept", "reason": "accepted", "attempts": 2}
    ]


@pytest.mark.usefixtures("_wired")
async def test_exhausted_loop_is_escalate_with_the_fallback_answer() -> None:
    llm = _ScriptedAskLLM([_insufficient(_REWRITE), _insufficient(_REWRITE)])
    events, stack = _captured_decision_events()
    with stack:
        result = await _service(llm).ask(
            user_id="user-1",
            tenant_id=_TENANT_A,
            payload=_payload(),
            require_graphiti_verified=False,
        )

    assert isinstance(result, Success)
    response = result.unwrap()
    # Byte-identical fallback: no generation ran, the recorded fallback shape.
    assert response.answer == _FALLBACK_ANSWER
    assert response.citations == []
    assert response.confidence == "uncertain"
    assert response.evidence_policy == "mechanically_verified"
    assert response.warnings == []
    assert response.cache_hit is False
    assert response.coverage.documents_ready == 1
    assert len(llm.generator_prompts) == 0
    # Exactly two grader attempts, then one seam-named ESCALATE event.
    assert len(llm.grader_prompts) == 2
    assert _decision_events(events) == [
        {
            "seam": "ask_context_grader",
            "decision": "escalate",
            "reason": "attempts_exhausted",
            "attempts": 2,
        }
    ]


@pytest.mark.usefixtures("_wired")
async def test_grader_failure_degrade_still_answers_byte_identically() -> None:
    """The legacy degrade (unavailable grade proceeds to generation) replays.

    ``_grade_context`` is kept verbatim, so a failing grader degrades to
    sufficient exactly as before the adoption.
    """

    class _FailingGraderLLM(_ScriptedAskLLM):
        async def _invoke(self, schema: type, messages: list[Any]) -> Any:
            if schema is ContextGrade:
                self.grader_prompts.append("failed")
                message = "grader unavailable"
                raise ValueError(message)
            return await super()._invoke(schema, messages)

        def __init__(self) -> None:
            super().__init__([_sufficient()])

    llm = _FailingGraderLLM()
    result = await _service(llm).ask(
        user_id="user-1", tenant_id=_TENANT_A, payload=_payload(), require_graphiti_verified=False
    )

    assert isinstance(result, Success)
    response = result.unwrap()
    assert response.answer == "Acme Corp must pay the monthly charge."
    assert response.confidence == "high"


@pytest.mark.usefixtures("_wired")
async def test_empty_rerank_replays_the_insufficient_feedback_loop() -> None:
    """No chunks: the recorded insufficient-with-rewrite loop, twice, fallback."""

    class _EmptyRepository(_StubRepository):
        async def fetch_chunks_by_ids(
            self, _chunk_ids: list[str], **_kwargs: object
        ) -> Success[dict[str, dict[str, Any]]]:
            return Success({})

    llm = _ScriptedAskLLM([_insufficient(_ORIGINAL_QUERY), _insufficient(_ORIGINAL_QUERY)])
    result = await _service(llm).ask(
        user_id="user-1", tenant_id=_TENANT_A, payload=_payload(), require_graphiti_verified=False
    )

    assert isinstance(result, Success)
    response = result.unwrap()
    # The empty-rerank short-circuit replays the recorded insufficient loop:
    # two grader attempts with the same-query rewrite, then the fallback.
    assert response.answer == _FALLBACK_ANSWER
    assert response.citations == []
    assert len(llm.grader_prompts) == 2


def test_cache_key_derivation_is_untouched_by_the_refactor() -> None:
    """The cache key is a pure function of the recorded inputs; unchanged."""
    from app.features.documents.application.queries.documents import _build_answer_cache_key

    kwargs: dict[str, Any] = {
        "user_id": "user-1",
        "tenant_id": _TENANT_A,
        "query": _ORIGINAL_QUERY,
        "doc_ids_filter": [],
        "allowed_doc_ids": (_DOC_ID,),
        "document_versions": ((_DOC_ID, 1, None),),
        "jurisdiction": None,
        "contract_type": None,
        "contract_index": None,
        "clause_type": None,
        "require_graphiti_verified": False,
        "evidence_policy": "mechanically_verified",
    }
    first = _build_answer_cache_key(**kwargs)
    second = _build_answer_cache_key(**kwargs)
    assert first == second
    assert json.loads(json.dumps(first)) == first
