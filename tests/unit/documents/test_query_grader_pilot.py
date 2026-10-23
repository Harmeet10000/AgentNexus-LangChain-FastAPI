"""Grader-path pilot tests: `_grade_context` consumes the structured seam.

Item 274 reopen condition. The legacy inline path stays exercised too, because
`grader=None` (seam disabled at lifespan) must replay the previous behavior
verbatim — including the parse-failure degrade that proceeds with
``sufficient=True`` so an unwired optional capability cannot turn `ask` red.
"""

from collections import deque
from types import SimpleNamespace
from typing import Any

from returns.result import Failure, Success

from app.ai.generation import (
    GeneratedStructured,
    TextGenerationRequest,
    TextGenerationStructuredResult,
    TextGenerationUnavailableError,
)
from app.features.documents.retrieval_graph import ContextGrade, RetrievedChunk
from app.features.documents.service.query import (
    _CONTEXT_GRADER_SYSTEM_PROMPT,
    ContextGrader,
    _grade_context,
)


class _RecordingGrader:
    """Structured-seam double matching the service's narrow ContextGrader shape."""

    def __init__(self, results: Any) -> None:
        self._results = deque(results)
        self.requests: list[TextGenerationRequest] = []

    async def generate_structured(
        self, request: TextGenerationRequest
    ) -> TextGenerationStructuredResult:
        self.requests.append(request)
        if self._results:
            return self._results.popleft()
        return Failure(TextGenerationUnavailableError())


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        chunk_text="Tenant obligations under the payment clause.",
        preamble="",
        clause_type="payment",
        parent_doc_id="doc-1",
        metadata_={},
        custom_metadata={},
        score=1.0,
    )


async def test_grade_context_routes_through_the_structured_seam() -> None:
    grade = ContextGrade(sufficient=False, missing_aspects=["termination"], rewrite_suggestion="q'")
    grader = _RecordingGrader([Success(GeneratedStructured(output=grade))])

    result = await _grade_context(
        grader_llm=SimpleNamespace(),
        grader=grader,  # type: ignore[arg-type]
        query="payment terms",
        reranked=[_chunk()],
    )

    assert result == grade
    request = grader.requests[0]
    assert request.response_schema is ContextGrade
    assert request.system == _CONTEXT_GRADER_SYSTEM_PROMPT
    assert "payment terms" in request.prompt  # the TOON payload, not a message object
    assert request.temperature == 0.1  # parity with the legacy grader (0.1 in dependencies.py), not the 0.7 default


async def test_grade_context_degrades_like_the_legacy_parse_failure() -> None:
    grader = _RecordingGrader([Failure(TextGenerationUnavailableError())])

    result = await _grade_context(
        grader_llm=SimpleNamespace(),
        grader=grader,  # type: ignore[arg-type]
        query="payment terms",
        reranked=[_chunk()],
    )

    assert result == ContextGrade(sufficient=True, missing_aspects=[])


async def test_grade_context_falls_back_to_the_legacy_path_when_the_seam_is_absent() -> None:
    grade = ContextGrade(sufficient=False, missing_aspects=["renewal"])

    class _LegacyGraderLLM:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        async def ainvoke(self, messages: Any) -> ContextGrade:
            self.calls.append(messages)
            return grade

    legacy_llm = _LegacyGraderLLM()

    result = await _grade_context(
        grader_llm=legacy_llm,
        grader=None,
        query="payment terms",
        reranked=[_chunk()],
    )

    assert result == grade
    assert len(legacy_llm.calls) == 1


async def test_grade_context_empty_rerank_short_circuits_without_any_generator_call() -> None:
    grader = _RecordingGrader([])

    result = await _grade_context(
        grader_llm=SimpleNamespace(),
        grader=grader,  # type: ignore[arg-type]
        query="payment terms",
        reranked=[],
    )

    assert result.sufficient is False
    assert result.missing_aspects == ["No retrieved chunks"]
    assert grader.requests == []


async def test_recording_grader_satisfies_the_runtime_protocol_shape() -> None:
    grader = _RecordingGrader([])

    assert isinstance(grader, ContextGrader)
