"""Grader-path pilot tests: the graph's `_grade_context` consumes the structured seam.

Item 274 reopen condition, retargeted to the retrieval graph
(mount-retrieval-graph): the inline ask body was deleted and the grader is now
the `context_grader` node's verbatim port. The seam-absent path stays exercised
too, because `grader=None` (seam disabled at lifespan) must replay the previous
behavior verbatim — including the parse-failure degrade that proceeds with
``sufficient=True`` so an unwired optional capability cannot turn `ask` red.
"""

from collections import deque
from typing import Any

from returns.result import Failure

from app.ai.generation import (
    TextGenerationRequest,
    TextGenerationStructuredResult,
    TextGenerationUnavailableError,
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
