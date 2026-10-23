"""Deterministic in-memory Adapter for the text-generation Seam."""

from collections import deque
from collections.abc import Iterable

from pydantic import BaseModel
from returns.result import Failure, Success

from .exceptions import TextGenerationUnavailableError
from .models import TextGenerationRequest
from .protocol import TextGenerationResult, TextGenerationStructuredResult


class InMemoryTextGenerator:
    """Replay typed outcomes while recording requests made through the Seam.

    Structured scripts are validated against the request's ``response_schema``
    when the call arrives, so a double scripted with the wrong value object
    fails its own test loudly instead of leaking a shape mismatch downstream.
    """

    def __init__(
        self,
        scripted_results: Iterable[TextGenerationResult] = (),
        structured_results: Iterable[TextGenerationStructuredResult] = (),
    ) -> None:
        self._scripted_results = deque(scripted_results)
        self._structured_results = deque(structured_results)
        self._calls: list[TextGenerationRequest] = []

    @property
    def calls(self) -> tuple[TextGenerationRequest, ...]:
        """Return an immutable snapshot of requests observed by the Adapter."""
        return tuple(self._calls)

    async def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        self._calls.append(request)
        if self._scripted_results:
            return self._scripted_results.popleft()
        return Failure(TextGenerationUnavailableError())

    async def generate_structured(
        self, request: TextGenerationRequest
    ) -> TextGenerationStructuredResult:
        self._calls.append(request)
        schema = request.response_schema
        if schema is None:
            msg = "generate_structured requires request.response_schema"
            raise RuntimeError(msg)
        if self._structured_results:
            result = self._structured_results.popleft()
            if isinstance(result, Success):
                output: BaseModel = result.unwrap().output
                if not isinstance(output, schema):
                    msg = (
                        "scripted structured output does not match the request schema: "
                        f"expected {schema.__name__}, got {type(output).__name__}"
                    )
                    raise TypeError(msg)
            return result
        return Failure(TextGenerationUnavailableError())
