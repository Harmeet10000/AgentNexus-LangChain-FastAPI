"""The external seam for provider-neutral text generation."""

from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from returns.result import Result

from .exceptions import TextGenerationError
from .models import GeneratedStructured, GeneratedText, TextGenerationRequest

type TextGenerationResult = Result[GeneratedText, TextGenerationError]
type TextGenerationStructuredResult = Result[GeneratedStructured, TextGenerationError]


@runtime_checkable
class TextGenerator(Protocol):
    """Asynchronous text-generation capabilities.

    Two methods, not one union contract: free text and schema-bound output have
    different successful payloads, and a separate method lets the type checker
    narrow each result at every call site without ``isinstance`` branching on
    Pydantic outputs.
    """

    def generate(self, request: TextGenerationRequest) -> Awaitable[TextGenerationResult]: ...

    def generate_structured(
        self, request: TextGenerationRequest
    ) -> Awaitable[TextGenerationStructuredResult]: ...
