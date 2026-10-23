"""Provider-neutral text-generation Module and its test Adapter."""

from .exceptions import (
    TextGenerationCode,
    TextGenerationError,
    TextGenerationProtocolError,
    TextGenerationRejectedError,
    TextGenerationUnavailableError,
)
from .in_memory import InMemoryTextGenerator
from .langchain_adapter import LangChainTextGenerator
from .models import GeneratedStructured, GeneratedText, TextGenerationRequest
from .protocol import (
    TextGenerationResult,
    TextGenerationStructuredResult,
    TextGenerator,
)

__all__ = [
    "GeneratedStructured",
    "GeneratedText",
    "InMemoryTextGenerator",
    "LangChainTextGenerator",
    "TextGenerationCode",
    "TextGenerationError",
    "TextGenerationProtocolError",
    "TextGenerationRejectedError",
    "TextGenerationRequest",
    "TextGenerationResult",
    "TextGenerationStructuredResult",
    "TextGenerationUnavailableError",
    "TextGenerator",
]
