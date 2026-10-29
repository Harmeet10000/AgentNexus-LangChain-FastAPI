"""Production text-generation Adapter backed by a LangChain chat model.

The public Seam deliberately exposes only prompt text, an optional system
message, and (for the structured method) the response schema. Model selection,
retry policy, breaker identity, and provider response parsing all live behind
this Adapter.  In particular, the breaker sees one logical provider request:
the bounded retry sequence is inside the breaker and parsing is outside it.

Schema-bound generation goes through ``with_structured_output(...,
include_raw=True)``: the raw provider message stays attached to the parse so
the Adapter can (a) map a parser failure to ``PROTOCOL_ERROR`` rather than
letting an ``OutputParserException`` masquerade as provider rejection, and
(b) capture token usage off the raw message.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, override

import anthropic
import httpx
import openai
from google.api_core import exceptions as google_exceptions
from langchain_core.exceptions import LangChainException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict
from returns.result import Failure, Success

from app.resilience.circuit_breaker import (
    CircuitBreakerBackendError,
    CircuitBreakerOpenError,
)

from .exceptions import (
    TextGenerationProtocolError,
    TextGenerationRejectedError,
    TextGenerationUnavailableError,
)
from .models import GeneratedStructured, GeneratedText
from .protocol import TextGenerator

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage

    from app.resilience.circuit_breaker import CircuitBreakerService

    from .exceptions import TextGenerationError
    from .models import TextGenerationRequest
    from .protocol import TextGenerationResult, TextGenerationStructuredResult

type ModelFactory = Callable[[str | None, float | None], BaseChatModel]

_BREAKER_NAME = "text-generation-provider"
_MAX_ATTEMPTS = 3
_INITIAL_BACKOFF_SECONDS = 0.25
_MAX_BACKOFF_SECONDS = 2.0
_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_MISSING = object()
_PROVIDER_EXCEPTION_TYPES = (
    httpx.HTTPError,
    anthropic.APIError,
    google_exceptions.GoogleAPIError,
    LangChainException,
    openai.APIError,
)
_TRANSIENT_PROVIDER_TYPES = (
    httpx.TransportError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.RateLimitError,
    google_exceptions.DeadlineExceeded,
    google_exceptions.ServiceUnavailable,
    google_exceptions.TooManyRequests,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.RateLimitError,
)


class _RawOutcome(BaseModel):
    """Private result of the provider call before parsing or error translation."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    value: object | None = None
    error: Exception | None = None
    unexpected: bool = False


class _ExhaustedProviderError(Exception):
    """Internal marker: only this failure is allowed to cross the breaker."""

    def __init__(self, original: Exception, *, attempts: int) -> None:
        super().__init__(str(original))
        self.original = original
        self.attempts = attempts


class LangChainTextGenerator(TextGenerator):
    """Generate plain or schema-bound text through an injected model and breaker.

    The constructor is intentionally narrow.  Callers cannot select a model,
    alter retry budgets, or invent a breaker namespace per request.  A per-call
    model choice still has to reach the provider, so the Adapter accepts an
    optional ``model_factory``: the composition root (lifespan) receives a
    builder bound to the deployment's provider settings, and request-level
    ``temperature``/``model`` overrides go through it. Requests without
    overrides never touch the factory.
    """

    def __init__(
        self,
        *,
        model: BaseChatModel,
        breaker: CircuitBreakerService,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._model = model
        self._breaker = breaker
        self._model_factory = model_factory
        self._override_models: dict[tuple[str | None, float | None], BaseChatModel] = {}

    @override
    async def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        """Return generated text or a typed expected provider failure."""
        messages = _messages(request)
        model = self._resolve_model(request)

        async def operation() -> _RawOutcome:
            return await _invoke_with_retries(lambda: model.ainvoke(messages))

        outcome = await self._call_within_breaker(operation)
        if isinstance(
            outcome,
            (
                TextGenerationUnavailableError,
                TextGenerationRejectedError,
                TextGenerationProtocolError,
            ),
        ):
            return Failure(outcome)

        input_tokens, output_tokens = _extract_usage(outcome.value)
        text = _extract_text(outcome.value)
        if text is None:
            return Failure(TextGenerationProtocolError())
        return Success(
            GeneratedText(text=text, input_tokens=input_tokens, output_tokens=output_tokens)
        )

    @override
    async def generate_structured(
        self, request: TextGenerationRequest
    ) -> TextGenerationStructuredResult:
        """Return the request's schema parsed from the provider output, or a typed failure."""
        schema = request.response_schema
        if schema is None:
            msg = "generate_structured requires request.response_schema"
            raise RuntimeError(msg)
        messages = _messages(request)
        model = self._resolve_model(request)
        structured = model.with_structured_output(schema, include_raw=True)

        async def operation() -> _RawOutcome:
            return await _invoke_with_retries(lambda: structured.ainvoke(messages))

        outcome = await self._call_within_breaker(operation)
        if isinstance(
            outcome,
            (
                TextGenerationUnavailableError,
                TextGenerationRejectedError,
                TextGenerationProtocolError,
            ),
        ):
            return Failure(outcome)

        value = outcome.value
        if not isinstance(value, Mapping):
            # No structured envelope: the provider produced nothing usable.
            return Failure(TextGenerationProtocolError())
        if value.get("parsing_error") is not None:
            # ``include_raw`` captures parser failures into the envelope, so a
            # response outside the schema is a wire-contract miss — a
            # protocol error, not provider unavailability or rejection.
            return Failure(TextGenerationProtocolError())
        parsed = value.get("parsed")
        if not isinstance(parsed, BaseModel):
            return Failure(TextGenerationProtocolError())
        input_tokens, output_tokens = _extract_usage(value.get("raw"))
        return Success(
            GeneratedStructured(
                output=parsed, input_tokens=input_tokens, output_tokens=output_tokens
            )
        )

    async def _call_within_breaker(
        self, operation: Callable[[], Awaitable[_RawOutcome]]
    ) -> _RawOutcome | TextGenerationError:
        """Run one logical provider request inside the breaker, translate failures.

        Returns the raw outcome on success, or an instance of the closed
        ``TextGenerationError`` union for every expected failure path.
        """
        try:
            outcome = await self._breaker.call(_BREAKER_NAME, operation)
        except CircuitBreakerOpenError:
            return TextGenerationUnavailableError()
        except CircuitBreakerBackendError:
            return TextGenerationUnavailableError()
        except _ExhaustedProviderError:
            return TextGenerationUnavailableError()

        if outcome.unexpected:
            # True application defects remain bugs.  They do not become a provider
            # failure and therefore never poison breaker state.
            if outcome.error is None:
                msg = "unexpected text-generation outcome had no exception"
                raise RuntimeError(msg)
            raise outcome.error
        if outcome.error is not None:
            return TextGenerationRejectedError()
        return outcome

    def _resolve_model(self, request: TextGenerationRequest) -> BaseChatModel:
        """Return the request's model, rebuilding one only for real overrides."""
        if request.temperature is None and request.model is None:
            return self._model
        if self._model_factory is None:
            # An override the Adapter cannot honor is a wiring bug: silently
            # running on the default model would change grading answers with no
            # signal at all. Fail loudly like any other local defect.
            msg = "request-level model override but the adapter was built without a model_factory"
            raise RuntimeError(msg)
        key = (request.model, request.temperature)
        override_model = self._override_models.get(key)
        if override_model is None:
            override_model = self._model_factory(request.model, request.temperature)
            self._override_models[key] = override_model
        return override_model


async def _invoke_with_retries(invoke: Callable[[], Awaitable[object]]) -> _RawOutcome:
    """Retry only named transient provider failures.

    Permanent provider failures are boxed so the existing breaker, which records
    every exception escaping its operation, does not mistake a deterministic
    rejection for provider unavailability.  Unexpected local defects are boxed
    for the same reason and re-raised by ``generate`` after breaker accounting.
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            value = await invoke()
            return _RawOutcome(value=value)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not _is_retryable_provider_failure(exc):
                return _RawOutcome(
                    error=exc,
                    unexpected=not _is_expected_provider_failure(exc),
                )
            if attempt == _MAX_ATTEMPTS:
                raise _ExhaustedProviderError(exc, attempts=attempt) from exc
            await asyncio.sleep(
                min(
                    _INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                    _MAX_BACKOFF_SECONDS,
                )
            )

    # The loop always returns or raises.  Keep the impossible state explicit so
    # a future edit cannot accidentally return None as a successful generation.
    msg = "text-generation retry loop ended without an outcome"
    raise AssertionError(msg)


def _messages(request: TextGenerationRequest) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    if request.system:
        messages.append(SystemMessage(content=request.system))
    messages.append(HumanMessage(content=request.prompt))
    return messages


def _status_code(exc: BaseException) -> int | None:
    if not isinstance(exc, _PROVIDER_EXCEPTION_TYPES):
        return None
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def _is_retryable_provider_failure(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT_PROVIDER_TYPES):
        return True
    return _status_code(exc) in _RETRYABLE_STATUS_CODES


def _is_expected_provider_failure(exc: BaseException) -> bool:
    """Recognize provider-shaped errors without relabelling local bugs."""
    return isinstance(exc, _PROVIDER_EXCEPTION_TYPES)


def _extract_text(value: object | None) -> str | None:
    """Normalize common LangChain content shapes without stringifying malformed data."""
    if isinstance(value, str):
        return value
    if value is None:
        return None

    content: object = getattr(value, "content", _MISSING)
    if content is _MISSING:
        return None
    return _content_text(content)


def _extract_usage(value: object | None) -> tuple[int | None, int | None]:
    """Read (input_tokens, output_tokens) off a LangChain message's usage metadata.

    Test doubles and older provider messages may carry none; absence stays
    ``None`` exactly as the value objects declare it.
    """
    if value is None:
        return (None, None)
    usage: object = getattr(value, "usage_metadata", _MISSING)
    if not isinstance(usage, Mapping):
        return (None, None)
    return (
        _token_count(usage.get("input_tokens")),
        _token_count(usage.get("output_tokens")),
    )


def _token_count(candidate: object) -> int | None:
    if isinstance(candidate, int) and not isinstance(candidate, bool):
        return candidate
    return None


def _content_text(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        text = content.get("text")
        return text if isinstance(text, str) else None
    if not isinstance(content, Sequence) or isinstance(content, bytes | bytearray):
        return None

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    result = "".join(parts)
    return result or None


__all__ = ["LangChainTextGenerator"]
