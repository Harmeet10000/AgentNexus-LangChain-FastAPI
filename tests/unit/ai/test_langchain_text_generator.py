"""Contract tests for the production text-generation Adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx
import pytest
from pydantic import BaseModel, Field
from returns.result import Failure, Success

from app.ai.generation.langchain_adapter import LangChainTextGenerator
from app.ai.generation.models import GeneratedText, TextGenerationRequest

if TYPE_CHECKING:
    from typing import Any


class _FakeModel:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[list[object]] = []
        self.structured_schemas: list[object] = []

    async def ainvoke(self, messages: list[object]) -> object:
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def with_structured_output(self, schema: object, *, include_raw: bool) -> _StructuredRunnable:
        assert include_raw is True
        self.structured_schemas.append(schema)
        return _StructuredRunnable(self)


class _StructuredRunnable:
    """Scriptable stand-in for a model's `with_structured_output` runnable."""

    def __init__(self, model: _FakeModel) -> None:
        self._model = model

    async def ainvoke(self, messages: list[object]) -> object:
        self._model.calls.append(messages)
        response = self._model.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, Mapping):
            # A scripted envelope (e.g. a captured parsing failure) flows as-is.
            return response
        return {
            "raw": SimpleNamespace(
                content="raw", usage_metadata={"input_tokens": 11, "output_tokens": 7}
            ),
            "parsed": response,
            "parsing_error": None,
        }


class _RecordingBreaker:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.failures = 0
        self.successes = 0

    async def call(self, name: str, operation: Any) -> object:
        self.names.append(name)
        try:
            result = await operation()
        except Exception:
            self.failures += 1
            raise
        self.successes += 1
        return result


class _Grade(BaseModel):
    sufficient: bool
    missing_aspects: list[str] = Field(default_factory=list)


def _status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.test")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"provider returned {status_code}",
        request=request,
        response=response,
    )


async def _no_wait(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_retries_transient_provider_failure_and_counts_one_logical_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.ai.generation.langchain_adapter.asyncio.sleep",
        _no_wait,
    )
    model = _FakeModel(httpx.ConnectError("offline"), SimpleNamespace(content="ok"))
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate(TextGenerationRequest(prompt="hello", system="be concise"))

    assert isinstance(result, Success)
    assert result.unwrap() == GeneratedText(text="ok")
    assert len(model.calls) == 2
    assert breaker.names == ["text-generation-provider"]
    assert breaker.failures == 0
    assert breaker.successes == 1


@pytest.mark.asyncio
async def test_exhausted_transient_sequence_is_one_breaker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.ai.generation.langchain_adapter.asyncio.sleep",
        _no_wait,
    )
    model = _FakeModel(
        httpx.ConnectError("offline"),
        httpx.ConnectError("offline"),
        httpx.ConnectError("offline"),
    )
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate(TextGenerationRequest(prompt="hello"))

    assert isinstance(result, Failure)
    assert result.failure().message == "Text generation provider is unavailable"
    assert len(model.calls) == 3
    assert breaker.failures == 1
    assert breaker.successes == 0


@pytest.mark.asyncio
async def test_permanent_provider_failure_does_not_trip_breaker() -> None:
    response = httpx.Response(400, request=httpx.Request("POST", "https://provider.test"))
    error = httpx.HTTPStatusError("bad request", request=response.request, response=response)
    model = _FakeModel(error)
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate(TextGenerationRequest(prompt="hello"))

    assert isinstance(result, Failure)
    assert result.failure().message == "Text generation request was rejected"
    assert len(model.calls) == 1
    assert breaker.failures == 0
    assert breaker.successes == 1


@pytest.mark.asyncio
async def test_retries_503_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.ai.generation.langchain_adapter.asyncio.sleep",
        _no_wait,
    )
    model = _FakeModel(_status_error(503), SimpleNamespace(content="ok"))
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate(TextGenerationRequest(prompt="hello"))

    assert isinstance(result, Success)
    assert result.unwrap() == GeneratedText(text="ok")
    assert len(model.calls) == 2
    assert breaker.failures == 0
    assert breaker.successes == 1


@pytest.mark.asyncio
async def test_exhausted_503_is_one_breaker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.ai.generation.langchain_adapter.asyncio.sleep",
        _no_wait,
    )
    model = _FakeModel(_status_error(503), _status_error(503), _status_error(503))
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate(TextGenerationRequest(prompt="hello"))

    assert isinstance(result, Failure)
    assert result.failure().message == "Text generation provider is unavailable"
    assert len(model.calls) == 3
    assert breaker.failures == 1
    assert breaker.successes == 0


@pytest.mark.asyncio
async def test_retries_429_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.ai.generation.langchain_adapter.asyncio.sleep",
        _no_wait,
    )
    model = _FakeModel(_status_error(429), SimpleNamespace(content="ok"))
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate(TextGenerationRequest(prompt="hello"))

    assert isinstance(result, Success)
    assert result.unwrap() == GeneratedText(text="ok")
    assert len(model.calls) == 2
    assert breaker.failures == 0
    assert breaker.successes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ConnectionError("local"), TimeoutError("local")])
async def test_local_transport_named_errors_propagate_without_retry_or_breaker_failure(
    error: Exception,
) -> None:
    model = _FakeModel(error)
    breaker = _RecordingBreaker()

    with pytest.raises(type(error), match="local"):
        await LangChainTextGenerator(
            model=model,  # type: ignore[arg-type]
            breaker=breaker,  # type: ignore[arg-type]
        ).generate(TextGenerationRequest(prompt="hello"))

    assert len(model.calls) == 1
    assert breaker.failures == 0
    assert breaker.successes == 1


@pytest.mark.asyncio
async def test_malformed_content_is_parsed_after_breaker() -> None:
    model = _FakeModel(SimpleNamespace(content=[{"type": "image", "url": "x"}]))
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate(TextGenerationRequest(prompt="hello"))

    assert isinstance(result, Failure)
    assert result.failure().message == "Text generation provider returned an invalid response"
    assert breaker.failures == 0
    assert breaker.successes == 1


@pytest.mark.asyncio
async def test_system_message_precedes_human_message() -> None:
    model = _FakeModel(SimpleNamespace(content="ok"))
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate(TextGenerationRequest(prompt="hello", system="be concise"))

    assert isinstance(result, Success)
    assert [message.content for message in model.calls[0]] == ["be concise", "hello"]


@pytest.mark.asyncio
async def test_cancellation_propagates_and_is_not_counted() -> None:
    model = _FakeModel(asyncio.CancelledError())
    breaker = _RecordingBreaker()

    with pytest.raises(asyncio.CancelledError):
        await LangChainTextGenerator(
            model=model,  # type: ignore[arg-type]
            breaker=breaker,  # type: ignore[arg-type]
        ).generate(TextGenerationRequest(prompt="hello"))

    assert breaker.failures == 0
    assert breaker.successes == 0


@pytest.mark.asyncio
async def test_unexpected_local_bug_propagates_without_poisoning_breaker() -> None:
    model = _FakeModel(TypeError("local bug"))
    breaker = _RecordingBreaker()

    with pytest.raises(TypeError, match="local bug"):
        await LangChainTextGenerator(
            model=model,  # type: ignore[arg-type]
            breaker=breaker,  # type: ignore[arg-type]
        ).generate(TextGenerationRequest(prompt="hello"))

    assert breaker.failures == 0
    assert breaker.successes == 1


# ---------------------------------------------------------------------------
# Structured generation, usage capture, request-level model overrides
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_free_text_captures_usage_metadata_when_present() -> None:
    model = _FakeModel(
        SimpleNamespace(content="ok", usage_metadata={"input_tokens": 4, "output_tokens": 9})
    )
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate(TextGenerationRequest(prompt="hello"))

    assert isinstance(result, Success)
    assert result.unwrap() == GeneratedText(text="ok", input_tokens=4, output_tokens=9)


@pytest.mark.asyncio
async def test_structured_output_success_returns_parsed_schema_instance_with_usage() -> None:
    grade = _Grade(sufficient=True, missing_aspects=[])
    model = _FakeModel(grade)
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate_structured(TextGenerationRequest(prompt="hello", response_schema=_Grade))

    assert isinstance(result, Success)
    structured = result.unwrap()
    assert structured.output is grade
    assert structured.input_tokens == 11
    assert structured.output_tokens == 7
    assert model.structured_schemas == [_Grade]
    assert breaker.failures == 0
    assert breaker.successes == 1


@pytest.mark.asyncio
async def test_structured_parse_failure_is_a_protocol_error_not_rejection() -> None:
    model = _FakeModel({"raw": None, "parsed": None, "parsing_error": ValueError("invalid json")})
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate_structured(TextGenerationRequest(prompt="hello", response_schema=_Grade))

    assert isinstance(result, Failure)
    assert result.failure().message == "Text generation provider returned an invalid response"
    assert breaker.failures == 0
    assert breaker.successes == 1


@pytest.mark.asyncio
async def test_structured_retries_transient_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.ai.generation.langchain_adapter.asyncio.sleep",
        _no_wait,
    )
    model = _FakeModel(httpx.ConnectError("offline"), _Grade(sufficient=True))
    breaker = _RecordingBreaker()

    result = await LangChainTextGenerator(
        model=model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
    ).generate_structured(TextGenerationRequest(prompt="hello", response_schema=_Grade))

    assert isinstance(result, Success)
    assert len(model.calls) == 2
    assert breaker.failures == 0
    assert breaker.successes == 1


@pytest.mark.asyncio
async def test_structured_requires_a_schema() -> None:
    model = _FakeModel()
    breaker = _RecordingBreaker()

    with pytest.raises(RuntimeError, match="response_schema"):
        await LangChainTextGenerator(
            model=model,  # type: ignore[arg-type]
            breaker=breaker,  # type: ignore[arg-type]
        ).generate_structured(TextGenerationRequest(prompt="hello"))

    assert len(model.calls) == 0


@pytest.mark.asyncio
async def test_request_level_overrides_rebuild_through_the_injected_factory() -> None:
    override_model = _FakeModel(_Grade(sufficient=True), _Grade(sufficient=False))
    base_model = _FakeModel()
    built: list[tuple[str | None, float | None]] = []

    def factory(model_name: str | None, temperature: float | None) -> _FakeModel:
        built.append((model_name, temperature))
        return override_model

    breaker = _RecordingBreaker()
    generator = LangChainTextGenerator(
        model=base_model,  # type: ignore[arg-type]
        breaker=breaker,  # type: ignore[arg-type]
        model_factory=factory,  # type: ignore[arg-type]
    )

    first = await generator.generate_structured(
        TextGenerationRequest(
            prompt="hello", response_schema=_Grade, temperature=0.2, model="gemini-lite"
        )
    )
    second = await generator.generate_structured(
        TextGenerationRequest(
            prompt="again", response_schema=_Grade, temperature=0.2, model="gemini-lite"
        )
    )

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert built == [("gemini-lite", 0.2)]  # cached: one build serves both calls
    assert override_model.structured_schemas == [_Grade, _Grade]
    assert base_model.calls == []  # the default model is never touched


@pytest.mark.asyncio
async def test_overrides_without_a_factory_fail_loudly() -> None:
    model = _FakeModel(_Grade(sufficient=True))
    breaker = _RecordingBreaker()

    with pytest.raises(RuntimeError, match="model_factory"):
        await LangChainTextGenerator(
            model=model,  # type: ignore[arg-type]
            breaker=breaker,  # type: ignore[arg-type]
        ).generate_structured(
            TextGenerationRequest(prompt="hello", response_schema=_Grade, temperature=0.2)
        )
