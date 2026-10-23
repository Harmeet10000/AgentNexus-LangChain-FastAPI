"""Behavior-first tests for the provider-neutral text-generation Seam."""

import pytest
from pydantic import BaseModel, Field, ValidationError
from returns.result import Failure, Success

from app.ai.generation import (
    GeneratedStructured,
    GeneratedText,
    InMemoryTextGenerator,
    TextGenerationRequest,
    TextGenerationUnavailableError,
    TextGenerator,
)


def test_request_and_output_are_immutable_value_objects() -> None:
    request = TextGenerationRequest(prompt="hello", system="be concise")
    output = GeneratedText(text="hi")

    assert request.prompt == "hello"
    assert request.system == "be concise"
    assert output.text == "hi"

    with pytest.raises(ValidationError):
        request.prompt = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        output.text = "changed"  # type: ignore[misc]


async def test_in_memory_success_uses_only_the_text_generator_interface() -> None:
    generator: TextGenerator = InMemoryTextGenerator([Success(GeneratedText(text="hi"))])

    result = await generator.generate(TextGenerationRequest(prompt="hello", system="be concise"))

    assert isinstance(result, Success)
    assert result.unwrap() == GeneratedText(text="hi")


async def test_in_memory_replays_typed_failures_without_provider_details() -> None:
    generator = InMemoryTextGenerator([Failure(TextGenerationUnavailableError())])

    result = await generator.generate(TextGenerationRequest(prompt="hello"))

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, TextGenerationUnavailableError)
    assert error.message == "Text generation provider is unavailable"
    assert error.details is None
    assert error.source == "text_generation"


async def test_in_memory_records_calls_without_changing_the_generator_interface() -> None:
    generator = InMemoryTextGenerator([Success(GeneratedText(text="hi"))])
    request = TextGenerationRequest(prompt="hello")

    await generator.generate(request)

    assert generator.calls == (request,)


async def test_in_memory_returns_typed_unavailable_when_script_is_exhausted() -> None:
    generator = InMemoryTextGenerator()

    result = await generator.generate(TextGenerationRequest(prompt="hello"))

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), TextGenerationUnavailableError)


class _Grade(BaseModel):
    sufficient: bool
    missing_aspects: list[str] = Field(default_factory=list)


class _Other(BaseModel):
    flag: bool


def test_structured_request_carries_schema_and_rejects_bad_overrides() -> None:
    request = TextGenerationRequest(
        prompt="hello", response_schema=_Grade, temperature=0.0, model="gemini-lite"
    )

    assert request.response_schema is _Grade
    assert request.temperature == pytest.approx(0.0)
    assert request.model == "gemini-lite"

    for bad_temperature in (-0.1, 2.1):
        with pytest.raises(ValidationError):
            TextGenerationRequest(prompt="hello", temperature=bad_temperature)
    with pytest.raises(ValidationError):
        TextGenerationRequest(prompt="hello", model="")
    # Free-text shape stays unchanged: every extension defaults to None.
    plain = TextGenerationRequest(prompt="hello", system="be concise")
    assert plain.response_schema is None
    assert plain.temperature is None
    assert plain.model is None


def test_outputs_carry_optional_usage_fields() -> None:
    text = GeneratedText(text="hi", input_tokens=3, output_tokens=9)
    structured = GeneratedStructured(output=_Grade(sufficient=True), input_tokens=3)

    assert text.input_tokens == 3
    assert text.output_tokens == 9
    assert structured.output.sufficient is True
    assert structured.output_tokens is None
    with pytest.raises(ValidationError):
        text.text = "changed"  # type: ignore[misc]


async def test_in_memory_structured_returns_validate_against_the_request_schema() -> None:
    grade = _Grade(sufficient=False, missing_aspects=["dep"])
    generator: TextGenerator = InMemoryTextGenerator(
        structured_results=[Success(GeneratedStructured(output=grade))]
    )

    result = await generator.generate_structured(
        TextGenerationRequest(prompt="hello", response_schema=_Grade)
    )

    assert isinstance(result, Success)
    assert result.unwrap().output is grade
    assert generator.calls[0].response_schema is _Grade


async def test_in_memory_structured_scripted_mismatch_fails_loudly() -> None:
    generator = InMemoryTextGenerator(
        structured_results=[Success(GeneratedStructured(output=_Other(flag=True)))]
    )

    with pytest.raises(TypeError, match="expected _Grade, got _Other"):
        await generator.generate_structured(
            TextGenerationRequest(prompt="hello", response_schema=_Grade)
        )


async def test_in_memory_structured_requires_a_schema() -> None:
    generator = InMemoryTextGenerator(
        structured_results=[Success(GeneratedStructured(output=_Grade(sufficient=True)))]
    )

    with pytest.raises(RuntimeError, match="response_schema"):
        await generator.generate_structured(TextGenerationRequest(prompt="hello"))


async def test_in_memory_structured_failure_replays_typed_outcome() -> None:
    generator = InMemoryTextGenerator(
        structured_results=[Failure(TextGenerationUnavailableError())]
    )

    result = await generator.generate_structured(
        TextGenerationRequest(prompt="hello", response_schema=_Grade)
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), TextGenerationUnavailableError)


async def test_in_memory_structured_exhausted_script_is_typed_unavailable() -> None:
    generator = InMemoryTextGenerator()

    result = await generator.generate_structured(
        TextGenerationRequest(prompt="hello", response_schema=_Grade)
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), TextGenerationUnavailableError)


async def test_runtime_protocol_holds_both_methods() -> None:
    generator = InMemoryTextGenerator()

    assert isinstance(generator, TextGenerator)
