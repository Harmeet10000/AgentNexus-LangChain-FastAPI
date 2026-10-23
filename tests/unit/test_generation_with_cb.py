"""The example endpoint translates the provider-neutral generation result."""

from __future__ import annotations

import pytest
from pydantic import SecretStr
from returns.result import Failure, Success
from starlette.requests import Request

import app.examples.generation_with_cb as gen_mod
import app.lifecycle.lifespan as lifespan_module
from app.ai.generation import (
    GeneratedText,
    InMemoryTextGenerator,
    TextGenerationProtocolError,
    TextGenerationRequest,
)
from app.utils import ServiceUnavailableException


async def test_provider_failure_translates_to_service_unavailable() -> None:
    generator = InMemoryTextGenerator([Failure(TextGenerationProtocolError())])

    with pytest.raises(ServiceUnavailableException, match="External API call failed"):
        await gen_mod.generate_text(generator)
    assert generator.calls == (TextGenerationRequest(prompt="Generate text."),)


def test_dependency_reads_the_lifespan_owned_generator() -> None:
    generator = InMemoryTextGenerator()
    app = type("App", (), {"state": type("State", (), {"text_generator": generator})()})()
    request = Request({"type": "http", "app": app})

    assert gen_mod.get_text_generator(request) is generator


def test_dependency_rejects_an_unwired_generator() -> None:
    app = type("App", (), {"state": type("State", (), {})()})()
    request = Request({"type": "http", "app": app})

    with pytest.raises(ServiceUnavailableException, match="Text generation provider"):
        gen_mod.get_text_generator(request)


def test_lifespan_disables_text_generation_without_redis() -> None:
    app = type("App", (), {"state": type("State", (), {"redis": None})()})()
    settings = type(
        "Settings",
        (),
        {"GEMINI_API_KEY": SecretStr("configured"), "GEMINI_PRO_MODEL": "gemini-test"},
    )()

    lifespan_module._setup_text_generator_state(app, settings)

    assert app.state.text_generator is None


def test_lifespan_disables_text_generation_without_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_client = object()
    app = type("App", (), {"state": type("State", (), {"redis": redis_client})()})()
    settings = type(
        "Settings",
        (),
        {"GEMINI_API_KEY": SecretStr(""), "GEMINI_PRO_MODEL": "gemini-test"},
    )()

    def fail_if_constructed(**_kwargs: object) -> object:
        pytest.fail("provider construction must not run without an API key")

    monkeypatch.setattr(lifespan_module, "build_chat_model", fail_if_constructed)
    lifespan_module._setup_text_generator_state(app, settings)

    assert app.state.text_generator is None


def test_lifespan_degrades_on_expected_provider_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RedisStub:
        def register_script(self, _script: str) -> object:
            return object()

    app = type("App", (), {"state": type("State", (), {"redis": RedisStub()})()})()
    settings = type(
        "Settings",
        (),
        {"GEMINI_API_KEY": SecretStr("configured"), "GEMINI_PRO_MODEL": "gemini-test"},
    )()

    def reject_configuration(**_kwargs: object) -> object:
        msg = "provider configuration is invalid"
        raise ValueError(msg)

    monkeypatch.setattr(lifespan_module, "build_chat_model", reject_configuration)
    lifespan_module._setup_text_generator_state(app, settings)

    assert app.state.text_generator is None


async def test_success_returns_generated_text() -> None:
    generator = InMemoryTextGenerator([Success(GeneratedText(text="hi"))])

    result = await gen_mod.generate_text(generator)
    assert result == {"generated_text": "hi"}
