from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING

import langextract as lx
import pytest
from pydantic import SecretStr, ValidationError

from app.extraction.langextract.service import (
    AsyncExtractionService,
    ExtractionFailed,
    ExtractionFailureCode,
    ExtractionRequest,
    ExtractionSucceeded,
    LangExtractClient,
)

if TYPE_CHECKING:
    from app.extraction.langextract.service import SyncExtractionProvider


def _settings(api_key: str) -> SimpleNamespace:
    return SimpleNamespace(LANGEXTRACT_API_KEY=SecretStr(api_key))


def test_extraction_request_rejects_unsupported_or_old_model() -> None:
    with pytest.raises(ValidationError):
        ExtractionRequest(
            text="text",
            prompt_description="prompt",
            model_id="gemini-2.5-flash",
        )


def test_settings_model_override_is_rejected_before_provider_creation() -> None:
    with pytest.raises(ValueError, match=r"gemini-3\.5-flash-lite"):
        AsyncExtractionService.from_settings(
            SimpleNamespace(
                LANGEXTRACT_API_KEY=SecretStr("api-key"),
                LANGEXTRACT_MODEL="gemini-2.5-flash",
            )
        )


def test_client_passes_the_configured_model_to_langextract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_extract(**kwargs: object) -> lx.data.AnnotatedDocument:
        captured.update(kwargs)
        return lx.data.AnnotatedDocument(text="text", extractions=[])

    monkeypatch.setattr(lx, "extract", fake_extract)
    client = LangExtractClient("api-key")

    documents = client.extract(ExtractionRequest(text="text", prompt_description="prompt"))

    assert documents
    assert captured["model_id"] == "gemini-3.5-flash-lite"


async def test_placeholder_key_is_an_unconfigured_provider() -> None:
    service = AsyncExtractionService.from_settings(_settings("empty-langextract-api-key"))

    outcome = await service.extract(ExtractionRequest(text="text", prompt_description="prompt"))

    assert service.configured is False
    assert isinstance(outcome, ExtractionFailed)
    assert outcome.code is ExtractionFailureCode.UNCONFIGURED


async def test_configured_settings_model_is_forwarded_to_langextract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_extract(**kwargs: object) -> lx.data.AnnotatedDocument:
        captured.update(kwargs)
        return lx.data.AnnotatedDocument(text="text", extractions=[])

    monkeypatch.setattr(lx, "extract", fake_extract)
    service = AsyncExtractionService.from_settings(
        SimpleNamespace(
            LANGEXTRACT_API_KEY=SecretStr("api-key"),
            LANGEXTRACT_MODEL="gemini-3.5-flash-lite",
        )
    )

    outcome = await service.extract(ExtractionRequest(text="text", prompt_description="prompt"))

    assert isinstance(outcome, ExtractionSucceeded)
    assert captured["model_id"] == "gemini-3.5-flash-lite"


async def test_synchronous_provider_runs_outside_the_event_loop_thread() -> None:
    event_loop_thread = threading.get_ident()
    provider_thread: int | None = None

    class RecordingProvider:
        def extract(self, _request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
            nonlocal provider_thread
            provider_thread = threading.get_ident()
            return ()

    provider: SyncExtractionProvider = RecordingProvider()
    service = AsyncExtractionService(provider)

    outcome = await service.extract(ExtractionRequest(text="text", prompt_description="prompt"))

    assert asyncio.get_running_loop().is_running()
    assert provider_thread is not None
    assert provider_thread != event_loop_thread
    assert isinstance(outcome, ExtractionSucceeded)
    assert outcome.documents == ()


async def test_provider_exception_becomes_a_typed_failure() -> None:
    provider_error = RuntimeError("provider unavailable")

    class FailingProvider:
        def extract(self, _request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
            raise provider_error

    service = AsyncExtractionService(FailingProvider())

    outcome = await service.extract(ExtractionRequest(text="text", prompt_description="prompt"))

    assert isinstance(outcome, ExtractionFailed)
    assert outcome.code is ExtractionFailureCode.PROVIDER_ERROR
    assert "provider unavailable" not in outcome.message
    assert any(
        "Traceback (most recent call last)" in note and "RuntimeError: provider unavailable" in note
        for note in provider_error.__notes__
    )
