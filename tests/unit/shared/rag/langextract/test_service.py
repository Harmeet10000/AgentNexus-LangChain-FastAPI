from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import langextract as lx
import pytest
from pydantic import SecretStr, ValidationError

from app.extraction.langextract.service import (
    AsyncExtractionService,
    ExtractionBudget,
    ExtractionBudgetDimension,
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


async def test_provider_failure_preserves_attempt_usage_in_the_budget() -> None:
    provider_error = RuntimeError("provider unavailable")

    class FailingProvider:
        def extract(self, _request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
            raise provider_error

    budget = ExtractionBudget(
        max_model_calls=10,
        cost_per_1k_tokens_usd=2.0,
        model_calls=1,
        input_tokens=10,
        cost_estimate_usd=0.02,
    )
    outcome = await AsyncExtractionService(FailingProvider()).extract(
        ExtractionRequest(text="abcd", prompt_description="p", budget=budget)
    )

    assert isinstance(outcome, ExtractionFailed)
    assert outcome.budget.model_calls == 4
    assert outcome.budget.input_tokens == 13
    assert outcome.budget.output_tokens == 0
    assert outcome.budget.cost_estimate_usd == pytest.approx(0.026)


async def test_budget_exhaustion_is_typed_and_does_not_call_provider() -> None:
    called = False

    class Provider:
        def extract(self, _request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
            nonlocal called
            called = True
            return ()

    outcome = await AsyncExtractionService(Provider()).extract(
        ExtractionRequest(
            text="text",
            prompt_description="prompt",
            budget=ExtractionBudget(max_model_calls=0),
        )
    )

    assert isinstance(outcome, ExtractionFailed)
    assert outcome.code is ExtractionFailureCode.BUDGET_EXHAUSTED
    assert outcome.budget_dimension is ExtractionBudgetDimension.MODEL_CALLS
    assert called is False


async def test_extraction_passes_consume_model_call_budget() -> None:
    called = False

    class Provider:
        def extract(self, _request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
            nonlocal called
            called = True
            return ()

    outcome = await AsyncExtractionService(Provider()).extract(
        ExtractionRequest(
            text="text",
            prompt_description="prompt",
            extraction_passes=3,
            budget=ExtractionBudget(max_model_calls=2),
        )
    )

    assert isinstance(outcome, ExtractionFailed)
    assert outcome.code is ExtractionFailureCode.BUDGET_EXHAUSTED
    assert outcome.budget_dimension is ExtractionBudgetDimension.MODEL_CALLS
    assert called is False


async def test_wall_clock_budget_cancels_a_slow_provider() -> None:
    class SlowProvider:
        def extract(self, _request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
            time.sleep(0.05)
            return ()

    outcome = await AsyncExtractionService(SlowProvider()).extract(
        ExtractionRequest(
            text="text",
            prompt_description="prompt",
            budget=ExtractionBudget(wall_clock_seconds=0.001),
        )
    )

    assert isinstance(outcome, ExtractionFailed)
    assert outcome.code is ExtractionFailureCode.BUDGET_EXHAUSTED
    assert outcome.budget_dimension is ExtractionBudgetDimension.WALL_CLOCK
