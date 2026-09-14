from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING

from pydantic import SecretStr

from app.shared.rag.langextract.service import (
    AsyncExtractionService,
    ExtractionFailed,
    ExtractionFailureCode,
    ExtractionRequest,
    ExtractionSucceeded,
)

if TYPE_CHECKING:
    import langextract as lx

    from app.shared.rag.langextract.service import SyncExtractionProvider


def _settings(api_key: str) -> SimpleNamespace:
    return SimpleNamespace(LANGEXTRACT_API_KEY=SecretStr(api_key))


async def test_placeholder_key_is_an_unconfigured_provider() -> None:
    service = AsyncExtractionService.from_settings(_settings("empty-langextract-api-key"))

    outcome = await service.extract(ExtractionRequest(text="text", prompt_description="prompt"))

    assert service.configured is False
    assert isinstance(outcome, ExtractionFailed)
    assert outcome.code is ExtractionFailureCode.UNCONFIGURED


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
    class FailingProvider:
        def extract(self, _request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
            msg = "provider unavailable"
            raise RuntimeError(msg)

    service = AsyncExtractionService(FailingProvider())

    outcome = await service.extract(ExtractionRequest(text="text", prompt_description="prompt"))

    assert isinstance(outcome, ExtractionFailed)
    assert outcome.code is ExtractionFailureCode.PROVIDER_ERROR
    assert "provider unavailable" not in outcome.message
