"""generation_with_cb classifies by name: provider failures trip the breaker,
project defects propagate unwrapped and never count toward the threshold."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
import pytest

import app.api.generation_with_cb as gen_mod
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any


class _RecordingBreaker:
    """Mimics CircuitBreakerService.protect semantics: any exception escaping
    the guarded block counts as a downstream failure."""

    def __init__(self) -> None:
        self.failures = 0

    @asynccontextmanager
    async def protect(self, *args: Any, **kwargs: Any) -> AsyncIterator[None]:
        try:
            yield
        except Exception:
            self.failures += 1
            raise


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeClient:
    def __init__(
        self, response: _FakeResponse | None = None, error: Exception | None = None
    ) -> None:
        self._response = response
        self._error = error

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    async def get(self, url: str) -> _FakeResponse:
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


async def test_provider_failure_trips_breaker() -> None:
    breaker = _RecordingBreaker()
    original = gen_mod.AsyncClient
    gen_mod.AsyncClient = lambda: _FakeClient(error=httpx.ConnectError("down"))  # type: ignore[assignment]
    try:
        with pytest.raises(ServiceUnavailableException):
            await gen_mod.generate_text(breaker)  # type: ignore[arg-type]
    finally:
        gen_mod.AsyncClient = original
    assert breaker.failures == 1


async def test_type_error_does_not_trip_breaker() -> None:
    """A TypeError in project code is not relabeled and trips nothing."""
    breaker = _RecordingBreaker()
    original = gen_mod.AsyncClient
    gen_mod.AsyncClient = lambda: _FakeClient(  # type: ignore[assignment]
        response=_FakeResponse(TypeError("project bug"))
    )
    try:
        with pytest.raises(TypeError, match="project bug"):
            await gen_mod.generate_text(breaker)  # type: ignore[arg-type]
    finally:
        gen_mod.AsyncClient = original
    assert breaker.failures == 0


async def test_success_returns_generated_text() -> None:
    breaker = _RecordingBreaker()
    original = gen_mod.AsyncClient
    gen_mod.AsyncClient = lambda: _FakeClient(response=_FakeResponse({"text": "hi"}))  # type: ignore[assignment]
    try:
        result = await gen_mod.generate_text(breaker)  # type: ignore[arg-type]
    finally:
        gen_mod.AsyncClient = original
    assert result == {"generated_text": "hi"}
    assert breaker.failures == 0
