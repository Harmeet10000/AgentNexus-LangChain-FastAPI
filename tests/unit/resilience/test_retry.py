"""Unit tests for the tenacity resilience helpers."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import httpx
import pytest
from returns.result import Failure, Success
from tenacity import wait_fixed

from app.kernel.errors import ErrorKind, FeatureError
from app.resilience.retry import (
    default_wait,
    is_transient,
    retry_result,
    transient_retry,
)

if TYPE_CHECKING:
    from returns.result import Result
    from tenacity.wait import wait_base

_BLIP = "blip"
_DOWN = "down"
_OUR_BUG = "our bug"


class _TestErrorCode(StrEnum):
    RETRYABLE = "test_retryable"
    FATAL = "test_fatal"


class _RetryableFeatureError(FeatureError):
    kind = ErrorKind.EXTERNAL_SERVICE
    code = _TestErrorCode.RETRYABLE
    retryable = True


class _FatalFeatureError(FeatureError):
    kind = ErrorKind.VALIDATION
    code = _TestErrorCode.FATAL


# --- is_transient ------------------------------------------------------------


def test_transport_failures_are_transient() -> None:
    assert is_transient(httpx.ConnectError(_BLIP)) is True
    assert is_transient(httpx.ReadTimeout("slow")) is True
    assert is_transient(ConnectionError("down")) is True


def test_deterministic_failures_are_not_transient() -> None:
    assert is_transient(ValueError(_OUR_BUG)) is False
    response = httpx.Response(418)
    request = httpx.Request("GET", "https://example.invalid")
    assert (
        is_transient(httpx.HTTPStatusError("418", request=request, response=response)) is False
    )


def test_feature_error_retryable_flag_is_honored() -> None:
    assert is_transient(_RetryableFeatureError(message="blip")) is True
    assert is_transient(_FatalFeatureError(message="no")) is False


# --- transient_retry decorator ------------------------------------------------


@pytest.mark.asyncio
async def test_transient_retry_retries_then_succeeds() -> None:
    calls = 0

    @transient_retry(wait=wait_fixed(0))
    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError(_BLIP)
        return "ok"

    assert await flaky() == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_transient_retry_reraises_original_type() -> None:
    @transient_retry(wait=wait_fixed(0))
    async def always_transient() -> None:
        raise httpx.ConnectError(_DOWN)

    with pytest.raises(httpx.ConnectError, match=_DOWN):
        await always_transient()


@pytest.mark.asyncio
async def test_transient_retry_fails_fast_on_deterministic_failure() -> None:
    calls = 0

    @transient_retry(wait=wait_fixed(0))
    async def deterministic() -> None:
        nonlocal calls
        calls += 1
        raise ValueError(_OUR_BUG)

    with pytest.raises(ValueError, match=_OUR_BUG):
        await deterministic()

    assert calls == 1


# --- retry_result ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_result_success_is_immediate() -> None:
    calls = 0

    async def op() -> Success[str]:
        nonlocal calls
        calls += 1
        return Success("ok")

    assert await retry_result(op, wait=wait_fixed(0)) == Success("ok")
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_result_retryable_failure_then_success() -> None:
    calls = 0

    async def op() -> Result[str, FeatureError]:
        nonlocal calls
        calls += 1
        if calls < 3:
            return Failure(_RetryableFeatureError(message=_BLIP))
        return Success("recovered")

    assert await retry_result(op, wait=wait_fixed(0)) == Success("recovered")
    assert calls == 3


@pytest.mark.asyncio
async def test_retry_result_non_retryable_failure_fails_fast() -> None:
    calls = 0
    fatal = _FatalFeatureError(message="deterministic")

    async def op() -> Result[str, FeatureError]:
        nonlocal calls
        calls += 1
        return Failure(fatal)

    outcome = await retry_result(op, wait=wait_fixed(0))
    assert isinstance(outcome, Failure)
    assert outcome.failure() is fatal
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_result_budget_exhaustion_returns_last_failure() -> None:
    calls = 0
    last = _RetryableFeatureError(message="still down")

    async def op() -> Result[str, FeatureError]:
        nonlocal calls
        calls += 1
        return Failure(last)

    outcome = await retry_result(op, wait=wait_fixed(0))
    assert isinstance(outcome, Failure)
    assert outcome.failure() is last
    assert calls == 3


# --- wait shape -----------------------------------------------------------------


def test_default_wait_is_jittered_and_bounded() -> None:
    wait: wait_base = default_wait()
    from tenacity import RetryCallState

    state = RetryCallState(None, fn=None, args=(), kwargs={})
    state.attempt_number = 5
    value = wait(retry_state=state)
    assert 0 <= value <= 5.0
