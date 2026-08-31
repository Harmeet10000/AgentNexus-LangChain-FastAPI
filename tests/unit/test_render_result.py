"""4.3, 4.4 renderer tests — transport status, not just body."""

from enum import StrEnum
from typing import ClassVar

from fastapi import Response
from returns.result import Failure, Success

from app.shared.result.errors import ErrorKind, FeatureError
from app.shared.result.render import render_result


class DummyCode(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    VALIDATION = "VALIDATION"
    INFRA_RETRYABLE = "INFRA_RETRYABLE"
    INFRA_DEAD = "INFRA_DEAD"


class NotFoundErr(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[DummyCode] = DummyCode.NOT_FOUND
    retryable: ClassVar[bool] = False
    identifier: str | None = None


class InfraRetryableErr(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[DummyCode] = DummyCode.INFRA_RETRYABLE
    retryable: ClassVar[bool] = True


class InfraDeadErr(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[DummyCode] = DummyCode.INFRA_DEAD
    retryable: ClassVar[bool] = False


def test_failure_renders_real_http_status_not_200():
    # 4.3 — returning http_error directly would be 200 with success false; render must set transport status
    resp = Response()
    result = Failure(NotFoundErr(message="not found"))
    envelope = render_result(result, resp)
    assert resp.status_code == 404
    assert envelope.status_code == 404
    assert envelope.success is False
    assert envelope.error is not None
    assert envelope.error.code == "NOT_FOUND"
    # body status and transport agree
    assert envelope.status_code == resp.status_code


def test_infrastructure_retryable_vs_dead():
    resp1 = Response()
    render_result(Failure(InfraRetryableErr(message="transient")), resp1)
    assert resp1.status_code == 503

    resp2 = Response()
    render_result(Failure(InfraDeadErr(message="dead")), resp2)
    assert resp2.status_code == 500


def test_success_uses_success_status():
    resp = Response()
    envelope = render_result(Success({"id": "1"}), resp, message="created", success_status=201)
    assert resp.status_code == 201
    assert envelope.status_code == 201
    assert envelope.success is True
    assert envelope.data == {"id": "1"}


def test_failure_status_not_overridable():
    # 4.4 — renderer offers no param to force failure status; even if caller passes success_status 201, failure still 404
    resp = Response()
    envelope = render_result(Failure(NotFoundErr(message="not found")), resp, success_status=201)
    assert resp.status_code == 404
    assert envelope.status_code == 404
    # ensure signature has no status_code param (would be ambiguous)
    import inspect

    sig = inspect.signature(render_result)
    assert "success_status" in sig.parameters
    assert "status_code" not in sig.parameters
