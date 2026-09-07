"""Tests for request correlation and logging context boundaries."""

from __future__ import annotations

from app.middleware.server_middleware import RequestStateLoggingMiddleware
from app.utils import request_state


def test_correlation_id_accepts_bounded_safe_values() -> None:
    scope = {"headers": [(b"x-correlation-id", b"request-123_abc.def")]}

    assert RequestStateLoggingMiddleware._read_correlation_id(scope) == "request-123_abc.def"


def test_correlation_id_rejects_control_characters_and_oversized_values() -> None:
    control_scope = {"headers": [(b"x-correlation-id", b"request\n123")]}
    oversized_scope = {"headers": [(b"x-correlation-id", b"x" * 129)]}

    assert RequestStateLoggingMiddleware._read_correlation_id(control_scope) is None
    assert RequestStateLoggingMiddleware._read_correlation_id(oversized_scope) is None


def test_request_actor_updates_context_used_by_downstream_logs() -> None:
    from app.utils import set_request_actor

    token = request_state.set({"request_id": "request-1", "user_id": None})
    try:
        set_request_actor("user-42")
        assert request_state.get()["user_id"] == "user-42"
    finally:
        request_state.reset(token)
