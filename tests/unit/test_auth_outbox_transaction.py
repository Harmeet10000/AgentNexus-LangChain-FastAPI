"""Auth's cross-store email intent boundary owns a real SQL transaction."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from returns.result import Failure, Success
from sqlalchemy.exc import SQLAlchemyError

from app.features.auth.adapters.persistence.mongodb.user_store import AuthAccount
from app.features.auth.application.service import AuthService
from app.features.auth.domain.user import UserRole
from app.security.passwords import hash_token

if TYPE_CHECKING:
    from typing import Any


class _Transaction:
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self) -> _Transaction:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.committed = True
        else:
            self.rolled_back = True


class _Session:
    def __init__(self) -> None:
        self.transaction = _Transaction()

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def begin(self) -> _Transaction:
        return self.transaction


class _SessionFactory:
    def __init__(self) -> None:
        self.sessions: list[_Session] = []

    def __call__(self) -> _Session:
        session = _Session()
        self.sessions.append(session)
        return session


class _AccountStore:
    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.account = AuthAccount(
            id="user-1",
            email="test@example.com",
            full_name=None,
            password_hash=None,
            is_active=True,
            is_verified=False,
            role=UserRole.USER,
            permissions=frozenset(),
            created_at=now,
            updated_at=now,
        )
        self.token_hashes: list[str] = []

    async def find_by_email(self, _email: str) -> Any:
        return Success(self.account)

    async def set_verification_token_hash(self, _account_id: str, token_hash_: str) -> Any:
        self.token_hashes.append(token_hash_)
        self.account = self.account.model_copy(update={"verification_token_hash": token_hash_})
        return Success(None)


def _service(factory: _SessionFactory) -> AuthService:
    return AuthService(_AccountStore(), object(), session_factory=factory)


@pytest.mark.asyncio
async def test_publish_outbox_commits_staged_email_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _SessionFactory()
    service = _service(factory)

    async def stage(**_kwargs: object) -> str:
        return "event-1"

    monkeypatch.setattr("app.events.with_outbox", stage)

    result = await service._publish_outbox_event(
        aggregate_type="auth_email",
        aggregate_id="user-1",
        event_type="auth.send_verification_email",
        payload={"user_id": "user-1"},
    )

    assert isinstance(result, Success)
    assert factory.sessions[0].transaction.committed
    assert not factory.sessions[0].transaction.rolled_back


@pytest.mark.asyncio
async def test_publish_outbox_rolls_back_and_returns_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _SessionFactory()
    service = _service(factory)

    async def stage(**_kwargs: object) -> str:
        error_message = "postgres unavailable"
        raise SQLAlchemyError(error_message)

    monkeypatch.setattr("app.events.with_outbox", stage)

    result = await service._publish_outbox_event(
        aggregate_type="auth_email",
        aggregate_id="user-1",
        event_type="auth.send_verification_email",
        payload={"user_id": "user-1"},
    )

    assert isinstance(result, Failure)
    assert result.failure().code.value == "DATABASE_ERROR"
    assert factory.sessions[0].transaction.rolled_back
    assert not factory.sessions[0].transaction.committed


@pytest.mark.asyncio
async def test_publish_outbox_preserves_diagnostics_in_internal_log_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _SessionFactory()
    service = _service(factory)
    bound_fields: dict[str, object] = {}

    class _BoundLogger:
        def exception(self, _message: str) -> None:
            return None

    class _Logger:
        def bind(self, **kwargs: object) -> _BoundLogger:
            bound_fields.update(kwargs)
            return _BoundLogger()

    monkeypatch.setattr("app.features.auth.application.service.logger", _Logger())

    async def stage(**_kwargs: object) -> str:
        exc = SQLAlchemyError("postgres unavailable")
        exc.add_note("driver diagnostic=connection refused")
        raise exc

    monkeypatch.setattr("app.events.with_outbox", stage)

    result = await service._publish_outbox_event(
        aggregate_type="auth_email",
        aggregate_id="user-1",
        event_type="auth.send_verification_email",
        payload={"user_id": "user-1"},
    )

    assert isinstance(result, Failure)
    error = result.failure()
    assert error.details == {"operation": "publish_outbox_event"}
    assert "Traceback" not in str(error.model_dump())
    assert "postgres unavailable" not in str(error.model_dump())

    diagnostic_notes = bound_fields["diagnostic_notes"]
    assert isinstance(diagnostic_notes, list)
    assert "driver diagnostic=connection refused" in diagnostic_notes
    assert any("Traceback" in note for note in diagnostic_notes)


@pytest.mark.asyncio
async def test_publish_outbox_does_not_swallow_unexpected_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _SessionFactory()
    service = _service(factory)

    async def stage(**_kwargs: object) -> str:
        error_message = "unexpected programmer failure"
        raise RuntimeError(error_message)

    monkeypatch.setattr("app.events.with_outbox", stage)

    with pytest.raises(RuntimeError, match="unexpected programmer failure"):
        await service._publish_outbox_event(
            aggregate_type="auth_email",
            aggregate_id="user-1",
            event_type="auth.send_verification_email",
            payload={"user_id": "user-1"},
        )

    assert factory.sessions[0].transaction.rolled_back
    assert not factory.sessions[0].transaction.committed


@pytest.mark.asyncio
async def test_resend_failure_is_visible_and_retry_stages_fresh_current_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account_store = _AccountStore()
    factory = _SessionFactory()
    service = AuthService(account_store, object(), session_factory=factory)
    tokens = iter(("first-token", "second-token"))
    payloads: list[dict[str, object]] = []
    attempts = 0

    monkeypatch.setattr("app.features.auth.application.service.generate_token", lambda: next(tokens))

    async def stage(**kwargs: object) -> str:
        nonlocal attempts
        attempts += 1
        payload = kwargs["payload"]
        assert isinstance(payload, dict)
        payloads.append(payload)
        if attempts == 1:
            error_message = "postgres unavailable"
            raise SQLAlchemyError(error_message)
        return "event-2"

    monkeypatch.setattr("app.events.with_outbox", stage)

    first = await service.resend_verification("test@example.com")
    second = await service.resend_verification("test@example.com")

    assert isinstance(first, Failure)
    assert isinstance(second, Success)
    assert account_store.token_hashes == [hash_token("first-token"), hash_token("second-token")]
    assert payloads[0]["token"] == "first-token"
    assert payloads[1]["token"] == "second-token"
    assert payloads[0]["idempotency_key"] != payloads[1]["idempotency_key"]
    assert factory.sessions[0].transaction.rolled_back
    assert factory.sessions[1].transaction.committed
