"""Fail-closed session validity proofs for the per-frame WS check.

Parity with agent_saul's handshake core (test_runtime_wiring.py): the
per-message check in ``_check_session_validity`` must agree with resolve_ws_session
semantics — an unreadable session (Redis outage) closes the connection, a
session row belonging to another user closes it with a distinct forensic log
event, and a healthy owned session passes through unchanged (the wire code
stays SESSION_REVOKED for every refusal).
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from returns.result import Failure, Success

from app.config import get_settings
from app.features.auth.exceptions import AuthInfrastructureError
from app.features.auth.repository import SessionData
from app.features.auth.websocket_security import (
    WebSocketSecurityContext,
    WebSocketSessionRevokedError,
    build_websocket_security_service,
)
from app.observability.logging import logger
from app.security.tokens import TokenClaims

pytestmark = pytest.mark.unit

_USER = "707f1f77bcf86cd799439099"
_OTHER = "707f1f77bcf86cd799439000"
_SESSION = "session-validity-1"


class _StaticSessionStore:
    """get_session-only stub of RefreshTokenRepository's session-read surface."""

    def __init__(self, result: object) -> None:
        self._result = result

    async def get_session(self, session_id: str) -> object:
        return self._result


def _claims() -> TokenClaims:
    return TokenClaims(
        sub=_USER,
        sid=_SESSION,
        jti=f"jti-{uuid4()}",
        role="user",
        permissions=[],
        token_type="access",
    )


def _context() -> WebSocketSecurityContext:
    connection_id = str(uuid4())
    return WebSocketSecurityContext(
        claims=_claims(),
        user_id=_USER,
        session_id=_SESSION,
        connection_id=connection_id,
        origin="https://example.com",
        user_rate_limit_key=f"user:{_USER}",
        connection_rate_limit_key=f"connection:{connection_id}",
    )


def _session(user_id: str) -> SessionData:
    return SessionData(
        session_id=_SESSION,
        user_id=user_id,
        device_id="device-1",
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(seconds=3600),
        ttl=3600,
    )


class _LogCollector:
    """Collect one frame per log call as the record's extras dict."""

    def __init__(self) -> None:
        self.frames: list[dict[str, str]] = []
        self._sink_id = logger.add(self._record, level="DEBUG")

    def _record(self, message: object) -> None:
        record = message.record["extra"]  # type: ignore[union-attr]
        self.frames.append({k: v for k, v in record.items() if isinstance(v, str)})

    def stop(self) -> None:
        logger.remove(self._sink_id)


@pytest.fixture
def logs():
    collector = _LogCollector()
    try:
        yield collector
    finally:
        collector.stop()


@pytest.fixture
async def service_with_store(redis):
    """Service with the session-repo slot ready for callers to swap a stub."""
    from app.features.auth.repository import RefreshTokenRepository

    service = await build_websocket_security_service(redis, get_settings())
    yield service, RefreshTokenRepository(redis)
    service.close()


class TestSessionValidFailClosed:
    async def test_outage_closes_connection_with_structured_log(
        self,
        service_with_store,
        logs,
    ):
        """Redis outage on the per-frame check must close, not continue."""
        service, _token_repo = service_with_store
        service._token_repo = _StaticSessionStore(  # type: ignore[assignment]
            Failure(AuthInfrastructureError(message="redis down"))
        )
        context = _context()
        logs.frames.clear()

        with pytest.raises(WebSocketSessionRevokedError):
            await service._check_session_validity(context, None)  # type: ignore[arg-type]

        frames = [f for f in logs.frames if f.get("event") == "session_check_unavailable"]
        assert frames, "outage must log a distinct session_check_unavailable event"
        assert frames[0]["session_id"] == _SESSION
        assert frames[0]["user_id"] == _USER
        assert frames[0]["operation"] == "check_session_validity"
        assert frames[0]["layer"] == "websocket"

    async def test_owner_mismatch_closes_connection_with_structured_log(
        self,
        service_with_store,
        logs,
    ):
        """A session row owned by another user must close, logged distinctly."""
        service, _token_repo = service_with_store
        service._token_repo = _StaticSessionStore(  # type: ignore[assignment]
            Success(_session(_OTHER))
        )
        context = _context()
        logs.frames.clear()

        with pytest.raises(WebSocketSessionRevokedError):
            await service._check_session_validity(context, None)  # type: ignore[arg-type]

        frames = [f for f in logs.frames if f.get("event") == "session_owner_mismatch"]
        assert frames, "owner mismatch must log a distinct session_owner_mismatch event"
        assert frames[0]["user_id"] == context.user_id
        assert frames[0]["owner_user_id"] == _OTHER
        assert frames[0]["session_id"] == _SESSION
        assert frames[0]["operation"] == "check_session_validity"

    async def test_happy_path_owned_session_passes_unchanged(
        self,
        service_with_store,
        logs,
    ):
        """A healthy, owned session still passes; no refusal events fire."""
        service, token_repo = service_with_store
        result = await token_repo.store_session(_session(_USER))
        assert isinstance(result, Success)
        service._token_repo = token_repo
        context = _context()
        logs.frames.clear()

        await service._check_session_validity(context, None)  # type: ignore[arg-type]

        events = {f.get("event", "") for f in logs.frames}
        assert "session_owner_mismatch" not in events
        assert "session_check_unavailable" not in events
