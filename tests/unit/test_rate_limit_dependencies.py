"""Rate-limiter identity: per-user from token claims, per-IP otherwise.

Pins the request-identity-from-token contract for
``app.utils.rate_limit.dependencies``: caller identity is derived from
validated access-token claims and never from per-request state, while
requests without a usable token still get limited by IP (the limiter
must not become an auth gate on unauthenticated endpoints).
"""

from types import SimpleNamespace

import pytest

from app.features.auth.security import TokenClaims
from app.utils.exceptions import UnauthorizedException
from app.utils.rate_limit import dependencies as rate_limit_deps
from app.utils.rate_limit.dependencies import get_rate_limiter


class _StubRequest:
    def __init__(
        self,
        headers=None,
        cookies=None,
        client_host="203.0.113.7",
        state=None,
    ) -> None:
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.client = SimpleNamespace(host=client_host) if client_host is not None else None
        self.app = SimpleNamespace(state=SimpleNamespace(redis=object()))
        # Deliberately spoofable: proves the limiter never consults it.
        self.state = state if state is not None else SimpleNamespace()


class _CapturingService:
    def __init__(self, redis, seen) -> None:
        self.redis = redis
        self.seen = seen

    async def check_limit(self, identifier, burst, rate, period_seconds) -> None:
        self.seen.append(identifier)


@pytest.fixture
def capture_identifiers(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        rate_limit_deps,
        "RateLimitService",
        lambda redis: _CapturingService(redis, seen),
    )
    return seen


def _access_claims(sub="user-123") -> TokenClaims:
    return TokenClaims(
        sub=sub,
        jti="jti-1",
        sid="sess-1",
        role="user",
        permissions=[],
        token_type="access",
    )


async def test_authenticated_request_is_keyed_by_user(capture_identifiers, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.features.auth.security.decode_token",
        lambda _token: _access_claims(),
    )
    limiter = get_rate_limiter()
    await limiter(_StubRequest(headers={"authorization": "Bearer valid-token"}))

    assert capture_identifiers == ["user:user-123"]


async def test_cookie_token_is_keyed_by_user(capture_identifiers, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.features.auth.security.decode_token",
        lambda _token: _access_claims(sub="cookie-user"),
    )
    limiter = get_rate_limiter()
    await limiter(_StubRequest(cookies={"access_token": "valid-token"}))

    assert capture_identifiers == ["user:cookie-user"]


async def test_unauthenticated_request_falls_back_to_ip(capture_identifiers) -> None:
    limiter = get_rate_limiter()
    await limiter(_StubRequest())

    assert capture_identifiers == ["203.0.113.7"]


async def test_forwarded_header_selects_client_ip(capture_identifiers) -> None:
    limiter = get_rate_limiter()
    await limiter(
        _StubRequest(
            headers={"X-Forwarded-For": "198.51.100.9, 203.0.113.7"},
        )
    )

    assert capture_identifiers == ["198.51.100.9"]


async def test_invalid_token_falls_back_to_ip_without_raising(
    capture_identifiers, monkeypatch
) -> None:
    def _boom(token: str):
        msg = "Invalid token"
        raise UnauthorizedException(msg)

    monkeypatch.setattr("app.features.auth.security.decode_token", _boom)
    limiter = get_rate_limiter()
    await limiter(_StubRequest(headers={"authorization": "Bearer garbage"}))

    assert capture_identifiers == ["203.0.113.7"]


async def test_wrong_token_type_falls_back_to_ip(capture_identifiers, monkeypatch) -> None:
    refresh = _access_claims().model_copy(update={"token_type": "refresh"})
    monkeypatch.setattr("app.features.auth.security.decode_token", lambda _token: refresh)
    limiter = get_rate_limiter()
    await limiter(_StubRequest(headers={"authorization": "Bearer refresh-token"}))

    assert capture_identifiers == ["203.0.113.7"]


async def test_request_state_identity_is_never_consulted(capture_identifiers, monkeypatch) -> None:
    """Even a populated per-request identity attribute must not affect the key."""
    monkeypatch.setattr(
        "app.features.auth.security.decode_token",
        lambda _token: _access_claims(),
    )
    limiter = get_rate_limiter()

    # Spoofed state + valid token: the token wins, not the state.
    await limiter(
        _StubRequest(
            headers={"authorization": "Bearer valid-token"},
            state=SimpleNamespace(user_id="spoofed"),
        )
    )
    # Spoofed state + no token: the IP wins, not the state.
    await limiter(_StubRequest(state=SimpleNamespace(user_id="spoofed")))

    assert capture_identifiers == ["user:user-123", "203.0.113.7"]
