"""Rate-limiter identity: per-user from token claims, per-IP otherwise.

Pins the request-identity-from-token contract for
``app.utils.rate_limit.dependencies``: caller identity is derived from
validated access-token claims and never from per-request state, while
requests without a usable token still get limited by IP (the limiter
must not become an auth gate on unauthenticated endpoints).
"""

from types import SimpleNamespace

import pytest
from fakeredis.aioredis import FakeRedis

import app.shared.rate_limit.adapters.redis.dependencies as rate_limit_deps
from app.features.auth.adapters.security.tokens import TokenClaims
from app.features.auth.api.http.router import (
    limit_forgot_password,
    limit_login,
    limit_register,
    limit_resend_verification,
)
from app.shared.rate_limit import get_rate_limiter
from app.utils.exceptions import TooManyRequestsException, UnauthorizedException


class _StubRequest:
    def __init__(
        self,
        headers=None,
        cookies=None,
        client_host="203.0.113.7",
        state=None,
        redis=None,
    ) -> None:
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.client = SimpleNamespace(host=client_host) if client_host is not None else None
        self.app = SimpleNamespace(state=SimpleNamespace(redis=redis or object()))
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
        "app.features.auth.adapters.security.tokens.decode_token",
        lambda _token: _access_claims(),
    )
    limiter = get_rate_limiter()
    await limiter(_StubRequest(headers={"authorization": "Bearer valid-token"}))

    assert capture_identifiers == ["user:user-123"]


async def test_scoped_quota_namespaces_the_authenticated_user(
    capture_identifiers, monkeypatch
) -> None:
    monkeypatch.setattr(
        "app.features.auth.adapters.security.tokens.decode_token",
        lambda _token: _access_claims(),
    )
    limiter = get_rate_limiter(scope="document-qa")
    await limiter(_StubRequest(headers={"authorization": "Bearer test-token"}))

    assert capture_identifiers == ["document-qa:user:user-123"]


async def test_cookie_token_is_keyed_by_user(capture_identifiers, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.features.auth.adapters.security.tokens.decode_token",
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

    monkeypatch.setattr("app.features.auth.adapters.security.tokens.decode_token", _boom)
    limiter = get_rate_limiter()
    await limiter(_StubRequest(headers={"authorization": "Bearer garbage"}))

    assert capture_identifiers == ["203.0.113.7"]


async def test_wrong_token_type_falls_back_to_ip(capture_identifiers, monkeypatch) -> None:
    refresh = _access_claims().model_copy(update={"token_type": "refresh"})
    monkeypatch.setattr(
        "app.features.auth.adapters.security.tokens.decode_token",
        lambda _token: refresh,
    )
    limiter = get_rate_limiter()
    await limiter(_StubRequest(headers={"authorization": "Bearer refresh-token"}))

    assert capture_identifiers == ["203.0.113.7"]


async def test_request_state_identity_is_never_consulted(capture_identifiers, monkeypatch) -> None:
    """Even a populated per-request identity attribute must not affect the key."""
    monkeypatch.setattr(
        "app.features.auth.adapters.security.tokens.decode_token",
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


async def test_register_and_login_are_scoped_per_endpoint_class(capture_identifiers) -> None:
    """Register and login each get their own per-IP bucket (no shared unscoped key)."""
    await limit_register(_StubRequest())
    await limit_login(_StubRequest())

    assert capture_identifiers == ["register:203.0.113.7", "login:203.0.113.7"]


async def test_forgot_password_and_resend_verification_are_scoped_per_endpoint_class(
    capture_identifiers,
) -> None:
    """The 300s-window auth email classes must not share one bucket either."""
    await limit_forgot_password(_StubRequest())
    await limit_resend_verification(_StubRequest())

    assert capture_identifiers == [
        "forgot-password:203.0.113.7",
        "resend-verification:203.0.113.7",
    ]


async def test_login_budget_is_independent_of_register_consumption() -> None:
    """Exhausting register's 3/60s quota must not throttle the first login."""
    redis = FakeRedis(decode_responses=True)
    request = _StubRequest(redis=redis)

    for _ in range(3):
        await limit_register(request)
    await limit_login(request)  # login's own 5/60s budget is untouched


async def test_exhausted_register_quota_still_429s_with_retry_after() -> None:
    redis = FakeRedis(decode_responses=True)
    request = _StubRequest(redis=redis)

    for _ in range(3):
        await limit_register(request)
    with pytest.raises(TooManyRequestsException) as raised:
        await limit_register(request)
    assert raised.value.headers is not None
    assert int(raised.value.headers["Retry-After"]) > 0
    # The scoped login bucket is still independent after register 429s.
    await limit_login(request)
