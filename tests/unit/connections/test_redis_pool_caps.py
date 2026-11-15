"""G8 regression: no unbounded Redis pool may exist in any process.

The 2026-11-15 smoke recorded a 40-parallel request burst pushing Redis Cloud
past its plan client cap ("max number of clients reached"); with no pool
budget and a broken TCP keepalive option (``{1: 120}`` set ``TCP_NODELAY``,
not ``TCP_KEEPIDLE``), every dead socket was held for the OS default ~2 h
idle window and the server refused all connections for ~22 minutes with no
self-recovery.  These tests pin the three structural facts that prevent a
repeat: every pool is capped, the caps sum to the documented deployment
budget, and dead sockets are reaped by real TCP keepalive probes — plus the
limiter's deliberate fail-open policy under a failing backend.
"""

from __future__ import annotations

import asyncio
import socket

import pytest
from redis import RedisError

from app.config import get_settings
from app.connections.redis import create_redis_client
from app.shared.rate_limit.adapters.redis.throttle import RateLimitService

pytestmark = pytest.mark.unit


class _FailingScript:
    """Registered Lua script that always fails like an unreachable backend."""

    def __init__(self, error: type[Exception]) -> None:
        self._error = error

    async def __call__(self, *args: object, **kwargs: object) -> object:
        message = "connection lost"
        raise self._error(message)


class _FailingClient:
    """Stand-in for a Redis client whose backend is down (G8 incident)."""

    def register_script(self, script: str) -> _FailingScript:
        return _FailingScript(RedisError)


def test_worker_client_pool_is_capped() -> None:
    """``create_redis_client`` defaults to the per-worker-process budget."""
    client = create_redis_client("redis://localhost:6379/0")
    pool = client.connection_pool

    assert pool.max_connections == get_settings().REDIS_WORKER_MAX_CONNECTIONS


def test_worker_client_honors_explicit_cap() -> None:
    """An explicit budget wins over the settings default."""
    client = create_redis_client("redis://localhost:6379/0", max_connections=3)

    assert client.connection_pool.max_connections == 3


def test_client_reaps_dead_sockets_with_real_keepalive_options() -> None:
    """Keepalive must configure idle/interval/count — not TCP_NODELAY.

    The old ``{1: 120}`` option dict set ``TCP_NODELAY`` (option 1 at the TCP
    level) instead of ``TCP_KEEPIDLE``, so no keepalive probe schedule was
    ever installed and dead connections lingered for the OS default (~2 h).
    """
    client = create_redis_client("redis://localhost:6379/0")
    kwargs = client.connection_pool.connection_kwargs

    assert kwargs["socket_keepalive"] is True
    keepalive = kwargs.get("socket_keepalive_options") or {}
    assert keepalive.get(socket.TCP_KEEPIDLE) == 120
    assert keepalive.get(socket.TCP_KEEPINTVL) == 30
    assert keepalive.get(socket.TCP_KEEPCNT) == 3
    assert socket.TCP_NODELAY not in keepalive
    # Periodic PING reaping as a second line of defense.
    assert kwargs["health_check_interval"] == 30


def test_documented_deployment_budget_stays_under_the_plan_client_cap() -> None:
    """1 API + 3 workers: the documented total must stay a bounded number.

    Redis Cloud refuses every connection once the plan client cap is hit and
    does not self-recover.  The budget is chosen below even the smallest paid
    tier's cap (256 clients), leaving headroom for operational tooling.
    """
    settings = get_settings()

    total = settings.REDIS_MAX_CONNECTIONS + 3 * settings.REDIS_WORKER_MAX_CONNECTIONS
    assert total == 98
    assert total < 256


def test_rate_limiter_fails_open_when_redis_is_down() -> None:
    """A failing backend must not turn limiter-guarded routes into 5xx.

    Policy (pinned in the throttle comment): rate limiting fails OPEN during
    a Redis outage — availability over enforcement.  The limiter raising
    ``TooManyRequestsException`` here would mean enforcement resurfaced an
    infrastructure failure as an application rejection.
    """
    service = RateLimitService(_FailingClient())  # type: ignore[arg-type]

    # Must NOT raise — the request passes through unenforced.
    asyncio.run(service.check_limit("user:1", burst=5, rate=10, period_seconds=60))


def test_api_sdk_pool_cap_reaches_the_sdk_env_contract(monkeypatch) -> None:
    """The API pool is SDK-managed and reads its budget from ``REDIS_MAX_CONNECTIONS``.

    ``create_app`` mirrors the app setting into that env var before the
    lifespan builds the pool.  A wrong variable name here would leave the
    API pool silently unbounded, so the contract is asserted through the
    SDK's own settings object — the exact reader used at startup.
    """
    from redis_fastapi.config import RedisSettings

    monkeypatch.delenv("REDIS_MAX_CONNECTIONS", raising=False)
    expected = get_settings().REDIS_MAX_CONNECTIONS

    # The same mirroring call app.main.create_app performs before the lifespan.
    import os

    os.environ.setdefault("REDIS_MAX_CONNECTIONS", str(expected))

    assert RedisSettings().max_connections == expected
