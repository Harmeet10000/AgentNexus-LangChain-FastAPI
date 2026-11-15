# app/db/redis_connect.py
import socket

from redis.asyncio import Redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry
from starlette.requests import HTTPConnection

from app.config import get_settings
from app.utils import InfrastructureException

# from tenacity import retry, stop_after_attempt, max_attempt_number, wait_exponential


def _tcp_keepalive_options() -> dict[int, int]:
    """Linux TCP keepalive knobs that reap dead connections in ~3.5 minutes.

    G8 (2026-11-15 smoke): the previous value here was ``{1: 120}`` — option
    ``1`` at the TCP level is ``TCP_NODELAY``, so the intended 120 s keepalive
    idle was never applied and every pool held dead sockets until the OS
    default (~2 h idle).  Those zombie sockets kept the Redis Cloud client
    count saturated for the whole outage even after traffic stopped.
    """

    if not hasattr(socket, "TCP_KEEPIDLE"):  # pragma: no cover — non-Linux hosts
        return {}
    return {
        socket.TCP_KEEPIDLE: 120,  # first probe after 120 s idle
        socket.TCP_KEEPINTVL: 30,  # probe every 30 s afterwards
        socket.TCP_KEEPCNT: 3,  # three failed probes -> socket is dead
    }


# @retry(stop=stop_after_attempt(max_attempt_number=3), wait=wait_exponential(multiplier=1, min=2, max=10))
def create_redis_client(url: str, *, max_connections: int | None = None) -> Redis:
    """
    Create and return a configured async Redis client.
    The client is connection-pooled and intended to live
    for the entire process lifetime.

    ``max_connections`` defaults to the documented per-worker-process budget
    (``REDIS_WORKER_MAX_CONNECTIONS``); the API process's SDK-managed pool is
    capped separately via ``REDIS_MAX_CONNECTIONS`` in ``app.main``.  Every
    pool in this deployment must be bounded: Redis Cloud refuses all
    connections once the plan's client cap is exceeded and does not recover
    on its own (G8).
    """

    retry_strategy = Retry(
        backoff=ExponentialBackoff(
            base=0.1,  # ~ retryDelayOnFailover (100ms)
            cap=2.0,
        ),
        retries=3,  # maxRetriesPerRequest
    )

    budget = max_connections if max_connections is not None else get_settings().REDIS_WORKER_MAX_CONNECTIONS

    return Redis.from_url(
        url,
        db=0,
        # Connection & command timeouts
        # Keep an unavailable Redis from pinning request workers for minutes.
        socket_connect_timeout=5.0,  # connectTimeout (ms → sec)
        socket_timeout=5.0,  # commandTimeout (ms → sec)
        # TCP keepalive
        socket_keepalive=True,
        socket_keepalive_options=_tcp_keepalive_options(),
        # Retry behavior
        retry=retry_strategy,
        retry_on_timeout=True,
        # Encoding
        decode_responses=True,
        # Health check
        health_check_interval=30,
        # Client budget (G8): bounded pools only.
        max_connections=budget,
    )


async def close_redis_client(client: Redis | None) -> None:
    """Close Redis clients and any connection pools they own."""
    if client is not None:
        await client.aclose()


async def get_redis(connection: HTTPConnection) -> Redis:
    redis = getattr(connection.app.state, "redis", None)
    if redis is None:
        # AUTH session storage failing here is honest typed degradation
        # (503), not a policy decision: without the session store neither
        # session reads nor login writes can be served.  This differs
        # deliberately from rate limiting, which fails OPEN (availability
        # over enforcement) — see the GCRA throttle in
        # app/shared/rate_limit/adapters/redis/throttle.py for that policy.
        raise InfrastructureException(
            detail="Redis session store unavailable",
            retryable=True,
            data={"operation": "redis_dependency"},
        )
    return redis
