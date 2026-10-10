from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from starlette.requests import Request

from app.connections.redis import close_redis_client, create_redis_client
from app.lifecycle.lifespan import setup_redis


@pytest.mark.anyio
async def test_standalone_client_factory_supports_redis_py_8() -> None:
    client = create_redis_client("redis://localhost:6379")

    await close_redis_client(client)


@pytest.mark.anyio
async def test_setup_redis_uses_sdk_managed_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    redis_client = MagicMock(spec=Redis)
    redis_client.ping = AsyncMock(return_value=True)
    resolved_request: Request | None = None

    async def resolve_redis(request: Request) -> Redis:
        nonlocal resolved_request
        resolved_request = request
        return redis_client

    monkeypatch.setattr("app.lifecycle.lifespan.get_async_redis", resolve_redis)

    assert await setup_redis(app) is redis_client
    assert resolved_request is not None
    assert resolved_request.app is app
    redis_client.ping.assert_awaited_once()


@pytest.mark.anyio
async def test_setup_redis_degrades_when_sdk_client_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    redis_client = MagicMock(spec=Redis)
    redis_client.ping = AsyncMock(side_effect=RedisConnectionError("offline"))

    async def resolve_redis(request: Request) -> Redis:
        _ = request
        return redis_client

    monkeypatch.setattr("app.lifecycle.lifespan.get_async_redis", resolve_redis)

    assert await setup_redis(app) is None
