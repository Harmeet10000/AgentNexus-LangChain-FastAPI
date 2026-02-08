# app/db/redis_connect.py
from fastapi import Request
from redis.asyncio import Redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry


def create_redis_client(url: str) -> Redis:
    """
    Create and return a configured async Redis client.
    The client is connection-pooled and intended to live
    for the entire process lifetime.
    """

    retry_strategy = Retry(
        backoff=ExponentialBackoff(
            base=0.1,      # ~ retryDelayOnFailover (100ms)
            cap=2.0,
        ),
        retries=3,        # maxRetriesPerRequest
    )

    # logger.info(
    #     "Successfully connected to Redis",
    #     host=get_settings().REDIS_HOST,
    #     max_connections=50,
    # )

    return Redis.from_url(
        url,
        db=0,
        # Connection & command timeouts
        socket_connect_timeout=120.0,  # connectTimeout (ms → sec)
        socket_timeout=5.0,            # commandTimeout (ms → sec)
        # TCP keepalive
        socket_keepalive=True,
        socket_keepalive_options={
            # 120s keepalive
            1: 120,
        },
        # Retry behavior
        retry=retry_strategy,
        retry_on_timeout=True,
        # Encoding
        decode_responses=True,
        # Health check
        health_check_interval=30,
    )

def get_redis(request: Request) -> Redis:
    return request.app.state.redis
