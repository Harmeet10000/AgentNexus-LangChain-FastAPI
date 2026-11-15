"""Task 3.4 — crawler Redis egress stays on the fixed mTLS relay.

These tests prove that relay routing changes only the network destination:
Redis credentials, database selection, and client behavior remain unchanged,
and relay or TLS failure never reopens a direct managed-Redis route.
"""

from __future__ import annotations

import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.config import Settings
from app.features.crawler.application.egress import (
    CrawlerRedisRelaySettings,
    build_egress_policy,
    relay_redis_url,
)
from app.worker.jobs import crawler as crawler_job

_BASE_SETTINGS: dict[str, object] = {
    "CRAWLER_EGRESS_ENABLED": True,
    "CRAWLER_EGRESS_PROXY_URL": "http://crawler-egress-proxy:4750",
    "CRAWLER_EGRESS_PUBLIC_PROBE_URL": "https://example.com/",
    "CRAWLER_EGRESS_LOOPBACK_PROBE_URL": "http://127.0.0.1:9/",
    "CRAWLER_EGRESS_ALLOWED_SCHEME_PORTS": "http:80,https:443",
    "CRAWLER_EGRESS_NO_PROXY": "rabbitmq,127.0.0.1",
    "REDIS_URL": "redis://user:pass@managed.example.com:6379/2",
}


def _settings(**overrides: object) -> Settings:
    return Settings(**{**_BASE_SETTINGS, **overrides})


def _relay() -> CrawlerRedisRelaySettings:
    return CrawlerRedisRelaySettings(
        host="crawler-redis-relay",
        port=6380,
        ca_file="/run/secrets/ca.pem",
        cert_file="/run/secrets/client.crt",
        key_file="/run/secrets/client.key",
    )


def test_relay_url_preserves_credentials_and_database() -> None:
    assert relay_redis_url("redis://user:pass@managed.example.com:6379/2", _relay()) == (
        "rediss://user:pass@crawler-redis-relay:6380/2"
    )


@pytest.mark.asyncio
async def test_worker_redis_client_uses_relay_mtls_and_preserves_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def settings() -> Settings:
        return _settings(
            CRAWLER_REDIS_RELAY_HOST="crawler-redis-relay",
            CRAWLER_REDIS_RELAY_PORT=6380,
            CRAWLER_REDIS_RELAY_CA_FILE="/run/secrets/ca.pem",
            CRAWLER_REDIS_RELAY_CERT_FILE="/run/secrets/client.crt",
            CRAWLER_REDIS_RELAY_KEY_FILE="/run/secrets/client.key",
        )

    configured = settings()
    monkeypatch.setattr(crawler_job, "get_settings", lambda: configured)
    monkeypatch.setattr(
        crawler_job,
        "_resolve_egress_policy",
        lambda: build_egress_policy(configured),
    )

    client = crawler_job._create_worker_redis()
    assert isinstance(client, Redis)
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["host"] == "crawler-redis-relay"
    assert kwargs["port"] == 6380
    assert kwargs["db"] == 2
    assert kwargs["username"] == "user"
    assert kwargs["password"] == "pass"
    assert kwargs["ssl_ca_certs"] == "/run/secrets/ca.pem"
    assert kwargs["ssl_certfile"] == "/run/secrets/client.crt"
    assert kwargs["ssl_keyfile"] == "/run/secrets/client.key"
    assert kwargs["ssl_cert_reqs"] == "required"
    await client.aclose(close_connection_pool=True)


@pytest.mark.asyncio
async def test_worker_redis_client_without_relay_uses_managed_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler_job, "get_settings", _settings)

    client = crawler_job._create_worker_redis()
    assert client.connection_pool.connection_kwargs["host"] == "managed.example.com"
    assert client.connection_pool.connection_kwargs["port"] == 6379
    await client.aclose(close_connection_pool=True)


def test_relay_failure_has_no_direct_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        CRAWLER_REDIS_RELAY_HOST="crawler-redis-relay",
        CRAWLER_REDIS_RELAY_PORT=6380,
        CRAWLER_REDIS_RELAY_CA_FILE="/run/secrets/ca.pem",
        CRAWLER_REDIS_RELAY_CERT_FILE="/run/secrets/client.crt",
        CRAWLER_REDIS_RELAY_KEY_FILE="/run/secrets/client.key",
    )
    calls: list[str] = []

    def create_client(url: str, **_kwargs: str) -> Redis:
        calls.append(url)
        message = "relay unavailable"
        raise RedisConnectionError(message)

    monkeypatch.setattr(crawler_job, "get_settings", lambda: settings)
    monkeypatch.setattr(
        crawler_job,
        "_resolve_egress_policy",
        lambda: build_egress_policy(settings),
    )
    monkeypatch.setattr(crawler_job, "create_redis_client", create_client)

    with pytest.raises(RedisConnectionError):
        crawler_job._create_worker_redis()

    assert calls == ["rediss://user:pass@crawler-redis-relay:6380/2"]
    assert all("managed.example.com" not in url for url in calls)
