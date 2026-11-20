"""Task 3.5 — provider and process egress remain bounded by deployment policy.

These tests prove that proxy environment variables are applied consistently,
broad bypass declarations are rejected, and optional external processing cannot
be requested while the deployment has it disabled.
"""

from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.features.crawler.application.egress import CrawlerEgressConfigError, build_egress_policy
from app.worker.capabilities.crawler import _apply_egress_environment
from app.worker.jobs import crawler as crawler_job

_BASE_SETTINGS: dict[str, object] = {
    "CRAWLER_EGRESS_ENABLED": True,
    "CRAWLER_EGRESS_PROXY_URL": "http://crawler-egress-proxy:4750",
    "CRAWLER_EGRESS_PUBLIC_PROBE_URL": "https://example.com/",
    "CRAWLER_EGRESS_LOOPBACK_PROBE_URL": "http://127.0.0.1:9/",
    "CRAWLER_EGRESS_ALLOWED_SCHEME_PORTS": "http:80,https:443",
    "CRAWLER_EGRESS_NO_PROXY": "rabbitmq,127.0.0.1",
    "CRAWLER_EXTERNAL_PROCESSING_ENABLED": False,
    "CRAWLER_REDIS_RELAY_HOST": None,
    "CRAWLER_REDIS_RELAY_CA_FILE": None,
    "CRAWLER_REDIS_RELAY_CERT_FILE": None,
    "CRAWLER_REDIS_RELAY_KEY_FILE": None,
    "REDIS_URL": "redis://127.0.0.1:6379/0",
}


def _settings(**overrides: object) -> Settings:
    return Settings(**{**_BASE_SETTINGS, **overrides})


def test_proxy_environment_matches_policy_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = build_egress_policy(_settings(CRAWLER_EGRESS_NO_PROXY="rabbitmq,127.0.0.1"))
    names = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy")
    for name in names:
        # Seed through monkeypatch so teardown records the original state even
        # when the variable did not exist before this test.  The production
        # helper writes os.environ directly, which would otherwise leak its
        # proxy values into every test that follows.
        monkeypatch.setenv(name, "__crawler_egress_test_sentinel__")

    _apply_egress_environment(policy)

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert os.environ[name] == policy.proxy_url
    assert os.environ["NO_PROXY"] == "rabbitmq,127.0.0.1"
    assert os.environ["no_proxy"] == "rabbitmq,127.0.0.1"


@pytest.mark.parametrize("bypass", ["example.com", "*"])
def test_public_or_wildcard_bypass_is_rejected(bypass: str) -> None:
    with pytest.raises(CrawlerEgressConfigError):
        build_egress_policy(_settings(CRAWLER_EGRESS_NO_PROXY=bypass))


@pytest.mark.asyncio
async def test_disabled_external_processing_is_terminal_without_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRedis:
        async def aclose(self, *, close_connection_pool: bool = False) -> None:
            _ = close_connection_pool

    class FakeStore:
        def __init__(self, redis: FakeRedis) -> None:
            self.redis = redis
            self.failed: list[tuple[str, str]] = []

        async def is_cancelled(self, _crawl_id: str) -> bool:
            return False

        async def is_cancel_requested(self, _crawl_id: str) -> bool:
            return False

        async def mark_running(self, _crawl_id: str) -> None:
            return None

        async def mark_failed(self, crawl_id: str, message: str) -> None:
            self.failed.append((crawl_id, message))

    store = FakeStore(FakeRedis())
    provider_calls = 0

    async def provider() -> None:
        nonlocal provider_calls
        provider_calls += 1

    monkeypatch.setattr(crawler_job, "get_settings", _settings)
    monkeypatch.setattr(crawler_job, "_create_worker_redis", lambda: store.redis)
    monkeypatch.setattr(crawler_job, "CrawlJobStore", lambda _redis: store)
    monkeypatch.setattr(crawler_job, "get_processor", provider)

    payload = {"url": "https://example.com/", "summary": True}
    await crawler_job._run_crawl_job(
        "crawl-provider-disabled", payload, final_attempt=True, browser=object()
    )

    assert provider_calls == 0
    assert store.failed == [
        ("crawl-provider-disabled", "Crawler external processing is disabled by the egress policy")
    ]
