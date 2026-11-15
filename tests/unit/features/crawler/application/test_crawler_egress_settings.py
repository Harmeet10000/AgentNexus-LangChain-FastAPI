"""Task 3.1 — typed crawler-egress settings fail closed on absence and unsafe values.

The crawler worker reaches public destinations only through the deployment's
egress boundary. These tests pin the declaration: a complete configuration
loads, and every incomplete or unsafe combination is rejected at load time
rather than degrading to direct networking.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.features.crawler.application.egress import (
    CrawlerEgressConfigError,
    build_egress_policy,
    parse_bypass_destinations,
)

pytestmark = pytest.mark.unit

_VALID: dict[str, object] = {
    "CRAWLER_EGRESS_ENABLED": True,
    "CRAWLER_EGRESS_PROXY_URL": "http://crawler-egress-proxy:4750",
    "CRAWLER_EGRESS_PUBLIC_PROBE_URL": "https://example.com/",
    "CRAWLER_EGRESS_LOOPBACK_PROBE_URL": "http://127.0.0.1:9/",
    "CRAWLER_EGRESS_ALLOWED_SCHEME_PORTS": "http:80,https:443",
    "CRAWLER_EGRESS_NO_PROXY": "rabbitmq,crawler-redis-relay,localhost,127.0.0.1,::1",
    "REDIS_URL": "redis://localhost:6379/0",
}


def _settings(**overrides: object) -> Settings:
    values = {**_VALID, **overrides}
    return Settings(**values)


def test_valid_configuration_loads_the_resolved_policy() -> None:
    policy = build_egress_policy(_settings())

    assert policy.proxy_url == "http://crawler-egress-proxy:4750"
    assert policy.public_probe_url == "https://example.com/"
    assert policy.loopback_probe_url == "http://127.0.0.1:9/"
    assert policy.allowed_scheme_ports == {"http": (80,), "https": (443,)}
    assert "rabbitmq" in policy.bypass_destinations
    assert policy.redis_relay is None
    assert policy.external_processing_enabled is False


def test_disabled_boundary_fails_closed() -> None:
    """The boundary is mandatory for the crawler worker, in every environment.

    The check is environment-independent, so production cannot reach a state
    where the crawler silently browses without a declared boundary.
    """
    with pytest.raises(CrawlerEgressConfigError):
        build_egress_policy(_settings(CRAWLER_EGRESS_ENABLED=False))


@pytest.mark.parametrize(
    "overrides",
    [
        {"CRAWLER_EGRESS_PROXY_URL": None},
        {"CRAWLER_EGRESS_PROXY_URL": ""},
        {"CRAWLER_EGRESS_PROXY_URL": "ftp://crawler-egress-proxy:4750"},
        {"CRAWLER_EGRESS_PUBLIC_PROBE_URL": None},
        {"CRAWLER_EGRESS_PUBLIC_PROBE_URL": "http://127.0.0.1:9/"},
        {"CRAWLER_EGRESS_PUBLIC_PROBE_URL": "http://169.254.169.254/latest/meta-data/"},
        {"CRAWLER_EGRESS_PUBLIC_PROBE_URL": "http://metadata.google.internal/"},
        {"CRAWLER_EGRESS_LOOPBACK_PROBE_URL": "http://example.com/"},
        {"CRAWLER_EGRESS_LOOPBACK_PROBE_URL": "http://10.0.0.1/"},
    ],
)
def test_absent_or_unsafe_configuration_fails_closed(overrides: dict[str, object]) -> None:
    with pytest.raises(CrawlerEgressConfigError):
        build_egress_policy(_settings(**overrides))


@pytest.mark.parametrize(
    "no_proxy",
    [
        "*",
        "example.com",
        "8.8.8.8",
        "http://rabbitmq",
        "rabbitmq:5672",
        "rabbitmq/path",
    ],
)
def test_broad_bypass_destination_fails_closed(no_proxy: str) -> None:
    with pytest.raises(CrawlerEgressConfigError):
        build_egress_policy(_settings(CRAWLER_EGRESS_NO_PROXY=no_proxy))


@pytest.mark.parametrize(
    "scheme_ports",
    [
        "http:8080",
        "https:8443",
        "ftp:21",
        "http",
        "",
    ],
)
def test_unsupported_browser_destination_fails_closed(scheme_ports: str) -> None:
    with pytest.raises(CrawlerEgressConfigError):
        build_egress_policy(_settings(CRAWLER_EGRESS_ALLOWED_SCHEME_PORTS=scheme_ports))


def test_bypass_destination_parsing_accepts_only_internal_names() -> None:
    assert parse_bypass_destinations("rabbitmq,127.0.0.1,::1") == (
        "rabbitmq",
        "127.0.0.1",
        "::1",
    )


def test_declared_relay_requires_deployment_issued_mtls_material() -> None:
    with pytest.raises(CrawlerEgressConfigError):
        build_egress_policy(_settings(CRAWLER_REDIS_RELAY_HOST="crawler-redis-relay"))


def test_external_redis_requires_the_fixed_destination_relay() -> None:
    with pytest.raises(CrawlerEgressConfigError, match="requires the fixed-destination"):
        build_egress_policy(
            _settings(
                REDIS_URL="rediss://managed.example.com:6379/0",
                CRAWLER_EGRESS_NO_PROXY="rabbitmq,localhost,127.0.0.1,::1",
            )
        )


def test_direct_redis_is_limited_to_an_explicit_internal_destination() -> None:
    policy = build_egress_policy(
        _settings(
            REDIS_URL="redis://redis:6379/0",
            CRAWLER_EGRESS_NO_PROXY="rabbitmq,redis,localhost,127.0.0.1,::1",
        )
    )

    assert policy.redis_relay is None


def test_declared_relay_resolves_its_endpoint_and_material() -> None:
    policy = build_egress_policy(
        _settings(
            CRAWLER_REDIS_RELAY_HOST="crawler-redis-relay",
            CRAWLER_REDIS_RELAY_PORT="6380",
            CRAWLER_REDIS_RELAY_CA_FILE="/run/secrets/crawler_redis_relay_ca",
            CRAWLER_REDIS_RELAY_CERT_FILE="/run/secrets/crawler_redis_client_cert",
            CRAWLER_REDIS_RELAY_KEY_FILE="/run/secrets/crawler_redis_client_key",
        )
    )

    assert policy.redis_relay is not None
    assert policy.redis_relay.endpoint == "crawler-redis-relay:6380"
    assert policy.redis_relay.ca_file == "/run/secrets/crawler_redis_relay_ca"


def test_external_processing_flag_is_carried_by_the_policy() -> None:
    policy = build_egress_policy(_settings(CRAWLER_EXTERNAL_PROCESSING_ENABLED=True))

    assert policy.external_processing_enabled is True
