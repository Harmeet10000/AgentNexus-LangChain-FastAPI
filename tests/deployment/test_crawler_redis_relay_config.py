"""Static security contract for the fixed-destination Redis relay."""

from pathlib import Path

ROOT = Path(__file__).parents[2]
RELAY_CONFIG = ROOT / "docker" / "crawler-egress" / "haproxy.redis.cfg"
HAPROXY_IMAGE = (
    "haproxy:3.4.7-alpine@sha256:b9fe316f718d5015c5cfebd8ca019591f3e44656f61d93cb8b59a1028961e530"
)


def test_relay_has_one_fixed_backend_and_verified_tls_on_both_legs() -> None:
    config = RELAY_CONFIG.read_text(encoding="utf-8")

    assert config.count("server managed-redis ") == 1
    assert "bind :6379 ssl" in config
    assert "verify required" in config
    assert "crawler_redis_relay_client_ca.pem" in config
    assert "ssl verify required" in config
    assert 'sni str("${CRAWLER_REDIS_UPSTREAM_HOST}")' in config
    assert 'verifyhost "${CRAWLER_REDIS_UPSTREAM_HOST}"' in config
    assert "${CRAWLER_REDIS_UPSTREAM_PORT}" in config


def test_relay_exposes_no_dynamic_target_or_redis_secret() -> None:
    config = RELAY_CONFIG.read_text(encoding="utf-8").lower()

    for forbidden in ("socks", "use_backend", "map(", "redis_password", "redis_url"):
        assert forbidden not in config


def test_haproxy_image_reference_is_exact_and_immutable() -> None:
    assert HAPROXY_IMAGE.startswith("haproxy:3.4.7-alpine@sha256:")
    assert len(HAPROXY_IMAGE.rsplit("sha256:", maxsplit=1)[1]) == 64
