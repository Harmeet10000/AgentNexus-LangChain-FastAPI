import asyncio
import json
import sys
import time
from unittest.mock import MagicMock

# Break circular import chains that fire when any app.connections or app.utils is imported
for _mod in (
    "app.connections.celery",
    "app.connections.crawl4ai",
    "app.connections.httpx_client",
    "app.connections.mongodb",
    "app.connections.neo4j",
    "app.connections.postgres",
    "app.connections.redis",
    "app.connections.tavily",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402

from app.connections.celery_reliability import (  # noqa: E402
    CircuitBreakerOpenError,
    IdempotencyLockError,
    RateLimiter,
    ReliabilitySystem,
    idempotency_manager,
)


def _make_settings(**overrides):
    defaults = {
        "CELERY_CIRCUIT_BREAKER_FAILURE_THRESHOLD": 5,
        "CELERY_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": 60,
        "CELERY_IDEMPOTENCY_TTL_SECONDS": 86400,
        "FASTAPI_GUARD_TRUSTED_PROXIES": [],
        "FASTAPI_GUARD_TRUSTED_PROXY_DEPTH": 1,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


class TestReliabilitySystemConfig:
    def test_config_loads_defaults_from_settings(self):
        mock_redis = MagicMock()
        system = ReliabilitySystem(
            mock_redis,
            circuit_breaker_name="test",
            settings=_make_settings(),
        )
        assert system._failure_threshold == 5
        assert system._recovery_timeout == 60
        assert system._default_idempotency_ttl == 86400

    def test_config_override_per_instance(self):
        mock_redis = MagicMock()
        system = ReliabilitySystem(
            mock_redis,
            circuit_breaker_name="test",
            failure_threshold=3,
            recovery_timeout_seconds=30,
            idempotency_ttl_seconds=100,
            settings=_make_settings(),
        )
        assert system._failure_threshold == 3
        assert system._recovery_timeout == 30
        assert system._default_idempotency_ttl == 100

    def test_config_validation_rejects_failure_threshold_lt_1(self):
        mock_redis = MagicMock()
        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            ReliabilitySystem(
                mock_redis,
                circuit_breaker_name="test",
                failure_threshold=0,
                settings=_make_settings(),
            )

    def test_config_validation_rejects_recovery_timeout_lt_1(self):
        mock_redis = MagicMock()
        with pytest.raises(ValueError, match="recovery_timeout_seconds must be >= 1"):
            ReliabilitySystem(
                mock_redis,
                circuit_breaker_name="test",
                recovery_timeout_seconds=0,
                settings=_make_settings(),
            )

    def test_config_validation_rejects_idempotency_ttl_lt_1(self):
        mock_redis = MagicMock()
        with pytest.raises(ValueError, match="idempotency_ttl_seconds must be >= 1"):
            ReliabilitySystem(
                mock_redis,
                circuit_breaker_name="test",
                idempotency_ttl_seconds=0,
                settings=_make_settings(),
            )


class TestReliabilitySystemCircuitBreaker:
    def test_check_circuit_breaker_allows_when_closed(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        system = ReliabilitySystem(
            mock_redis,
            circuit_breaker_name="test",
            settings=_make_settings(),
        )
        system.check_circuit_breaker()

    def test_check_circuit_breaker_raises_when_open(self):
        mock_redis = MagicMock()
        state = {
            "state": "open",
            "failures": 5,
            "opened_at": time.time(),
        }
        mock_redis.get.return_value = json.dumps(state).encode()
        system = ReliabilitySystem(
            mock_redis,
            circuit_breaker_name="my-service",
            settings=_make_settings(),
        )
        with pytest.raises(CircuitBreakerOpenError, match="my-service"):
            system.check_circuit_breaker()

    def test_record_success_deletes_redis_key(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        system = ReliabilitySystem(
            mock_redis,
            circuit_breaker_name="test",
            settings=_make_settings(),
        )
        system.record_success()
        assert mock_redis.delete.called

    def test_record_failure_writes_to_redis(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        system = ReliabilitySystem(
            mock_redis,
            circuit_breaker_name="test",
            settings=_make_settings(),
        )
        system.record_failure()
        assert mock_redis.set.called


class TestReliabilitySystemIdempotency:
    def test_get_idempotency_status_returns_none_when_no_record(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        system = ReliabilitySystem(
            mock_redis,
            circuit_breaker_name="test",
            settings=_make_settings(),
        )
        result = system.get_idempotency_status("key-1")
        assert result is None

    def test_get_idempotency_status_returns_status_when_record_exists(self):
        mock_redis = MagicMock()
        record = {
            "status": "completed",
            "task_id": "t-1",
            "updated_at": "2026-01-01T00:00:00",
            "metadata": {},
        }
        mock_redis.get.return_value = json.dumps(record).encode()
        system = ReliabilitySystem(
            mock_redis,
            circuit_breaker_name="test",
            settings=_make_settings(),
        )
        result = system.get_idempotency_status("key-1")
        assert result == "completed"

    def test_default_idempotency_ttl_property(self):
        mock_redis = MagicMock()
        system = ReliabilitySystem(
            mock_redis,
            circuit_breaker_name="test",
            idempotency_ttl_seconds=123,
            settings=_make_settings(),
        )
        assert system.default_idempotency_ttl == 123


class TestIdempotencyManager:
    def test_normal_flow_marks_completed(self):
        mock_redis = MagicMock()
        mock_redis.set.side_effect = [True, None]

        async def _run():
            async with idempotency_manager(mock_redis, "op-1"):
                pass

        asyncio.run(_run())
        assert mock_redis.set.call_count == 2

    def test_lock_already_held_raises(self):
        mock_redis = MagicMock()
        mock_redis.set.return_value = None

        async def _run():
            async with idempotency_manager(mock_redis, "op-1"):
                pass

        with pytest.raises(IdempotencyLockError, match="op-1"):
            asyncio.run(_run())

    def test_retryable_exception_releases_lock(self):
        mock_redis = MagicMock()
        mock_redis.set.return_value = True
        msg = "transient"

        async def _run():
            async with idempotency_manager(
                mock_redis, "op-1", retryable_exceptions=(ValueError,)
            ):
                raise ValueError(msg)

        with pytest.raises(ValueError, match=msg):
            asyncio.run(_run())
        assert mock_redis.delete.called

    def test_non_retryable_exception_marks_failed(self):
        mock_redis = MagicMock()
        mock_redis.set.return_value = True
        msg = "permanent"

        async def _run():
            async with idempotency_manager(
                mock_redis, "op-1", retryable_exceptions=(ValueError,)
            ):
                raise TypeError(msg)

        with pytest.raises(TypeError, match=msg):
            asyncio.run(_run())
        assert mock_redis.set.call_count == 2


class TestRateLimiterConfig:
    def test_build_key_format(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        limiter = RateLimiter(
            mock_redis,
            scope="upload",
            rate=10,
            period_seconds=60,
            burst=15,
            settings=_make_settings(),
        )
        key = limiter._build_key()
        assert key == "celery:ratelimit:upload:rate=10:period=60:burst=15"

    def test_parse_key_round_trip(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        limiter = RateLimiter(
            mock_redis,
            scope="upload",
            rate=10,
            period_seconds=60,
            burst=15,
            settings=_make_settings(),
        )
        key = limiter._build_key()
        parsed = RateLimiter._parse_key(key)
        assert parsed == {"scope": "upload", "rate": 10, "period": 60, "burst": 15}

    def test_config_validation_rejects_rate_lt_1(self):
        mock_redis = MagicMock()
        with pytest.raises(ValueError, match="rate must be >= 1"):
            RateLimiter(mock_redis, scope="x", rate=0, period_seconds=60, settings=_make_settings())

    def test_config_validation_rejects_period_lt_1(self):
        mock_redis = MagicMock()
        with pytest.raises(ValueError, match="period_seconds must be >= 1"):
            RateLimiter(mock_redis, scope="x", rate=1, period_seconds=0, settings=_make_settings())

    def test_config_validation_rejects_burst_lt_rate(self):
        mock_redis = MagicMock()
        with pytest.raises(ValueError, match="burst must be >= rate"):
            RateLimiter(mock_redis, scope="x", rate=10, period_seconds=60, burst=5, settings=_make_settings())


class TestRateLimiterExtractClientIp:
    def test_no_proxy_returns_direct_ip(self):
        mock_redis = MagicMock()
        limiter = RateLimiter(
            mock_redis,
            scope="x",
            rate=1,
            period_seconds=60,
            settings=_make_settings(),
        )
        assert limiter.extract_client_ip(None, "1.2.3.4") == "1.2.3.4"

    def test_chain_long_enough_uses_depth(self):
        mock_redis = MagicMock()
        limiter = RateLimiter(
            mock_redis,
            scope="x",
            rate=1,
            period_seconds=60,
            settings=_make_settings(
                FASTAPI_GUARD_TRUSTED_PROXIES=["10.0.0.1"],
                FASTAPI_GUARD_TRUSTED_PROXY_DEPTH=2,
            ),
        )
        forwarded = "10.0.0.1, 172.16.0.1, 1.2.3.4"
        result = limiter.extract_client_ip(forwarded, "127.0.0.1")
        assert result == "172.16.0.1"

    def test_chain_short_returns_first_ip(self):
        mock_redis = MagicMock()
        limiter = RateLimiter(
            mock_redis,
            scope="x",
            rate=1,
            period_seconds=60,
            settings=_make_settings(
                FASTAPI_GUARD_TRUSTED_PROXIES=["10.0.0.1"],
                FASTAPI_GUARD_TRUSTED_PROXY_DEPTH=3,
            ),
        )
        forwarded = "10.0.0.1, 172.16.0.1"
        result = limiter.extract_client_ip(forwarded, "127.0.0.1")
        assert result == "10.0.0.1"


class TestRateLimiterCheckAndIncrement:
    def test_allowed_when_under_limit(self):
        from fakeredis.aioredis import FakeRedis

        async def _run():
            fake = FakeRedis()
            limiter = RateLimiter(
                fake,
                scope="test",
                rate=5,
                period_seconds=60,
                settings=_make_settings(),
            )
            result = await limiter.check_and_increment(direct_ip="1.2.3.4")
            assert result.allowed is True
            assert result.remaining == 4
            assert result.scope == "test:ip=1.2.3.4"
            await fake.aclose()

        asyncio.run(_run())

    def test_rejected_when_at_burst_limit(self):
        from fakeredis.aioredis import FakeRedis

        async def _run():
            fake = FakeRedis()
            limiter = RateLimiter(
                fake,
                scope="test",
                rate=2,
                period_seconds=60,
                burst=2,
                settings=_make_settings(),
            )
            await limiter.check_and_increment(direct_ip="1.2.3.4")
            await limiter.check_and_increment(direct_ip="1.2.3.4")
            result = await limiter.check_and_increment(direct_ip="1.2.3.4")
            assert result.allowed is False
            assert result.remaining == 0
            await fake.aclose()

        asyncio.run(_run())
