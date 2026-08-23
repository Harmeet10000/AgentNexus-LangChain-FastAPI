from unittest.mock import MagicMock

import pytest

from app.connections.celery_reliability import (
    CircuitBreakerState,
    is_circuit_breaker_open,
    record_circuit_breaker_failure,
    record_circuit_breaker_success,
    run_with_circuit_breaker,
)


class TestCircuitBreakerState:
    def test_acquire_returns_allow_when_closed(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        result = is_circuit_breaker_open(
            mock_redis,
            "test-service",
            namespace="test",
            recovery_timeout_seconds=60,
        )
        assert result is False


class TestCircuitBreakerFailure:
    def test_failure_increments_count(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        record_circuit_breaker_failure(
            mock_redis,
            "test-service",
            namespace="test",
            failure_threshold=3,
            recovery_timeout_seconds=60,
        )

        assert mock_redis.set.called

    def test_opens_after_threshold_breached(self):
        mock_redis = MagicMock()
        import json
        import time

        now = time.time()
        state_value = json.dumps(
            {
                "state": "open",
                "failures": 5,
                "opened_at": now,
            }
        ).encode()
        mock_redis.get.return_value = state_value

        result = is_circuit_breaker_open(
            mock_redis,
            "test-service",
            namespace="cb_test",
            recovery_timeout_seconds=3600,
        )
        assert result is True


class TestCircuitBreakerSuccess:
    def test_success_resets_failures(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        record_circuit_breaker_success(
            mock_redis,
            "test-service",
            namespace="test",
        )

        assert mock_redis.delete.called


class TestCircuitBreakerProbe:
    def test_probe_allows_one_request_in_half_open(self):
        mock_redis = MagicMock()
        state = CircuitBreakerState(state="half_open", failures=0, opened_at=0.0)
        mock_redis.get.return_value = state.model_dump_json().encode()

        result = is_circuit_breaker_open(
            mock_redis,
            "test-service",
            namespace="test",
            recovery_timeout_seconds=60,
        )
        assert result is False


class TestCircuitBreakerRun:
    def test_run_with_circuit_breaker_success(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        result = run_with_circuit_breaker(
            mock_redis,
            "test-operation",
            lambda: "success",
            namespace="test",
            failure_threshold=3,
            recovery_timeout_seconds=60,
        )
        assert result == "success"

    def test_run_with_circuit_breaker_failure_raises(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        def failing_op() -> None:
            msg = "operation failed"
            raise ValueError(msg)

        with pytest.raises(ValueError):
            run_with_circuit_breaker(
                mock_redis,
                "test-operation",
                failing_op,
                namespace="test",
                failure_threshold=3,
                recovery_timeout_seconds=60,
            )
