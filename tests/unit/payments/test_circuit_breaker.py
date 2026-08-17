from app.features.payments.clients.razorpay_client import _CircuitBreaker


class TestCircuitBreaker:
    def test_allows_until_threshold(self) -> None:
        breaker = _CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=60)
        assert breaker.allow("op")
        breaker.record_failure("op")
        breaker.record_failure("op")
        assert breaker.allow("op")
        breaker.record_failure("op")
        assert not breaker.allow("op")

    def test_success_resets_failures(self) -> None:
        breaker = _CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=60)
        breaker.record_failure("op")
        breaker.record_success("op")
        breaker.record_failure("op")
        assert breaker.allow("op")

    def test_open_is_per_operation(self) -> None:
        breaker = _CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=60)
        breaker.record_failure("a")
        breaker.record_failure("a")
        assert not breaker.allow("a")
        assert breaker.allow("b")