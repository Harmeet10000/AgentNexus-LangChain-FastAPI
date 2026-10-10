"""Transport-independent reliability failures shared with the HTTP boundary."""


class CircuitBreakerOpenError(RuntimeError):
    """An external provider circuit is open."""


class IdempotencyLockError(RuntimeError):
    """Another execution holds the operation lease."""
