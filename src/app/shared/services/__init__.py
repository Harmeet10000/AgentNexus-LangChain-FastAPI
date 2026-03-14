"""Shared services module."""

from app.shared.services.celery_reliability import (
    CircuitBreakerOpenError,
    acquire_idempotency_lock,
    build_circuit_breaker_key,
    build_idempotency_key,
    get_circuit_breaker_state,
    get_idempotency_status,
    is_circuit_breaker_open,
    mark_idempotency_completed,
    mark_idempotency_failed_permanently,
    record_circuit_breaker_failure,
    record_circuit_breaker_success,
    release_idempotency_processing_lock,
    run_redis_call,
    run_with_circuit_breaker,
    set_circuit_breaker_state,
)
from app.shared.services.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitScope,
    get_rate_limiter,
)
from app.shared.services.tavily import (
    SearchResponse,
    SearchResult,
    TavilyClient,
    get_tavily_client,
)

__all__ = [
    "CircuitBreakerOpenError",
    "RateLimitConfig",
    "RateLimitScope",
    "RateLimiter",
    "SearchResponse",
    "SearchResult",
    "TavilyClient",
    "acquire_idempotency_lock",
    "build_circuit_breaker_key",
    "build_idempotency_key",
    "get_circuit_breaker_state",
    "get_idempotency_status",
    "get_rate_limiter",
    "get_tavily_client",
    "is_circuit_breaker_open",
    "mark_idempotency_completed",
    "mark_idempotency_failed_permanently",
    "record_circuit_breaker_failure",
    "record_circuit_breaker_success",
    "release_idempotency_processing_lock",
    "run_redis_call",
    "run_with_circuit_breaker",
    "set_circuit_breaker_state",
]
