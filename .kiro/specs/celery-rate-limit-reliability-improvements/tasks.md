# Implementation Plan: Celery Rate Limit Reliability Improvements

## Overview

This implementation consolidates circuit breaker and idempotency patterns into a unified `ReliabilitySystem` base class, introduces an `IdempotencyManager` context manager for automatic lock lifecycle management, and adds a `RateLimiter` with Redis-scoped rate limiting and proxy-aware IP extraction. All components delegate to existing functional helpers in `celery_reliability.py` for backward compatibility.

## Tasks

- [ ] 1. Implement ReliabilitySystem unified base class
  - [ ] 1.1 Create ReliabilitySystem class with constructor and configuration
    - Implement `__init__` accepting redis_client, circuit_breaker_name, and optional overrides
    - Add `_validate_config()` method to check threshold and timeout ranges
    - Load defaults from Settings (CELERY_CIRCUIT_BREAKER_FAILURE_THRESHOLD, CELERY_CIRCUIT_BREAKER_RECOVERY_TIMEOUT, CELERY_IDEMPOTENCY_TTL_SECONDS)
    - Store configuration and Redis client reference as instance attributes
    - _Requirements: 1.3, 1.4, 1.5, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_
  
  - [ ] 1.2 Add circuit breaker methods to ReliabilitySystem
    - Implement `check_circuit_breaker()` delegating to `is_circuit_breaker_open()`
    - Implement `record_success()` delegating to `record_circuit_breaker_success()`
    - Implement `record_failure()` delegating to `record_circuit_breaker_failure()`
    - Raise `CircuitBreakerOpenError` with descriptive message including circuit breaker name
    - _Requirements: 1.1, 1.6, 1.7, 1.8, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 9.1, 9.3_
  
  - [ ] 1.3 Add idempotency methods to ReliabilitySystem
    - Implement `get_idempotency_status(idempotency_key)` delegating to functional helper
    - Add `default_idempotency_ttl` property returning configured TTL
    - Return `IdempotencyStatus | None` from status check
    - _Requirements: 1.2, 1.9, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_
  
  - [ ]* 1.4 Write unit tests for ReliabilitySystem
    - Test configuration validation (invalid thresholds, timeouts, TTL)
    - Test circuit breaker methods with mocked Redis client
    - Test idempotency status checking
    - Test settings defaults and per-instance overrides
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 8.4, 8.5, 8.6_

- [ ] 2. Implement IdempotencyManager context manager
  - [ ] 2.1 Create IdempotencyLockError exception class
    - Define as subclass of RuntimeError
    - Include descriptive error message with idempotency key
    - _Requirements: 9.2, 9.3, 9.4_
  
  - [ ] 2.2 Implement idempotency_manager async context manager
    - Create async context manager function accepting redis_client, idempotency_key, task_id, ttl_seconds, metadata, retryable_exceptions
    - On entry: call `acquire_idempotency_lock()` and raise `IdempotencyLockError` if lock already held
    - On normal exit: call `mark_idempotency_completed()`
    - On exception (non-retryable): call `mark_idempotency_failed_permanently()`
    - On exception (retryable): call `release_idempotency_processing_lock()`
    - Add structured logging for lock acquired, operation completed, retryable failure, permanent failure
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_
  
  - [ ]* 2.3 Write unit tests for IdempotencyManager
    - Test successful lock acquisition and completion
    - Test lock already held scenario (IdempotencyLockError raised)
    - Test retryable exception handling (lock released)
    - Test non-retryable exception handling (marked failed permanently)
    - Verify structured logging output
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [ ] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Implement RateLimiter with Redis-scoped rate limiting
  - [ ] 4.1 Create RateLimitResult dataclass
    - Define frozen dataclass with allowed, remaining, reset_at, scope fields
    - _Requirements: 3.12_
  
  - [ ] 4.2 Create RateLimiter class with key building logic
    - Implement `__init__` accepting redis_client, scope, rate, period_seconds, burst, settings
    - Add `_validate_config()` to check rate >= 1, period >= 1, burst >= rate
    - Implement `_build_key()` returning format `celery:ratelimit:{scope}:rate={rate}:period={period}:burst={burst}`
    - Implement static `_parse_key()` method to extract configuration from key
    - Load FASTAPI_GUARD_TRUSTED_PROXIES and FASTAPI_GUARD_TRUSTED_PROXY_DEPTH from settings
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_
  
  - [ ] 4.3 Implement IP extraction with proxy trust
    - Add `extract_client_ip(forwarded_for, direct_ip)` method
    - If no proxy trust configured, return direct_ip
    - Parse X-Forwarded-For header chain
    - Extract client IP using configured proxy depth
    - Add structured logging for IP extraction
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  
  - [ ] 4.4 Implement sliding window rate limiting with Redis
    - Add `check_and_increment()` async method accepting forwarded_for and direct_ip
    - Extract client IP and append to key if provided
    - Use Redis sorted set (ZREMRANGEBYSCORE, ZCARD, ZADD) for sliding window
    - Remove entries outside the current window
    - Check if count < burst limit
    - If allowed: add current timestamp, set expiry, calculate remaining
    - If rejected: set remaining to 0
    - Return RateLimitResult with allowed, remaining, reset_at, scope
    - Add structured logging for rate limit checks
    - Import and use `run_redis_call` helper for async Redis operations
    - _Requirements: 3.7, 3.8, 3.9, 3.10, 3.11, 3.12_
  
  - [ ]* 4.5 Write unit tests for RateLimiter
    - Test key building format
    - Test key parsing
    - Test configuration validation
    - Test IP extraction with and without proxies
    - Test sliding window algorithm (allow, reject, window expiry)
    - Test remaining capacity calculation
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 5. Integration and backward compatibility verification
  - [ ] 5.1 Update celery_reliability.py with new exports
    - Add RateLimitResult, ReliabilitySystem, idempotency_manager, RateLimiter, IdempotencyLockError to module exports
    - Ensure RedisClientProtocol is extended with zremrangebyscore, zcard, zadd, expire methods for RateLimiter
    - Add async `run_redis_call` overload or ensure compatibility with async Redis operations
    - _Requirements: 10.1, 10.2_
  
  - [ ] 5.2 Verify backward compatibility with existing functional helpers
    - Confirm ReliabilitySystem delegates to existing functions without modifying signatures
    - Confirm IdempotencyManager uses existing acquire/mark/release functions
    - Confirm Redis key formats remain unchanged (IDEMPOTENCY_NAMESPACE, CIRCUIT_BREAKER_NAMESPACE)
    - Confirm IdempotencyRecord and CircuitBreakerState models are unchanged
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_
  
  - [ ]* 5.3 Write integration tests
    - Test ReliabilitySystem with real Redis instance (circuit breaker flow)
    - Test IdempotencyManager with real Redis instance (lock lifecycle)
    - Test RateLimiter with real Redis instance (rate limiting)
    - Test combined usage: circuit breaker + idempotency + rate limiting in a single task
    - _Requirements: 1.10, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.7, 3.8, 3.9, 3.10, 3.11, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [ ] 6. Documentation and usage examples
  - [ ] 6.1 Create usage example for ReliabilitySystem in CELERY.md
    - Show basic task with circuit breaker
    - Show task with idempotency checking
    - Show combined usage
    - _Requirements: 1.1, 1.2, 6.1, 6.2, 6.3, 6.4_
  
  - [ ] 6.2 Create usage example for IdempotencyManager in CELERY.md
    - Show context manager usage with retryable exceptions
    - Show error handling
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  
  - [ ] 6.3 Create usage example for RateLimiter in CELERY.md
    - Show basic rate limiting per scope
    - Show IP-based rate limiting with proxy extraction
    - Show handling of rate limit exceeded
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ] 6.4 Create comprehensive example combining all components
    - Show task using ReliabilitySystem, IdempotencyManager, and RateLimiter together
    - Include error handling and structured logging
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 5.1, 6.1, 6.2, 6.3, 6.4, 9.5, 9.6, 9.7_

- [ ] 7. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- All components delegate to existing functional helpers in `celery_reliability.py` for backward compatibility
- New components are added to `celery_reliability.py` to maintain single module for reliability patterns
- Python 3.12+ type syntax used throughout (`type`, `|` union, PEP 695 generics)
- Structured logging with loguru used for all state transitions and events
- Configuration embedded in Redis keys for self-documentation and operational visibility

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "4.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "4.2"] },
    { "id": 2, "tasks": ["1.4", "2.2", "4.3"] },
    { "id": 3, "tasks": ["2.3", "4.4"] },
    { "id": 4, "tasks": ["4.5", "5.1"] },
    { "id": 5, "tasks": ["5.2"] },
    { "id": 6, "tasks": ["5.3", "6.1", "6.2", "6.3"] },
    { "id": 7, "tasks": ["6.4"] }
  ]
}
```
