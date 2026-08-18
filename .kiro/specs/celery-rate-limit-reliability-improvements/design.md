# Design Document

## Overview

This document describes the design for improving Celery task execution reliability and rate limiting. The design consolidates circuit breaker and idempotency patterns into a unified base class, introduces a context manager for idempotency control, implements Redis-scoped rate limiting with embedded configuration, and integrates with existing FASTAPI_GUARD settings for proxy trust configuration.

The key architectural decisions are:

- **Unified Base Class**: Combine circuit breaker and idempotency into a single `ReliabilitySystem` class
- **Context Manager Pattern**: Automate idempotency lock lifecycle with proper cleanup
- **Config-in-Key**: Embed rate limit configuration directly in Redis keys
- **Backward Compatibility**: All new components delegate to existing functional helpers in `celery_reliability.py`

## Architecture

### Architecture Overview

The reliability improvements follow a layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     Celery Task Layer                        │
│  (Task definitions using ReliabilitySystem & IdempotencyMgr) │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│              ReliabilitySystem (Unified Base)                │
│  • Circuit Breaker Control                                   │
│  • Idempotency Status Checking                               │
│  • Configuration Management                                  │
│  • Delegates to functional helpers                           │
└───────┬─────────────────────────────┬───────────────────────┘
        │                             │
┌───────▼──────────────┐    ┌─────────▼────────────────┐
│ IdempotencyManager   │    │    RateLimiter           │
│ (Context Manager)    │    │ (Redis-scoped limits)    │
│ • Lock lifecycle     │    │ • Sliding window         │
│ • Auto cleanup       │    │ • Config in key          │
│ • Retry handling     │    │ • Proxy IP extraction    │
└───────┬──────────────┘    └─────────┬────────────────┘
        │                             │
┌───────▼─────────────────────────────▼─────────────────────┐
│            celery_reliability.py (Functional Helpers)      │
│  • acquire_idempotency_lock()                              │
│  • mark_idempotency_completed()                            │
│  • is_circuit_breaker_open()                               │
│  • record_circuit_breaker_failure()                        │
│  • get_idempotency_status()                                │
└───────────────────────┬────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────────┐
│                    Redis Storage                            │
│  • Circuit breaker state (celery:circuit:{name})           │
│  • Idempotency records (celery:idempotency:{key})          │
│  • Rate limit counters (celery:ratelimit:{scope}:...)      │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**ReliabilitySystem**: Unified facade providing circuit breaker and idempotency functionality to Celery tasks. Composes functional helpers from `celery_reliability.py`.

**IdempotencyManager**: Async context manager that automates idempotency lock lifecycle (acquire, complete, fail, release on retry).

**RateLimiter**: Redis-based rate limiting with configuration embedded in keys, supporting sliding window algorithm and proxy-aware IP extraction.

**Functional Helpers**: Stateless functions in `celery_reliability.py` that perform atomic Redis operations for circuit breaker and idempotency.

### Data Flow

#### Task Execution with Circuit Breaker and Idempotency

```
1. Task invoked
2. ReliabilitySystem.check_circuit_breaker()
   └─> is_circuit_breaker_open() → Redis query
3. If open: raise CircuitBreakerOpenError
4. async with IdempotencyManager(key):
   └─> acquire_idempotency_lock() → Redis SET NX
5. If lock acquired:
   └─> Execute task logic
6. On success:
   ├─> mark_idempotency_completed() → Redis SET
   └─> record_circuit_breaker_success() → Redis DELETE
7. On failure (non-retryable):
   ├─> mark_idempotency_failed_permanently() → Redis SET
   └─> record_circuit_breaker_failure() → Redis incr + state update
8. On failure (retryable):
   ├─> release_idempotency_processing_lock() → Redis DELETE
   └─> record_circuit_breaker_failure()
```

#### Rate Limiting Check

```
1. Task requests execution
2. RateLimiter.check_and_increment(scope)
   ├─> Extract client IP (with proxy trust)
   ├─> Build key: celery:ratelimit:{scope}:rate={r}:period={p}:burst={b}
   └─> Redis sliding window check
3. If rate exceeded: reject request
4. If allowed: increment counter, return remaining capacity
```

### Integration Points

**Settings**: Configuration values from `Settings` class (Celery, FASTAPI_GUARD).

**Redis**: Single Redis client instance passed to all components, conforming to `RedisClientProtocol`.

**Existing Functional Helpers**: All new components delegate to functions in `celery_reliability.py` for backward compatibility.

**FASTAPI_GUARD**: Proxy trust configuration reused for rate limiter IP extraction.

## Components and Interfaces

### Component: ReliabilitySystem

#### Purpose
Unified base class providing circuit breaker and idempotency functionality to Celery tasks.

#### Class Structure

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from app.config.settings import Settings
from app.connections.celery_reliability import (
    CircuitBreakerOpenError,
    get_idempotency_status,
    is_circuit_breaker_open,
    record_circuit_breaker_failure,
    record_circuit_breaker_success,
)

if TYPE_CHECKING:
    from app.connections.celery_reliability import (
        IdempotencyStatus,
        RedisClientProtocol,
    )


class ReliabilitySystem:
    """Unified reliability base for Celery tasks.
    
    Provides circuit breaker and idempotency functionality by composing
    functional helpers from celery_reliability.py.
    """

    def __init__(
        self,
        redis_client: RedisClientProtocol,
        *,
        circuit_breaker_name: str,
        failure_threshold: int | None = None,
        recovery_timeout_seconds: int | None = None,
        idempotency_ttl_seconds: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize reliability system.
        
        Args:
            redis_client: Redis client conforming to RedisClientProtocol
            circuit_breaker_name: Name for circuit breaker state key
            failure_threshold: Max failures before opening breaker (default from settings)
            recovery_timeout_seconds: Timeout before half-open (default from settings)
            idempotency_ttl_seconds: Default TTL for idempotency records (default from settings)
            settings: Settings instance (defaults to get_settings())
        """
        self._redis = redis_client
        self._circuit_breaker_name = circuit_breaker_name
        
        # Load settings
        if settings is None:
            from app.config.settings import get_settings
            settings = get_settings()
        
        self._failure_threshold = (
            failure_threshold
            if failure_threshold is not None
            else settings.CELERY_CIRCUIT_BREAKER_FAILURE_THRESHOLD
        )
        self._recovery_timeout = (
            recovery_timeout_seconds
            if recovery_timeout_seconds is not None
            else settings.CELERY_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        )
        self._default_idempotency_ttl = (
            idempotency_ttl_seconds
            if idempotency_ttl_seconds is not None
            else settings.CELERY_IDEMPOTENCY_TTL_SECONDS
        )
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration values are within acceptable ranges."""
        if self._failure_threshold < 1:
            msg = f"failure_threshold must be >= 1, got {self._failure_threshold}"
            raise ValueError(msg)
        if self._recovery_timeout < 1:
            msg = f"recovery_timeout_seconds must be >= 1, got {self._recovery_timeout}"
            raise ValueError(msg)
        if self._default_idempotency_ttl < 1:
            msg = f"idempotency_ttl_seconds must be >= 1, got {self._default_idempotency_ttl}"
            raise ValueError(msg)
    
    def check_circuit_breaker(self) -> None:
        """Check if circuit breaker allows execution.
        
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
        """
        if is_circuit_breaker_open(
            self._redis,
            self._circuit_breaker_name,
            recovery_timeout_seconds=self._recovery_timeout,
        ):
            msg = f"Circuit breaker '{self._circuit_breaker_name}' is open"
            raise CircuitBreakerOpenError(msg)
    
    def record_success(self) -> None:
        """Record successful operation, resetting circuit breaker."""
        record_circuit_breaker_success(self._redis, self._circuit_breaker_name)
    
    def record_failure(self) -> None:
        """Record failed operation, potentially opening circuit breaker."""
        record_circuit_breaker_failure(
            self._redis,
            self._circuit_breaker_name,
            failure_threshold=self._failure_threshold,
            recovery_timeout_seconds=self._recovery_timeout,
        )
    
    def get_idempotency_status(self, idempotency_key: str) -> IdempotencyStatus | None:
        """Check idempotency status for a given key.
        
        Args:
            idempotency_key: Business-level identifier
            
        Returns:
            Status string or None if no record exists
        """
        return get_idempotency_status(self._redis, idempotency_key)
    
    @property
    def default_idempotency_ttl(self) -> int:
        """Get default TTL for idempotency records."""
        return self._default_idempotency_ttl
```

#### Key Design Decisions

1. **Composition over Inheritance**: Delegates to functional helpers rather than reimplementing logic
2. **Configuration Override**: Allows per-instance override of settings defaults
3. **Validation**: Validates configuration ranges at construction time
4. **Minimal State**: Only stores configuration and Redis client reference

### Component: IdempotencyManager

#### Purpose
Async context manager that automates idempotency lock lifecycle with proper cleanup.

#### Class Structure

```python
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from app.connections.celery_reliability import (
    acquire_idempotency_lock,
    mark_idempotency_completed,
    mark_idempotency_failed_permanently,
    release_idempotency_processing_lock,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from app.connections.celery_reliability import (
        JsonMetadata,
        RedisClientProtocol,
    )


class IdempotencyLockError(RuntimeError):
    """Raised when idempotency lock cannot be acquired."""


@asynccontextmanager
async def idempotency_manager(
    redis_client: RedisClientProtocol,
    idempotency_key: str,
    *,
    task_id: str | None = None,
    ttl_seconds: int = 86400,
    metadata: JsonMetadata | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (),
) -> AsyncIterator[None]:
    """Context manager for idempotency lock lifecycle.
    
    Automatically acquires lock on entry, marks completed on normal exit,
    marks failed on exception, and releases lock for retryable failures.
    
    Args:
        redis_client: Redis client conforming to RedisClientProtocol
        idempotency_key: Business-level identifier for operation
        task_id: Optional Celery task ID for tracking
        ttl_seconds: TTL for idempotency record
        metadata: Optional metadata to store with record
        retryable_exceptions: Exception types that should release lock for retry
        
    Raises:
        IdempotencyLockError: If lock is already held by another process
        
    Example:
        ```python
        async with idempotency_manager(
            redis,
            f"process_payment:{payment_id}",
            task_id=self.request.id,
            retryable_exceptions=(TimeoutError, ConnectionError),
        ):
            await process_payment(payment_id)
        ```
    """
    # Attempt to acquire lock
    acquired = acquire_idempotency_lock(
        redis_client,
        idempotency_key,
        task_id=task_id,
        ttl_seconds=ttl_seconds,
        metadata=metadata,
    )
    
    if not acquired:
        logger.warning(
            "Idempotency lock already held",
            idempotency_key=idempotency_key,
            task_id=task_id,
        )
        msg = f"Operation '{idempotency_key}' is already processing"
        raise IdempotencyLockError(msg)
    
    logger.info(
        "Idempotency lock acquired",
        idempotency_key=idempotency_key,
        task_id=task_id,
        ttl_seconds=ttl_seconds,
    )
    
    try:
        yield
        
        # Normal exit: mark completed
        mark_idempotency_completed(
            redis_client,
            idempotency_key,
            task_id=task_id,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )
        logger.info(
            "Operation completed successfully",
            idempotency_key=idempotency_key,
            task_id=task_id,
        )
        
    except Exception as exc:
        # Determine if failure is retryable
        is_retryable = isinstance(exc, retryable_exceptions) if retryable_exceptions else False
        
        if is_retryable:
            # Release lock so retry can acquire it
            release_idempotency_processing_lock(redis_client, idempotency_key)
            logger.warning(
                "Retryable failure, released processing lock",
                idempotency_key=idempotency_key,
                task_id=task_id,
                exception_type=type(exc).__name__,
            )
        else:
            # Mark as permanently failed
            mark_idempotency_failed_permanently(
                redis_client,
                idempotency_key,
                task_id=task_id,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            )
            logger.error(
                "Permanent failure, marked as failed",
                idempotency_key=idempotency_key,
                task_id=task_id,
                exception_type=type(exc).__name__,
            )
        
        raise
```

#### Key Design Decisions

1. **Context Manager Pattern**: Guarantees cleanup even on exception
2. **Retryable Exception Handling**: Allows caller to specify which exceptions should release lock
3. **Structured Logging**: All lifecycle events logged with context
4. **Clear Error Messages**: Includes operation name and context in exceptions

### Component: RateLimiter

#### Purpose
Redis-based rate limiting with configuration embedded in keys and proxy-aware IP extraction.

#### Class Structure

```python
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from app.config.settings import Settings

if TYPE_CHECKING:
    from app.connections.celery_reliability import RedisClientProtocol


@dataclass(frozen=True)
class RateLimitResult:
    """Result of rate limit check."""
    
    allowed: bool
    remaining: int
    reset_at: float
    scope: str


class RateLimiter:
    """Redis-based rate limiter with config embedded in keys.
    
    Uses sliding window algorithm for accurate rate calculation.
    Integrates with FASTAPI_GUARD proxy trust settings for IP extraction.
    """
    
    def __init__(
        self,
        redis_client: RedisClientProtocol,
        *,
        scope: str,
        rate: int,
        period_seconds: int,
        burst: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize rate limiter.
        
        Args:
            redis_client: Redis client conforming to RedisClientProtocol
            scope: Rate limit scope (e.g., "api:user:{user_id}", "task:process_doc")
            rate: Number of requests allowed per period
            period_seconds: Time period in seconds
            burst: Optional burst allowance (defaults to rate)
            settings: Settings instance (defaults to get_settings())
        """
        self._redis = redis_client
        self._scope = scope
        self._rate = rate
        self._period = period_seconds
        self._burst = burst if burst is not None else rate
        
        # Load proxy trust settings
        if settings is None:
            from app.config.settings import get_settings
            settings = get_settings()
        
        self._trusted_proxies = settings.FASTAPI_GUARD_TRUSTED_PROXIES
        self._proxy_depth = settings.FASTAPI_GUARD_TRUSTED_PROXY_DEPTH
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate rate limit configuration."""
        if self._rate < 1:
            msg = f"rate must be >= 1, got {self._rate}"
            raise ValueError(msg)
        if self._period < 1:
            msg = f"period_seconds must be >= 1, got {self._period}"
            raise ValueError(msg)
        if self._burst < self._rate:
            msg = f"burst must be >= rate, got burst={self._burst}, rate={self._rate}"
            raise ValueError(msg)
    
    def _build_key(self) -> str:
        """Build Redis key with embedded configuration.
        
        Format: celery:ratelimit:{scope}:rate={rate}:period={period}:burst={burst}
        """
        return (
            f"celery:ratelimit:{self._scope}"
            f":rate={self._rate}"
            f":period={self._period}"
            f":burst={self._burst}"
        )
    
    @staticmethod
    def _parse_key(key: str) -> dict[str, int | str]:
        """Parse configuration from Redis key.
        
        Args:
            key: Redis key with embedded config
            
        Returns:
            Dictionary with scope, rate, period, burst
        """
        # celery:ratelimit:{scope}:rate={rate}:period={period}:burst={burst}
        parts = key.split(":")
        
        scope_parts = []
        config_parts = {}
        
        # Find where config starts (first part with =)
        config_start = next(
            (i for i, p in enumerate(parts[2:], start=2) if "=" in p),
            len(parts),
        )
        
        # Extract scope
        scope_parts = parts[2:config_start]
        scope = ":".join(scope_parts)
        
        # Parse config
        for part in parts[config_start:]:
            if "=" in part:
                key_part, value = part.split("=", 1)
                config_parts[key_part] = int(value)
        
        return {
            "scope": scope,
            "rate": config_parts.get("rate", 0),
            "period": config_parts.get("period", 0),
            "burst": config_parts.get("burst", 0),
        }
    
    def extract_client_ip(self, forwarded_for: str | None, direct_ip: str) -> str:
        """Extract client IP considering proxy trust configuration.
        
        Args:
            forwarded_for: X-Forwarded-For header value
            direct_ip: Direct connection IP
            
        Returns:
            Client IP address
        """
        # If no proxy trust configured, use direct IP
        if not self._trusted_proxies or not forwarded_for:
            return direct_ip
        
        # Parse X-Forwarded-For chain
        ip_chain = [ip.strip() for ip in forwarded_for.split(",")]
        
        # Extract client IP using configured depth
        # depth=1 means rightmost trusted proxy's left neighbor
        # depth=2 means second-rightmost trusted proxy's left neighbor
        if len(ip_chain) >= self._proxy_depth:
            client_ip = ip_chain[-(self._proxy_depth)]
            logger.debug(
                "Extracted client IP from proxy chain",
                forwarded_for=forwarded_for,
                proxy_depth=self._proxy_depth,
                client_ip=client_ip,
            )
            return client_ip
        
        # Fallback to leftmost IP if chain is shorter than depth
        return ip_chain[0] if ip_chain else direct_ip
    
    async def check_and_increment(
        self,
        *,
        forwarded_for: str | None = None,
        direct_ip: str | None = None,
    ) -> RateLimitResult:
        """Check rate limit and increment counter if allowed.
        
        Uses sliding window algorithm for accurate rate calculation.
        
        Args:
            forwarded_for: Optional X-Forwarded-For header for IP extraction
            direct_ip: Optional direct connection IP
            
        Returns:
            RateLimitResult with allowed status and remaining capacity
        """
        now = time.time()
        key = self._build_key()
        
        # If IP-based scope, append extracted IP
        final_scope = self._scope
        if direct_ip:
            client_ip = self.extract_client_ip(forwarded_for, direct_ip)
            final_scope = f"{self._scope}:ip={client_ip}"
            key = f"{key}:ip={client_ip}"
        
        # Sliding window: track timestamps in sorted set
        window_start = now - self._period
        
        # Import redis async helpers
        from app.connections.celery_reliability import run_redis_call
        
        # Remove old entries outside window
        await run_redis_call(self._redis.zremrangebyscore(key, "-inf", window_start))
        
        # Count current requests in window
        current_count = await run_redis_call(self._redis.zcard(key))
        
        # Check if under limit
        allowed = current_count < self._burst
        
        if allowed:
            # Add current request with timestamp score
            await run_redis_call(self._redis.zadd(key, {str(now): now}))
            # Set expiry to period + buffer
            await run_redis_call(self._redis.expire(key, self._period * 2))
            remaining = self._burst - current_count - 1
        else:
            remaining = 0
        
        reset_at = now + self._period
        
        logger.debug(
            "Rate limit check",
            scope=final_scope,
            allowed=allowed,
            current=current_count,
            limit=self._burst,
            remaining=remaining,
        )
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=reset_at,
            scope=final_scope,
        )
```

#### Key Design Decisions

1. **Config in Key**: Configuration embedded in Redis key for self-documentation
2. **Sliding Window**: Uses Redis sorted sets for accurate time-based windowing
3. **Proxy Integration**: Reuses FASTAPI_GUARD proxy trust configuration
4. **IP Extraction**: Handles X-Forwarded-For chains with configurable depth
5. **Validation**: Validates configuration at construction and key parsing

## Data Models

### Existing Models (Maintained for Backward Compatibility)

```python
# From celery_reliability.py

class IdempotencyRecord(BaseModel, frozen=True):
    """Serialized idempotency record stored in Redis."""
    status: IdempotencyStatus
    task_id: str | None = None
    updated_at: str
    metadata: JsonMetadata


class CircuitBreakerState(BaseModel, frozen=True):
    """Circuit breaker state snapshot."""
    state: str = "closed"
    failures: int = 0
    opened_at: float | None = None
```

### New Models

```python
@dataclass(frozen=True)
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_at: float
    scope: str
```

## Error Handling

### Exception Hierarchy

```python
# From celery_reliability.py
class CircuitBreakerOpenError(RuntimeError):
    """Raised when the circuit breaker is open."""

# New exceptions
class IdempotencyLockError(RuntimeError):
    """Raised when idempotency lock cannot be acquired."""

class RateLimitExceededError(RuntimeError):
    """Raised when rate limit is exceeded."""
```

### Error Context

All exceptions include:
- Operation/circuit breaker/idempotency key name
- Relevant identifiers (task_id, scope)
- Contextual information (current state, threshold values)

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Idempotency Lock Acquisition Round Trip

*For any* valid idempotency key and Redis client, when the IdempotencyManager successfully acquires a lock, the lock SHALL be present in Redis with "processing" status, and when the context exits normally, the status SHALL transition to "completed".

**Validates: Requirements 2.2, 2.4**

### Property 2: Idempotency Lock Prevents Duplicate Processing

*For any* idempotency key that already has an active lock, attempting to acquire the lock again SHALL raise IdempotencyLockError.

**Validates: Requirements 2.3**

### Property 3: Retryable Failure Releases Lock

*For any* exception type specified in retryable_exceptions, when that exception occurs within the IdempotencyManager context, the processing lock SHALL be released (deleted from Redis).

**Validates: Requirements 2.6**

### Property 4: Non-Retryable Failure Marks Permanent

*For any* exception that is not in retryable_exceptions, when that exception occurs within the IdempotencyManager context, the idempotency record SHALL be marked as "failed_permanent" in Redis.

**Validates: Requirements 2.5**

### Property 5: Rate Limit Key Embeds Configuration

*For any* combination of scope, rate, period_seconds, and burst parameters, the RateLimiter SHALL construct a Redis key in the exact format "celery:ratelimit:{scope}:rate={rate}:period={period}:burst={burst}".

**Validates: Requirements 4.1, 3.1**

### Property 6: Rate Limit Key Parsing Round Trip

*For any* valid rate limit Redis key with embedded configuration, parsing the key SHALL extract the original scope, rate, period, and burst values exactly.

**Validates: Requirements 4.6**

### Property 7: Rate Limit Enforcement

*For any* rate limiter with configured rate R and burst B, after B successful requests within a period, the next request SHALL be rejected (allowed=False).

**Validates: Requirements 3.8, 3.7**

### Property 8: Rate Limit Counter Increment

*For any* allowed rate limit check, the request count in the Redis sorted set SHALL increase by exactly 1.

**Validates: Requirements 3.9**

### Property 9: Sliding Window Cleanup

*For any* rate limit check at time T with period P, all Redis sorted set entries with timestamp < (T - P) SHALL be removed before counting.

**Validates: Requirements 3.10**

### Property 10: Proxy IP Extraction Depth

*For any* X-Forwarded-For chain with length >= proxy_depth, extracting the client IP with depth D SHALL return the IP at position -(D) in the chain.

**Validates: Requirements 5.3**

### Property 11: Direct IP Fallback

*For any* rate limit check where forwarded_for is None or empty, the client IP SHALL be the direct_ip value.

**Validates: Requirements 5.4**

### Property 12: Client IP in Scope

*For any* rate limit check with a direct_ip provided, the final scope used in the Redis key SHALL include ":ip={client_ip}".

**Validates: Requirements 5.5**

### Property 13: Circuit Breaker Rejection When Open

*For any* circuit breaker in "open" state, calling check_circuit_breaker() SHALL raise CircuitBreakerOpenError.

**Validates: Requirements 6.2, 6.1**

### Property 14: Circuit Breaker Success Reset

*For any* circuit breaker with recorded failures, calling record_success() SHALL delete the circuit breaker state from Redis (resetting to closed with 0 failures).

**Validates: Requirements 6.3**

### Property 15: Circuit Breaker Failure Increment

*For any* circuit breaker with F failures where F < threshold, calling record_failure() SHALL increment the failure count to F+1.

**Validates: Requirements 6.4**

### Property 16: Circuit Breaker Opens At Threshold

*For any* circuit breaker with failures equal to (threshold - 1), calling record_failure() SHALL transition the state to "open" with opened_at timestamp.

**Validates: Requirements 6.5**

### Property 17: Circuit Breaker Half-Open Transition

*For any* circuit breaker in "open" state where (current_time - opened_at) >= recovery_timeout, checking the state SHALL transition it to "half_open".

**Validates: Requirements 6.6**

### Property 18: Circuit Breaker Recovery From Half-Open

*For any* circuit breaker in "half_open" state, calling record_success() SHALL delete the state (closing the breaker).

**Validates: Requirements 6.7**

### Property 19: Circuit Breaker Re-Open From Half-Open

*For any* circuit breaker in "half_open" state, calling record_failure() SHALL transition the state back to "open" with a new opened_at timestamp.

**Validates: Requirements 6.8**

### Property 20: Idempotency Status Deserialization

*For any* valid IdempotencyRecord JSON in Redis, calling get_idempotency_status() SHALL deserialize it to an IdempotencyRecord model and return the status field.

**Validates: Requirements 7.7, 7.2**

### Property 21: Configuration Override Takes Precedence

*For any* ReliabilitySystem initialized with explicit failure_threshold T, the instance SHALL use T instead of the settings default for all circuit breaker operations.

**Validates: Requirements 8.4, 8.5**

### Property 22: Configuration Validation Rejects Invalid Values

*For any* configuration value V that violates range constraints (e.g., failure_threshold < 1), initialization SHALL raise ValueError.

**Validates: Requirements 8.6**

### Property 23: Exception Messages Include Operation Context

*For any* CircuitBreakerOpenError or IdempotencyLockError raised, the exception message SHALL contain the circuit breaker name or idempotency key.

**Validates: Requirements 9.3, 9.4**

### Property 24: Idempotency Key Format Preservation

*For any* idempotency_key K, the Redis key constructed SHALL be exactly "celery:idempotency:{K}", maintaining backward compatibility.

**Validates: Requirements 10.3**

### Property 25: Circuit Breaker Key Format Preservation

*For any* circuit breaker name N, the Redis key constructed SHALL be exactly "celery:circuit:{N}", maintaining backward compatibility.

**Validates: Requirements 10.3**

## Security Considerations

**Redis Access Control**: All components assume Redis client has appropriate permissions for SET, GET, DELETE, ZADD, ZCARD operations.

**Idempotency Key Security**: Idempotency keys should not contain sensitive data as they're stored in Redis unencrypted.

**Rate Limit Scope**: Scopes should be designed to prevent cross-user pollution (e.g., include user_id in scope).

**Proxy Trust**: FASTAPI_GUARD_TRUSTED_PROXIES must be configured correctly to prevent IP spoofing. Incorrect configuration allows attackers to bypass rate limits.

**TTL Configuration**: Idempotency TTLs should balance between preventing duplicate processing and not blocking legitimate retries indefinitely.

## Performance Considerations

**Redis Operations**: All components perform 1-3 Redis operations per call. Rate limiter performs 3-4 operations (ZREMRANGEBYSCORE, ZCARD, ZADD, EXPIRE).

**Sliding Window Memory**: Rate limiter stores one sorted set entry per request within the window. Memory usage scales with rate × period.

**Circuit Breaker Overhead**: Minimal - single GET operation to check state, single SET/DELETE on state transitions.

**Idempotency Overhead**: Two Redis operations per task (SET NX on entry, SET on exit).

**Key Parsing**: Key parsing is O(n) where n is the number of key segments. Should be negligible for typical key lengths.

## Migration Strategy

**Phase 1**: Deploy new classes alongside existing code. No breaking changes.

**Phase 2**: Migrate tasks incrementally to use ReliabilitySystem and IdempotencyManager.

**Phase 3**: Deploy RateLimiter for tasks requiring rate limiting.

**Phase 4**: Monitor Redis memory usage and adjust TTLs/periods as needed.

**Rollback**: All components delegate to existing functional helpers. Rolling back means reverting to direct helper calls.

## Testing Strategy

**Unit Tests**: Test each component in isolation with Redis mocks. Focus on configuration validation, key format construction, error handling.

**Property Tests**: Generate random configurations, keys, and operation sequences. Verify invariants hold across all inputs. Minimum 100 iterations per property test.

**Integration Tests**: Test against real Redis instance. Verify sliding window behavior, TTL expiration, concurrent access patterns.

**Load Tests**: Simulate high request rates to verify rate limiter accuracy under load.

## Open Questions

None at this time.
