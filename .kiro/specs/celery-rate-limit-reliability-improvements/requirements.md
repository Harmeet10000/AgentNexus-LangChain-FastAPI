# Requirements Document

## Introduction

This document specifies requirements for improving the Celery task execution reliability and rate limiting system. The feature consolidates the existing circuit breaker and idempotency mechanisms into a unified base class, adds a context manager pattern for idempotency control, implements Redis-scoped rate limiting with embedded configuration, and integrates with existing FASTAPI_GUARD settings for proxy trust configuration.

## Glossary

- **Reliability_System**: The unified base class that provides circuit breaker and idempotency functionality for Celery tasks
- **Idempotency_Manager**: Context manager that controls idempotency lock lifecycle (acquire, release, mark completed/failed)
- **Rate_Limiter**: Redis-based rate limiting mechanism that prevents task execution beyond configured thresholds
- **Redis_Client**: The Redis client instance conforming to RedisClientProtocol used for state storage
- **Celery_Task**: A background task executed by Celery workers
- **Circuit_Breaker**: Mechanism that prevents execution of failing operations by tracking failure counts and state transitions
- **Idempotency_Key**: Business-level identifier used to prevent duplicate processing of the same operation
- **Rate_Limit_Key**: Redis key that embeds configuration parameters and scope for rate limit enforcement
- **Proxy_Trust_Config**: Configuration from FASTAPI_GUARD settings that determines trusted proxy behavior

## Requirements

### Requirement 1: Unified Reliability Base Class

**User Story:** As a Celery task developer, I want a single base class that provides both circuit breaker and idempotency functionality, so that I can apply reliability patterns consistently without duplicating code

#### Acceptance Criteria

1. THE Reliability_System SHALL provide circuit breaker functionality
2. THE Reliability_System SHALL provide idempotency checking functionality
3. THE Reliability_System SHALL accept Redis_Client as a constructor parameter
4. THE Reliability_System SHALL accept circuit breaker configuration as constructor parameters
5. THE Reliability_System SHALL accept idempotency configuration as constructor parameters
6. THE Reliability_System SHALL expose a method to check circuit breaker state
7. THE Reliability_System SHALL expose a method to record operation success
8. THE Reliability_System SHALL expose a method to record operation failure
9. THE Reliability_System SHALL expose a method to check idempotency status
10. THE Reliability_System SHALL use the existing functional helpers from celery_reliability.py

### Requirement 2: Idempotency Context Manager

**User Story:** As a Celery task developer, I want a context manager for idempotency control, so that locks are automatically acquired and released with proper cleanup

#### Acceptance Criteria

1. THE Idempotency_Manager SHALL be implemented as an async context manager
2. WHEN the Idempotency_Manager enters context, THE Idempotency_Manager SHALL attempt to acquire the idempotency lock
3. IF the idempotency lock is already held, THEN THE Idempotency_Manager SHALL raise an exception or return a skip signal
4. WHEN the Idempotency_Manager exits context normally, THE Idempotency_Manager SHALL mark the operation as completed
5. IF an exception occurs within the context, THEN THE Idempotency_Manager SHALL mark the operation as failed permanently
6. IF an exception occurs within the context and the failure is retryable, THEN THE Idempotency_Manager SHALL release the processing lock
7. THE Idempotency_Manager SHALL accept idempotency_key as a required parameter
8. THE Idempotency_Manager SHALL accept Redis_Client as a required parameter
9. THE Idempotency_Manager SHALL accept ttl_seconds as an optional parameter with default value from settings
10. THE Idempotency_Manager SHALL accept metadata as an optional parameter

### Requirement 3: Redis-Scoped Rate Limiter

**User Story:** As a Celery task developer, I want rate limiting with configuration embedded in Redis keys, so that different tasks can have different rate limits without external state management

#### Acceptance Criteria

1. THE Rate_Limiter SHALL embed rate limit configuration in the Redis key
2. THE Rate_Limiter SHALL use Redis for rate limit state storage
3. THE Rate_Limiter SHALL accept scope as a required parameter
4. THE Rate_Limiter SHALL accept rate as a required parameter (requests per period)
5. THE Rate_Limiter SHALL accept period_seconds as a required parameter
6. THE Rate_Limiter SHALL accept burst as an optional parameter
7. WHEN a task requests execution, THE Rate_Limiter SHALL check the current rate against the configured limit
8. IF the rate limit is exceeded, THEN THE Rate_Limiter SHALL reject the request
9. WHEN a request is allowed, THE Rate_Limiter SHALL increment the request count
10. WHEN the period expires, THE Rate_Limiter SHALL reset the request count
11. THE Rate_Limiter SHALL use a sliding window algorithm for rate calculation
12. THE Rate_Limiter SHALL return remaining capacity information

### Requirement 4: Rate Limit Key Format

**User Story:** As a system operator, I want rate limit configuration embedded in Redis keys, so that I can inspect and debug rate limit state without external documentation

#### Acceptance Criteria

1. THE Rate_Limiter SHALL construct Redis keys in the format "celery:ratelimit:{scope}:rate={rate}:period={period}:burst={burst}"
2. THE Rate_Limiter SHALL include the scope parameter in the key
3. THE Rate_Limiter SHALL include the rate parameter in the key
4. THE Rate_Limiter SHALL include the period_seconds parameter in the key
5. THE Rate_Limiter SHALL include the burst parameter in the key
6. THE Rate_Limiter SHALL parse configuration from existing keys when checking limits
7. IF configuration in the key differs from the requested configuration, THEN THE Rate_Limiter SHALL use the configuration from the key

### Requirement 5: Proxy Trust Configuration Integration

**User Story:** As a system administrator, I want rate limiting to use existing FASTAPI_GUARD proxy trust settings, so that client IP detection is consistent across the application

#### Acceptance Criteria

1. THE Rate_Limiter SHALL read FASTAPI_GUARD_TRUSTED_PROXIES from settings
2. THE Rate_Limiter SHALL read FASTAPI_GUARD_TRUSTED_PROXY_DEPTH from settings
3. WHERE proxy trust is configured, THE Rate_Limiter SHALL extract client IP using the configured proxy depth
4. WHERE proxy trust is not configured, THE Rate_Limiter SHALL use the direct connection IP
5. THE Rate_Limiter SHALL use the extracted client IP as part of the rate limit scope
6. THE Rate_Limiter SHALL validate that the proxy configuration is consistent with the expected request path

### Requirement 6: Circuit Breaker Integration

**User Story:** As a Celery task developer, I want circuit breaker state to prevent task execution when external services are failing, so that I avoid cascading failures

#### Acceptance Criteria

1. WHEN a Celery_Task is executed, THE Reliability_System SHALL check the circuit breaker state before proceeding
2. IF the circuit breaker is open, THEN THE Reliability_System SHALL reject the task execution
3. WHEN an operation succeeds, THE Reliability_System SHALL record success in the circuit breaker
4. WHEN an operation fails, THE Reliability_System SHALL record failure in the circuit breaker
5. IF the failure threshold is reached, THEN THE Reliability_System SHALL open the circuit breaker
6. WHEN the recovery timeout elapses, THE Reliability_System SHALL transition the circuit breaker to half-open state
7. IF an operation succeeds in half-open state, THEN THE Reliability_System SHALL close the circuit breaker
8. IF an operation fails in half-open state, THEN THE Reliability_System SHALL re-open the circuit breaker

### Requirement 7: Idempotency Status Checking

**User Story:** As a Celery task developer, I want to check idempotency status before executing a task, so that I can skip already-processed operations

#### Acceptance Criteria

1. THE Reliability_System SHALL provide a method to check idempotency status
2. WHEN checking idempotency status, THE Reliability_System SHALL query Redis using the idempotency_key
3. IF no record exists, THEN THE Reliability_System SHALL return None
4. IF a record exists with status "processing", THEN THE Reliability_System SHALL return "processing"
5. IF a record exists with status "completed", THEN THE Reliability_System SHALL return "completed"
6. IF a record exists with status "failed_permanent", THEN THE Reliability_System SHALL return "failed_permanent"
7. THE Reliability_System SHALL deserialize the record using IdempotencyRecord model

### Requirement 8: Configuration Management

**User Story:** As a system administrator, I want centralized configuration for reliability settings, so that I can tune behavior without code changes

#### Acceptance Criteria

1. THE Reliability_System SHALL read CELERY_IDEMPOTENCY_TTL_SECONDS from settings
2. THE Reliability_System SHALL read CELERY_CIRCUIT_BREAKER_FAILURE_THRESHOLD from settings
3. THE Reliability_System SHALL read CELERY_CIRCUIT_BREAKER_RECOVERY_TIMEOUT from settings
4. THE Reliability_System SHALL allow override of default TTL per operation
5. THE Reliability_System SHALL allow override of circuit breaker thresholds per operation
6. THE Reliability_System SHALL validate that configuration values are within acceptable ranges

### Requirement 9: Error Handling and Recovery

**User Story:** As a Celery task developer, I want clear error messages and proper exception types, so that I can handle reliability failures appropriately

#### Acceptance Criteria

1. IF the circuit breaker is open, THEN THE Reliability_System SHALL raise CircuitBreakerOpenError
2. IF idempotency lock acquisition fails, THEN THE Idempotency_Manager SHALL raise a descriptive exception
3. WHEN an exception is raised, THE Reliability_System SHALL include the operation name in the error message
4. WHEN an exception is raised, THE Reliability_System SHALL include relevant context (circuit breaker name, idempotency key) in the error message
5. THE Reliability_System SHALL use structured logging for all state transitions
6. THE Reliability_System SHALL log circuit breaker state changes with context
7. THE Reliability_System SHALL log idempotency lock lifecycle events with context

### Requirement 10: Backward Compatibility

**User Story:** As a maintainer, I want the new reliability system to be compatible with existing functional helpers, so that migration is incremental and safe

#### Acceptance Criteria

1. THE Reliability_System SHALL use the existing functional helpers from celery_reliability.py
2. THE Reliability_System SHALL not modify the existing function signatures
3. THE Reliability_System SHALL not change the Redis key format for existing idempotency and circuit breaker keys
4. THE Reliability_System SHALL maintain the existing IdempotencyRecord model structure
5. THE Reliability_System SHALL maintain the existing CircuitBreakerState model structure
6. THE Reliability_System SHALL maintain the existing namespace constants (IDEMPOTENCY_NAMESPACE, CIRCUIT_BREAKER_NAMESPACE)
