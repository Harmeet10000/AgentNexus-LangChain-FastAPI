# Bugfix Requirements Document

## Introduction

WebSocket security implementation has multiple critical and high-priority issues affecting session revocation, presence tracking, rate limiting, and configuration. These issues create production security vulnerabilities where authenticated WebSocket connections can persist beyond their intended lifetime, presence tracking has race conditions, and configuration is incomplete.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN a user's session is revoked via `AuthService.revoke_session()` THEN the system does NOT close live WebSocket connections associated with that session - connections remain authenticated indefinitely

1.2 WHEN a JWT access token expires (after 15 minutes) THEN the system does NOT re-validate the token on active WebSocket connections - connections opened at minute 0 stay authenticated for the entire process lifetime

1.3 WHEN `touch_connection()` is called on every message (in/out) THEN the system performs Redis roundtrips for each message - creating latency in the hot path of every streamed message

1.4 WHEN `ensure_connection_capacity()` checks connection count THEN the system has a TOCTOU (time-of-check-to-time-of-use) race condition because it performs a read-then-write operation that is not atomic

1.5 WHEN WebSocket rate limiting is applied THEN the system incorrectly mutates `websocket.state` between limiter calls instead of using direct `Limiter.try_acquire()` calls - rate limiting is bypassed

1.6 WHEN `ws_url` configuration is used with `{host}` template placeholder THEN the system does NOT resolve the actual URL using `X-Forwarded-Proto` and `X-Forwarded-Host` headers - proxy-aware resolution is missing

1.7 WHEN checking production secret field compliance THEN the system's `PRODUCTION_SECRET_FIELDS` list is incomplete - missing `RABBITMQ_DEFAULT_PASS` and `POSTGRES_PASSWORD` fields

1.8 WHEN `MCP_CLIENT_RETRY_ATTEMPTS` configuration is read THEN the system defaults to 1 (no retry) despite circuit breaker config existing - retry is disabled by default

### Expected Behavior (Correct)

2.1 WHEN a user's session is revoked via `AuthService.revoke_session()` THEN the system SHALL close all live WebSocket connections associated with that session by implementing pull-based revocation check every 30 seconds

2.2 WHEN a JWT access token expires (after 15 minutes) THEN the system SHALL re-validate the token on active WebSocket connections by re-reading from Redis and closing connections if the session no longer exists

2.3 WHEN presence tracking updates are needed THEN the system SHALL use atomic Redis sorted set operations instead of separate Redis keys - reducing key patterns from three to one sorted set per user

2.4 WHEN connection capacity is checked THEN the system SHALL perform atomic check-and-increment operations to prevent TOCTOU race conditions in `ensure_connection_capacity()`

2.5 WHEN WebSocket rate limiting is applied THEN the system SHALL use direct `Limiter.try_acquire()` calls instead of mutating `websocket.state` between calls - proper rate limiter usage

2.6 WHEN `ws_url` configuration is used WITH `{host}` template placeholder THEN the system SHALL resolve the actual URL using `X-Forwarded-Proto` and `X-Forwarded-Host` headers for proxy-aware resolution

2.7 WHEN checking production secret field compliance THEN the system's `PRODUCTION_SECRET_FIELDS` list SHALL include `RABBITMQ_DEFAULT_PASS` and `POSTGRES_PASSWORD` fields to ensure production security

2.8 WHEN `MCP_CLIENT_RETRY_ATTEMPTS` configuration is read THEN the system SHALL default to a value greater than 1 to enable retry despite circuit breaker config existing

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a WebSocket connection is established with valid credentials THEN the system SHALL continue to authenticate the connection using JWT validation

3.2 WHEN `touch_connection()` is called for existing connections THEN the system SHALL continue to maintain Redis presence keys for active connections

3.3 WHEN multiple WebSocket connections exist for the same user THEN the system SHALL continue to enforce `WEBSOCKET_MAX_CONNECTIONS_PER_USER` limit

3.4 WHEN rate limiting is applied to WebSocket messages THEN the system SHALL continue to track and limit message rates per user and per connection

3.5 WHEN session revocation is requested via `AuthService.revoke_session()` THEN the system SHALL continue to invalidate refresh tokens in Redis

3.6 WHEN WebSocket connections are closed THEN the system SHALL continue to clean up Redis presence keys via `unregister_connection()`

3.7 WHEN `PRODUCTION_SECRET_FIELDS` validation runs in production THEN the system SHALL continue to raise errors for known insecure default values

3.8 WHEN MCP client requests are made THEN the system SHALL continue to use circuit breaker configuration for retry behavior
