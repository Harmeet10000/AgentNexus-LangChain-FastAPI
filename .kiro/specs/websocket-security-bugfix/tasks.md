# Implementation Plan

## Bug Condition Exploration Test

**Property 1: Bug Condition** - WebSocket Security Vulnerabilities

- [ ] 1. Write bug condition exploration tests (BEFORE implementing fix)
  - **Property 1: Bug Condition** - Session Revocation Gap and TOCTOU Vulnerabilities
  - **IMPORTANT**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **GOAL**: Surface counterexamples that demonstrate the security vulnerabilities exist
  - **Test Categories**:
    - **Session Revocation Gap**: Create WebSocket connection, revoke session, verify connection remains open (P0)
    - **JWT Expiry Ignored**: Create connection with token, advance clock, verify connection persists (P0)
    - **TOCTOU Capacity Overflow**: 4 simultaneous connections for max=3, verify all 4 accepted (P0)
    - **Rate Limiter Bypass**: Send 65 messages in 60s (limit 60), verify all accepted (P1)
  - **Expected Outcome**: Tests FAIL on unfixed code (confirms vulnerabilities exist)
  - **Documentation Required**: Record counterexamples for each vulnerability
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

## Preservation Property Tests

**Property 2: Preservation** - Valid Connection Handling

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Normal Connection Flow and Rate Limiting
  - **IMPORTANT**: Follow observation-first methodology
  - **Observe Unfixed Behavior**:
    - Normal connection establishment with valid credentials
    - Message handling for valid sessions
    - Presence updates via touch_connection()
    - Rate limiting behavior (60/60s per user, 20/10s per connection)
    - Connection capacity enforcement for non-buggy inputs
  - **Property-Based Tests**:
    - **Property 2.1**: Valid connections maintain authentication and message handling
    - **Property 2.2**: Presence keys updated correctly during touch_connection()
    - **Property 2.3**: Rate limiting enforces limits for normal traffic patterns
    - **Property 2.4**: Connection capacity enforced for valid arrival patterns
  - **Verification Required**: Tests PASS on unfixed code (baseline behavior)
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

## Implementation Tasks

- [ ] 3. Fix WebSocket Security Vulnerabilities

  - [ ] 3.1 Implement pull-based session revocation check loop
    - Create 30-second periodic background task in lifespan
    - Re-read `token_repo.get_session(context.session_id)` for each active connection
    - Close connection with `SESSION_REVOKED` violation if session gone
    - Maintain bounded staleness: 30-second check cycle
    - _Bug_Condition: Session revocation or JWT expiry leaves connections open_
    - _Expected_Behavior: Connection closes within 30 seconds of revocation/expiry_
    - _Preservation: Valid connections continue normal operation_
    - _Requirements: 2.1, 2.2_

  - [ ] 3.2 Refactor presence tracking to sorted sets for atomic operations
    - Replace `_USER_CONNECTIONS_KEY`, `_SESSION_CONNECTIONS_KEY`, `_CONNECTION_KEY` patterns
    - Add `_USER_PRESENCE_KEY`, `_SESSION_PRESENCE_KEY` sorted sets
    - Member = `connection_id`, score = last-touch epoch (Unix timestamp)
    - Use `ZADD` + `ZCARD` atomic pipeline in `ensure_connection_capacity()`
    - Use `ZREMRANGEBYSCORE` for TTL-based eviction in `touch_connection()`
    - Update `register_connection()`, `unregister_connection()`, `get_active_connection_count()`
    - _Bug_Condition: TOCTOU race condition in capacity check_
    - _Expected_Behavior: Atomic check-and-increment prevents overflow_
    - _Preservation: Presence tracking still works for valid connections_
    - _Requirements: 2.4_

  - [ ] 3.3 Fix rate limiter to use direct try_acquire() calls
    - Replace `WebSocketRateLimiter` wrapper with direct `Limiter.try_acquire()` calls
    - Remove `websocket.state.ws_rate_limit_id` mutation
    - Use explicit identifier keys: `f"user:{user_id}"` and `f"connection:{connection_id}"`
    - Pass explicit identifiers to `try_acquire()` instead of relying on state
    - _Bug_Condition: Rate limiter bypassed due to incorrect state mutation_
    - _Expected_Behavior: Rate limiting enforces 60/60s per user, 20/10s per connection_
    - _Preservation: Rate limiting works correctly for valid traffic patterns_
    - _Requirements: 2.5_

  - [ ] 3.4 Add session revocation lookup from sorted sets
    - When `AuthService.revoke_session()` is called, look up connection IDs from `ws:session:{session_id}`
    - Trigger closure of all connections in the set
    - Return list of closed connection IDs for audit logging
    - _Expected_Behavior: All connections for revoked session are closed immediately_
    - _Preservation: Non-revoked sessions unaffected_
    - _Requirements: 2.3_

  - [ ] 3.5 Update configuration for production secrets and proxy resolution
    - Add `RABBITMQ_DEFAULT_PASS` and `POSTGRES_PASSWORD` to `PRODUCTION_SECRET_FIELDS`
    - Split `validate_embedding_dimension` into two validators
    - Change `MCP_CLIENT_RETRY_ATTEMPTS` default from 1 to 3
    - Share `FASTAPI_GUARD_TRUSTED_PROXIES` logic with WebSocket path
    - Honor `X-Forwarded-Proto` and `X-Forwarded-Host` headers in `get_websocket_url()`
    - _Expected_Behavior: Production configuration includes all required secrets and proxy handling_
    - _Preservation: Development configuration unchanged_
    - _Requirements: 2.6, 2.7, 2.8_

- [ ] 4. Verify Bug Condition Exploration Test Now Passes
  - **Property 1: Expected Behavior** - Session Revocation Gap and TOCTOU Fixed
  - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
  - The test from task 1 encodes the expected behavior
  - When this test passes, it confirms the expected behavior is satisfied
  - Run all bug condition exploration tests from step 1
  - **EXPECTED OUTCOME**: All tests PASS (confirms vulnerabilities are fixed)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

- [ ] 5. Verify Preservation Tests Still Pass
  - **Property 2: Preservation** - Normal Connection Flow Preserved
  - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
  - Run preservation property tests from step 2
  - **EXPECTED OUTCOME**: All tests PASS (confirms no regressions)
  - Confirm all tests still pass after fix (no regressions)
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 6. Checkpoint - Ensure All Tests Pass
  - Ensure all tests pass
  - Run `uv run ruff check --fix src/`
  - Run `uv run ty check src/`
  - Ask user if questions arise
