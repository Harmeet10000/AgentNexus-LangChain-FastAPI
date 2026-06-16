# Capability: auth-integration-tests

## Purpose

Integration tests for the auth lifecycle: register, login, refresh, logout, email verification, password reset, OAuth, session management. Catches regressions in the highest-risk auth paths.

## Requirements

### R1: Test Fixtures
- `tests/conftest.py` with session-scoped PostgreSQL async session
- `tests/conftest.py` with fakeredis instance for session storage
- `tests/conftest.py` with `httpx.AsyncClient` for endpoint tests
- Per-test cleanup: rollback PostgreSQL transactions, flush Redis

### R2: Register Flow
- Test: successful registration creates user with hashed password
- Test: duplicate email raises `ConflictException`
- Test: password is argon2-hashed (not stored in plaintext)
- Test: verification token is generated and stored

### R3: Login Flow
- Test: successful login returns access + refresh tokens
- Test: wrong password raises `UnauthorizedException`
- Test: non-existent email raises `UnauthorizedException` (same message as wrong password — timing-attack safe)
- Test: disabled account raises `UnauthorizedException`
- Test: unverified email raises `UnauthorizedException`
- Test: transparent rehash when argon2 params are outdated

### R4: Token Refresh
- Test: valid refresh token returns new access token
- Test: expired refresh token raises `UnauthorizedException`
- Test: revoked refresh token raises `UnauthorizedException`
- Test: refresh token for non-existent user raises `UnauthorizedException`

### R5: Logout
- Test: logout revokes refresh token in Redis
- Test: accessing revoked refresh token after logout fails

### R6: Session Management
- Test: `list_sessions` returns active sessions
- Test: `revoke_session` removes specific session
- Test: `revoke_all_sessions` removes all sessions except current
- Test: session metadata (device_name, ip, user_agent) is stored

### R7: Password Reset
- Test: forgot_password generates reset token and stores hash
- Test: reset_password updates password and revokes all sessions
- Test: expired reset token raises `UnauthorizedException`

## Acceptance Criteria
- [ ] All R2-R7 tests pass
- [ ] Tests use real PostgreSQL (asyncpg) + fakeredis
- [ ] No LLM calls (Gemini mocked)
- [ ] Test execution time < 30s for full auth suite

## Non-Goals
- OAuth provider integration (mock userinfo)
- Rate limiting tests
- WebSocket auth tests
