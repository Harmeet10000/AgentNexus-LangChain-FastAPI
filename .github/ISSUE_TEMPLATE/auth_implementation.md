---
name: 🔐 Implement Production-Ready Authentication System
about: Build complete auth system with JWT, OAuth2, session management, and email verification
title: '[Feature]: Production-Ready Authentication System'
labels: ['enhancement', 'security', 'auth']
assignees: []
---

## Problem Statement

The application needs a production-grade authentication system that supports:
- Email/password registration and login with secure password hashing
- JWT-based access and refresh tokens with proper expiration
- OAuth2 integration (Google, GitHub)
- Email verification and password reset flows
- Session management with device tracking and revocation
- Role-based access control (RBAC) with granular permissions
- Resilient background email tasks with idempotency and circuit breakers

## Proposed Solution

Implement a complete auth feature module under `src/app/features/auth/` following the project's modular monolith architecture.

### Core Components

#### 1. Data Models (`model.py`)
- **User document** (MongoDB/Beanie):
  - Email (unique indexed), hashed password (Argon2id)
  - OAuth accounts (embedded list for Google/GitHub)
  - Verification and reset token hashes (SHA-256)
  - Role (admin/moderator/user) with permission mapping
  - Active/verified flags, timestamps
- **Indexes**: email (unique), verification_token_hash (sparse), reset_token_hash (sparse)

#### 2. DTOs (`dto.py`)
- **Requests**: RegisterRequest, LoginRequest, RefreshRequest, LogoutRequest, VerifyEmailRequest, ResendVerificationRequest, ForgotPasswordRequest, ResetPasswordRequest
- **Responses**: TokenResponse (camelCase aliases), UserResponse, SessionResponse, OAuthAuthorizeResponse
- All request models use `extra="forbid"` and strong password validation

#### 3. Security Layer (`security.py`)
- **Password hashing**: Argon2id with transparent rehashing on login
- **Token generation**: Cryptographically secure URL-safe tokens for email flows
- **JWT**: HS256 with configurable expiration, issuer validation, embedded session_id
- **OAuth2**: State signing with itsdangerous, provider configs (Google/GitHub), userinfo normalization
- **Token claims**: Structured dataclass with sub, jti, sid, role, permissions, token_type

#### 4. Repository Layer (`repository.py`)
- **UserRepository**: CRUD, email/token lookups, OAuth user creation with upsert logic
- **RefreshTokenRepository**: Redis-backed session storage with TTL, revocation by session/user/device

#### 5. Service Layer (`service.py`)
- **Registration**: Hash password, store verification token hash, dispatch email task
- **Login**: Constant-time email existence check, password verify, transparent rehash, session creation
- **Token refresh**: Redis session validation, new access token generation (no rotation)
- **Email verification**: Token hash lookup, mark verified
- **Password reset**: Token generation with expiration, reset flow, revoke all sessions on success
- **OAuth2**: Authorization URL generation with signed state, callback with userinfo fetch, find-or-create user
- **Session management**: List, revoke single, revoke all (with optional keep-current)

#### 6. Router (`router.py`)
- **POST /auth/register**: Create account, send verification email
- **POST /auth/login**: Authenticate, return tokens, set httpOnly cookie
- **POST /auth/logout**: Revoke session, clear cookie
- **POST /auth/refresh**: Issue new access token from refresh token
- **POST /auth/verify-email**: Verify email with token
- **POST /auth/resend-verification**: Resend verification email
- **POST /auth/forgot-password**: Send password reset email
- **POST /auth/reset-password**: Reset password, revoke all sessions
- **GET /auth/oauth/{provider}/authorize**: Generate OAuth URL, set signed state cookie
- **GET /auth/oauth/{provider}/callback**: Exchange code, create/link user, redirect with tokens
- **GET /auth/me**: Return current user profile (protected)
- **GET /auth/sessions**: List active sessions (protected)
- **DELETE /auth/sessions/{session_id}**: Revoke specific session (protected)
- **DELETE /auth/sessions**: Revoke all sessions with optional keep-current (protected)

#### 7. Dependencies (`dependencies.py`)
- **CurrentUser**: Extract and validate JWT from Authorization header or cookie
- **CurrentVerifiedUser**: Require verified email
- **CurrentClaims**: Return raw token claims for session operations
- **RequirePermission**: Dependency factory for permission checks
- **AuthServiceDep**: Compose UserRepository + RefreshTokenRepository + AuthService

#### 8. Background Tasks (`src/tasks/auth_email_tasks.py`)
- **send_verification_email**: Uses ResilientTask base, idempotency lock, circuit breaker
- **send_password_reset_email**: Same resilience patterns
- Both tasks accept `idempotency_key` parameter and return structured dict
- Placeholder for real mailer integration

#### 9. Configuration (`settings.py` + `.env.development`)
- JWT_SECRET_KEY, JWT_ISSUER, JWT_ALGORITHM
- ACCESS_TOKEN_EXPIRE_MINUTES (15), REFRESH_TOKEN_EXPIRE_DAYS (30)
- PASSWORD_RESET_EXPIRE_MINUTES (30)
- OAUTH_STATE_SECRET (separate from JWT secret)
- GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
- GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET
- BACKEND_URL, FRONTEND_URL

### Security Features

- **Argon2id** password hashing with transparent rehashing
- **Constant-time** email existence checks during login
- **SHA-256** hashed tokens stored in DB; raw tokens only in email
- **JWT** with issuer validation, embedded session_id for revocation
- **OAuth2** state CSRF protection with signed cookies
- **Session revocation** via Redis with TTL
- **httpOnly, secure, samesite=lax** cookies for access tokens
- **Refresh tokens** in URL fragment (never hits server logs)
- **Permission-based** authorization with role mapping

### Resilience Features

- **ResilientTask** base for email tasks with exponential backoff + jitter
- **Idempotency locks** prevent duplicate emails on redelivery
- **Circuit breaker** protects against flaky email providers
- **Permanent vs transient** failure handling
- **Structured logging** with user_id, email, session_id context

## Implementation Checklist

### Phase 1: Core Auth
- [ ] Create User model with Beanie indexes
- [ ] Implement security.py (password hashing, JWT, token generation)
- [ ] Build UserRepository with email/token lookups
- [ ] Build RefreshTokenRepository with Redis session storage
- [ ] Implement AuthService (register, login, logout, refresh)
- [ ] Create auth DTOs with validation
- [ ] Build auth router with registration and login endpoints
- [ ] Add auth dependencies (CurrentUser, CurrentVerifiedUser, CurrentClaims)
- [ ] Wire auth router into main.py

### Phase 2: Email Flows
- [ ] Add email verification flow (verify, resend)
- [ ] Add password reset flow (forgot, reset)
- [ ] Implement auth_email_tasks.py with ResilientTask
- [ ] Update service.py to dispatch email tasks with idempotency keys
- [ ] Add verification/reset token hash indexes to User model

### Phase 3: OAuth2
- [ ] Add OAuth provider configs (Google, GitHub)
- [ ] Implement OAuth state signing and verification
- [ ] Build OAuth userinfo fetching and normalization
- [ ] Add OAuth authorize and callback endpoints
- [ ] Update User model with oauth_accounts field
- [ ] Implement find_or_create_oauth_user in repository

### Phase 4: Session Management
- [ ] Add session listing endpoint
- [ ] Add single session revocation endpoint
- [ ] Add bulk session revocation endpoint
- [ ] Embed session_id in access token claims
- [ ] Add device tracking (device_id, device_name, ip, user_agent)

### Phase 5: RBAC
- [ ] Define UserRole enum (admin, moderator, user)
- [ ] Define Permission enum with granular permissions
- [ ] Create ROLE_PERMISSIONS mapping
- [ ] Add get_permissions and has_permission methods to User model
- [ ] Implement RequirePermission dependency factory
- [ ] Add role and permissions to JWT claims

### Phase 6: Testing & Documentation
- [ ] Unit tests for security.py (hashing, JWT, OAuth state)
- [ ] Unit tests for service.py (all flows)
- [ ] Integration tests for auth router endpoints
- [ ] E2E tests for OAuth callback flow
- [ ] API documentation with examples
- [ ] Add auth usage example to src/app/examples/

## Configuration Requirements

Add to `.env.development`:
```env
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ISSUER=your-app
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=30
PASSWORD_RESET_EXPIRE_MINUTES=30
OAUTH_STATE_SECRET=your-oauth-state-secret
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
BACKEND_URL=http://localhost:5000
FRONTEND_URL=http://localhost:3000
```

Add to `settings.py`:
```python
JWT_SECRET_KEY: str
JWT_ISSUER: str = "your-app"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
REFRESH_TOKEN_EXPIRE_DAYS: int = 30
PASSWORD_RESET_EXPIRE_MINUTES: int = 30
OAUTH_STATE_SECRET: str
GOOGLE_CLIENT_ID: str = ""
GOOGLE_CLIENT_SECRET: str = ""
GITHUB_CLIENT_ID: str = ""
GITHUB_CLIENT_SECRET: str = ""
BACKEND_URL: str = "http://localhost:5000"
FRONTEND_URL: str = "http://localhost:3000"
```

## Dependencies

Add to `pyproject.toml`:
```toml
argon2-cffi = "^23.1.0"
authlib = "^1.3.0"
itsdangerous = "^2.2.0"
pyjwt = "^2.9.0"  # or use authlib.jose
```

## API Endpoints

### Public
- `POST /api/v1/auth/register` - Create account
- `POST /api/v1/auth/login` - Authenticate
- `POST /api/v1/auth/refresh` - Refresh access token
- `POST /api/v1/auth/verify-email` - Verify email
- `POST /api/v1/auth/resend-verification` - Resend verification
- `POST /api/v1/auth/forgot-password` - Request password reset
- `POST /api/v1/auth/reset-password` - Reset password
- `GET /api/v1/auth/oauth/{provider}/authorize` - Get OAuth URL
- `GET /api/v1/auth/oauth/{provider}/callback` - OAuth callback

### Protected
- `POST /api/v1/auth/logout` - Logout (revoke session)
- `GET /api/v1/auth/me` - Get current user
- `GET /api/v1/auth/sessions` - List active sessions
- `DELETE /api/v1/auth/sessions/{session_id}` - Revoke session
- `DELETE /api/v1/auth/sessions` - Revoke all sessions

## Architecture Decisions

### Why Argon2id over bcrypt?
- Memory-hard algorithm resistant to GPU/ASIC attacks
- Transparent rehashing allows upgrading params without forcing password resets
- Recommended by OWASP for password storage

### Why Redis for sessions instead of JWT-only?
- Instant revocation without token blacklisting
- Device tracking and session management
- Supports "logout everywhere" and "logout other devices"
- TTL-based automatic cleanup

### Why separate access and refresh tokens?
- Short-lived access tokens (15min) limit exposure window
- Long-lived refresh tokens (30d) reduce login friction
- Refresh tokens can be revoked; access tokens are stateless

### Why hash email verification/reset tokens?
- Raw tokens only travel via email (single exposure point)
- DB compromise doesn't leak usable tokens
- SHA-256 is sufficient for one-way hashing of high-entropy tokens

### Why signed OAuth state cookie?
- Stateless CSRF protection without server-side storage
- Binds state to provider to prevent cross-provider attacks
- 5-minute expiration limits replay window

### Why ResilientTask for email tasks?
- Idempotency prevents duplicate emails on redelivery
- Circuit breaker protects against flaky email providers
- Exponential backoff with jitter reduces thundering herd
- Permanent failure handling prevents infinite retries

## Alternatives Considered

### JWT in localStorage vs httpOnly cookie
- **Chosen**: httpOnly cookie for access token, refresh token in memory/URL fragment
- **Rejected**: localStorage vulnerable to XSS; cookies with httpOnly + secure + samesite=lax provide better protection

### Token rotation on refresh
- **Chosen**: No rotation (simpler, fewer edge cases)
- **Rejected**: Rotation adds complexity with concurrent refresh requests and requires careful handling of race conditions

### OAuth state in Redis vs signed cookie
- **Chosen**: Signed cookie (stateless, no Redis dependency for OAuth flow)
- **Rejected**: Redis state requires cleanup logic and adds latency

## Additional Context

This implementation follows the project's architecture rules:
- Thin router handlers, business logic in service layer
- Repository layer for persistence only
- Explicit dependency passing via FastAPI Depends
- Async-first with motor (MongoDB) and redis.asyncio
- Structured logging with context binding
- Typed exceptions from utils/exceptions.py
- APIResponse envelope for consistent responses
- ResilientTask base for background jobs

## Priority

**High** - Authentication is a foundational feature required for user management, authorization, and protected endpoints.

## Success Criteria

- [ ] All endpoints return consistent APIResponse envelope
- [ ] Password hashing uses Argon2id with transparent rehashing
- [ ] JWT tokens include issuer validation and embedded session_id
- [ ] OAuth2 flows complete successfully with Google and GitHub
- [ ] Email verification and password reset work end-to-end
- [ ] Session revocation is instant via Redis
- [ ] Email tasks use idempotency and circuit breaker patterns
- [ ] All auth endpoints pass integration tests
- [ ] No secrets or credentials in logs or error messages
- [ ] Timing attacks mitigated in login flow
- [ ] RBAC permissions enforced on protected endpoints
