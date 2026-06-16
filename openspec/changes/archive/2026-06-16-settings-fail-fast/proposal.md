## Why

The `Settings` class at `src/app/config/settings.py` has insecure defaults for production secrets:
- `JWT_SECRET_KEY` defaults to `"super-secret-change-this-in-production"`
- `NEO4J_PASSWORD` defaults to `"password"`
- `GEMINI_API_KEY` defaults to `""`
- `RESEND_API_KEY` defaults to `""`

If someone deploys with `ENVIRONMENT=production` without setting these env vars, the app runs silently with known secrets. This is a security incident waiting to happen. The fix: fail fast at import time if production mode is detected with default secrets.

## What Changes

### Settings Validation
- Add `model_post_init()` to `Settings` that checks secret fields when `ENVIRONMENT=production`
- Define a list of "secret fields" that must not be defaults in production
- Raise `ValueError` at import time if validation fails
- Development/staging modes: warn but allow startup

## Capabilities

### New Capabilities
- `settings-fail-fast`: Startup validation that prevents running production with default secrets

### Modified Capabilities
- (none)

## Impact

### Affected Code
- `src/app/config/settings.py` — add `model_post_init()` validation
- No other files changed

### Affected APIs
- None

### Dependencies Added
- None

### Systems
- CI: settings validation runs at import time (no separate test needed)
