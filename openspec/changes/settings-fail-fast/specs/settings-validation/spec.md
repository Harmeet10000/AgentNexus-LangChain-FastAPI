# Capability: settings-validation

## Purpose

Prevent running the application in production with default/insecure secrets. Fail fast at import time with a clear error message.

## Requirements

### R1: Secret Field Registry
Define a list of fields that must not be defaults in production:
```python
_PRODUCTION_SECRET_FIELDS: list[tuple[str, str]] = [
    ("JWT_SECRET_KEY", "super-secret-change-this-in-production"),
    ("NEO4J_PASSWORD", "password"),
    ("OAUTH_STATE_SECRET", "your-oauth-state-secret"),
    ("GEMINI_API_KEY", ""),
    ("RESEND_API_KEY", ""),
    ("S3_ACCESS_KEY_ID", ""),
    ("S3_SECRET_ACCESS_KEY", ""),
    ("PINECONE_API_KEY", ""),
    ("TAVILY_API_KEY", ""),
    ("LANGSMITH_API_KEY", ""),
]
```

### R2: Validation Logic
- Add `model_post_init(self, __context: object)` to `Settings`
- If `self.ENVIRONMENT == "production"`:
  - Check each field in `_PRODUCTION_SECRET_FIELDS`
  - If `getattr(self, field_name).get_secret_value() == bad_default`:
    - Collect into `bad_fields` list
  - If `bad_fields` is non-empty:
    - Raise `ValueError` with message: `f"Production secrets not configured: {', '.join(bad_fields)}. Set these environment variables before starting the app."`
- If `self.ENVIRONMENT in ("development", "staging")`:
  - Same check, but `logger.warning()` instead of raising

### R3: Error Message Format
```
Production secrets not configured: JWT_SECRET_KEY, NEO4J_PASSWORD, GEMINI_API_KEY.
Set these environment variables before starting the app.
```

### R4: Logging
- Production error: `logger.error("Settings validation failed", bad_fields=[...])`
- Dev/staging warning: `logger.warning("Settings validation warning", bad_fields=[...])`

## Acceptance Criteria
- [ ] `Settings(ENVIRONMENT="production")` raises `ValueError` if secrets are defaults
- [ ] `Settings(ENVIRONMENT="development")` logs warning but succeeds
- [ ] Error message lists all bad field names
- [ ] No secret values are logged or included in error message
- [ ] Existing tests still pass (no regression)

## Non-Goals
- Secret format validation (length, complexity)
- Runtime secret rotation
- Secrets manager integration
