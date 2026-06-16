## Context

`Settings` at `src/app/config/settings.py:10` is a frozen Pydantic `BaseSettings` model with 100+ fields. Secret fields use `SecretStr` with empty-string or placeholder defaults:
- `JWT_SECRET_KEY: SecretStr = Field(default=SecretStr("super-secret-change-this-in-production"))`
- `NEO4J_PASSWORD: SecretStr = Field(default=SecretStr("password"))`
- `GEMINI_API_KEY: SecretStr = Field(default=SecretStr(""))`
- `RESEND_API_KEY: SecretStr = Field(default=SecretStr(""))`
- `OAUTH_STATE_SECRET: SecretStr = Field(default=SecretStr("your-oauth-state-secret"))`

The `get_settings()` function at line 308 is cached — `Settings()` is constructed once at import time. If validation runs in `model_post_init()`, it catches misconfiguration before any request is served.

## Goals / Non-Goals

**Goals:**
- Fail fast in production if secrets are defaults
- Warn in development/staging (allow startup)
- Clear error message listing which fields need to be set
- Zero impact on existing behavior when secrets are properly configured

**Non-Goals:**
- Validate secret format/strength (e.g., minimum length)
- Rotate secrets at runtime
- Add a secrets manager integration (AWS Secrets Manager, etc.)
- Change the Settings model structure

## Decisions

### D1: Validation in `model_post_init()`, not `__init__`

**Decision:** Add `model_post_init(self, __context: object)` to `Settings` that checks secret fields. Pydantic v2 calls this after field validation, so all values are already parsed.

**Rationale:** `model_post_init()` is the Pydantic v2 pattern for post-validation logic. It runs after all fields are validated, so we can safely inspect `self.ENVIRONMENT` and compare secret values.

**Alternatives considered:**
- *Custom `__init__'*: bypasses Pydantic validation — rejected
- *Validator on each field*: too noisy, can't access `ENVIRONMENT` — rejected
- *Separate startup check*: runs after import, could miss — rejected

### D2: Secret field list — hardcoded, not introspected

**Decision:** Define a `PRODUCTION_SECRET_FIELDS` list of field names that must not be defaults in production. Check each field's `.get_secret_value()` against a list of known-bad defaults.

**Rationale:** Explicit list is clear and maintainable. Introspection (checking all `SecretStr` fields) would catch too many fields (e.g., `S3_SECRET_ACCESS_KEY` might legitimately be empty in dev).

**Alternatives considered:**
- *Introspect all `SecretStr` fields*: too broad — rejected
- *Config-file based*: adds complexity — rejected
- *Environment variable presence check*: doesn't catch wrong values — rejected

### D3: Warning in non-production, error in production

**Decision:** 
- `ENVIRONMENT=production`: raise `ValueError` with full list of bad fields
- `ENVIRONMENT=development` or `staging`: log `logger.warning()` with the list

**Rationale:** Production must fail fast. Development should warn but not block (developers might forget to set up env vars locally).

## Risks / Trade-offs

- **[False positive]** A developer might have a legitimate reason to use a default secret in staging. **Mitigation:** Only check in `production`; staging gets a warning, not an error.
- **[Cached settings]** `get_settings()` is `@cache`d. If `model_post_init()` raises, the exception happens at import time. **Mitigation:** This is the desired behavior — fail before any request is served.
- **[SecretStr comparison]** Comparing `SecretStr` values requires `.get_secret_value()`. **Mitigation:** Only compare against known-bad defaults, never log the actual secret value.
