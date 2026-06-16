# Capability: settings-validation

## Purpose

Fail fast at import time if `ENVIRONMENT=production` and secret fields still have default/insecure values.

## ADDED Requirements

### Requirement: Secret Field Registry

The `Settings` module SHALL define `PRODUCTION_SECRET_FIELDS` as a dict mapping field name to a list of known-bad default values. The registry MUST include: `JWT_SECRET_KEY`, `NEO4J_PASSWORD`, `GEMINI_API_KEY`, `RESEND_API_KEY`, `OAUTH_STATE_SECRET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `TAVILY_API_KEY`, `PINECONE_API_KEY`.

#### Scenario: Registry defines known-bad defaults

- Given `PRODUCTION_SECRET_FIELDS` dict at module level
- When the Settings class is instantiated
- Then each field is checked against its known-bad default list

### Requirement: Production Fail-Fast Validation

The `model_post_init()` method SHALL check `self.ENVIRONMENT`. If `production`, it SHALL iterate `PRODUCTION_SECRET_FIELDS`, compare `.get_secret_value()` against bad defaults, and raise `ValueError` with a message listing all bad field names. The actual secret values MUST NOT be exposed in the error message.

#### Scenario: Production fails with bad secrets

- Given `ENVIRONMENT=production`
- When a secret field has its known-bad default value
- Then `Settings()` raises `ValueError`

#### Scenario: Production succeeds with proper secrets

- Given `ENVIRONMENT=production`
- When all secret fields have non-default values
- Then `Settings()` succeeds

#### Scenario: Error lists field names without exposing values

- Given production with bad secrets
- When `Settings()` raises
- Then the error message contains "The following secret fields have default/insecure values"
- And the error message does NOT contain the actual secret values

### Requirement: Non-Production Warning

If `ENVIRONMENT` is not `production` and bad defaults are found, the system SHALL log a `logger.warning()` with the list of bad field names and allow startup.

#### Scenario: Development succeeds with bad secrets

- Given `ENVIRONMENT=development`
- When secret fields have default values
- Then `Settings()` succeeds without error

#### Scenario: Secrets never logged in warnings

- Given development with bad secrets
- When the warning is logged
- Then actual secret values are not included in the log message

## Non-Goals

- Secret strength validation (length, complexity)
- Runtime secret rotation
- Secrets manager integration
