## Why

391 `str()` coercion calls exist in the codebase, but only 31 use the explicit `.get_secret_value()` pattern for `SecretStr` fields. Some of these `str()` calls likely wrap `SecretStr` objects, which either leaks secrets into logs/traces (if the object is serialized before masking) or produces `**********` output (masking happens too early, losing the value). The `SECRET_FIELDS` constant in `settings.py` defines which fields are secrets, but nothing enforces that consumers use `.get_secret_value()`.

## What Changes

- Add a project convention: all `SecretStr` fields MUST be accessed via `.get_secret_value()`, never via `str()` or f-string interpolation
- Audit all 391 `str()` calls in `src/` and fix any that wrap `SecretStr` objects
- Add a ruff-style or grep-based CI check that flags `str()` calls on known secret field names
- Document the pattern in `.opencode/instructions/PYTHON-TYPING-RULES.md`

## Capabilities

### New Capabilities
- `secretstr-access-pattern`: Enforces `.get_secret_value()` access for SecretStr fields and audits existing str() coercions

### Modified Capabilities

_(none)_

## Impact

- **Files**: `src/app/config/settings.py` (documentation), files under `src/app/features/` and `src/app/shared/` with `str()` on secrets
- **Dependencies**: None
- **APIs**: None
- **Risk**: Medium — incorrect fix could break secret passing at runtime; each `str()` → `.get_secret_value()` change needs manual review to confirm the field is actually `SecretStr`
