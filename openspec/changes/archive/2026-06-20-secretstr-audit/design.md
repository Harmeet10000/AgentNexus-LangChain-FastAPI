## Context

Pydantic v2's `SecretStr` type automatically masks values in `repr()`, `model_dump()`, and `model_dump_json()`. To extract the actual string, consumers MUST call `.get_secret_value()`. The project's `settings.py` defines `SECRET_FIELDS` listing all secret field names, but there's no enforcement — code can (and does) call `str()` on `SecretStr` objects, which either masks too early or leaks depending on context.

ast-grep found 391 `str()` calls vs 31 `.get_secret_value()` calls. Manual audit confirmed several files call `str()` on settings fields that are `SecretStr`.

## Goals / Non-Goals

**Goals:**
- Audit all 391 `str()` calls and fix any that coerce `SecretStr` objects
- Establish a project convention: `SecretStr` fields always accessed via `.get_secret_value()`
- Add a CI check that flags `str()` on known secret field names

**Non-Goals:**
- Encrypting `SecretStr` at rest (already handled by Pydantic v2)
- Replacing `SecretStr` with a custom encrypted type
- Auditing non-secret `str()` calls

## Decisions

1. **Manual audit + grep-based CI check, not ast-grep rule**
   - Rationale: The set of secret field names is defined in `settings.py` (`SECRET_FIELDS`). A grep for `str(settings.SECRET_FIELD_NAME)` is simpler and more maintainable than an ast-grep AST rule for this pattern.
   - Alternative considered: Custom ast-grep rule — rejected because the pattern is field-name-dependent, not structure-dependent.

2. **Fix: replace `str(settings.FIELD)` with `settings.FIELD.get_secret_value()`**
   - Rationale: This is the canonical Pydantic v2 pattern. The return type changes from `str` to `str`, so no downstream type changes needed.

3. **CI check as a simple shell script or ruff custom rule**
   - Rationale: Keep it minimal. A grep-based check in CI that fails if `str()` is called on known secret field names is enough.

4. **Document in `PYTHON-TYPING-RULES.md` under a new `### SecretStr` section**
   - Rationale: The existing typing rules already cover `PrivateAttr` and `Field` patterns — `SecretStr` access is a natural addition.

## Risks / Trade-offs

- **Risk**: Incorrectly identifying a `str()` call that's actually on a non-secret. **Mitigation**: Audit is per-file, each fix is manual and verifiable. CI check uses exact field names from `SECRET_FIELDS`.
- **Risk**: Breaking code that passes `str(secret)` to a function expecting `str`. **Mitigation**: `.get_secret_value()` returns `str`, so types are compatible. No downstream changes needed.
