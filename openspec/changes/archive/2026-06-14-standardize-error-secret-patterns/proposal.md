## Why

Error codes are currently string literals scattered across 20+ exception classes in two separate hierarchies (`exceptions.py` and `errors.py`) with no single source of truth, making them impossible to type-check, match exhaustively, or enumerate. Secret fields in `settings.py` are stored as plain `str`, leaking credentials in `repr()`, logs, and serialization. PrivateAttr usage has no documented standard, risking inconsistent application.

## What Changes

- Create a single `ErrorCode` StrEnum centralizing all error codes from both `exceptions.py` and `errors.py`
- Migrate exception classes to reference the enum instead of string literals
- Move dead error-message constants in `enums.py` into `exceptions.py` as `__init__` defaults
- Migrate all sensitive `str` fields in `Settings` to `SecretStr`
- Document SecretStr and PrivateAttr usage patterns in `PYTHON-TYPING-RULES.md`
- Remove unused message constants from `enums.py`

## Capabilities

### New Capabilities
- `error-code-enum`: Single `ErrorCode` StrEnum in `src/app/utils/exceptions.py` covering all error codes from both the API exception hierarchy and the internal `AppError` hierarchy. Codes are the source of truth; `exceptions.py` and `errors.py` reference the enum.
- `error-message-defaults`: Migrate dead string constants from `src/app/config/enums.py` into `exceptions.py` as `__init__` parameter defaults. Remove unused constants from `enums.py`.
- `secret-str-migration`: Migrate all secret-holding `str` fields in `Settings` to Pydantic's `SecretStr` type, preventing credential leakage in logs/repr/serialization.
- `pydantic-pattern-documentation`: Document `SecretStr`, `PrivateAttr`, `Field(default_factory=...)`, and Pydantic v2 conventions in `.opencode/instructions/PYTHON-TYPING-RULES.md`.

### Modified Capabilities
- *(None — no existing spec files found)*

## Impact

- **Files modified**: `src/app/utils/exceptions.py`, `src/app/shared/result/errors.py`, `src/app/config/settings.py`, `src/app/config/enums.py`, `.opencode/instructions/PYTHON-TYPING-RULES.md`
- **API surface**: No breaking changes — `ErrorCode` enum values match existing string literals exactly. `SecretStr` changes may require `.get_secret_value()` at usage boundaries.
- **Dependencies**: None new. `SecretStr` is already in Pydantic v2.
- **Tooling impact**: Serialized config output (if any) will redact secrets. Code that reads secrets from settings must call `.get_secret_value()`.
