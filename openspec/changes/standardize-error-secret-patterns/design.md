## Context

The codebase has two error hierarchies — `APIException` (FastAPI HTTP exceptions in `exceptions.py`, 14 classes) and `AppError` (internal Result payloads in `errors.py`, 6 classes) — each carrying its own `code: str` field with string literal defaults. These 20+ codes must remain synchronized manually because there is no shared enum. A `StrEnum` in `enums.py` (`SOMETHING_WENT_WRONG`, `VALIDATION_ERROR`, etc.) holds message strings that are dead code — no exception class references them.

Settings stores 15+ secret/credential fields (`str`) that leak in `repr()`, loguru serialization, and Pydantic `model_dump()`. `PrivateAttr` is used in 2 files (`S3ClientWrapper`, `StorageService`) with no documented convention.

## Goals / Non-Goals

**Goals:**
- Single `ErrorCode` StrEnum that both `exceptions.py` and `errors.py` reference, eliminating duplicate string literals
- Migrate dead message constants from `enums.py` into `exceptions.py` as `__init__` defaults
- Remove unused constants from `enums.py`
- Migrate all secret-holding `str` fields in `Settings` to Pydantic `SecretStr`
- Update usage sites to call `.get_secret_value()` where secrets are read
- Document `SecretStr`, `PrivateAttr`, and related Pydantic v2 patterns in `.opencode/instructions/PYTHON-TYPING-RULES.md`

**Non-Goals:**
- Refactoring exception hierarchy structure (class names, inheritance, status codes)
- Changing the API response envelope shape (error codes in response body remain unchanged)
- Converting all `str` fields in `Settings` — only fields holding secrets/tokens/credentials
- Adding error-code enum usage to non-exception code (routes, services, etc.)
- Changing how the global exception handler formats error responses

## Decisions

**1. `ErrorCode` StrEnum lives in `exceptions.py` (re-exported from a new file)**
- **Why**: The enum must be accessible from both `exceptions.py` and `errors.py`. Placing it in a shared module avoids circular imports (`errors.py` currently imports nothing from `utils/`).
- **Decision**: Create `src/app/utils/error_codes.py` containing the `ErrorCode(StrEnum)` class. Both `exceptions.py` and `errors.py` import from it. Re-export from `exceptions.py` for existing import chains.
- **Alternative considered**: Inline in `exceptions.py`. Rejected: `errors.py` would need to import from `exceptions.py`, creating an unwanted dependency from internal result types to HTTP exception types.

**2. ErrorCode values match existing string literals exactly**
- **Why**: Zero behavioral change — the same string value flows to the API response `error_code` field. No client-side updates needed.
- **Example**: `ErrorCode.VALIDATION_ERROR = "VALIDATION_ERROR"` (identity pattern).

**3. SecretStr fields use `Field(exclude=True)` in `model_config` serialization instead of — actually, `SecretStr` handles this natively**
- **Why**: Pydantic v2 `SecretStr` automatically redacts in `repr()`, `model_dump()`, and `.model_dump_json()` by emitting `'**********'`. No extra config needed.
- **Usage pattern**: At every site that reads a secret (e.g., `settings.GEMINI_API_KEY.get_secret_value()`), add `.get_secret_value()`. This is a mechanical, grep-able change.
- **Fields to migrate**: All fields whose name contains `api_key`, `secret`, `password`, `token` (case-insensitive) — ~15 fields.

**4. `PrivateAttr` usage is niche — document with examples, not exhaustive rules**
- **Why**: `PrivateAttr` is only appropriate for Pydantic models that hold runtime-only state (e.g., cached clients, connection pools) that should not participate in serialization or validation.
- **Documented pattern**: Use `PrivateAttr` only when (a) the value is a non-serializable runtime object AND (b) it is derived from other validated fields. Prefer `@property` or `field(default=..., exclude=True)` when the value is serializable.

## Risks / Trade-offs

- **[Breakage] `.get_secret_value()` calls may be missed at some usage sites** → Mitigation: After migration, grep for `settings\.\w+API_KEY`, `settings\.\w+PASSWORD`, `settings\.\w+SECRET` to find every reading site. Each must be converted. `ty` type checker will catch `str` vs `SecretStr` mismatches at all usage sites.
- **[Leakage] ErrorCode enum becomes a dumping ground for ad-hoc codes** → Mitigation: Add a comment in `error_codes.py` — "Add codes here only when they map to a typed exception class. For truly one-off codes in routes, use inline strings with a `# noqa: S` comment."
- **[Churn] Secret migration touches ~15 settings fields + all their downstream consumers** → Mitigation: Do this as a single focused commit. The migration is mechanical (add `SecretStr`, add `.get_secret_value()`) and testable.
- **[Circular import] `error_codes.py` imported by both `utils/` and `shared/result/`** → Mitigation: `error_codes.py` lives at `src/app/utils/` with zero dependencies beyond `enum` and `StrEnum`. No import chain to either `utils/` or `shared/` packages.
