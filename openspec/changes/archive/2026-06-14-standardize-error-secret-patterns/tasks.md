## 1. ErrorCode Enum — Create and Wire Core

- [x] 1.1 Create `src/app/utils/error_codes.py` with `ErrorCode(StrEnum)` containing all codes from `exceptions.py` and `errors.py` (VALIDATION_ERROR, NOT_FOUND, UNAUTHORIZED, FORBIDDEN, CONFLICT, TOO_MANY_REQUESTS, SERVICE_UNAVAILABLE, DATABASE_ERROR, EXTERNAL_SERVICE_ERROR, INVALID_TOKEN, TOKEN_EXPIRED, REFRESH_TOKEN_INVALID, INFRASTRUCTURE_ERROR)
- [x] 1.2 Re-export `ErrorCode` from `src/app/utils/exceptions.py` for existing import chains
- [x] 1.3 Migrate all 14 `APIException` subclasses in `exceptions.py` to use `ErrorCode` enum members as `error_code` defaults
- [x] 1.4 Migrate all 6 `AppError` subclasses in `errors.py` to use `ErrorCode` enum members as `code` defaults
- [x] 1.5 Update inline error_code literals in `main.py` (line 121, `error_code="NOT_FOUND"`) and `global_exception_handler.py` (lines 55, 106) to use `ErrorCode`
- [x] 1.6 Update inline error_code literals in repository files (`documents/repository.py`: `DOCUMENT_NOT_FOUND`, `STATUS_NOT_FOUND`; `search/repository.py`: `SEARCH_DOCUMENT_NOT_FOUND`; `auth/repository.py`: `USER_NOT_FOUND`) to use `ErrorCode` — add new codes to enum as needed
- [x] 1.7 Update inline error_code in `agent_saul/router.py` (line 166, `VALIDATION_ERROR`) and `search/repository.py` (lines 62, 96) to use `ErrorCode`

## 2. Error Message Defaults — Consolidate and Clean

- [x] 2.1 Remove unused `SOMETHING_WENT_WRONG`, `INTERNAL_SERVER_ERROR`, `VALIDATION_ERROR`, `NOT_FOUND`, `UNAUTHORIZED`, `FORBIDDEN` constants from `src/app/config/enums.py`
- [x] 2.2 Verify no codebase imports reference the removed constants (`grep -r` across src/)
- [x] 2.3 Verify `ValidatorVisitor` in `global_exception_handler.py` and all exception `__init__` defaults produce identical detail messages before/after

## 3. SecretStr — Migrate Settings Fields

- [x] 3.1 Add `from pydantic import SecretStr` to `src/app/config/settings.py`
- [x] 3.2 Migrate GEMINI_API_KEY, POSTGRES_PASSWORD, NEO4J_PASSWORD, REDIS_PASSWORD, RABBITMQ_DEFAULT_PASS, PINECONE_API_KEY, LANGSMITH_API_KEY, TAVILY_API_KEY, PAGEINDEX_API_KEY, LANGEXTRACT_API_KEY, RESEND_API_KEY to `SecretStr`
- [x] 3.3 Migrate JWT_SECRET_KEY, OAUTH_STATE_SECRET, GOOGLE_CLIENT_SECRET, GITHUB_CLIENT_SECRET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, FASTAPI_GUARD_IPINFO_TOKEN to `SecretStr`
- [x] 3.4 Update `src/app/features/documents/service.py` — remove `SecretStr(...)` double-wrap, use `.get_secret_value()` or rely on `SecretStr` directly
- [x] 3.5 Update `src/app/features/search/service.py` — same pattern
- [x] 3.6 Update `src/app/features/search/embeddings.py` — same pattern
- [x] 3.7 Update `src/app/shared/services/tavily.py` — add `.get_secret_value()` for TAVILY_API_KEY
- [x] 3.8 Update `src/app/shared/services/storage.py` — add `.get_secret_value()` for S3 credentials
- [x] 3.9 Update `src/app/shared/services/mailer.py` — add `.get_secret_value()` for RESEND_API_KEY
- [x] 3.10 Update `src/app/features/auth/security.py` — add `.get_secret_value()` for JWT_SECRET_KEY, OAUTH_STATE_SECRET, GOOGLE_CLIENT_SECRET, GITHUB_CLIENT_SECRET
- [x] 3.11 Update `src/app/shared/rag/graphiti/client.py` — add `.get_secret_value()` for GEMINI_API_KEY
- [x] 3.12 Update `src/app/shared/langchain_layer/models.py` — add `.get_secret_value()` for GEMINI_API_KEY
- [x] 3.13 Update `src/app/shared/langchain_layer/agents/memory/cognee_client.py` — add `.get_secret_value()` for GEMINI_API_KEY
- [x] 3.14 Update `src/app/shared/rag/pageindex/client.py` — add `.get_secret_value()` for PAGEINDEX_API_KEY
- [x] 3.15 Update `src/app/shared/rag/pageindex/functions.py` — add `.get_secret_value()` for PAGEINDEX_API_KEY
- [x] 3.16 Update `src/app/shared/langchain_layer/callback.py` — add `.get_secret_value()` for LANGSMITH_API_KEY
- [x] 3.17 Update `src/app/middleware/server_middleware.py` — add `.get_secret_value()` for FASTAPI_GUARD_IPINFO_TOKEN
- [x] 3.18 Update `src/app/connections/neo4j.py` — add `.get_secret_value()` for NEO4J_PASSWORD
- [x] 3.19 Update `src/app/lifecycle/lifespan.py` — add `.get_secret_value()` for NEO4J_PASSWORD and commented-out GEMINI_API_KEY

## 4. Pydantic Pattern Documentation

- [x] 4.1 Add SecretStr usage rules to `.opencode/instructions/PYTHON-TYPING-RULES.md` — when to use, how to consume via `.get_secret_value()`
- [x] 4.2 Add PrivateAttr usage rules — only for non-serializable runtime state, prefer `@property` or `Field(exclude=True)` when possible
- [x] 4.3 Add Field pattern rules — `default_factory` for mutable defaults, `ConfigDict(frozen=True)`, `extra="forbid"` for request models

## 5. Verify

- [x] 5.1 Run `uv run ruff check src/` — zero new violations
- [x] 5.2 Run `uv run ty check src/` — zero new errors
- [x] 5.3 Run `uv run ruff format src/` — formatting clean
- [x] 5.4 Run `grep -r "from src\.app\.config\.enums import" src/` to confirm no remaining imports of removed constants
