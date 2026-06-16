## Why

Two quality issues that create runtime risk:

1. **LLM client instantiated per request.** `search/service.py:236` and `documents/service.py:326,437` create `ChatGoogleGenerativeAI(...)` directly on every call instead of using the existing `_build_chat_model()` factory at `shared/langchain_layer/models.py:88`. This wastes resources (new HTTP client per request) and bypasses the centralized model configuration.

2. **`datetime.utcnow()` deprecated since Python 3.12.** Used in 7 locations across auth, users, and RAG models. Returns naive datetimes that break DST-aware comparisons. The codebase already uses `datetime.now(datetime.timezone.utc)` correctly in most places — these are stragglers.

## What Changes

### LLM Client Injection
- `SearchService.__init__` takes `llm: BaseChatModel` parameter
- `ask_legal()` uses `self._llm` instead of creating new instance
- `documents/service.py` functions take `llm: BaseChatModel` parameter
- Remove direct `ChatGoogleGenerativeAI(...)` from service methods

### datetime.utcnow() Cleanup
- Replace all `datetime.utcnow()` with `datetime.now(datetime.timezone.utc)` in:
  - `features/auth/service.py:221`
  - `features/users/repository.py:66,71`
  - `features/auth/repository.py:407,513`
  - `shared/rag/document_processing/models.py:289,301`

## Capabilities

### New Capabilities
- (none — these are refactorings, not new capabilities)

### Modified Capabilities
- (none)

## Impact

### Affected Code
- `src/app/features/search/service.py` — `SearchService.__init__` + `ask_legal()`
- `src/app/features/documents/service.py` — document processing functions
- `src/app/features/auth/service.py:221` — `datetime.utcnow()` → `datetime.now(timezone.utc)`
- `src/app/features/users/repository.py:66,71` — same
- `src/app/features/auth/repository.py:407,513` — same
- `src/app/shared/rag/document_processing/models.py:289,301` — same
- `src/app/features/search/dependencies.py` — update `SearchService` factory to inject LLM
- `src/app/features/documents/dependencies.py` — update document service factory

### Affected APIs
- None (internal refactoring)

### Dependencies Added
- None

### Systems
- CI: `uv run ruff check` and `uv run ty check` must pass
