## 1. LLM Client Injection

- [x] 1.1 Update `SearchService.__init__` to accept `llm: BaseChatModel` parameter
- [x] 1.2 Update `ask_legal()` to use `self._llm` instead of creating new `ChatGoogleGenerativeAI`
- [x] 1.3 Remove `from langchain_google_genai import ChatGoogleGenerativeAI` from `search/service.py`
- [x] 1.4 Update `search/dependencies.py` to create LLM via `_build_chat_model()` and pass to `SearchService`
- [x] 1.5 Update document service functions in `documents/service.py` to accept `llm: BaseChatModel` parameter
- [x] 1.6 Remove direct `ChatGoogleGenerativeAI(...)` from document service functions
- [x] 1.7 Update `documents/dependencies.py` to create LLM once and pass to document functions
- [x] 1.8 Update any existing tests that create `SearchService` to provide mock LLM

## 2. datetime.utcnow() Cleanup

- [x] 2.1 Fix `features/auth/service.py:221`: `datetime.utcnow()` → `datetime.now(UTC)`
- [x] 2.2 Fix `features/users/repository.py:66,71`: `datetime.utcnow()` → `datetime.now(UTC)`
- [x] 2.3 Fix `features/auth/repository.py:407,513`: `datetime.utcnow()` → `datetime.now(UTC)`
- [x] 2.4 Fix `shared/rag/document_processing/models.py:289,301`: `datetime.utcnow()` → `datetime.now(UTC)`
- [x] 2.5 Update imports in each file: add `UTC` to `from datetime import UTC, datetime`
- [x] 2.6 Verify no `datetime.utcnow()` calls remain in `src/` (only in examples/REDIS_USAGE.md)

## 3. Verification

- [x] 3.1 Run `uv run ruff check src/` (no new errors from our changes)
- [ ] 3.2 Run `uv run ty check src/`
- [ ] 3.3 Run `uv run pytest tests/ -v` (ensure no regressions)
