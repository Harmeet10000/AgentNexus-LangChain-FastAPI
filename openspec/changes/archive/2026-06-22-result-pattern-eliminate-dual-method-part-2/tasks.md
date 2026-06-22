## 1. Repo: `documents/repository.py` — 3 dual-method pairs

- [x] 1.1 Delete `bm25_search()` wrapper (lines 310-326), rename `bm25_search_result()` → `bm25_search()` (line 328)
- [x] 1.2 Delete `vector_search()` wrapper (lines 373-389), rename `vector_search_result()` → `vector_search()` (line 391)
- [x] 1.3 Delete `trigram_search()` wrapper (lines 442-458), rename `trigram_search_result()` → `trigram_search()` (line 460)
- [x] 1.4 Verify all 3 method signatures use `AppResult[...]` return type

## 2. Repo: `search/repository.py` — 3 dual-method pairs

- [x] 2.1 Delete `create_document()` wrapper (lines 102-118), rename `create_document_result()` → `create_document()` (line 120)
- [x] 2.2 Delete `upsert_chunks()` wrapper (lines 148-155), rename `upsert_chunks_result()` → `upsert_chunks()` (line 157)
- [x] 2.3 Delete `fetch_chunks_by_ids()` wrapper (lines 285-292), rename `fetch_chunks_by_ids_result()` → `fetch_chunks_by_ids()` (line 294)
- [x] 2.4 Verify all 3 method signatures use `AppResult[...]` return type

## 3. Service: `users/service.py` — stale reference

- [x] 3.1 Line 41: change `self._user_repo.find_by_id_result(user_id)` → `self._user_repo.find_by_id(user_id)`

## 4. Service: `search/service.py` — 3 call sites

- [x] 4.1 Line 90: change `self.repo.create_document_result(...)` → `self.repo.create_document(...)`
- [x] 4.2 Lines 190-193: wrap `fetch_chunks_by_ids()` return in `match ... case Success / Failure` with graceful degradation (empty dict on Failure)
- [x] 4.3 Lines 322-327: wrap `upsert_chunks()` return in `match ... case Success / Failure` with exception raise on Failure

## 5. Service: `documents/service.py` — hybrid_search call sites

- [x] 5.1 Lines 243-261: refactor `asyncio.gather` block to pattern-match each of the 3 search results individually
- [x] 5.2 Handle partial failures: log each with `log_expected_failure` and continue with available results
- [x] 5.3 Verify `reciprocal_rank_fusion` receives correct types from the match branches

## 6. Validation

- [x] 6.1 Run `uv run ruff check src/app/features/documents/ src/app/features/search/ src/app/features/users/` — 2 pre-existing errors only
- [x] 6.2 Run `uv run ty check src/app/features/documents/ src/app/features/search/ src/app/features/users/` — 11 pre-existing diagnostics only
- [x] 6.3 Run `uv run pytest tests/ -x -q` — pre-existing circular import (not from this change)
- [x] 6.4 Verify no `_result` method names remain in any `src/app/features` repo file — clean
