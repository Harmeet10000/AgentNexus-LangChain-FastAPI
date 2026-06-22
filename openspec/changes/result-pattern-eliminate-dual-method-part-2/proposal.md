## Why

The previous refactor (`result-pattern-eliminate-dual-method`, archived 2026-06-21) eliminated the dual-method pattern from `auth/repository.py`, `users/repository.py`, and partial areas of `documents/repository.py` and `search/repository.py`. Six dual-method pairs remain in the documents and search repos — wrappers that unwrap `AppResult` and re-raise as exceptions, defeating the purpose of the single-method pattern. The previous refactor also left behind one stale method reference (`find_by_id_result`) in `users/service.py` that calls a no-longer-existing method.

## What Changes

### Repo method cleanup (6 pairs)
For each pair: delete the thin public wrapper and rename the `_result` variant to drop the suffix, making it the sole method returning `AppResult[T]`.

| File | Public wrapper → | AppResult method (renamed) |
|------|-----------------|---------------------------|
| `documents/repository.py:310` | `bm25_search()` returning `list[...]` | `bm25_search_result()` → `bm25_search()` |
| `documents/repository.py:373` | `vector_search()` returning `list[...]` | `vector_search_result()` → `vector_search()` |
| `documents/repository.py:442` | `trigram_search()` returning `list[...]` | `trigram_search_result()` → `trigram_search()` |
| `search/repository.py:102` | `create_document()` returning `SearchDocument` | `create_document_result()` → `create_document()` |
| `search/repository.py:148` | `upsert_chunks()` returning `None` | `upsert_chunks_result()` → `upsert_chunks()` |
| `search/repository.py:285` | `fetch_chunks_by_ids()` returning `dict[...]` | `fetch_chunks_by_ids_result()` → `fetch_chunks_by_ids()` |

### Service call site updates (6 sites)
Each caller that consumed the old public wrapper return value must now pattern-match on `AppResult`:

| File | Location | Current pattern | New pattern |
|------|----------|----------------|-------------|
| `users/service.py:41` | `_get_user_or_raise()` | `find_by_id_result()` → stale reference | `find_by_id()` — already returns `AppResult` |
| `search/service.py:90` | `ingest_document()` | `create_document_result()` (old naming) | `create_document()` |
| `search/service.py:190` | `hybrid_search()` | `fetch_chunks_by_ids()` → unwrapped dict | `match ... case Success` |
| `search/service.py:322` | `process_ingestion_document()` | `upsert_chunks()` → unwrapped None | `match ... case Success / Failure` |
| `documents/service.py:244-261` | `search()` | `bm25_search/vector_search/trigram_search` → gathered, used raw | `match each case Success/Failure` |

### `documents/repository.py:fetch_chunks_by_ids`
This method is NOT part of the refactor — it is a single method returning `dict[str, dict[str, Any]]` with no dual pattern. Left unchanged.

## Capabilities

### New Capabilities
- (none — pure refactor)

### Modified Capabilities
- `documents-repository` — 3 methods change return type from `list[...]` to `AppResult[list[...]]`
- `search-repository` — 3 methods change return type from mixed types to `AppResult[T]`
- `documents-search-service` — 4 call sites adopt `match` on `AppResult`

## Impact

### Affected Code
- `src/app/features/documents/repository.py` — delete 3 wrappers, rename 3 methods
- `src/app/features/documents/service.py` — 1 async-gather block + 1 call site adopt match
- `src/app/features/search/repository.py` — delete 3 wrappers, rename 3 methods
- `src/app/features/search/service.py` — 3 call sites adopt match
- `src/app/features/users/service.py` — 1 method reference fixed

### Affected APIs
- No breaking changes to HTTP request/response contracts
- Repository return types change internally; all error paths convert to HTTP exceptions via existing `match Failure(error): raise app_error_to_exception(error)` pattern

### Dependencies Added
- None

### Systems
- CI: `uv run ruff check src/ && uv run ty check src/` must pass
