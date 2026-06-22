## Context

The previous refactor established the pattern:

**Single-method return** — all repository methods return `AppResult[T]` directly. No public wrapper. Callers pattern-match:

```python
match await repo.method(...):
    case Success(value):
        ...
    case Failure(error):
        log_expected_failure(error, operation="...")
        raise app_error_to_exception(error)
```

Six methods in `documents/repository.py` and `search/repository.py` still use the old pattern:

```python
# OLD — dual method
async def bm25_search(self, ...) -> list[dict[str, Any]]:
    result = await self.bm25_search_result(...)
    if isinstance(result, Failure):
        raise app_error_to_exception(result.failure())
    return result.unwrap()

async def bm25_search_result(self, ...) -> AppResult[list[dict[str, Any]]]:
    # actual implementation
    ...
```

The only reason these survived the first pass is that the callers consume the return value as a plain type (e.g., passing search results directly to `reciprocal_rank_fusion()` in `asyncio.gather`). These callers must be updated to pattern-match on `AppResult` **within each gathered coroutine** rather than expecting unwrapped values.

Additionally, `users/service.py:41` calls `find_by_id_result()` which was removed in the previous refactor — the repo now has only `find_by_id()` returning `AppResult`. This is a latent runtime bug.

## Goals / Non-Goals

**Goals:**
- Eliminate all 6 remaining dual-method pairs
- Fix the stale `find_by_id_result` reference in `users/service.py`
- All 5 service call sites correctly pattern-match on `AppResult`
- Zero runtime behavior changes

**Non-Goals:**
- Change error types or error messages
- Refactor unrelated methods or files
- Add new capabilities
- Change `documents/repository.py:fetch_chunks_by_ids` (single method, no dual pattern)

## Decisions

### D1: Inline `Failure` handling in asyncio.gather

**Decision:** For `documents/service.py:search()`, replace the 3-way `asyncio.gather` with individual `match` on each result, then filter to `Success` branches before fusion.

```python
results = await asyncio.gather(
    self.repo.bm25_search(...),
    self.repo.vector_search(...),
    self.repo.trigram_search(...),
)
bm25_results, vector_results, trigram_results = [], [], []
for r in results:
    match r:
        case Success(value):
            # dispatch to the correct accumulator based on position
            ...
        case Failure(error):
            log_expected_failure(error, operation="hybrid_search")

fused_results = reciprocal_rank_fusion(
    _to_ranked_rows(bm25_results),
    _to_ranked_rows(vector_results),
    _to_ranked_rows(trigram_results),
    ...
)
```

**Rationale:** Each search can fail independently (partitioned PG, index rebuild). Failing the entire search because one of three indexes is down is wrong behavior. The `documents/service.py` search path is legal-domain (not general search) and can tolerate partial failures.

**Alternatives considered:**
- *Flatten via `Success.unwrap()` on each* — loses error logging — rejected
- *Propagate first Failure* — wrong for tolerance — rejected
- *Wrap gather in try/except* — loses per-index granularity — rejected

### D2: search/service.py `_run_parallel_search` — already correct

**Decision:** No change needed. `_run_parallel_search` already calls `repo.bm25_search`/`vector_search`/`trigram_search` which ALREADY return `AppResult` in the search repo (these were refactored in the first pass). The existing `match r: case Failure(error): ...` handler works correctly.

**Rationale:** These three methods in `search/repository.py` already use the single-method pattern. Only `create_document`, `upsert_chunks`, and `fetch_chunks_by_ids` in the same file still have the dual pattern.

### D3: `users/service.py` `find_by_id_result` → `find_by_id`

**Decision:** Simple rename. The repo method `find_by_id()` already returns `AppResult[User | None]`. The service already pattern-matches correctly:

```python
result = await self._user_repo.find_by_id(user_id)
match result:
    case Success(user) if user is not None:
        ...
    case Success():
        ...
    case Failure(error):
        ...
```

**Rationale:** The existing match arms already handle all three branches correctly. This is a pure cleanup.

### D4: `search/repository.py` `fetch_chunks_by_ids` — use existing `match` pattern

**Decision:** The `search/service.py:hybrid_search()` call site currently does:
```python
chunk_lookup = await self.repo.fetch_chunks_by_ids([item.chunk_id for item in fused_results])
```
After refactor, this returns `AppResult[dict[str, SearchChunkRecord]]`. Add a `match` on the result:

```python
match await self.repo.fetch_chunks_by_ids([item.chunk_id for item in fused_results]):
    case Success(chunk_lookup):
        items = _build_search_items(fused_results, chunk_lookup)
        ...
    case Failure(error):
        log_expected_failure(error, operation="hybrid_search")
        items = _build_search_items(fused_results, {})
```

**Rationale:** Chunk lookups failing should degrade gracefully. Return empty `{}` and continue with `chunk_text` fallbacks in the response.

### D5: `search/service.py` `upsert_chunks` — add match

**Decision:** The `process_ingestion_document()` call site currently does:
```python
await repo.upsert_chunks(build_chunk_rows(...))
```
After refactor, returns `AppResult[None]`. Add match:

```python
match await repo.upsert_chunks(build_chunk_rows(...)):
    case Success():
        pass
    case Failure(error):
        log_expected_failure(error, operation="search_ingestion")
        raise app_error_to_exception(error)
```

**Rationale:** Upsert failure should fail the ingestion task. This matches the pattern used in `documents/service.py:482` and `documents/service.py:650`.

## Risks / Trade-offs

- **[Partial search failure]** Legal-domain search in `documents/service.py` may produce fewer results if one of three indexes fails. **Mitigation:** Logged per-index as expected failure; fused results still work on 2-of-3 or 1-of-3. User gets partial results — acceptable for legal search.
- **[Latent bug fix]** `users/service.py:find_by_id_result()` would raise `AttributeError` at runtime. **Mitigation:** Immediate fix, no deployment needed.
- **[Test coverage]** These 6 methods are integration-tested via the search and document endpoints. **Mitigation:** Run full test suite after refactor.
