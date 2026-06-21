## Context

Every repository has dual-method pairs (public wrapper + `_result` variant). The wrappers are inconsistent:

```python
# Queries swallow: user gets None, no idea if DB was down
async def find_by_email(self, email: str) -> User | None:
    result = await self.find_by_email_result(email=email)
    if isinstance(result, Failure):
        return None
    return result.unwrap()

# Writes raise: infrastructure error propagates
async def create(self, user: User) -> User:
    result = await self.create_result(user=user)
    if isinstance(result, Failure):
        raise app_error_to_exception(result.failure())
    return result.unwrap()
```

Beyond repos, the Result pattern leaks into 2 LangGraph node modules (`ingestion_kb/nodes.py`, `reconciliation/nodes.py`) and the `IngestionService`. A bug exists: all 7 LangGraph node callers use `result.failure` (property) instead of `result.failure()` (method), which silently returns the bound method object instead of the error payload. And some repo methods (`DocumentRepository.bm25_search`) lack `_result` variants entirely — DB errors crash instead of being captured.

## Verified Audit (graphify + ast-grep)

Scanned `src/app/features/` (auth, users, documents, search, ingestion, crawler, health, knowledge_base, profile, web_scraping) and `src/app/shared/` (circuit_breaker, crawler, langchain_layer, langgraph_layer, mcp, rag, result, services, vectorstore).

| Metric | Value | Source |
|--------|-------|--------|
| Files importing `returns.result` | 10 | `rg -l "from returns.result import" src/app/ | wc -l` |
| `_result` method definitions to rename | 11 (documents:5, search:5, users:1) | `rg "def .*_result\(" --glob **/repository.py` |
| Public wrappers to delete | 11 | 1:1 with `_result` methods |
| Uncovered methods needing wrapping | 6 (docs:3, search:3) | `rg "def (bm25\|vector\|trigram\|create_document\|upsert_chunks\|fetch_chunks_by_ids)" --glob **/repository.py` |
| `app_error_to_exception` in repos | 2 files, 5 calls | auth(already cleaned), documents(2), search(3) |
| Service call sites to update | 34 total | auth(21+1), documents(8), search(4) |
| `.failure`→`.failure()` bug | 20 occurrences (10 per node file) | `rg -n "\.failure[^(]" ingestion_kb/nodes.py reconciliation/nodes.py` |
| Features already correct | users/service, ingestion/service | Already pattern-match, no change |

## Goals / Non-Goals

**Goals:**
- All 4 repos: single method returning `AppResult[T]`, no wrappers, no `_result` suffix
- All 6 service files + 1 dependency: pattern-match `AppResult`
- All LangGraph nodes: use `result.failure()` method convention
- `log_expected_failure` on every `Failure` match branch at ownership boundaries
- `_result` coverage for `DocumentRepository.bm25/vector/trigram` and `SearchRepository.create_document/upsert_chunks/fetch_chunks_by_ids`

**Non-Goals:**
- Add Result to simple repo methods that can't fail (e.g., `analyze_chunks`, `update_document_status`)
- Change the `AppError` hierarchy or `returns.result` version
- Change the `IngestionService` pattern (it reads failures from graph state, which is correct)
- Add Result to FastAPI router handlers or lifespan wiring

## Decisions

### D1: Unify repos — one method per operation

**Before:**
```python
async def find_by_email(self, email: str) -> User | None:
    result = await self.find_by_email_result(email=email)
    if isinstance(result, Failure):
        return None
    return result.unwrap()

async def find_by_email_result(self, email: str) -> AppResult[User | None]:
    try: ...
    except PyMongoError as exc:
        return Failure(InfrastructureAppError(...))
```

**After:**
```python
async def find_by_email(self, email: str) -> AppResult[User | None]:
    try: ...
    except PyMongoError as exc:
        return Failure(InfrastructureAppError(...))
```

### D2: Service callers pattern-match with consistent error handling

**Pattern for all service callers:**
```python
match await repo.find_by_email(email):
    case Success(user) if user is not None:
        ...do work...
    case Success():                   # not found — caller decides
        return None / raise NotFoundException(...) / etc
    case Failure(error):
        log_expected_failure(error, operation="operation_name")
        raise app_error_to_exception(error)
```

Exception: `auth/service.py` login/forgot-password/resend-verification paths that must stay constant-time use `case _` wildcard — they SHOULD NOT log or differentiate error types (timing leak).

### D3: Fix `result.failure` → `result.failure()`

In `ingestion_kb/nodes.py` and `reconciliation/nodes.py`, 7 occurrences use `result.failure` (property access) instead of `result.failure()` (method call). In the `returns` version installed, `Failure.failure` is a method, not a property — so `result.failure` returns the bound method object, not the error payload.

**Before:** `return _state_failure(result.failure)`
**After:** `return _state_failure(result.failure())`

### D4: Add `_result` variants to uncovered methods

`DocumentRepository.bm25_search/vector_search/trigram_search` return `list[...]` directly. If the DB goes down during a search, the user gets an unhandled `SQLAlchemyError`. These need `_result` variants:

```python
async def bm25_search(
    self, *, user_id: str, query: str, candidate_limit: int, filter_params: dict[str, Any]
) -> list[dict[str, Any]]:
    result = await self.bm25_search_result(...)
    if isinstance(result, Failure):
        raise app_error_to_exception(result.failure())  # or return []
    return result.unwrap()

async def bm25_search_result(
    self, *, user_id: str, query: str, candidate_limit: int, filter_params: dict[str, Any]
) -> AppResult[list[dict[str, Any]]]:
    try: ...
    except SQLAlchemyError as exc:
        return Failure(InfrastructureAppError(...))
```

Same for `SearchRepository.create_document`, `upsert_chunks`, `fetch_chunks_by_ids`.

### D5: Remove `app_error_to_exception` from repos

After D1, no repo file calls `app_error_to_exception`. Remove the import from all 4 repos.

## Risks / Trade-offs

- **Risk**: `auth/service.py` constant-time paths leak timing if we differentiate `NotFoundAppError` vs `InfrastructureAppError`. → **Mitigation**: Use `case _` wildcard. Both errors produce identical response.
- **Risk**: Callers that previously got `None` from swallow wrappers now see `Failure`. → **Mitigation**: Match pattern with `case _` for constant-time paths, explicit match for admin/user-facing paths.
- **Risk**: `DocumentRepository` search methods are used in `asyncio.gather` — one Failure crashes all three. → **Mitigation**: The `_run_parallel_search` in both docs and search services wraps each gathered result.
- **Risk**: The `result.failure` bug silently returned bound methods — fixing to `result.failure()` changes behavior. → **Mitigation**: This is a bugfix, not a refactor. The error payload was being silently converted to a string representation of the bound method.

## Migration Plan

1. **Sl 5**: Add `_result` variants to `DocumentRepository.bm25/vector/trigram` + `SearchRepository.create/upsert/fetch`
2. **Sl 1**: Refactor all 4 repos — delete wrappers, rename `_result` → primary
3. **Sl 2**: Update all service/dependency callers to `match` on `AppResult`
4. **Sl 3**: Fix `result.failure` → `result.failure()` in LangGraph nodes
5. **Sl 4**: Add `log_expected_failure` to all service `Failure` match branches
6. **Sl 6**: Remove unused `app_error_to_exception` imports from repos
7. Verify with ruff + ty
