## Why

The `returns.Result` pattern is used across 4 repositories, 6 services, 2 LangGraph node modules, and the ingestion service. But it's applied inconsistently:

1. **Dual-method**: Every repo has 2 methods per operation — a public wrapper (swallow/raise) and a `_result` variant. The wrappers are inconsistent (some swallow, some raise), and callers don't know which without reading source. 2x code for marginal benefit.
2. **LangGraph nodes**: Node entrypoints receive `Failure` but use `result.failure` (property access) instead of `result.failure()` (method call) — the `returns` version used here requires the latter.
3. **`log_expected_failure`**: Used in some callers (auth/refresh, users/admin, ingestion) but missing in others (auth/login, documents, search).
4. **Uncovered methods**: `DocumentRepository.bm25_search/vector_search/trigram_search` return `list[...]` directly — no `_result` variant, no error wrapping. DB errors crash instead of being captured as `Failure`.
5. **IngestionService**: Handles `AppError` from graph state correctly, but follows a unique pattern (reads failure from state-dict) that should be documented.

## What Changes

**BREAKING** — every repository method now returns `AppResult[T]` instead of `T | None` or `T`.

| Slice | Files | Changes (verified by graphify + ast-grep) |
|-------|-------|---|
| **Sl1** Repos unify | `users/repo.py`, `documents/repo.py`, `search/repo.py` | 11 `_result` methods renamed to primary + 11 public wrappers deleted |
| **Sl2** Callers match | `auth/service.py`(21 sites), `auth/deps.py`(1), `documents/service.py`(8), `search/service.py`(4) | 34 call sites updated to pattern-match `AppResult` |
| **Sl3** Bugfix | `ingestion_kb/nodes.py`(10 lines), `reconciliation/nodes.py`(10 lines) | 20 occurrences `result.failure`→`result.failure()` |
| **Sl4** Logging | Same services as Sl2 | `log_expected_failure` added to all `Failure` branches (except constant-time paths) |
| **Sl5** Coverage | `documents/repo.py`, `search/repo.py` | 6 new `_result` variants wrapping currently-uncaught `SQLAlchemyError` |
| **Sl6** Cleanup | `documents/repo.py`, `search/repo.py` | 2 files remove `app_error_to_exception` import |

## Capabilities

### New Capabilities
- `repo-unify-result`: Drop public wrappers, rename `_result` methods, all repos return `AppResult[T]`
- `service-result-handling`: Update all callers to pattern-match `AppResult`
- `langgraph-node-result`: Fix `Failure.failure` method-call convention in LangGraph nodes
- `log-failure-consistency`: Add `log_expected_failure` to all service `Failure` match branches
- `result-coverage`: Add `_result` variants to uncovered repo methods

### Modified Capabilities

(none)

## Impact

- **11 files affected**: 3 repos (users/documents/search) + 4 services/deps (auth/service, auth/deps, documents/service, search/service) + 2 LangGraph node modules (ingestion_kb, reconciliation) + 2 repo cleanup (documents, search)
- **Code delta**: ~200 lines removed (boilerplate) + ~120 lines added (uncovered wrappers)
- **Behavior change**: Infrastructure errors (`Failure`) no longer silently swallowed by "not found" paths
- **Bug fix**: `Failure.failure()` method call convention fixed in LangGraph nodes — was silently returning bound method object instead of error payload
- **New coverage**: 6 repo methods now catch `SQLAlchemyError`/`IntegrityError` as `Failure` instead of crashing
