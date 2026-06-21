## Why

Every repository has 2 methods per operation: a public wrapper (swallow or raise) and a `_result` variant returning `AppResult[T]`. The public wrappers are inconsistent — some swallow `Failure` silently (returning `None`/`[]`), others raise via `app_error_to_exception`. Callers don't know which is which without reading the source. This is 2x code for marginal benefit: the `_result` suffix is a signal that every caller should already follow.

## What Changes

**BREAKING** — every repository method now returns `AppResult[T]` instead of `T | None` or `T`.

- Delete all 32 public wrapper methods across 4 repositories
- Rename all `_result` methods to be the primary method (drop `_result` suffix)
- Update all 20+ service/dependency callers to pattern-match on `AppResult`
- Remove `app_error_to_exception` imports from repositories (no longer used there)
- Standardize service-layer error handling: `match` on `Failure(error)` → `log_expected_failure` + `raise app_error_to_exception`

## Capabilities

### New Capabilities
- `repo-unify-result`: Drop public wrappers, rename `_result` methods, all repos return `AppResult[T]`
- `service-result-handling`: Update services/dependencies to pattern-match `AppResult`

### Modified Capabilities

(none)

## Impact

- **6 files**: 4 repositories (auth, users, documents, search) + 2 services (auth, documents/search) + 1 dependency (auth) + 1 service (users already matches)
- **~200 lines removed** (boilerplate wrappers)
- **Behavior change**: Infrastructure errors (`Failure`) no longer silently swallowed by "not found" paths — they propagate as exceptions
