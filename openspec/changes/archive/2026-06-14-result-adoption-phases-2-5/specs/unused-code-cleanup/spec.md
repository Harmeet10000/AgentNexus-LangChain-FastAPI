## REMOVED Requirements

### Requirement: Remove AppFutureResult type alias
**Reason**: `AppFutureResult` is defined in `src/app/shared/result/types.py` and re-exported in `__init__.py` but has zero usages across the entire codebase. The project's documented policy prefers ordinary async functions over `FutureResult` ("only when async composition is materially clearer than ordinary async code"). No async composition path uses `AppFutureResult`. Keeping it creates dead code and misleading API surface.

**Migration**: Remove the `AppFutureResult` line from `types.py`. Remove its re-export from `__init__.py`. No callers require migration since none exist.

