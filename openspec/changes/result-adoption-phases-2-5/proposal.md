## Why

The `returns`/Result infrastructure (Phase 1) is fully built but only 2 of 6 repositories and 0 LangGraph internal helpers use it. This leaves ~80% of persistence error sites untyped — callers cannot distinguish "not found" from "infrastructure failure" from "validation error" at the repository boundary. LangGraph nodes silently rely on blanket exception handling, preventing deterministic retry and typed error recovery.

## What Changes

- Convert `documents/repository.py` and `search/repository.py` to expose `AppResult`-returning methods with thin exception-based wrappers
- Complete the partial conversion of `auth/repository.py` (13 unconverted methods)
- Convert internal helper functions in ingestion and reconciliation LangGraph nodes to return `AppResult` instead of raising or returning plain dicts
- Add `match/case` unwrapping at the service boundary for converted paths
- Clean up `AppFutureResult` (defined but unused)

## Capabilities

### New Capabilities
- `repository-result-pattern`: Adopt `AppResult[T]` in `documents` and `search` repositories — `NotFoundAppError` for missing rows, `InfrastructureAppError` for DB failures, `ConflictAppError` for constraint violations. Each method gets a `_result` variant + thin public wrapper.
- `auth-repository-completion`: Complete partial conversion of `auth/repository.py` — add `_result` variants to `UserRepository` unconverted methods and entire `RefreshTokenRepository`.
- `langgraph-node-result-pattern`: Convert sync helper functions in `ingestion_kb/nodes.py` and `reconciliation/nodes.py` to return `AppResult[...]` — reduce reliance on blanket exception handling in graph error paths.
- `unused-code-cleanup`: Remove `AppFutureResult` type alias (defined in `types.py`, imported in `__init__.py`, zero usages).

### Modified Capabilities
*(None — no existing spec files found)*

## Impact

- **Repositories modified**: `documents/repository.py`, `search/repository.py`, `auth/repository.py`
- **LangGraph layers modified**: `ingestion_kb/nodes.py`, `reconciliation/nodes.py`
- **Result infrastructure modified**: `shared/result/types.py`, `shared/result/__init__.py` (remove `AppFutureResult`)
- **Dependencies**: None new. `returns` already in `pyproject.toml`.
- **API surface**: New `_result` method variants added. Existing public method signatures unchanged (thin wrappers). LangGraph node return shapes unchanged (results mapped back to plain dicts at node boundary).
