## Context

The project has a complete `AppResult[T]` / `AppError` infrastructure (`returns`-based Result type, frozen Pydantic error models, `log_expected_failure`, `app_error_to_exception` boundary mapper) with documented adoption rules in `.github/copilot-instructions.md`. Adoption is at ~60%: foundation done, 2 repos converted, 0 LangGraph internal helpers converted.

The unconverted repositories (`documents`, `search`) use SQLAlchemy async and let exceptions fly — callers cannot distinguish `NotFound` from `InfrastructureError`. The partially converted `auth` repository has 13 unconverted methods. LangGraph ingestion/reconciliation nodes use `AppError` types in state but return plain dicts without `Success`/`Failure` containers.

## Goals / Non-Goals

**Goals:**
- Add `AppResult`-returning `_result` method variants to all critical persistence methods in `documents`, `search`, and `auth` repositories
- Add thin public wrappers that unwrap `Failure` to `None` or raise for non-recoverable errors (matching existing `users/repository.py` pattern)
- Convert sync helper functions in `ingestion_kb/nodes.py` and `reconciliation/nodes.py` to return `AppResult` — map back to plain dicts at node boundary
- Remove unused `AppFutureResult` type alias

**Non-Goals:**
- Converting the entire codebase to Result pattern (async service/graph entrypoints stay exception-based per documented policy)
- Changing FastAPI router, Celery task, or middleware signatures (they remain exception-based)
- Converting SQLAlchemy repositories to use raw exceptions for non-recoverable errors — they keep `let exceptions fly` for truly unexpected failures
- Adding Result variants to `scalar_one_or_none()` calls that legitimately return `None` for absent data

## Decisions

**1. Dual-method pattern (`find_by_id` + `find_by_id_result`) over single Result-returning method**
- **Why**: Keeps existing callers unchanged. Callers that benefit from typed recovery use the `_result` variant. New callers default to the simple unwrap. Proven pattern from `users/repository.py`.
- **Alternative considered**: Single `Result`-returning method with `unwrap()` at call sites. Rejected: too much churn for callers that don't need typed recovery.

**2. `NotFoundAppError` for SQLAlchemy `scalar_one_or_none()` returning `None`**
- **Why**: Distinguishes "row not found" from "database connection failed" at the type level. Callers can react differently.
- **Important**: Only apply to methods where absence is exceptional (e.g., `get_document_by_id`). Skip methods where `None` is a valid query result (e.g., `email_exists` returning `False` is not a failure).

**3. LangGraph node helpers return `AppResult`, mapped back to plain dicts at node boundary**
- **Why**: LangGraph nodes must return plain dicts for state updates. Internal helpers benefit from typed error handling. The `_ingestion_failure()` pattern already exists — make it return `AppResult` for composability.
- **Alternative considered**: Making nodes themselves return `AppResult`. Rejected: LangGraph's state machine requires dict return values; wrapping/unwrapping at every node adds ceremony without benefit.

**4. `InfrastructureAppError` wraps `SQLAlchemyError` and similar DB/network exceptions**
- **Why**: Gives callers a typed branch for retry vs fail decisions. The existing `app_error_to_exception` mapper converts these back to `ServiceUnavailableException` at the boundary.
- **Scope**: Only for unexpected DB failures (connection pool, deadlock, serialization). Expected failures like constraint violations use `ConflictAppError`.

## Risks / Trade-offs

- **[Churn] Adding `_result` variants doubles the method count** → Mitigation: each variant is 3-5 lines. The thin wrapper pattern is mechanical and testable. Total additions ~150 lines across ~3500 lines of repo code.
- **[Inconsistency] Some methods get Result treatment, others don't** → Mitigation: apply the pattern consistently to all methods that perform I/O or validation. Pure query builders and trivial property access remain untouched.
- **[LangGraph coupling] Node helpers returning `AppResult` may break if graph structure changes** → Mitigation: helpers are internal module functions, not part of the public API. The `Success`/`Failure` wrapping is contained within each helper.
