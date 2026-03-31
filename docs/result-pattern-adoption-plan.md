# Result Pattern Adoption Plan

## Recommendation

Use the `result` pattern selectively, not as a repo-wide replacement for exceptions.

This codebase already has a clear HTTP error boundary through `APIException` in
`src/app/utils/exceptions.py` and the global exception handler in
`src/app/middleware/global_exception_handler.py`. Keep that design. Introduce
`Result[T, DomainError]` only in internal code paths where failure is expected,
recoverable, and part of normal business flow.

todo
Stop swallowing infrastructure errors as “not found”.
In users/repository.py, find_by_id catches broad Exception and returns None. That conflates invalid IDs, driver failures, and true absence. Repositories should distinguish:

invalid identifier
not found
database failure
Standardize one internal error taxonomy.
You already have good HTTP-facing exceptions in exceptions.py and a clean boundary in global_exception_handler.py. The missing piece is a non-HTTP domain/infrastructure error layer for internal code. That will make selective Result adoption much cleaner.



Define transaction and rollback rules explicitly.
If you adopt Result, document one rule for DB code: inside transaction scopes, prefer exceptions when rollback depends on them. This matters in workflow code like ingestion/pipeline.py, where partial-step failures and DB persistence are close together.

## Where To Use Result

- Internal workflow and graph code where intermediate failures are expected and
  should be propagated explicitly instead of raised.
- Repository helpers and adapters that normalize recoverable persistence or
  upstream failures.
- External-service adapters where third-party operational failures can be
  converted into typed domain errors.
- Internal helper functions where the caller can handle failure without
  unwinding the whole stack.
- Ingestion, document-processing, and orchestration flows that already model
  partial failure as returned state.

## Where Not To Use Result

- FastAPI routers, dependencies, middleware, and global exception handlers.
- Transport and HTTP boundary logic that should continue using project exception
  types from `src/app/utils/exceptions.py`.
- Programmer errors, invariant violations, and misconfiguration.
- Cancellation propagation such as `asyncio.CancelledError`.
- Transaction scopes where rollback semantics depend on exceptions being raised.

## Benefits For This Repo

- Expected failure becomes explicit in function signatures instead of being
  hidden behind broad `try/except` blocks or `None` returns.
- Internal workflows become easier to compose, especially in ingestion and
  graph-style pipelines where failure is often recoverable.
- Service and repository tests can assert typed failure values directly instead
  of only checking raised exceptions.
- The code can better align with Ruff `TRY` intent by avoiding exception-based
  control flow for normal internal outcomes.
- Boundary ownership becomes cleaner: internal layers return domain outcomes,
  while HTTP-facing layers decide how to convert them into API exceptions.

## Edge Cases And Nuances

- `None` and `Err(...)` are different signals. Keep “missing but acceptable”
  distinct from “operation failed”.
- Do not convert cancellation into `Err`. Cancellation should propagate.
- Third-party libraries will still raise exceptions. Convert only expected
  operational failures at adapter boundaries.
- Prefer `Err(DomainError(...))` over `Err(Exception(...))` or `Err("...")`.
- Never return `Err(APIException(...))`; transport concerns must stay at the
  HTTP boundary.
- Keep logging at the ownership boundary to avoid duplicate logs for the same
  failure as it moves across layers.
- Do not force `Result` into code paths where it adds unwrap boilerplate without
  improving clarity.
- In DB transaction scopes, raising may remain the correct choice because the
  transaction manager already uses exceptions for rollback behavior.

## Migration Plan

1. Verify dependency installation and active runtime.
   Confirm that the `result` package is actually installed in the runtime used
   by `uv`, tests, and tooling before introducing usage.
2. Add a small internal result convention module.
   Define shared `DomainError` types, `Result` aliases if useful, and boundary
   mappers from domain errors to `APIException` subclasses.
3. Pilot one vertical slice in ingestion or workflow code.
   Start in an area such as ingestion where expected failure is already modeled
   as returned state instead of raised exceptions.
4. Add tests around success and failure mapping.
   Verify that internal `Err` values map cleanly to the existing API exception
   system and that HTTP responses remain unchanged.
5. Expand only if complexity decreases.
   If the pilot reduces broad exception handling and makes control flow clearer,
   extend the pattern to similar internal subsystems. Otherwise stop there.

## Copilot instructions
## Result Pattern And Exceptions

- Use `Result[T, DomainError]` for expected, recoverable failures in internal workflow code, repository helpers, and external-service adapters.
- Keep FastAPI routers, dependencies, middleware, and global exception handling exception-based. Use project exceptions from `src/app/utils/exceptions.py` at those boundaries.
- Prefer `Result` when failure is part of normal business flow and the caller can handle it without unwinding the stack.
- Do not use `Result` for programmer errors, broken invariants, misconfiguration, or cancellation. Those should still raise.
- Do not return `Err(Exception)` or raw string errors by default. Prefer structured domain error types.
- Never use `Err(APIException(...))`; transport-layer exceptions must be created only at the HTTP boundary.
- Map `DomainError -> APIException` once at the service/router boundary.
- In transaction scopes where rollback depends on exceptions, prefer raising over returning `Err`.
- Catch third-party operational failures at adapter boundaries and convert only expected cases into `Err`; let unexpected exceptions propagate.
- Log failures at the boundary that owns the decision, not at every layer that forwards an `Err`.

Rule of thumb: exception outside-in, `Result` inside where it improves explicitness and composition. 

## Default Policy

Exception outside-in, `Result` inside where it improves explicitness and
composition.


# Selective `Result` Adoption, Not Repo-Wide Replacement

## Summary
- Recommendation: start using the `result` pattern selectively, not as a wholesale replacement for exceptions.
- Keep raised exceptions at HTTP/framework boundaries because the current API contract is built around [exceptions.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/utils/exceptions.py), [global_exception_handler.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/middleware/global_exception_handler.py), and FastAPI’s exception flow.
- Introduce `Result[T, E]` first in internal workflow code where failure is expected and recoverable, especially ingestion/workflow-style logic like [pipeline.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/features/ingestion/pipeline.py).
- Current repo fact: `result>=0.17.0` is declared in `pyproject.toml`, but `python3` in this environment cannot import it. First migration step must verify lockfile/install state before any code change.

## Why This Repo Benefits
- Expected failures become explicit in function signatures instead of being hidden in `try/except` or `None` returns.
- Internal pipelines can compose success/failure branches more cleanly than today’s mix of raised exceptions and `"error"` fields in returned dicts.
- Repository/client code gets a cleaner split between business failure and programmer failure.
- Ruff `TRY` rules become easier to satisfy in internal code because “expected bad outcome” stops being modeled as raising.
- Tests become more direct for failure cases because you assert `Err(...)` values instead of exception side effects.

## Where To Use It
- Use `Result` in internal service helpers, repositories, external-client adapters, and graph/pipeline nodes where failure is normal, domain-specific, and often recoverable.
- Good first targets:
  - ingestion/document workflows that already return failure state instead of crashing
  - adapter/client code that normalizes third-party errors
  - repositories that currently collapse “invalid input / not found / upstream failure” into `None` or broad `except`
- Do not use `Result` for:
  - FastAPI routers, dependencies, middleware, or exception handlers
  - true programmer errors, invariant violations, or misconfiguration
  - cancellation/timeout control flow that should propagate
  - transaction/context-manager boundaries that rely on exceptions to rollback

## Key Design Decisions
- Standardize on `Result[T, DomainError]`, not `Result[T, Exception]`.
- Define a small internal error ADT/dataclass set for expected failures such as validation, not-found, conflict, upstream-unavailable, and transient-infra failure.
- Convert `DomainError -> APIException` only at the HTTP boundary.
- Preserve exceptions for unexpected bugs and invariant breaks; do not wrap everything into `Err`.
- Ban raw string `Err("...")` outside trivial leaf helpers; require structured error values for logging, mapping, and tests.

## Migration Plan
1. Environment check
- Confirm `result` is actually installed in the active runtime/lock workflow and fix dependency sync first.

2. Add a small result convention layer
- Create a lightweight internal module with:
  - `Result` import re-export
  - shared `DomainError` types
  - helper mappers from `DomainError` to `APIException`
  - optional helper utilities for async composition if the package API is too bare

3. Pilot one vertical slice
- Start with ingestion because it already mixes explicit failure state with caught exceptions.
- Refactor one path end-to-end so internal nodes/helpers return `Result`, while the router/service still returns the same HTTP contract.

4. Refactor client/repository adapters
- Move broad `except Exception` normalization into adapter/repository edges.
- Return typed `Err` for expected operational failures and re-raise unexpected bugs.

5. Add HTTP-boundary mapping
- In service/router boundaries, unwrap `Result` once and convert `Err` to your existing `APIException` subclasses.
- Keep global exception middleware unchanged.

6. Expand only after pilot criteria pass
- Apply to other workflow-heavy or external-service-heavy areas.
- Leave straightforward CRUD service methods exception-based unless they gain real composition benefits.

## Edge Cases And Nuances
- `None` is not the same as `Err`: keep “not found but acceptable” distinct from “operation failed”.
- Async cancellation must propagate. Never convert `CancelledError` into `Err`.
- Transaction rollbacks usually depend on exceptions. Inside a DB transaction, raising may still be the correct control flow.
- Third-party libraries will keep raising. Adapter layers should translate only expected operational failures.
- Overuse can make code worse: if every layer returns `Result`, you can end up with repetitive unwrap/match noise.
- Logging should happen at ownership boundaries, not at every `Err` construction, or you will duplicate logs.
- Mixing `Result` with current `APIException` subclasses is fine, but only if boundary ownership is strict.
- A bad migration pattern would be `Err(APIException(...))`; that leaks transport concerns into domain code.

## Test Plan
- Unit tests for `DomainError -> APIException` mapping.
- Pilot-path tests for success, validation failure, not-found, upstream failure, and unexpected exception propagation.
- Regression tests that HTTP responses remain unchanged for the migrated slice.
- Type-check/lint checks confirming migrated functions no longer rely on broad exception control flow for expected outcomes.

## Assumptions
- Default approach: selective migration, not repo-wide rewrite.
- Default boundary rule: routers/middleware keep exceptions; internal workflow code may return `Result`.
- Default error shape: structured domain errors, not raw exceptions or strings in `Err`.
- If the pilot shows mostly boilerplate and little readability gain, stop after the pilot and keep exceptions as the dominant pattern elsewhere.
