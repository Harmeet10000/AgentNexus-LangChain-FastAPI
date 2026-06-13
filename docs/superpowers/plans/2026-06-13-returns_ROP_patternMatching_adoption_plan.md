# `returns` / Result Pattern Adoption Plan

This document is the canonical repository-specific policy and phased migration plan for adopting `returns` in this codebase.

## Current State

- Repository policy selects `returns` as the Result-style library for this codebase.
- `result` was the older Result-style dependency and should remain absent once runtime usage is verified as removed.
- Migration is no longer only a dependency declaration. Code adoption should still be evaluated against this document, not assumed from dependency presence.
- This document consolidates previously duplicated policy and migration content into a single source of truth.

## Adoption Policy

Default policy:

- Use exceptions outside-in.
- Use `returns.Result` only for expected, recoverable internal failures where callers benefit from explicit branching on failure meaning.
- Keep `None` when absence is acceptable and callers do not need to know why the value is missing.
- Unexpected failures should still raise and be logged.

In this repository, the existing HTTP exception boundary is already clear and should remain intact:

- `src/app/utils/exceptions.py`
- `src/app/middleware/global_exception_handler.py`
- `src/app/utils/http_response.py`

`returns` is an internal modeling tool in this repository. It is not a replacement for FastAPI's exception model, the global exception handler, or existing transport contracts.

Rule of thumb:

- exceptions outside-in
- `returns` inside where explicit expected failures improve correctness, composition, and readability
- `None` where absence is the only signal that matters

The real boundary is not "Result vs exception". The boundary is who still owns recovery. Use `Result` only while a caller can still make a meaningful local decision. Once the decision is "tell FastAPI", "tell the client", or "tell the task runner", convert back to the existing boundary contract.

## Boundary Rules

### Where `Result` Should Be Used

Use `returns.Result` in internal code paths that already need explicit, typed recoverable failures.

Good candidates in this repository:

- sync pure helpers that validate, normalize, parse, or transform data before I/O
- internal workflow helpers where multiple steps can fail in expected ways and the caller must react differently depending on failure type
- repository helper boundaries that need to distinguish normal absence, invalid identifiers, and infrastructure failure
- external-service adapters when a known subset of operational failures should be normalized into typed internal errors
- internal graph and pipeline helpers that currently use ad-hoc `error` fields, status flags, or manual branching after every step

### Where `Result` Should Not Be Used

Do not use `Result` as a repo-wide replacement for exceptions.

Do not use `Result` in:

- FastAPI routers
- FastAPI dependencies
- middleware
- lifespan wiring
- Celery task entrypoints
- WebSocket frame contracts
- LangChain tool contracts
- `APIResponse`
- `ToolOutput`
- `MCPToolResponse`
- diagnostics-heavy paths where real traceback ownership matters most
- fail-fast situations such as startup failures, misconfiguration, broken invariants, programmer mistakes, or cancellation
- rollback-sensitive transaction scopes where exception propagation is part of correctness

Do not use `Result` when callers do not care about the error shape:

- silent lookup or cache-miss style flows where `None` is already the correct contract
- local private control flow where ordinary branching or a local exception is clearer

### Failure Payload Rules

Do not return or propagate:

- `Failure(Exception(...))`
- `Failure(APIException(...))`
- traceback strings inside failures
- raw string errors as the long-term contract
- transport-layer errors inside internal Result values

Expected failures should carry structured, typed meaning rather than diagnostics-heavy payloads.

Required shared convention:

- create `src/app/shared/result/`
- define `AppResult[T] = Result[T, AppError]`
- define `AppFutureResult[T]` only as an available convention, not as the default async style
- use frozen Pydantic models for expected internal error types
- include fields appropriate to boundary mapping and observability, such as `code`, `message`, `details`, `retryable`, `source`, operation name, correlation ID if available, and flow/execution path
- add mapper functions from internal Result errors to existing project exceptions
- add a narrow structured logging helper for expected failures at ownership boundaries only

`Failure` must not become a stealth logging system. A `Failure` should carry business or operational meaning, not forensic detail. Stack traces belong to raised exceptions and boundary logs.

Boundary logging for expected failures should bind at least:

- `error_code`
- `retryable`
- `source`
- `operation`
- `correlation_id` when available from request or task context
- `flow` or `execution_path` when available

Keep `logger.exception(...)` for unexpected crashes. Expected failures should be logged once at the ownership boundary, not at every `Failure(...)` construction site.

### Service Boundary Rule

This is a hard rule for this repository:

- expected internal `Failure(...)` values must be mapped to project exceptions before leaving the service layer
- services should not return failure DTOs such as `status="failed"` for these paths unless the endpoint contract is intentionally designed as a non-exception summary or status API

Result-aware services must unwrap or translate failures before returning. `global_exception_handler` remains the HTTP exception boundary and must not be taught to understand `Result` or `Failure` directly.

### Transaction Rule

If rollback semantics depend on exceptions, keep raising inside the transaction boundary. Translate to `Failure(...)` only outside that rollback-sensitive block if the surrounding ownership boundary needs a typed recoverable outcome.

This is especially important in:

- `src/app/shared/langgraph_layer/ingestion_kb/pipeline_node.py`
- `src/app/shared/langgraph_layer/reconciliation`

## Async And Pattern Matching Guidance

This repository is async-first. That matters more than library style.

Default async policy:

- keep async service, repository, and LangGraph node entrypoints as normal async functions
- use `Result` mainly inside extracted sync helpers
- use `FutureResult` only when composition becomes materially clearer than ordinary async code
- do not treat `FutureResult` as the default abstraction
- avoid advanced `returns` containers in the first pass unless a later slice proves a concrete need

Plain `Result` is most appropriate for:

- validation
- parsing
- normalization
- mapping
- domain decision helpers

Use `flow`, `bind`, `map_`, `@safe`, or related `returns` composition helpers only when they make the pipeline clearer than ordinary Python. They are optional tools, not a style mandate. In async code, prefer ordinary `async def` plus local `Result` helpers unless `FutureResult` materially removes repeated branching.

Pattern matching guidance:

- use `match` / `case` at ownership boundaries to unwrap `Success(...)` / `Failure(...)`
- use it when mapping internal failures to project exceptions, DTO contracts, or graph-state updates
- prefer matching typed domain errors over raw strings
- keep one generic fallback branch at the boundary
- do not use pattern matching to swallow unexpected exceptions
- do not use pattern matching inside every step of a pipeline if ordinary composition is clearer

Pattern matching is guidance, not a mandate. If a small mapper function is clearer than `match`, use the smaller clearer function.

## Tooling And Enforcement

There are no first-class Ruff or `ty` rules that enforce selective ROP architecture directly. In this repository, tooling is approximation and guardrail, not the source of truth. This document and code review remain the source of truth.

### Ruff Guardrails

Keep `pyproject.toml` rules intact. Do not weaken checks for this migration.

The most relevant Ruff families are:

- `BLE`, `TRY`, `RSE`, `EM` for deliberate exception handling
- `ANN`, `RET`, `B`, `SIM`, `C4` for explicit typed control flow
- `ASYNC` for avoiding hidden async mistakes
- `LOG` and `G` for boundary logging hygiene

Repo-specific note:

- `PLR0911` is intentionally ignored in this repository. Many returns are not, by themselves, a reason to force code into `Result`.

### `ty` Guardrails

The most relevant `ty` rules are:

- `invalid-return-type`, `invalid-argument-type`, `no-matching-overload` for Result aliases and mappers
- `invalid-raise`, `invalid-exception-caught` for exception boundary correctness
- `await-on-non-awaitable`, `non-awaitable-in-async-function`, `unused-awaitable` for async correctness
- `missing-typed-dict-key`, `invalid-key`, `possibly-missing-attribute` to expose ad-hoc graph-state error plumbing

### Review Checklist

Use review and tests rather than custom lint in this pass.

Review expectations:

- no `returns` imports in routers, middleware, lifespan, dependencies, Celery tasks, or transport-contract modules
- no `Failure(Exception)` or `Failure(APIException)`
- all public Result-returning helpers have explicit return annotations
- every service boundary either unwraps success or maps failure to existing project exceptions or intentionally stable DTO contracts
- expected failures are logged once at the ownership boundary, not at every `Failure` construction point

## Migration Phases

### Phase 1: Foundation

Goals:

- keep this document as the source of truth for the migration
- update `.github/copilot-instructions.md` so generated code follows the same selective-adoption rules
- add `src/app/shared/result/` as the shared internal convention module
- define `AppResult[T] = Result[T, AppError]`
- define a small internal error taxonomy using frozen Pydantic models
- add boundary mappers to existing project exceptions
- add a structured logging helper for expected failures at ownership boundaries
- start using `trace_layer(...)` or equivalent flow tracking in real orchestration entrypoints where it materially improves diagnostics
- include correlation ID and flow/execution-path context in expected-failure logs when available

### Phase 2: Ingestion Vertical Slice

Target files:

- `src/app/shared/langgraph_layer/ingestion_kb/pipeline_node.py`
- `src/app/features/ingestion/service.py`

Goals:

- replace ad-hoc `error` and `extraction_error` plumbing with typed internal failures
- keep LangGraph node signatures unchanged and compatible with LangGraph
- keep `DocumentUploadResponse` unchanged unless an explicit contract change is approved later
- map final `Failure` values to project exceptions at the service boundary so upload failures continue through `global_exception_handler`
- use the actual active ingestion node module in this checkout (`nodes.py`) if `pipeline_node.py` is stale or absent

### Phase 3: Reconciliation Vertical Slice

Target area:

- `src/app/shared/langgraph_layer/reconciliation`

Goals:

- replace `fetch_error`, `reconcile_error`, and `apply_error` plumbing internally with typed failures
- keep graph-state and task-summary outer contracts stable unless explicitly changed later
- preserve exception-based rollback behavior inside SQLAlchemy transaction-sensitive sections

### Phase 4: Repository Ambiguity Cleanup

Target files:

- `src/app/features/users/repository.py`
- `src/app/features/auth/repository.py`

Goals:

- remove broad `except Exception: return None` behavior around lookup and identifier-parsing paths
- distinguish malformed identifier, normal absence, and infrastructure failure
- map repository failures once at the service boundary

This phase has especially high production value because collapsing invalid identifiers and infrastructure failures into `None` can silently change authorization, not-found, and outage behavior.

### Phase 5: Selective Expansion

Expand only into modules that already emulate Result manually with:

- sentinel `None`
- empty-string or fallback status values
- ad-hoc `error` fields
- repeated manual branching after each step

Do not expand by default into:

- FastAPI routers
- middleware
- lifespan
- auth and profile flows already clear with typed exceptions
- MCP and tool interop contracts
- fail-fast infrastructure code

Good later candidates, subject to local inspection:

- internal parser/normalizer helpers that currently return ad-hoc `success` / `error` shapes before being mapped to stable DTOs
- external-service adapter helpers where a known subset of upstream failures should become typed, retryable internal failures
- graph or workflow helpers where repeated manual branching hides the meaningful failure categories

Keep these contracts unchanged unless a separate contract decision approves a change:

- HTTP response DTOs
- health-check summary payloads
- WebSocket frame contracts
- LangChain tool return contracts
- MCP tool contracts
- existing `ToolResult` / `MCPToolResponse` style interop envelopes

### Phase 6: Dependency Cleanup

- once migration code is in place and runtime usage is verified, remove the unused `result` dependency
- keep `returns` as the only Result-style library in the repository
- verify there are no `from result ...` or `import result` runtime imports before removing the dependency

## Documentation Source Of Truth

This file is the canonical migration plan. Do not maintain a second `returns` adoption plan with conflicting paths, checklist state, or policy wording.

If older notes are useful, fold their durable guidance into this document and delete or archive the duplicate. Stale paths such as `pipeline_node.py` or `pipeline (1).py` must be reconciled against the current repository layout before implementation.

## Test And Verification Plan

### Shared Result Module Tests

- invalid identifier maps to `ValidationException`
- not found maps to `NotFoundException`
- infrastructure failure maps to `DatabaseException` or `ExternalServiceException`
- retryable upstream failure maps to `ServiceUnavailableException` where appropriate

### Repository Tests

- malformed Beanie object ID is not collapsed into not-found
- missing user remains a normal absence or not-found path according to the target contract
- simulated Beanie or Motor failure is mapped as infrastructure failure

### Ingestion Tests

- empty document or missing required intermediate state produces typed failure internally
- service boundary maps expected failures into project exceptions
- unexpected graph exceptions still log with `logger.exception(...)` and become the existing HTTP error envelope behavior

### Reconciliation Tests

- fetch, reconcile, and apply expected failures produce typed internal errors
- SQL transaction failures still raise within transaction scope so rollback behavior is preserved
- task-facing summary behavior remains stable unless intentionally changed

### Verification Commands

- `uv run ruff format src/`
- `uv run ruff check src/`
- `uv run ty check src/`
- targeted `pytest` commands for each changed slice

## Implementation Checklist

- [ ] Keep this document as the source of truth.
- [ ] Update `.github/copilot-instructions.md` with the same selective-adoption rules.
- [ ] Add `src/app/shared/result/`.
- [ ] Define `AppResult[T] = Result[T, AppError]`.
- [ ] Define frozen Pydantic error models for expected failures.
- [ ] Add boundary mappers to existing project exceptions.
- [ ] Add a structured logging helper for expected failures.
- [ ] Start using `trace_layer(...)` or equivalent only where it materially improves orchestration diagnostics.
- [ ] Convert the ingestion slice.
- [ ] Convert the reconciliation slice.
- [ ] Clean up repository ambiguity paths.
- [ ] Expand only where Result improves clarity more than it adds abstraction.
- [ ] Remove `result` after migration confirms no runtime usage.

## Assumptions And Non-Goals

Assumptions:

- `returns` remains the selected Result library for this repository
- `result` should be removed after migration confirms there is no runtime usage that must be preserved
- HTTP service boundaries should raise mapped project exceptions for expected internal failures
- `FutureResult` is not the default; ordinary async functions remain the baseline
- no custom lint checker or Ruff plugin will be added in this pass
- this document is repo-specific and intentionally names concrete files, boundaries, and anti-patterns in this codebase

Non-goals:

- rewriting FastAPI routers around `Result`
- teaching `global_exception_handler` to understand `Failure` directly
- replacing every exception with `Result`
- converting diagnostics-heavy failures into value objects
- changing stable transport contracts unless a separate decision explicitly approves that work

## Examples

### Service Boundary Mapping Example

```python
match result:
    case Success(value):
        return value
    case Failure(UserLookupError(code="invalid_id", details={"identifier": identifier})):
        raise ValidationException(
            detail="Invalid user identifier",
            data={"identifier": identifier},
        )
    case Failure(UserLookupError(code="not_found", details={"identifier": identifier})):
        raise NotFoundException("User", identifier)
    case Failure(error):
        log_expected_failure(error)
        raise DatabaseException(detail="User lookup failed")
```

### Repository Ambiguity Example

Prefer this distinction:

- malformed identifier -> typed invalid-identifier failure
- valid identifier with no row or document -> normal absence or not-found path
- database or driver failure -> typed infrastructure failure

Do not collapse all three into `None`.

### Anti-Example

Do not do this:

```python
return Failure(APIException(detail="Upload failed"))
```

Transport-layer exceptions must be created only at the HTTP ownership boundary.
