# `returns` / Railway-Oriented Programming Adoption Plan

This document defines how `returns` should be adopted in this repository.

The policy is selective, not repo-wide. The goal is to make expected internal failures explicit where that improves clarity, while preserving the existing exception-based boundaries, diagnostics, and framework integration that already work well in this codebase.

## Purpose And Default Policy

Default policy:

- Use exceptions outside-in.
- Use `Result` inside only where failure is expected, recoverable, and part of normal control flow.
- Keep `None` when absence is acceptable and callers do not care why the value is missing.
- Keep diagnostics boundary-only. Real stack traces should come from raised exceptions and structured logs, not from values stored inside `Failure(...)`.

In this repository, the current HTTP error boundary is already clear:

- `src/app/utils/exceptions.py`
- `src/app/middleware/global_exception_handler.py`
- `src/app/utils/http_response.py`

That design should remain intact. `returns` is an internal modeling tool here, not a replacement for the FastAPI error model.

Rule of thumb:

- Exceptions outside-in
- `returns` inside where explicit expected failures improve correctness and readability
- `None` where absence is the only signal that matters

## Where `Result` / ROP Should Be Used In This Repo

Use `returns` in internal code paths that are already doing ad-hoc Result-style work today.

Good candidates:

- Sync pure helpers that validate, normalize, parse, or transform data before I/O.
- Internal workflow helpers where multiple steps can fail in expected ways and the caller needs to react to those failures explicitly.
- Repository helper boundaries that need to distinguish:
  - normal absence
  - invalid input such as malformed identifiers
  - infrastructure or driver failures
- External-service adapters when a specific set of operational failures should be normalized into typed internal errors.
- Internal graph and pipeline helpers that currently return `"error"` fields, status flags, or other ad-hoc failure payloads.

Best first migration slices:

- `src/app/shared/langgraph_layer/ingestion_kb/pipeline_node.py`
- `src/app/features/ingestion/service.py`
- `src/app/features/reconciliation/pipeline (1).py`
- `src/app/features/users/repository.py`
- `src/app/features/auth/repository.py`

Why these are first:

- They already emulate Result manually with error strings, status flags, and broad `except Exception` blocks.
- Moving these modules to typed internal failures reduces ambiguity without changing public HTTP contracts.

## Where `Result` / ROP Should Not Be Used In This Repo

Do not use `Result` as a universal replacement for exceptions.

Do not use it for diagnostics-heavy paths:

- If the primary need is a real traceback, exception chaining, or crash diagnostics, raise and log the exception.

Do not use it to reinvent exceptions:

- Unexpected failures will still exist.
- Top-level boundaries will still need exception handling.
- Wrapping everything in `Failure(Exception(...))` adds ceremony without improving the design.

Do not use it for fail-fast situations:

- Startup failures in `src/app/lifecycle/lifespan.py`
- Misconfiguration
- Broken invariants
- Programmer mistakes
- Cancellation such as `asyncio.CancelledError`

Do not use it for private control flow where local exceptions are simpler:

- If the logic is local, hidden from consumers, and clearer with a local exception or simple branching, keep it imperative.

Do not use it where callers do not care about the error shape:

- Silent flows where absence is enough
- Internal lookups where `None` is already the right contract

Do not model every I/O failure as a domain result:

- Infrastructure surfaces such as raw database, Redis, HTTP SDK, Celery, or startup wiring should still raise on unexpected failures.
- Only normalize the subset of failures that the domain actually wants to treat as expected.

Be cautious in performance-sensitive code:

- Do not add container churn to hot paths unless the clarity gain is real and measured.

Do not force `Result` into interop-facing contracts:

- `ToolOutput`
- `MCPToolResponse`
- existing DTOs with `success` / `error`
- HTTP response envelopes

Those boundaries should stay idiomatic for their consumers.

## Diagnostics And Stack-Trace Policy

The main criticism of Railway-Oriented Programming is valid: when errors become ordinary values, you lose automatic stack traces unless you deliberately preserve diagnostics elsewhere.

This repository should handle that by keeping diagnostics at ownership boundaries, not by putting exceptions inside `Failure(...)`.

Policy:

- Do not store traceback strings inside `Failure(...)`.
- Do not store full exceptions inside domain errors by default.
- Do not return `Failure(APIException(...))`.
- Unexpected exceptions should still be raised and logged with `logger.exception(...)`.

Expected failures should carry structured metadata instead:

- `code`
- `message`
- `details`
- `retryable`
- operation name
- correlation ID if available
- flow or logical execution path if available

Existing logging infrastructure that should support this:

- `src/app/utils/logger.py`
- `request_state`
- `execution_path`
- `trace_layer(...)`
- correlation ID set in `src/app/middleware/server_middleware.py`
- `LOG_BACKTRACE` and `LOG_DIAGNOSE` settings in `src/app/config/settings.py`

Recommended implementation policy for the later code migration:

- Keep `logger.exception(...)` for unexpected crashes.
- Add one boundary logging helper for expected failures that binds:
  - `error_code`
  - `retryable`
  - `correlation_id`
  - `flow`
- Start using `trace_layer(...)` or an equivalent wrapper in real service/orchestration entrypoints so `execution_path` becomes meaningful outside the example file.
- Keep HTTP trace exposure limited to the existing non-production path in `global_exception_handler`.

Boundary-only diagnostics means:

- `Failure(...)` gives typed business or operational meaning.
- raised exceptions give stack traces.
- structured logging connects the two.

## Pattern Matching Guidance

Structural pattern matching is part of the migration policy, but it has a narrow role.

Use `match` / `case` at ownership boundaries to unwrap `Success(...)` / `Failure(...)`, not inside every step of a pipeline.

Good uses:

- Mapping internal failures to `APIException`
- Mapping internal failures to DTO error payloads
- Mapping internal failures to graph state updates
- Matching on typed domain errors for targeted recovery or translation

Do not use pattern matching:

- inside every step of a pipeline
- as a substitute for composition
- for deep nested control flow that is harder to read than ordinary branching

Repository rules for pattern matching:

- Prefer specific cases before generic ones.
- Match on typed domain errors, not raw strings, whenever possible.
- Keep one final `Failure(_)` catch-all branch at the boundary.
- Do not use pattern matching to swallow unexpected exceptions. Unexpected exceptions should still propagate and be logged.

Pattern matching is expected to help first in:

- the ingestion service boundary
- the reconciliation graph boundary
- service-layer mapping from repository/internal failures to HTTP exceptions

Example boundary shape:

```python
match result:
    case Success(value):
        return value
    case Failure(UserLookupError(kind="invalid_id", identifier=identifier)):
        raise ValidationException(
            detail="Invalid user identifier",
            data={"identifier": identifier},
        )
    case Failure(UserLookupError(kind="not_found", identifier=identifier)):
        raise NotFoundException("User", identifier)
    case Failure(error):
        log_expected_failure(error)
        raise DatabaseException(
            detail="User lookup failed",
        )
```

Pattern matching is guidance, not a mandate. If a simple mapper function is clearer than `match`, use the simpler code.

## Async-First `returns` Guidance

This repository is async-first. That matters.

Plain `Result` is appropriate for sync pure helpers.

Examples:

- validation
- parsing
- normalization
- mapping
- domain decision helpers

For async flows, use `FutureResult` or `@future_safe` only when composition becomes materially clearer than ordinary async code.

Policy for async use:

- Keep async service, repository, and LangGraph node entrypoints as normal async functions.
- Use `Result` inside extracted sync helpers.
- Use `FutureResult` only for clearly compositional async workflows where the extra abstraction removes repeated branching.
- Avoid advanced containers such as `IOResult`, `RequiresContext`, and similar abstractions in the first pass.

Important nuance for transaction scopes:

- If rollback semantics depend on exceptions, keep raising inside the transaction boundary.
- Translate to `Failure(...)` only outside that rollback-sensitive block.

This is especially important in:

- `src/app/shared/langgraph_layer/ingestion_kb/pipeline_node.py`
- `src/app/features/reconciliation/pipeline (1).py`

## Ruff / `ty` Rules That Best Approximate ROP Discipline In This Repo

There are no first-class Ruff or `ty` rules that enforce Railway-Oriented Programming, `returns` adoption, or a selective exception-versus-Result architecture directly.

In this repository, tooling should be treated as approximation and guardrail, not as the source of truth. The source of truth is this document plus code review.

### Ruff Rules That Approximate The Policy

The current Ruff configuration already enables the rule families that best support the intended discipline:

- `TRY`
  - Helps keep exception handling deliberate.
  - Useful for catching noisy or poorly-structured `try` / `except` usage in code that should either fail fast or return a typed failure cleanly.
- `BLE`
  - Important for this migration because broad `except Exception` blocks are one of the main anti-patterns being cleaned up.
  - Especially relevant in repository methods that currently collapse infrastructure failure into `None`.
- `RET`
  - Helps keep branching and returned values explicit.
  - Useful when replacing ad-hoc status strings or `"error"` payloads with clearer success/failure paths.
- `ANN`
  - Supports the policy that public helpers should make success and failure shapes explicit in their signatures.
  - This matters when introducing `AppResult[T]` and `AppFutureResult[T]`.
- `ASYNC`
  - Important because this repo is async-first and `returns` should not become an excuse for hiding async misuse.
  - Supports the rule that async entrypoints remain ordinary async functions even when internal helpers use `Result`.
- `EM` and `RSE`
  - Reinforce cleaner exception construction at boundaries where exceptions should still be raised.
  - Useful for the "exceptions outside-in" side of the policy.
- `LOG` and `G`
  - Support the boundary-only diagnostics policy by nudging logging toward structured, machine-parseable patterns.
  - Helpful where expected failures are logged once at the ownership boundary.

Useful repo-specific note:

- `PLR0911` is explicitly ignored in this repo.
  - That is reasonable for graph and orchestration code.
  - Do not treat "many returns" as a signal that code should automatically become ROP; use `Result` only where it improves the design.

### `ty` Rules That Approximate The Policy

The current `ty` configuration gives the strongest support on typing and async correctness, not on ROP semantics.

The most relevant rules are:

- `invalid-return-type`
  - Helps enforce that helper functions actually return the declared `Result` or non-Result type.
  - Useful when stabilizing shared aliases such as `AppResult[T]`.
- `invalid-argument-type`, `missing-argument`, `no-matching-overload`, `unknown-argument`
  - Help catch refactor drift when functions start accepting typed error containers or typed mappers.
- `invalid-raise` and `invalid-exception-caught`
  - Support disciplined exception usage at boundaries.
  - Useful for ensuring that exception-based paths remain valid while internal expected failures move to `Result`.
- `await-on-non-awaitable`, `non-awaitable-in-async-function`, `unused-awaitable`
  - Critical for the async-first guidance.
  - These rules matter more than library style rules when introducing `FutureResult` or `@future_safe`.
- `missing-typed-dict-key`, `invalid-key`, `possibly-missing-attribute`
  - Very relevant to the first migration slices because those modules currently pass ad-hoc error-bearing dict shapes through pipelines and graph state.
  - These rules help expose the exact ambiguity that typed failures are meant to replace.
- `unresolved-reference`, `unresolved-import`, `possibly-missing-import`
  - Useful for keeping the migration safe while adding shared convention modules and error types.

### Pattern Matching And Tooling

Tooling support for structural pattern matching is limited but still useful:

- `ty` can catch invalid match patterns via `invalid-match-pattern`.
- Neither Ruff nor `ty` currently gives you a meaningful "use pattern matching here" or "this match is the right boundary-unwrapping style" rule.

That means the pattern matching policy remains architectural guidance:

- use `match` / `case` at boundaries
- prefer typed domain errors over raw strings
- keep one generic fallback branch
- do not hide unexpected exceptions behind pattern matching

### Practical Policy

Use Ruff and `ty` to enforce the approximations:

- explicit signatures
- disciplined exceptions
- structured logging
- async correctness
- fewer broad catch-all handlers
- less ad-hoc dict-based error plumbing

Do not expect Ruff or `ty` to enforce:

- when to use `Result` versus exceptions
- whether a `returns` abstraction is adding clarity
- whether pattern matching is being used at the correct ownership boundary

Those decisions still require repo policy and code review.

## Migration Phases

### Phase 1: Foundation

- Rewrite this document as the source of truth for the migration.
- Update `.github/copilot-instructions.md` so generated code follows the same rules.
- Add one shared internal convention module under `src/app/shared` containing:
  - `AppResult[T]` for sync pure helpers
  - `AppFutureResult[T]` or equivalent async alias
  - shared internal error types
  - boundary mappers
  - logging helper for expected failures

- Use `Result[ValueType, DomainError]` (from `returns.result`) for expected, recoverable failures in internal workflow code, repository helpers, and external-service adapters.
- Leverage `flow`, `bind`, `map_`, and `@safe` for clean Railway Oriented pipelines.
- Keep FastAPI routers, dependencies, middleware, and global exception handling **exception-based**. Use project exceptions from `src/app/utils/exceptions.py` at those boundaries.
- Prefer `returns` when failure is part of normal business flow **and** the caller benefits from composition without unwinding the stack.
- Do not use `Result` for programmer errors, broken invariants, misconfiguration, or cancellation — those should raise.
- Do not return `Failure(Exception)` or raw strings by default. Prefer structured `DomainError` types (dataclasses recommended).
- Never return `Failure(APIException(...))`; transport-layer exceptions must be created only at the HTTP boundary.
- Map `Failure(DomainError) → APIException` **once** at the service/router boundary (unwrap with `.unwrap()` or pattern matching / `.lash()` for recovery).
- In transaction scopes where rollback depends on exceptions, prefer raising over returning `Failure`.
- Catch third-party operational failures at adapter boundaries (use `@safe` where appropriate) and convert **only** expected cases into `Failure`; let unexpected exceptions propagate.
- Log failures at the ownership boundary, not at every `Failure` construction.

**Rule of thumb**: Exceptions outside-in (HTTP/framework boundaries), **`returns` Result** inside where it improves explicitness, composition, and readability

### Phase 2: Ingestion Vertical Slice

Target files:

- `src/app/shared/langgraph_layer/ingestion_kb/pipeline_node.py`
- `src/app/features/ingestion/service.py`

Goals:

- replace ad-hoc `"error"` and `"extraction_error"` plumbing with typed internal failures
- keep LangGraph node signatures unchanged
- keep `DocumentUploadResponse` unchanged
- use pattern matching only at the service or node boundary where results are translated back into the existing response/state contract

### Phase 3: Reconciliation Vertical Slice

Target file:

- `src/app/shared/langgraph_layer/reconciliation`

Goals:

- replace `fetch_error`, `reconcile_error`, and `apply_error` string plumbing internally with typed failures
- keep graph state shape unchanged at the outer boundary
- preserve exception-based rollback inside transaction-sensitive sections

### Phase 4: Repository Ambiguity Cleanup

Target files:

- `src/app/features/users/repository.py`
- `src/app/features/auth/repository.py`

Goals:

- remove broad `except Exception: return None`
- distinguish invalid identifier, not found, and infrastructure failure
- keep service-layer HTTP behavior unchanged by mapping failures once at the service boundary

### Phase 5: Selective Expansion

Expand only into modules that are already emulating Result with:

- sentinel `None`
- empty string or status fallback
- ad-hoc `error` fields
- manual branching after every step

Do not expand by default into:

- FastAPI routers
- middleware
- lifespan
- auth and profile flows that are already clear with typed exceptions
- MCP and tool interop contracts
- fail-fast infrastructure code

### Phase 6: Dependency Cleanup

- Once migration code is in place and runtime usage is verified, remove the unused `result` dependency.
- Keep `returns` as the only Result-style library in the repository.

## What Remains Unchanged

These parts of the architecture should stay exception-based and should not be rewritten around `returns`:

- `APIException`
- `http_error(...)`
- `APIResponse`
- `global_exception_handler`
- FastAPI routers
- FastAPI dependencies
- middleware
- lifespan wiring
- Celery task entrypoints
- MCP tool contracts
- `ToolOutput`
- `MCPToolResponse`

Existing service flows that are already clear with typed exceptions should generally remain the same:

- `src/app/features/auth/service.py`
- `src/app/features/profile/service.py`

The migration target is ambiguity and ad-hoc error signaling, not code that is already explicit.

## Where To Use Returns / Result

- Internal workflow, pipeline, and graph-style code where intermediate failures are expected and should be propagated explicitly (perfect for `flow` + `bind` pipelines).
- Repository helpers and adapters that normalize recoverable persistence or upstream failures.
- External-service adapters where third-party operational failures can be converted into typed `DomainError`s.
- Internal helper functions where the caller can handle failure without unwinding the whole stack.
- Ingestion, document-processing, and orchestration flows that already model partial failure as returned state (these will benefit the most from ROP).

## Implementation Checklist

- [ ] Keep this document as the source of truth for the migration.
- [ ] Update `.github/copilot-instructions.md` with the same selective-adoption rules.
- [ ] Add a shared internal convention module under `src/app/shared`.
- [ ] Define a small internal error taxonomy for expected failures.
- [ ] Add boundary mappers to HTTP exceptions, DTOs, and graph-state updates.
- [ ] Add a logging helper for expected failures using structured fields.
- [ ] Start using `trace_layer(...)` or equivalent flow tracking in real orchestration entrypoints.
- [ ] Convert the ingestion slice.
- [ ] Convert the reconciliation slice.
- [ ] Clean up broad `except Exception: return None` repository paths.
- [ ] Expand only where Result improves clarity more than it adds abstraction.
- [ ] Remove the old `result` dependency after the code migration is stable.

## Final Position

`returns` is useful in this repository, but only in a limited role.

Use it for expected internal control-flow failures where explicit modeling improves the code.

Do not use it for:

- diagnostics-heavy failures
- framework boundaries
- fail-fast paths
- rollback-sensitive exception handling
- interop contracts

Use pattern matching as a boundary tool, not as a blanket style.

Keep real stack traces tied to raised exceptions and structured logs.

That gives this codebase the right split:

- typed expected failures where the domain benefits
- exceptions where Python and the framework already provide the right tool
