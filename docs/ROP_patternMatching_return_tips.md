# Return use

- `Success` / `Failure`, not `Ok` / `Err`
- stronger composition: `bind`, `flow`, `pipeline`, `pointfree`, `@safe`
- better Railway Oriented Programming (ROP) for multi-step workflows
- richer typing + mypy integration
- extra containers: `IOResult`, `RequiresContext`

### Updated Plan

# Returns (dry-python/returns) Adoption Plan

## Recommendation

Use the **`returns`** library selectively, not as a repo-wide replacement for exceptions.

This codebase already has a clean HTTP error boundary through `APIException` in  
`src/app/utils/exceptions.py` and the global exception handler in  
`src/app/middleware/global_exception_handler.py`. Keep that design intact.

Introduce `Result[Value, DomainError]` **only** in internal code paths where failure is expected, recoverable, and part of normal business flow. Leverage **`returns`** for its superior Railway Oriented Programming capabilities (`bind`, `flow`, `map_`, `@safe`, etc.), which will make complex ingestion, document-processing, and orchestration pipelines significantly more readable and composable than the original `result` package allowed.

**todo**  
Stop swallowing infrastructure errors as “not found”.  
In `users/repository.py`, `find_by_id` catches broad `Exception` and returns `None`. That conflates invalid IDs, driver failures, and true absence. Repositories should distinguish:

- invalid identifier
- not found
- database / infrastructure failure

Standardize one internal error taxonomy using structured `DomainError` dataclasses or exceptions. You already have good HTTP-facing exceptions. The missing piece is a clean non-HTTP domain/infrastructure error layer for internal code. **`returns`** makes propagating and composing these errors much more ergonomic.

Define transaction and rollback rules explicitly.  
If you adopt `returns`, document this rule for DB code: inside transaction scopes, prefer raising exceptions when rollback semantics depend on them (e.g., in workflow code like `ingestion/pipeline.py`). `returns` works beautifully for recoverable business failures, but does not replace the transaction manager’s exception-based rollback behavior.

## Where To Use Returns / Result

- Internal workflow, pipeline, and graph-style code where intermediate failures are expected and should be propagated explicitly (perfect for `flow` + `bind` pipelines).
- Repository helpers and adapters that normalize recoverable persistence or upstream failures.
- External-service adapters where third-party operational failures can be converted into typed `DomainError`s.
- Internal helper functions where the caller can handle failure without unwinding the whole stack.
- Ingestion, document-processing, and orchestration flows that already model partial failure as returned state (these will benefit the most from ROP).

**Key advantage over the old `result` package**: You can now write linear, readable pipelines instead of manual `if isinstance(result, Err)` checks or repetitive unwrapping.

## Where NOT To Use Returns / Result

- FastAPI routers, dependencies, middleware, and global exception handlers (keep using project exceptions).
- Transport and HTTP boundary logic — continue using exceptions from `src/app/utils/exceptions.py`.
- Programmer errors, invariant violations, misconfiguration, and assertion-style failures (these should still raise).
- Cancellation propagation such as `asyncio.CancelledError`.
- Transaction scopes where rollback semantics depend on exceptions being raised (raising remains appropriate here).

## Benefits For This Repo (Enhanced by returns)

- Expected failures become explicit in function signatures **and** composable via true Railway Oriented Programming.
- Internal workflows (especially ingestion and graph-style pipelines) become dramatically cleaner — no more manual error checking at every step.
- Service and repository tests can assert `Success(...)` / `Failure(...)` values directly and test pipelines end-to-end.
- Better alignment with Ruff `TRY` rules by reducing exception-based control flow for normal internal outcomes.
- Boundary ownership stays clean: internal layers return `Result`, while HTTP-facing layers decide how to convert `Failure` into `APIException`.
- Future-proof: If you later need `IOResult`, `Maybe`, or dependency injection via `RequiresContext`, the same library already supports it.

## Edge Cases and Nuances

- `None` and `Failure(...)` are different signals — keep “missing but acceptable” distinct from “operation failed”.
- Do not convert cancellation into `Failure`. `CancelledError` (and similar) must still propagate.
- Third-party libraries will still raise exceptions. Convert **only** expected operational failures at adapter boundaries using `@safe` or explicit try/except → `Failure`.
- Prefer `Failure(DomainError(...))` over `Failure(Exception(...))` or `Failure("...")`. Use structured domain errors for logging, mapping, and testing.
- Never return `Failure(APIException(...))` — transport concerns must stay at the HTTP boundary.
- Keep logging at the ownership boundary to avoid duplicate logs as a `Failure` travels through the pipeline.
- Do not force `Result` into code paths where it adds noise without improving clarity or composition.
- In DB transaction scopes, raising may still be the correct choice because the transaction manager relies on exceptions for rollback.
- Use `returns.pointfree.bind`, `map_`, `flow`, and the `@safe` decorator heavily for pipelines. This is where **`returns`** shines over the lighter `result` package.

## Migration Plan

1. **Environment & Dependency Check**  
   Confirm that `returns` (not `result`) is installed in the runtime used by `uv`, tests, and tooling.  
   Update `pyproject.toml` if needed:  

   ```toml
   dependencies = ["returns>=0.25.0"]  # or latest stable
   ```  

   Remove or deprecate the old `result>=0.17.0` dependency once migration of the pilot is complete. Run `uv sync` and verify import works.

2. **Add a Small Internal Returns Convention Module**  
   Create e.g. `src/app/utils/returns.py` or `src/app/domain/result.py` with:

   ```python
   from returns.result import Result, Success, Failure
   from returns.pipeline import flow
   from returns.pointfree import bind, map_
   from returns.curry import curry  # if needed
   from returns.decorators import safe  # for converting exceptions

   # Shared domain errors (dataclasses or exceptions)
   # Helper mappers: DomainError → APIException at boundaries
   ```

   This becomes the single source of truth for how `Result` is used in the project.

3. **Pilot One Vertical Slice**  
   Start in an area with heavy expected failure modeling, such as `ingestion/pipeline.py` or document-processing workflows.  
   Refactor one end-to-end path to use `Result` internally + `flow` / `bind` for composition, while the public service/router layer still returns the same HTTP contract.

4. **Add / Update Tests**  
   - Unit tests for `DomainError` → `APIException` mapping at boundaries.
   - Pipeline tests asserting `Success` and `Failure` cases.
   - Regression tests ensuring HTTP responses and behavior remain unchanged.

5. **Expand Only If Value Is Clear**  
   If the pilot reduces broad exception handling, eliminates manual error checks, and makes control flow clearer, extend to similar subsystems (repositories, external adapters, other workflows).  
   Otherwise, keep the pilot and continue using exceptions where `returns` does not provide a net win.

## Copilot / LLM Instructions (Updated for returns)

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

**Rule of thumb**: Exceptions outside-in (HTTP/framework boundaries), **`returns` Result** inside where it improves explicitness, composition, and readability.

## Default Policy

**Exception outside-in, `returns` Result inside** where it improves explicitness and composition via Railway Oriented Programming.

---

## Summary of Changes from Original `result` Plan

- **Package**: Switched from `rustedpy/result` (`Ok`/`Err`) → `dry-python/returns` (`Success`/`Failure`).
- **Composition**: Replaced manual checks with `flow(...)`, `bind(...)`, `map_`, etc. This is the biggest win for your ingestion/pipeline code.
- **Decorators**: Added mention of `@safe` for cleanly turning exception-throwing code into `Result`.
- **Error Types**: Still recommend structured `DomainError` (no change).
- **Boundary Mapping**: Still happens once at the HTTP edge (no change).
- **Migration Steps**: Updated dependency name, added convention module for ROP helpers, and emphasized pipeline benefits in the pilot.

This converted plan keeps the selective, pragmatic spirit of the original while unlocking the full power of Railway Oriented Programming that `returns` was designed for.

Would you like me to also provide:

- A concrete example of refactoring one function/pipeline from the old style to the new `returns` style?
- The suggested content for the new `src/app/utils/returns.py` convention module?
- Type hints and mypy configuration tips for `returns`?

Just let me know the next piece you'd like to tackle. This gives your codebase (and your agent persona) a much more senior, strategic, and composable error-handling foundation.

`ROP` + structural pattern matching give clear, explicit error handling for multi-step Python business logic.

### Core Principles of Railway Oriented Programming (ROP)

ROP uses 2 tracks:

1. **Success Track**: happy path, data move forward.
2. **Failure Track**: failure short-circuit success path, propagate error to end.

**Key Principles**:

- Return explicit two-track values: `Result[SuccessType, ErrorType]`, `Success(value)`, `Failure(error)`.
- Compose small functions:
  - one-track function -> `map`
  - two-track function -> `bind`
  - helpers like `tee`, `alt`, `lash` handle side effects, error transforms, recovery
- Failure short-circuit removes repeated manual checks.
- Separate concerns:
  - expected business/domain failures -> `Result`
  - programmer errors, invariants, misconfiguration -> raise exceptions
- Prefer explicit composition over hidden control flow.
- Use `lash` or explicit recovery when needed.

**Visual Metaphor**:

- Success stay on green track.
- `Failure` moves to red track until end or recovery.
- Composition operators act like switches between tracks.

Best fit: validation, ingestion pipelines, API orchestration, document processing, financial transactions, external service calls.

**Nuances and Edge Cases**:

- Do not force railway everywhere.
- Container overhead usually small.
- Use `@safe` at adapter boundaries for expected exceptions.
- Log at ownership boundaries, not every function.
- In transaction scopes, raising may still be better for rollback.

### How ROP Works in Modern Python with the `returns` Library

`dry-python/returns` gives:

- `Result[Value, Error]` with `Success(value)` and `Failure(error)`
- `flow()`, `bind()`, `map_()`, `alt()`, `lash()`
- `@safe` to convert exception-based code into `Result`
- strong mypy support + async helpers like `IOResult`

**Example: Multi-Step User Onboarding Pipeline**

```python
from returns.result import Result, Success, Failure
from returns.pipeline import flow
from returns.pointfree import bind, map_

# Each step returns Result[...] 
def validate_input(data: dict) -> Result[dict, str]:
    if not data.get("email"):
        return Failure("missing_email")
    return Success(data)

def enrich_user(data: dict) -> Result[dict, str]:
    # ... business logic, possibly calling external services
    return Success({**data, "enriched": True})

def save_to_db(user: dict) -> Result[int, str]:
    # ... DB operation
    return Success(123)

def send_welcome_email(user_id: int) -> Result[None, str]:
    # ...
    return Success(None)

# Clean ROP pipeline
onboard_user = flow(
    validate_input,
    bind(enrich_user),   # only runs if previous succeeded
    bind(save_to_db),
    bind(send_welcome_email),
)

result: Result[int, str] = onboard_user(raw_data)

# Handling the result (see pattern matching below)
```

Pipeline reads linear. Any failure short-circuits automatically.

### Pattern Matching for Handling Results in Modern Python (3.10+)

Python 3.10 `match` / `case` fits ROP well. It matches structure, type, bindings.

**Recommended Way to Unwrap Results**:

```python
from returns.result import Result, Success, Failure

def handle_onboarding(raw_data: dict):
    result = onboard_user(raw_data)
    
    match result:
        case Success(user_id):                  # Success track
            print(f"User {user_id} onboarded successfully!")
            # proceed with success logic
            
        case Failure(error_msg):                # Failure track
            print(f"Onboarding failed: {error_msg}")
            # handle specific errors, log, recover, or convert to APIException
            
        # Optional: more specific error patterns
        case Failure("missing_email"):
            # tailored response
            pass
```

**Advanced Patterns**:

```python
from dataclasses import dataclass

@dataclass
class ValidationError:
    field: str
    message: str

match result:
    case Success(user_id):
        ...
    case Failure(ValidationError(field="email", message=msg)):
        # specific handling for email validation
        ...
    case Failure(error) if isinstance(error, str):
        # generic string error
        ...
    case Failure(_):  # catch-all
        ...
```

**Why This Combination is Powerful**:

- `Result` makes both tracks explicit in types and values.
- `match` gives clean handling at boundaries.
- Less repetitive `isinstance(...)` or `.unwrap()` logic.
- Forces explicit failure modeling.

**Nuances and Edge Cases for Pattern Matching with ROP**:

- Put specific cases before generic cases.
- Use guards for extra conditions.
- For async pipelines, combine with async helpers.
- Use `.lash()` or `.alt()` before final `match` when partial recovery helps.
- `match` needs Python 3.10+; older versions can use `.unwrap_or()`, `.map_or()`, or `isinstance`.
- Keep `match` blocks shallow.

**Broader Implications and Considerations**:

- Fits `exceptions outside-in, Result inside`.
- Easy to test `Success(...)` and `Failure(...)`.
- Makes failure modes clearer for team members.
- May not help simple CRUD, one-off scripts, or hot paths.
- Learning curve exists; start with a small pilot.
- Pair with `@safe`, `flow`, and mypy.

If you’d like:

- A full side-by-side refactoring of one of your ingestion/pipeline functions using ROP + `match`
- More advanced `returns` patterns (e.g., `IOResult`, `RequiresContext`, parallel validation)
- Integration tips with your existing `APIException` boundary
- Examples of specific `DomainError` hierarchies

Share code snippet or scenario. I can compress next piece same way.
