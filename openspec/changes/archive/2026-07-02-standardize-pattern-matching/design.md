## Context

The codebase has grown to use 5 different pattern matching approaches without an explicit standard. The dominant pattern (`match`/`case` on `returns.Result`) is used in ~50+ blocks across the service layer and reads well. However, ~80+ `isinstance` checks exist alongside it, many redundant with the type system. This creates confusion for contributors and makes error-handling flow harder to trace.

Current state:
- `features/auth/service.py`: 30+ match/case blocks on Result — consistent, clean
- `features/agent_saul/service.py`: 15+ isinstance checks on LangChain/WebSocket types — legitimate dynamic data
- `middleware/global_exception_handler.py`: 4-branch isinstance chain on external exceptions — idiomatic
- `features/ingestion/service.py`: 1 isinstance+model_validate hybrid — inconsistent
- `features/documents/service.py`, `features/search/service.py`: mix of match/case and isinstance — needs alignment

## Goals / Non-Goals

**Goals:**
- Establish the 5 pattern approaches as documented project standard with a decision matrix
- Remove ~20-30 redundant isinstance checks that duplicate what the type system already guarantees
- Replace the 1 isinstance+model_validate hybrid with match/case
- Convert 2-3 service-layer isinstance guards on Result types to match/case for consistency
- Preserve all legitimate dynamic-data isinstance guards (Redis, LangChain, WebSocket, Celery)

**Non-Goals:**
- Rewriting the global_exception_handler.py isinstance chain (it's correct for external hierarchies)
- Converting all 80+ isinstance checks (many are legitimate)
- Adding new lint rules or pre-commit hooks (documented convention is sufficient)
- Changing any public API, return types, or error contracts
- Migrating away from the `returns` library

## Decisions

### 1. Keep match/case as the service-layer standard (not isinstance)

**Choice:** `match`/`case` on `returns.Result` for all service-layer unwrapping.

**Rationale:** Already used in 50+ blocks. Exhaustive matching catches missing cases at development time. Guard conditions (`case Success(x) if x is not None`) combine unwrap + validation in one step. The `case _:` wildcard handles the "anything else" path clearly.

**Alternatives considered:**
- `isinstance(result, Failure)` + if/elif: More verbose, doesn't leverage exhaustiveness, inconsistent with existing code.
- Monadic chaining (`.value_or()`, `.map()`): More functional but harder to read for Python developers unfamiliar with `returns`.

### 2. Remove redundant isinstance on typed data

**Choice:** Delete isinstance checks where the type is already guaranteed by annotations or Pydantic validation.

**Rationale:** These checks are noise. If a function parameter is `list[dict]`, checking `isinstance(x, list)` adds nothing. If a Pydantic model field is `str | None`, use `if x is not None:` instead of `isinstance(x, str)`.

**Scope of removals:**
- `isinstance(exc.detail, dict)` — exc.detail is typed
- `isinstance(group, dict)` — iterating a typed list
- `isinstance(value, str)` on typed parameters
- `isinstance(raw_groups, list)` on Pydantic fields

### 3. Preserve isinstance on genuinely dynamic data

**Choice:** Keep all isinstance checks on data from external boundaries.

**These are NOT being removed:**
- Redis: `isinstance(cached, bytes)` — Redis returns bytes/str/None depending on decoder
- LangChain: `isinstance(m, SystemMessage)` — message lists contain mixed types
- WebSocket: `isinstance(inbound, WSPingMessage)` — discriminated union from wire
- Celery: `isinstance(task_result.result, dict)` — result could be any pickled type

### 4. Replace isinstance+model_validate hybrid with match/case

**Choice:** Use `match failure: case AppError(): ... case dict() as raw: AppError.model_validate(raw)` instead of `isinstance(failure, AppError) else AppError.model_validate(failure)`.

**Rationale:** The match/case version is more explicit about what's being handled and scales cleanly if more types are added to the union.

## Risks / Trade-offs

- **[Risk] Removing isinstance checks reveals latent type bugs** → Mitigation: Run `ty check src/` and `ruff check src/` after each removal batch. The type checker will catch real issues.
- **[Risk] Contributors may not read the pattern standard** → Mitigation: The decision matrix in RESULT-PATTERN.md is concise and grep-friendly. Add a note in ARCHITECTURE-RULES.md linking to it.
- **[Trade-off] match/case is more verbose than isinstance for simple checks** → Accepted: The exhaustiveness benefit and consistency outweigh the verbosity for service-layer code.
