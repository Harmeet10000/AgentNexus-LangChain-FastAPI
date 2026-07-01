## Why

The codebase uses 5 distinct pattern matching approaches (match/case on Results, match/case on enums, isinstance for exception dispatch, isinstance for type guards, and isinstance+model_validate hybrids) with no documented standard. ~80+ isinstance checks exist across the codebase, many redundant with the type system, creating noise that hides the legitimate dynamic-data guards. New contributors don't know which pattern to use where, and the inconsistency makes the service layer harder to grep for error-handling flow.

## What Changes

- **Document the pattern matching taxonomy** in `.opencode/instructions/RESULT-PATTERN.md` with a decision matrix (already done).
- **Retire redundant isinstance checks** (~20-30 removals) on already-typed data (Pydantic model outputs, function signature guarantees) while preserving legitimate dynamic-data guards (Redis bytes, LangChain messages, WebSocket frames, Celery results).
- **Replace the isinstance+model_validate hybrid** in `features/ingestion/service.py` with match/case + `case dict() as raw:` for clarity.
- **Convert 2-3 isinstance checks in service layer** to match/case where they guard Result types for consistency with the dominant pattern.
- **Leave untouched:** global_exception_handler.py isinstance chain (external exception hierarchy, idiomatic), legitimate dynamic data guards, all existing match/case blocks.

## Capabilities

### New Capabilities
- `pattern-matching-standard`: Documents the 5 pattern matching approaches, declares the project standard, and provides a decision matrix for which pattern to use in each scenario.

### Modified Capabilities
<!-- None — this is a documentation + cleanup change, not a spec-level behavior change -->

## Impact

- **Files modified:** ~15-20 Python files (mostly removals of redundant isinstance checks)
- **Files created:** `.opencode/instructions/RESULT-PATTERN.md` (already written)
- **No API changes** — all changes are internal code style
- **No dependency changes** — `returns` library already installed
- **Risk:** Low — removing redundant type checks doesn't change behavior; the type system already guarantees what isinstance was checking
