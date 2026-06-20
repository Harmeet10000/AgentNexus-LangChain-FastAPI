## Why

6 wildcard imports (`from x import *`) found across the codebase. These pollute the module namespace, make it impossible to grep for actual usage of imported names, defeat static analysis by `ruff` and `ty`, and hide dependency edges between modules.

## What Changes

- Replace all `from X import *` with explicit imports of only the names actually used
- Run `ruff check --fix F403` to auto-detect which names are used
- Add `# noqa: F403` only where wildcard import is intentionally required (if any)

## Capabilities

### New Capabilities
- `wildcard-import-cleanup`: Eliminates wildcard imports and enforces explicit imports

### Modified Capabilities

_(none)_

## Impact

- **Files**: 6 files across `src/` with wildcard imports
- **Dependencies**: None
- **APIs**: None
- **Risk**: Low — mechanical refactor; `ruff check` verifies correctness after change
