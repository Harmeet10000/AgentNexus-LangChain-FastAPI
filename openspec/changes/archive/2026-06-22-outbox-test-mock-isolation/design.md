## Context

`tests/unit/test_outbox.py` has 70 lines and 2 test cases that verify `with_outbox` behavior (INSERT + pg_notify in transaction, and rollback on exception). To prevent circular imports, the test file installs a blanket mock:

```python
sys.modules["app.utils"] = AsyncMock()
```

This replaces the entire `app.utils` module — including `logger`, `NotFoundException`, `ValidationException`, `ErrorCode`, and all other symbols — with a single `AsyncMock` object. The import `from app.shared.outbox.helper import with_outbox` ultimately imports `from app.utils import logger`, which resolves to `AsyncMock().logger` (which is a default attribute on the mock, so it doesn't fail). The test runs correctly because none of the current test paths exercise the logger, but:

1. Any future test that triggers a logging path will silently pass with a mock logger instead of exercising real behavior.
2. Import-time failures inside `app.utils` are masked.
3. The pattern is fragile — if `with_outbox` ever adds an import from `app.utils` that resolves to a non-callable type (e.g., a class or constant), the mock might not behave like it.

## Goals / Non-Goals

**Goals:**
- Replace the blanket `sys.modules` mock with a targeted `unittest.mock.patch` on the specific symbol `with_outbox` uses from `app.utils`.
- Preserve all existing assertions and test logic.
- Make the test infrastructure resilient to changes in `app.utils`.

**Non-Goals:**
- No change to test coverage (2 tests, 70 lines — fine for the scope).
- No change to production code.
- No change to pytest configuration or conftest.

## Decisions

### Decision 1: Use `@patch` at the import site, not the definition site

**Choice:** `@patch("app.shared.outbox.helper.logger")` — patch the `logger` symbol at the module where `with_outbox` is defined, not where `logger` itself is defined (`app.utils`).

**Rationale:** Python's import system resolves `from app.utils import logger` at import time inside `helper.py`. Once imported, `helper.logger` is a module-level reference, disconnected from the `app.utils.logger` namespace. Patching `app.utils.logger` after import would have no effect on `helper.logger`. Patching `app.shared.outbox.helper.logger` replaces the specific reference `with_outbox` actually uses.

**Alternatives considered:**
- *Keep `sys.modules` mock but narrow it.* Same fragility — any new import from `app.utils` in `helper.py` would still resolve to a mock.
- *Refactor `with_outbox` to accept `logger` as a parameter.* Over-engineering for test convenience. The function is stable and has no other testability issues.

### Decision 2: Keep the import guard, but structured as a `@patch` decorator on each test method

**Choice:** Apply `@patch("app.shared.outbox.helper.logger")` as a class-level decorator or per-method decorator, and remove the `sys.modules` block entirely.

**Rationale:** The `@patch` decorator handles setup and teardown automatically — the mock is installed before the test runs and restored after. No need for manual `sys.modules` manipulation.

## Risks / Trade-offs

- **[None] Fully backwards-compatible.** The mock replaces the same symbol with the same behavior (an `AsyncMock`), just with narrower scope. All existing assertions pass unchanged.
- **[Low] If `with_outbox` adds imports from other `app.utils` symbols in the future, each new symbol needs its own `@patch`.** This is a feature, not a bug — explicit patching makes test dependencies visible.
