## Why

`tests/unit/test_outbox.py` sets up its mock by replacing the entire `app.utils` module in `sys.modules` with an `AsyncMock()` before any outbox imports:

```python
sys.modules["app.utils"] = AsyncMock()
```

This is a wide-spectrum suppression that replaces every symbol from `app.utils` (including `logger`, `NotFoundException`, `ValidationException`, and all other utilities) with a single mock object. The test only needs to isolate `with_outbox` from the `logger` import, but this approach nukes the entire module. Any test case that inadvertently triggers logging or an exception path will silently pass on mock objects instead of exercising real types. It also masks import-time failures in `app.utils` itself. The pattern is fragile and makes the test harder to debug when it breaks for an unrelated reason.

## What Changes

- **Replace `sys.modules["app.utils"] = AsyncMock()` with `unittest.mock.patch("app.shared.outbox.helper.logger")`.** This targets only the specific symbol `with_outbox` imports from `app.utils` (the `logger`) and leaves the rest of the module intact.
- **No spec-level capability changes** — pure test infrastructure cleanup.

## Capabilities

### New Capabilities
*(none — test infrastructure cleanup only)*

### Modified Capabilities
*(none — no spec-level behavior changes)*

## Impact

- **File:** `tests/unit/test_outbox.py` — replace mock setup; verify all existing assertions pass unchanged.
- **No changes to production code.** No new dependencies.
