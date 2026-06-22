## 1. Replace mock setup

- [x] 1.1 Keep `import sys` — still needed for `sys.modules` bypass (circular import in `app.utils`)
- [x] 1.2 Keep `import asyncio` — still used by `asyncio.run()`
- [x] 1.3 Replace blanket `sys.modules["app.utils"] = AsyncMock()` with a minimal proxy `ModuleType` that only exposes `logger`
- [x] 1.4 Add `import types` for `types.ModuleType`

## 2. Apply minimal proxy module

- [x] 2.1 Create `_app_utils = types.ModuleType("app.utils")` and set `_app_utils.logger = AsyncMock()`
- [x] 2.2 Assign `sys.modules["app.utils"] = _app_utils` before importing `with_outbox`
- [x] 2.3 Comment explains the circular import bypass
- [x] 2.4 No `@patch` decorator needed — the minimal proxy is sufficient (the `@patch` approach was rejected because the circular import is in `app.utils` itself, not in `helper.py`)

## 3. Verify and run

- [x] 3.1 Run `uv run python tests/unit/test_outbox.py` — verify both tests pass (pytest conftest has pre-existing circular import, run directly)
- [x] 3.2 Run `uv run ruff check tests/unit/test_outbox.py` — ensure no lint regressions (only pre-existing PT015/B011/PT017)
- [x] 3.3 Run `uv run ruff format tests/unit/test_outbox.py` — ensure formatting is clean

## 4. Clean up

- [x] 4.1 Update the comment from `# Mock app.utils before any app import to avoid circular import` to explain the minimal proxy pattern
- [x] 4.2 Remove the `patch` import (not used)
