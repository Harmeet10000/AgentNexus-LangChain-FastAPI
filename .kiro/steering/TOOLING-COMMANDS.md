# Required Commands

| Command | Description |
|---|---|
| `uv sync` | Install/update dependencies |
| `uv run ruff format src/` | Format all source code |
| `uv run ruff check src/` | Lint check (no auto-fix) |
| `uv run ruff check --fix src/` | Auto-fix safe lint issues |
| `uv run ty check src/` | Type check |

## Baseline Expectations

- `ruff` is source of truth for formatting and linting.
- `ty` is source of truth for static typing.
- Use `uv run` for ALL tooling; never bare `ruff` or `ty`.
- `pyproject.toml` is authoritative for enabled rules (see `[tool.ruff.lint]` and `[tool.ty.rules]` sections).
- Before PR/merge: run BOTH `uv run ruff check src/` and `uv run ty check src/`.
- Prefer patterns that satisfy async, security, import-order, and typing rules without needing `# noqa` or `# type: ignore`.
- Do not weaken configured checks in examples, generated commands, CI snippets, or review advice.

## Ruff Quick Reference

```text
I, F401, UP, C4, SIM, PTH, RUF, TC  → safe autofix (run check --fix)
B, ANN, S                             → requires review, not autofix
FAST                                  → FastAPI antipatterns
N                                     → PEP8 naming (mandatory)
ASYNC                                 → blocking calls in async functions
LOG                                   → logging/loguru patterns
```

Respect configured ignores in `pyproject.toml` (e.g., `E501`, `ANN401`, `ISC001`, `TRY003`).

## Ty Quick Reference

**Blocking errors:** `unresolved-import`, `possibly-missing-attribute`, `possibly-missing-import`, `invalid-assignment`, `unresolved-reference`, `await-on-non-awaitable`, `non-awaitable-in-async-function`, `possibly-unbound-variable`.

**Key principles:**
- Pydantic v2 dynamic attributes → `unresolved-attribute` is warn, not error.
- Strict async correctness for `asyncpg`, `motor`, Redis, etc.
- No dataclass/protocol misuse, no parameter default mismatches, no context manager violations.
