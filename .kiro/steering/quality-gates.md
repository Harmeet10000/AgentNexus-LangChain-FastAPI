---
inclusion: always
---

# Quality Gates & Tooling

## Required Commands

Before commit and PR:

```bash
uv sync
uv run ruff format src/
uv run ruff check src/
uv run ruff check --fix src/
uv run ty check src/
```

## Ruff Rules

Active rule families: `E`, `W`, `F`, `I`, `UP`, `B`, `A`, `C4`, `PERF`, `TRY`, `ASYNC`, `RUF`, `PL`, `ANN`, `S`, `SIM`, `PTH`, `TCH`, `RET`, `ARG`

### Safe Autofix
- `I` (import sorting)
- `F401` (unused imports)
- `UP` (syntax upgrades)
- `C4` (comprehensions)
- `SIM` (simplifications)
- `PTH` (pathlib)
- `RUF` (Ruff-specific)
- Type-checking import cleanup

### Require Review
- `B` (bugbear)
- `ANN` (annotations)
- `S` (security)

### Configured Ignores
`E501`, `ANN401`, `ISC001`, `TRY003`, `PLR0913`, `PLR2004`, `PLR0911`, `ANN001`, `ANN002`, `ANN003`, `ANN204`

### Special Cases
- `fastapi.Depends`, `fastapi.Query`, `pydantic.Field` are allowed immutable-style calls
- Exempt modules: `pydantic`, `fastapi`, `langchain`, `langgraph`, `docling`

## Ty (Type Checker) Rules

### Blocking Errors
- `unresolved-import`
- `possibly-missing-attribute`
- `possibly-missing-import`
- `invalid-assignment`
- `unresolved-reference`
- `await-on-non-awaitable`
- `non-awaitable-in-async-function`
- `possibly-unbound-variable`

### Warnings (Clean Up When Practical)
- `unresolved-attribute`
- `redundant-cast`
- `unused-ignore-comment`

### Special Handling
- Account for Pydantic v2 dynamic attributes
- Be strict about async correctness: `asyncpg`, `motor`, Redis, database clients
- All I/O must be awaitable

## Never Weaken Checks

- Don't invent alternative lint/type baselines
- Treat `pyproject.toml` rules as authoritative
- Don't weaken checks in examples, commands, CI, or advice unless explicitly requested
- Prefer patterns that satisfy active rules without needing ignores

## Use `uv` for Tooling

Always use `uv run ...` instead of bare commands:
- `uv run ruff format src/` not `ruff format src/`
- `uv run ty check src/` not `ty check src/`
