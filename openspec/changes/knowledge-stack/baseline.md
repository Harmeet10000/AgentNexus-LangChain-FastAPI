# Baseline — knowledge-stack (measured 2026-09-14)

## Tooling and schema

- `uv lock --check`: exit 0.
- `uv run python -c "import app.main"`: exit 0.
- Live migration head before this change: `0019`; knowledge-stack storage revision: `0020`.
- `uv run alembic check` at revision `0020`: `No new upgrade operations detected.`

## Structural persistence measurement

- `rg -n 'DoclingDocument' src/app/features/documents/` found the parser-local Docling object, but
  no write into `UnifiedDocument`.
- The pre-change live document sample had `metadata_ = {}` and no structural-tree column.
- Answer: **no**, the parser's structural tree was not persisted. Task group 2 was therefore
  mandatory. Revision `0020` adds the owned JSONB column and the live ingestion acceptance test
  proves a non-empty tree survives the database round trip.

## Stage boundary

The extraction stage precedes `segment_chunks(...)` inside
`features/documents/service.py::process_document_ingestion`. This is a stage edge: parsing produces
one `ParsedDocument`, extraction consumes its markdown, and chunking consumes that same object only
after the typed extraction outcome has returned.

## Gate numbers (post-hoc: 1.1 recorded tooling but no counts; first counted values, 2026-09-14)

- `uv run pytest -q`: `697 passed, 43 deselected`, **0 failures**
- `uv run ruff check --no-cache src/`: `All checks passed!`
- `uv run ty check src/`: `Found 4 diagnostics` (all in `src/alembic/env.py`, outside this change)
