# Baseline — ingestion-chunking (task 1.x, measured 2026-09-13)

## 1.1 Tooling

| Command | Result |
|---|---|
| `uv lock --check` | exit `0` (Resolved 593 packages) |
| `uv run python -c "import app.main"` | exit `0` |
| `uv run ruff check --no-cache src/ 2>&1 \| tail -1` | `No fixes available (6 hidden fixes can be enabled with the \`--unsafe-fixes\` option).` — preceding line: `Found 9 errors.` (pre-existing WIP; `docs/relay/baseline-ruff-after.txt` still reads `All checks passed!`) |
| `uv run pytest -q 2>&1 \| tail -1` | `1 failed, 626 passed, 39 deselected, 16 warnings in 55.10s` — the single failure is `test_throwaway_graph_resilience.py::test_permanent_failure_is_not_retried`, matching `docs/relay/baseline-pytest.txt` |
| `uv run ty check src/` | `Found 10 diagnostics` (matches prior cluster baseline of 10) |

## 1.2 Migration head

Live `uv run alembic current` output (final revision line): **`0016`**.

That revision literal is present at `src/alembic/versions/0016_add_statute_identity_index.py`.

**Code head ahead of the live stamp:** `0017` and `0018` exist in `src/alembic/versions/` (`0018_retrieval_kind_partial_indexes.py` from `retrieval-sql`) but are **not** applied on the live instance (`alembic_version` = `0016`). Task 2.2's new migration therefore chains from **`0018`** (the tree's tip), after applying/stamping through the already-authored revisions — not from a guessed head, and not by inventing a parallel branch off `0016`.

## 1.3 Chunks population

Live `SELECT count(*) FROM chunks` → **`1`** (non-zero).

Consequence: task 5.2's generated-column / content-preamble rewrite is a **backfill** situation for the existing row, not a free empty-table rewrite.

## 1.4 Tier-1 retrieval baseline

From `evals/reports/baseline.json` (`rag-eval-harness` 4.1 / re-confirmed by `agentic-retrieval` 5.3):

- Golden-set version: `legal_retrieval_v1`
- Commit identifier: `0653503845c2`
- Aggregates: recall_at_k 0.0, reciprocal_rank 0.0, ndcg_at_k 0.0, precision_at_k 0.0
