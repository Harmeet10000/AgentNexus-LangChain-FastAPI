# Verification — consolidate-worker-feature-layout (task 4.2)

Recorded 2026-10-10, working tree at task-group 4 (all migration code present,
tasks 1.1–4.1 complete). No stash/baseline capture from before the migration
exists in this session; "baseline" below means failures proven unrelated to
this change (untouched files, or environment limits).

## Gates

| Gate | Result |
|---|---|
| `uv run ruff check src/` | pass — all checks passed |
| `uv run ty check src/` | 1 diagnostic, pre-existing (see below) |
| `uv run pytest -q -p no:cacheprovider` (full, `tests/integration` not separately excluded; 44 deselected by markers) | 766 passed, 3 failed — all 3 pre-existing (see below) |
| `uv run lint-imports` | 3 contracts kept, 0 broken |
| `ast-grep scan src/` | 3 errors, all pre-existing (see below) |
| `openspec validate consolidate-worker-feature-layout` | valid |
| `codegraph sync` | done |

## Fixed during 4.2 (migration regressions)

- `src/app/worker/reliability.py:399` — `ty invalid-return-type` on
  `_resolve_redis_result` (`await value` narrows to `object`); fixed with
  `cast("T", await value)` (`cast` was already imported).
- `src/app/worker/replay_outbox.py:19` — `ty missing-type-argument`
  (`AsyncGenerator[dict, None]`); fixed to `dict[str, Any]`.

## Pre-existing failures (not this change; files untouched by the migration diff)

- `ty`: `src/app/features/auth/repository.py:317` — `redundant-cast` warning.
  File has no working-tree diff.
- `ast-grep no-feature-error-subclassing` (×2):
  `src/app/examples/logger_usage_example.py:50,55` — example-only
  `FeatureError` subclasses; file has no working-tree diff.
- `ast-grep no-raise-in-result-service` (×1):
  `src/app/features/health/service.py:86` (`raise AssertionError`); file has
  no working-tree diff.
- `pytest tests/unit/test_migration_chain.py` (2 tests) —
  `ModuleNotFoundError: No module named 'alembic.script'`; `alembic` is not
  installed in this environment, so these cannot pass here regardless of tree
  state.
- `pytest tests/unit/test_postgres_db_shape.py::
  test_get_postgres_db_rolls_back_on_exception` — asserts the source contains
  the literal `except Exception:`; the dependency uses a narrower handler.
  Test/source drift in files untouched by this change.

## Operational limitations

- The documented-command proofs that require running a worker against the
  live broker were not executed (would consume real queued work); substitute
  evidence is the `test_documented_worker_command` equality assertions plus
  broker-free registration tests, as recorded in that module's docstring.
- `docs/` (relay/baseline) comparisons from the retrieval-sql change are
  unaffected; this change adds no columns, queues, or wire names.
