# Tasks

Sixteen ordered steps. Each leaves the repository importable and the suite collecting, and each carries a **Proof**
that is a command you can run — no step is complete because someone read it and agreed.

**Three rules every Proof below obeys.**

1. **Compare against a captured baseline, never an absolute number.** Deleting roughly eleven hundred lines moves
   every lint, type and test count for reasons unrelated to correctness. Step 1 captures the baselines; later steps
   assert `<=` or `identical` against those files.

   **Corrected 2026-08-23 — this rule was written correctly and then violated three times by its own Proofs.**
   Steps 10, 14 and 16 each annotated a `pytest` invocation with `0 failed, 0 errors`, which is exactly the
   absolute this rule forbids. It is also unreachable: the suite is **red at baseline** and has been since before
   this change was drafted. Measured on a clean tree at `e9c78db`:

   ```
   3 failed, 264 passed, 1 skipped, 48 deselected, 5 warnings, 9 errors
   ```

   The twelve red items are websocket fixture drift, they are owned by no change in this refactor, and no step
   here touches them. An executor who read `0 failed` as the bar would either report false success or spend the
   step repairing tests unrelated to the work. All three annotations are corrected below to compare against
   `/tmp/c2-baseline/pytest.txt`. **The failure count must not increase and the pass count must not decrease** —
   that is the real assertion, and it is the one rule 1 asked for in the first place.
2. **Read the printed summary line for `pytest`, and know what its exit code actually means.**

   **Corrected 2026-08-23 — the instruction survives, the reason given for it was false.** An earlier draft of
   this rule said: "`--cov-fail-under=80` in `pyproject.toml:759` makes a fully green suite exit non-zero. `0
   failed` in the summary is the signal; `$?` is noise." Measured: **there is no coverage flag in `addopts` and
   no coverage threshold anywhere in the repository.** `pyproject.toml:758-763` reads `addopts =
   ["--strict-markers", "--strict-config", "-m", "not integration"]`. `[tool.coverage.run]` and
   `[tool.coverage.report]` configure coverage but set no `fail_under`, `--cov` is never passed, and there is no
   `pytest.ini`, `setup.cfg`, `tox.ini` or `.coveragerc`. `pytest-cov` is installed as a test dependency and is
   inert.

   So `$?` is **not** noise — it is a reliable signal, and it is `1` today for an honest reason: the twelve
   pre-existing websocket fixture-drift items. Of the two errors this rule contained, telling a reader to
   discard a working signal was the more dangerous.

   What survives, and why the instruction is still worth obeying: the exit code collapses to one bit, while the
   summary line carries the composition — and composition is what these Proofs assert against. The `-m "not
   integration"` in `addopts` deselects 48 tests on **every** run, so a green summary here never means
   "everything ran". Read the summary, diff it against the step-1 baseline, and treat `$?` as the coarse check
   it genuinely is.

   (Exit codes are likewise meaningful for `rg`, `alembic` and the gate's own CLI, and are used there.)
3. **No Proof renders migration history from base.** Offline `alembic upgrade head --sql` always starts at base
   (D14.3), so `alembic upgrade head --sql | grep -c 'CREATE TABLE search_'` returns `2` forever and is **forbidden**
   as a proof anywhere in this change. Where a migration's output must be asserted, the Proof either reads the revision
   file statically or renders a **scoped range** (`alembic upgrade <down>:<rev> --sql`), which is the re-scoped form
   D14.3 makes binding.

   **Corrected 2026-08-18 — the rule stands, but one reason given for it was false.** An earlier draft of this rule
   added that "a from-base render cannot even complete, because `9f4a1b7c6d2e:103` alters the phantom `clauses`
   relation." That is wrong, and it was asserted without being run. Measured: `alembic upgrade heads --sql` exits
   **0** and emits **697 lines** with a single `COMMIT;`, `clauses` ALTERs included — offline rendering emits DDL as
   *text* and never executes it, so a phantom target is not an error in a render. The real property is narrower and
   is the whole reason for the rule: an offline render is **never incremental**, because there is no database from
   which to read `alembic_version`. Keeping the false reason attached to a correct rule is a hazard — anyone who
   tests the reason, finds it wrong, and discards the rule loses a constraint that is genuinely binding.

   **Re-measured 2026-08-23 — same verdict, different number.** The render now exits **0** and emits **974 lines**
   with a single `COMMIT;` and 37 `CREATE TABLE` statements. The figure moved because change 0 added migrations
   between the two measurements; nothing about the rule changed. The line count is recorded here only as evidence
   that the render completes, so **do not treat either number as an assertion** — a Proof that pinned it would fail
   on the next migration and would be measuring the wrong thing. The single `COMMIT;` across 37 tables is the
   durable observation: an offline render emits one transaction for the whole walk regardless of how many revisions
   it spans, which is why it can never tell you what a partially-migrated database still needs.

---

## 1. Capture the baselines and the pre-change commit

Capture on a clean tree, before the first edit. The commit SHA is not bookkeeping — step 9's red-before proof scans
that tree, and step 13 diffs its route set.

- [ ] Capture test summary, collection summary, lint count, type-diagnostic count, formatter check, the mounted
      route set, and the pre-change SHA.

**Proof**

```bash
mkdir -p /tmp/c2-baseline
git rev-parse HEAD                       > /tmp/c2-baseline/sha.txt
uv run pytest -q                  2>&1 | tail -3 > /tmp/c2-baseline/tests.txt
uv run pytest --collect-only -q   2>&1 | tail -2 > /tmp/c2-baseline/collect.txt
uv run ruff check src/ tests/     2>&1 | tail -2 > /tmp/c2-baseline/ruff.txt
uv run ty check src/              2>&1 | tail -2 > /tmp/c2-baseline/ty.txt
uv run ruff format --check src/ tests/ 2>&1 | tail -2 > /tmp/c2-baseline/fmt.txt
uv run python -c "
from app.main import app
for p in sorted({r.path for r in app.routes}): print(p)" > /tmp/c2-baseline/routes.txt
wc -l /tmp/c2-baseline/*
```

Passes when every file is non-empty and `sha.txt` holds the commit the change starts from.

## 2. Create the unified feature's constants module

Carries the retrieval constants plus the two load-bearing identifier names, and drops the superseded index-name
constant. The names exist **for the gate to assert against**, not to interpolate into SQL — Decision 10 rejects
interpolation, so the literals stay literals in query text and the constants are what the gate compares them to.

- [ ] Add `src/app/features/documents/constants.py`.

**Proof**

```bash
uv run python -c "
from app.features.documents.constants import (
    RRF_K, HYBRID_CANDIDATE_LIMIT, INGEST_CHUNK_SIZE, INGEST_CHUNK_OVERLAP,
    DEFAULT_SEARCH_CACHE_TTL_SECONDS, CHUNKS_BM25_INDEX_NAME, CHUNKS_UNIQUE_CONSTRAINT_NAME)
assert RRF_K == 60, RRF_K
assert CHUNKS_BM25_INDEX_NAME == 'chunks_bm25_idx'
assert CHUNKS_UNIQUE_CONSTRAINT_NAME == 'uq_chunks_document_chunk_index'
print('constants ok')"
```

## 3. Relocate chunking, rank fusion and RAG assembly behind a re-export shim

The global conftest imports **twenty-one** symbols from the module being deleted, at module level, so relocation and
deletion cannot share a step. The shim makes both import paths resolve until step 10 removes it.

- [ ] Move `chunking.py`, `fusion.py`, `rag.py` into `src/app/features/documents/`; leave re-export shims at the old
      paths. The embedding client stays in `search/` — it is change 1's unification target.

**Proof**

```bash
uv run python -c "
from app.features.documents.chunking import chunk_text as a; from app.features.search.chunking import chunk_text as b
from app.features.documents.fusion import reciprocal_rank_fusion as c; from app.features.search.fusion import reciprocal_rank_fusion as d
from app.features.documents.rag import assemble_rag_context as e; from app.features.search.rag import assemble_rag_context as f
assert a is b and c is d and e is f, 'shim re-exports a different object'
print('both import paths resolve to the same objects')"
diff /tmp/c2-baseline/collect.txt <(uv run pytest --collect-only -q 2>&1 | tail -2) && echo "collection unchanged"
```

The identity assertion is the point: a shim that re-exports a *copy* would let the conftest and the feature diverge
silently.

## 4. Flip the unified feature's imports off the superseded feature

- [ ] Repoint `documents/dto.py`, `documents/service.py`, `documents/repository.py` at the relocated helpers and the
      new constants module.

**Proof**

```bash
rg -n 'features\.search' src/app/features/documents/
```

Passes when the only surviving matches name the embedding client — which is flagged, not fixed, and is change 1's.
Any other match is a miss.

## 5. Retarget the retrieval graph's fused search onto the unified chunk store

The hybrid node calls `legal_rrf_search`, which reads `FROM clauses` — a table no migration creates. Pass no clause
filter and no verification filter (the graph's plan object forbids extra fields), and clear the residual untyped
attribute in the same module.

- [ ] Retarget `src/app/shared/langgraph_layer/retrieval_kb/nodes.py`.

**Proof**

```bash
rg -n 'legal_rrf_search|clauses' src/app/shared/langgraph_layer/retrieval_kb/; test $? -eq 1 && echo "no clause reader left"
uv run python -c "import app.shared.langgraph_layer.retrieval_kb.nodes; print('imports')"
uv run ty check src/ 2>&1 | tail -2   # <= /tmp/c2-baseline/ty.txt
```

## 6. Retarget the one remaining phantom-index literal in the ingestion graph

**Depends on change 1 by ownership, not by ordering.** `shared/langgraph_layer/ingestion_kb/nodes.py` runs
`SELECT bm25_force_merge('clauses_bm25_idx')` — the second source *reader* of the only identifier the drift gate finds
red. This change touches **exactly one string literal** in change 1's module, because the gate at step 9 cannot go
green while that reader survives, and a gate that ships red is an expected-fail list waiting to be written. Change 1
must not revert it; the Coordination points record the handoff.

**Amended 2026-08-23 — the line moved, and the Proof cannot pass in this step's position.** Measured at `e9c78db`:

- The literal is at **`ingestion_kb/nodes.py:773`**, not `:751`.
- There are **four** occurrences of `clauses_bm25_idx` in `src/` outside frozen migration history, not one: the
  ingestion one, plus three in `src/app/features/search/repository.py:356,361,362`. Calling the ingestion module the
  "second reader" is defensible if you count *files*, and this step still owns exactly one literal — but the count
  matters for the Proof.
- **The Proof as written asserts zero occurrences across all of `src/`** (`rg …; test $? -eq 1`). Those three
  survive until **step 10** deletes `search/repository.py`, and step 6 runs before step 10. So the Proof fails at
  step 6 no matter how correctly the step is executed. It is not a stale reference; it is an assertion about a
  later step's outcome, placed in this one.

Corrected below to assert what this step owns — that the ingestion module's literal is retargeted and no
`clauses_bm25_idx` survives *in that module*. The repo-wide zero-reader assertion is what **step 11** is for, and it
runs after the deletion that makes it true.

- [ ] Change that literal to `chunks_bm25_idx`. It stays a literal — Decision 10 forbids interpolating the constant.

**Proof**

```bash
# Scoped to this step's own module — the repo-wide form belongs to step 11.
rg -n 'clauses_bm25_idx' src/app/shared/langgraph_layer/ingestion_kb/nodes.py; test $? -eq 1 && echo "module clean"
rg -n "bm25_force_merge\('chunks_bm25_idx'\)" src/app/shared/langgraph_layer/ingestion_kb/nodes.py

# Recorded, not asserted: the three survivors and the step that removes them.
rg -c 'clauses_bm25_idx' src/app/features/search/repository.py   # expect 3, deleted by step 10
```

The `--glob` exclusion in the original repo-wide form was deliberate and not a loophole, and the same reasoning
carries to step 11: `9f4a1b7c6d2e` is frozen, editing it is rejected by decision, and the gate's rule is about
*query text in source*, not about migration bodies.

## 7. Make a failed retrieval branch fail the request

Today the fused-search path logs a branch failure and appends an empty rank list, so a request whose keyword branch
raises returns `200` with a result silently fused from two modes. That is what
`document-retrieval-schema`'s three-mode requirement forbids, and it is now settled policy rather than an open
question: fail loudly is the ruling, and this change owns it. **An empty result from a healthy branch is not a
failure** and must keep degrading gracefully — that distinction is the whole content of this step.

- [ ] In `documents/service.py`, propagate a branch `Failure` as the failure of the whole retrieval call, naming the
      branch that failed. Keep the empty-success path fusing normally.

**Proof**

```bash
rg -n 'row_sets\.append\(\[\]\)' src/app/features/documents/service.py; test $? -eq 1 && echo "degrade path gone"
uv run pytest tests/unit/documents/test_hybrid_search_failure.py -q 2>&1 | tail -3
```

The test carries both halves, and neither is optional: a stub repository whose keyword branch returns `Failure` and
whose other two return `Success([])` makes the service return a `Failure` naming the branch; a stub whose three
branches all return `Success([])` makes it return a `Success` with an empty result set.

## 8. Add the static identifier gate

A pure function of paths: scan source for index and constraint names appearing inside query text, scan migration
files for the identifiers they create and the tables they create, and report any named identifier that no migration
creates, or that a migration creates on a table no migration creates. It must never open a connection — the spec
requires the failure not depend on a reachable database. Expose it as both a pytest test and a `python -m` CLI
taking a source root and a migrations root, so step 9 can point it at another tree.

- [ ] Add the gate module and `tests/unit/test_schema_identifier_gate.py`.

**Proof**

```bash
uv run pytest tests/unit/test_schema_identifier_gate.py -q 2>&1 | tail -3
rg -n 'sqlalchemy|asyncpg|psycopg|create_engine|get_session' src/app/utils/schema_identifier_gate.py; test $? -eq 1 \
  && echo "gate imports no database machinery"
```

The unit test asserts the gate on **synthetic fixtures**, not on the live tree, which is what makes it a regression
guard rather than a snapshot of today: an index named in query text with no creating migration is reported; an index
created by a migration on a table that same migration creates is not reported; and an index created on a table no
migration creates **is** reported even though the `CREATE INDEX` exists. That third case is the defect class that
produced the clause-index hole, and a gate that passes it while failing the first two is not the gate this change
promised.

## 9. Prove the gate red before the retarget and green after

Red-before is not decoration — without it the gate is a snapshot that would have passed on the broken tree too.

- [ ] Run the gate against the pre-change tree and the current tree.

**Proof**

```bash
git worktree add /tmp/c2-pre "$(cat /tmp/c2-baseline/sha.txt)"
uv run python -m app.utils.schema_identifier_gate /tmp/c2-pre/src /tmp/c2-pre/src/alembic/versions; echo "pre exit=$?"
uv run python -m app.utils.schema_identifier_gate src src/alembic/versions;                          echo "post exit=$?"
git worktree remove /tmp/c2-pre --force
```

Expected, and the counts are the assertion: `pre exit` is non-zero and names **`clauses_bm25_idx` at two reader
locations** — `features/search/repository.py` and `shared/langgraph_layer/ingestion_kb/nodes.py`. `post exit` is `0`
and names nothing.

**One identifier, not three.** Under the gate's own rule, `search_chunks_bm25_idx` and
`uq_search_chunks_document_chunk_index` are **green**: `8a7d9b1c2e3f` creates both of them on `search_chunks`, which
that same revision creates at `:45`. Their real problem is that the revision was stamped and never applied, which is
a live-database fact the spec forbids this gate from consulting and which belongs to change 0. If the gate reports
three, it has been built database-aware and is wrong. Note also that the gate does **not** need a copy of itself
inside the worktree — it takes paths, so the current tree's gate scans the old tree's source.

## 10. Delete the schema-bound twin and rewrite the test surface in the same commit

The single highest-risk moment in this change. The conftest is global, so one missing symbol is a collection error
for **every** test in the repository, not just this feature's.

- [ ] Delete the superseded models, repository, router, dependency layer, ingest service path, ingest DTOs, the
      relocated modules' shims and the superseded constants module. Move the graph-backed ask path onto the document
      query service, **left unexposed by any router** — it holds the only caller of the retrieval graph builder,
      which is change 1's foundation, so deleting it would orphan the graph. Rewrite the global conftest, the
      feature's integration test and the relocated unit tests **in this same commit**.

**Amended 2026-08-23 — three external importers are not in that list, and one of them breaks Alembic.** Measured
across the tree at `e9c78db`: 38 import sites outside the package reference modules this step deletes. Most are
covered above or belong to steps 4 and 12. Three are not covered anywhere, and they fail through **three different
gates**, which is why no single proof here catches them:

| Unnamed site | What it holds | Which gate catches its deletion |
|---|---|---|
| `src/alembic/env.py:18` and `:65` | `app.features.search.model`, imported and listed in the runtime metadata tuple | **none in this change** |
| `src/app/shared/langgraph_layer/retrieval_kb/nodes.py:28` | `SearchRepository`, under `TYPE_CHECKING` | `ty check src/` only |
| `tests/unit/documents/test_vector_width_configured.py:27` | `search.model.SearchChunk` | `pytest` collection |

**The Alembic one is the serious one.** `env.py:65` is a plain runtime tuple entry, not a conditional import, so
deleting `search/model.py` without editing `env.py` makes **every** `alembic` invocation fail with an `ImportError`
— `upgrade`, `current`, `heads`, `history`, and every migration proof in the remaining steps and in change 0's
frozen history. Nothing in step 10's proof runs `alembic`, so the step would report success and the breakage would
first surface in whatever later step happens to touch a migration. Delete both lines in the same commit.

**The `TYPE_CHECKING` one is the instructive one.** A type-only import is invisible to every runtime probe this
document favours: `pytest --collect-only`, the `importlib` loops, `uv run python -c "import …"` all pass with a
dangling `TYPE_CHECKING` import, because the block never executes. Only `ty check src/` sees it, and step 10's proof
does not run `ty` — step 16 does, at the very end. So add `ty` to this step's proof rather than discovering it six
steps later. This generalises past this change: **a proof built from imports cannot verify an import that is
declared not to happen.**

- [ ] Delete `app.features.search.model` from both `src/alembic/env.py:18` and its entry at `:65`, in this commit.
- [ ] Resolve `retrieval_kb/nodes.py:28` — the retrieval graph is change 1's and must keep type-checking.
- [ ] Repoint `tests/unit/documents/test_vector_width_configured.py:27`.
- [ ] Rewrite or delete `src/app/features/search/__init__.py`. As it stands it re-exports from `.chunking`,
      `.constants`, `.fusion`, `.model`, `.rag` and `.router` — **six of the modules this step relocates or
      deletes** — so leaving it untouched is itself the repo-wide collection error this step's own preamble warns
      about. It is also what `tests/unit/search/test_{rag,chunking,fusion}.py` import through.

**Corrected 2026-08-23 — the `ls` expectation names a module that is no longer there.** The proof expected
"`__init__.py` and the embedding client only". There is no embedding client in `src/app/features/search/`:
`embeddings.py` was deleted and its logic collapsed into `src/app/shared/langchain_layer/embeddings.py` by commit
`eb4f14d`, *"refactor(embeddings): collapse six embedding paths into one"* — change 1's own Band B, landed during
this refactor and after this step was drafted. What survives in that directory is a stale
`__pycache__/embeddings.cpython-312.pyc` with no source beside it, which is **inert**: PEP 3147 bytecode in
`__pycache__` is not importable without its `.py`, and `importlib.util.find_spec("app.features.search.embeddings")`
returns `None`. It is misleading to a reader, not a hazard to the interpreter. The corrected expectation is
`__init__.py` alone.

**Proof**

```bash
ls src/app/features/search/     # expect: __init__.py alone (see the correction above)
uv run pytest --collect-only -q 2>&1 | rg -ci 'error'   # expect 0
uv run pytest -q 2>&1 | tail -3                          # vs /tmp/c2-baseline/pytest.txt: failures must not increase
uv run ty check src/ 2>&1 | tail -2                      # the only gate that sees a dangling TYPE_CHECKING import
uv run alembic heads 2>&1 | tail -2                      # env.py still imports; catches the metadata-tuple omission
uv run python -c "
from app.shared.langgraph_layer.retrieval_kb.graph import build_retrieval_graph
from app.features.documents.service import DocumentQueryService
assert any('ask' in n for n in dir(DocumentQueryService)), 'ask path did not land'
print('graph builder still has a caller')"
```

`rg -ci 'error'` returning `0` on collection is the specific guard for this step; a green `pytest` summary alone
would not distinguish "tests pass" from "tests were never collected". `alembic heads` is the cheapest command that
forces `env.py` to import — it reads the revision files and touches no database.

## 11. Assert no superseded search table survives outside frozen migration history

- [ ] Confirm the deletion is complete at the identifier level, not just the file level.

**Proof**

```bash
rg -n 'search_documents|search_chunks' src/ tests/ --glob '!src/alembic/**'; test $? -eq 1 && echo "clean"
```

Scoped to the **search-specific** tables on purpose, matching the spec scenario's wording. `clauses`,
`parent_documents` and `statutes` still have readers in the ingestion graph and `src/database/schemas/`; those are
change 1's to retarget against the ADR, and this change deliberately touched only the one `clauses_bm25_idx` literal
at step 6. Widening this grep would make it fail for work this change does not own.

## 12. Delete the Celery ingest task, its registration and its conftest stub

The conftest stub is not incidental — left in place it would satisfy the import of a module that no longer exists and
mask the deletion.

**Amended 2026-08-23 — this step was written before change 1's Band C existed, and now names four of nine sites.**
Commit `d175dda` (change 1, C9) replaced Celery's implicit registration with a single task-name definition module,
and commit for C7 added a derived routing table on top of it. `tasks.search_ingest` is therefore no longer one
module plus one include entry; it is a name with a definition site, a module mapping, a queue-set membership, a
payload model, and two test files that consume it. The full set, measured against the tree at `e9c78db`:

| # | Site | Named by the original step? |
|---|---|---|
| 1 | `src/tasks/search_tasks.py` — the task module | yes |
| 2 | `src/tasks/__init__.py:6` — the re-export | yes |
| 3 | `src/app/connections/celery.py:273` — the `include` entry | yes |
| 4 | `tests/conftest.py:31` — the `sys.modules` stub | yes |
| 5 | `src/app/connections/celery_task_names.py:47` — `SEARCH_INGEST` definition | **no** |
| 6 | `celery_task_names.py:102` — `_SEARCH_TASKS` + its `TASK_DECLARING_MODULES` entry | **no** |
| 7 | `celery_task_names.py:62-64` — `SEARCH_INGEST` inside `INGESTION_TASK_NAMES` | **no** |
| 8 | `src/app/connections/celery_registry.py` — the registered payload model | **no** |
| 9 | `tests/integration/conftest.py:29` — a **second** stub the step's singular "the conftest stub line" hides | **no** |

Two consequences change how this step must be executed, and neither is cosmetic.

**Site 7 is an import-time hard failure.** `INGESTION_TASK_NAMES` is a `frozenset` literal that references
`SEARCH_INGEST` by name. Deleting site 5 without editing site 7 raises `NameError` while importing
`celery_task_names`, which `connections/celery.py` imports at module scope, which `app.main` reaches — so the whole
application stops importing, not just the worker. Edit site 7 in the same edit as site 5.

**The step's own Proof cannot see the largest sites.** `rg -n 'search_tasks'` misses site 5 outright, because the
constant's *value* is `tasks.search_ingest` and the string `search_tasks` does not occur in it. Worse,
`tests/unit/celery/test_task_registration.py:53` does `@pytest.mark.parametrize("task_name", [DOCUMENTS_INGEST,
SEARCH_INGEST, PAGEINDEX_INGEST])` — it imports the constant **as a symbol**, so deleting site 5 turns that file
into an `ImportError` at *collection*, and the file contains no matchable literal at all. A grep for a string
cannot find a dependency expressed as an import. Six further assertions in that file iterate
`TASK_DECLARING_MODULES` at runtime and re-scope themselves silently once site 6 is gone, which is correct
behaviour and is why they need no edit — but `:50` asserts
`set(TASK_DECLARING_MODULES.values()) == set(conf.include)` as an equality in **both** directions, so it fails on a
half-deletion that removes site 3 without site 6 or the reverse. That guard is doing its job; do not route around
it by deleting the assertion.

**No route entry needs deleting.** C7's `_task_routes()` builds the table by comprehension over
`TASK_DECLARING_MODULES`, so the route disappears with site 6. A reader who goes looking for a literal route in
`celery.py` will correctly find none.

- [ ] Delete all nine sites. Sites 5 and 7 must land in the same edit, or the application stops importing.
- [ ] Update `tests/unit/celery/test_task_registration.py:53` to drop `SEARCH_INGEST` from the parametrize list and
      from its import — the symbol import, not a string, is what breaks.
- [ ] Repoint or neutralise `tests/unit/test_outbox.py:34,48,68`, which use `"tasks.search_ingest"` as an outbox
      `event_type` fixture. The tests are about the outbox, not about this task, so a live task name or an obviously
      synthetic one both work — but a dead name left in a fixture reads as a live dispatch path to the next reader.

**Proof**

```bash
# Both spellings, because the constant's value and its module mapping share no substring.
rg -n 'search_ingest|search_tasks' src/ tests/ --glob '!src/alembic/**'; test $? -eq 1 && echo "no references"

# The symbol-level check the grep above structurally cannot perform.
uv run python -c "
import app.connections.celery_task_names as n
assert not hasattr(n, 'SEARCH_INGEST'), 'constant survives'
assert not any('search' in v for v in n.TASK_DECLARING_MODULES.values()), 'module mapping survives'
print('name definition site is clean')"

uv run python -c "
import importlib
from app.connections.celery import celery_app
for m in celery_app.conf.include: importlib.import_module(m)
print(f'all {len(celery_app.conf.include)} include targets import cleanly')"

# C9's and C7's own guards, which fail on a partial deletion in either direction.
uv run pytest tests/unit/celery/ -q 2>&1 | tail -3   # compare to /tmp/c2-baseline/pytest.txt
```

The import loop is still the one that matters most for the worker: a dangling entry in the include list kills every
worker command with an import error, and no grep catches a typo'd replacement. The `hasattr` probe is the one that
matters most for *this* deletion, because it is the only check here that inspects names rather than text.

## 13. Prove no endpoint became newly reachable

The deleted router was never mounted, so the mounted route set must be **byte-identical** before and after — a
stronger claim than "no new document paths", and cheaper to check.

- [ ] Enumerate and diff.

**Proof**

```bash
uv run python -c "
from app.main import app
for p in sorted({r.path for r in app.routes}): print(p)" > /tmp/c2-routes-after.txt
diff /tmp/c2-baseline/routes.txt /tmp/c2-routes-after.txt && echo "route set identical"
```

This proves the mounted-owner requirement's **first** scenario.

Its **second** scenario — that an unauthenticated call is refused as an authorization failure rather than an
unhandled internal error — was recorded here as **not provable and not to be claimed**, on the grounds that
"`documents/dependencies.py:61-62` reads `request.state.user_id` unguarded and no middleware assigns it, so it
raises `AttributeError` today", with the fix attributed to change 0's `UserIdDep` work (D5.2) and listed as an
explicit Non-Goal.

**Corrected 2026-08-23 — the blocker is historical and the scenario is now provable.** That rewrite already
shipped, in commit `7fc0ab5`, *"fix(features): replace request.state.user_id auth stubs with real token claims"*.
`documents/dependencies.py:62` now reads `async def get_current_user_id(claims: CurrentClaims) -> str: return
claims.sub`, and `UserIdDep` is built from it — there is no `request.state` read left to raise. So the Non-Goal
carve-out is void: this change inherits a working authorization path rather than waiting on one, and the second
scenario should be asserted rather than deferred.

- [ ] Assert the second scenario too: an unauthenticated call to a mounted document route is refused with an
      authorization status, not a 500.

Assert the **status code only**, not the response body. The body shape is changing under a separate app-wide
envelope fix (registering `APIException`, `RequestValidationError` and `StarletteHTTPException` against the
project's handler, which until now was reachable only for unhandled 500s), and a body assertion written here would
be asserting whichever shape happened to be live on the day it was written. The status code is stable across that
change; the JSON keys are not.

This is the third time in this refactor that a step's stated blocker turned out to be already fixed, and the second
time removing one **shortened** the work rather than adding to it — the same pattern change 0 step 8 hit, where a
BREAKING notice and an ordering gate both dissolved on inspection. Re-check a gate before honouring it.

## 14. Wire the static gate into the default suite and add the real-database gate behind a marker

`--strict-markers` is enabled, so an unregistered marker is a hard error rather than a warning — register
`requires_db` in `pyproject.toml`'s `markers` list in the same step. (**Corrected 2026-08-23:** the flag is passed
through `addopts` at `pyproject.toml:758-763`, not at the `:753` this step used to cite, and the `markers` list it
must be added to is at `:769-773`. The `[tool.pytest.ini_options]` table starts at `:756`. Same class of staleness
as the one rule 2 records; the instruction itself was always correct.)

- [ ] Run the static gate in the default suite. Add the real-database gate marked `requires_db`, deselected by
      default so the default suite stays offline. **Autogenerate comparison is deliberately not added**: it cannot
      distinguish a drifted model from migrations that never ran, and stays unusable until change 0 rebuilds the
      database by upgrade rather than stamp.

**Proof**

```bash
rg -n 'requires_db' pyproject.toml                      # marker registered, or --strict-markers fails
uv run pytest -q -m 'not requires_db' 2>&1 | tail -3    # vs baseline; and it opens no connection
uv run pytest -q -m 'requires_db' --collect-only 2>&1 | tail -2   # the db gate exists and is collected only here
```

**Depends on change 0** for the marked gate to *pass* — it needs a database where the two tables and four extensions
actually exist, which has never been true in any environment. Until then the marked gate is expected to be
deselected, and `design.md`'s Risks section is the standing record that this change's SQL ships unexecuted.

## 15. Align the ORM with what change 0 creates: chunk `updated_at` and the three statute attributes

Four columns, all ORM-side only — **no `ALTER` accompanies any of them**; they ship in change 0's `CREATE TABLE`.
`updated_at` is D16. The statute attributes (`instrument_name`, `section_ref`, `instrument_year`) are nullable and
write-through: nothing in this change populates them, and change 3's `legal-corpus-retrieval` is their reader.

**Amended 2026-08-23 — three columns, not four, and the one that already shipped is the broken one.** The D16 trap
this step describes at the bottom is not a hazard to avoid; it is **already present in the tree**. Measured at
`e9c78db`:

| | `UnifiedDocument.updated_at` | `UnifiedChunk.updated_at` |
|---|---|---|
| column present | yes | **yes** |
| `nullable` | `False` | `False` |
| Python-side `default` | yes | **no** |
| `server_default` | no | **yes** (`func.now()`) |
| `onupdate` | yes | yes |

So `UnifiedChunk.updated_at` exists and needs no adding. What it needs is the *other two halves*: it is **absent from
the `on_conflict_do_update` set_** at `documents/repository.py:292-304`, and `build_chunk_rows`
(`repository.py:676-679`) is `[{**chunk, "document_id": …, "user_id": …} for chunk in chunks]`, which passes the
chunk dict through and adds nothing. Chunks have exactly one write path and it is that upsert.

**`onupdate=True` is why this looks fine and is not.** SQLAlchemy's `onupdate` is applied to Core- and
ORM-generated `UPDATE` statements; `insert().on_conflict_do_update(set_={…})` carries an **explicit** SET clause and
SQLAlchemy does not merge `onupdate` defaults into it. The hook is therefore present and dead for the only path that
writes a chunk twice. `server_default=func.now()` fills the column on first insert, so every chunk has a plausible
non-null `updated_at` that equals its creation time forever. The column is populated, non-nullable, indexed by
nothing, and never changes — indistinguishable from a working one until someone diffs it against `created_at`.

**The step's own Proof would report this as done.** Its first assertion is `assert c in cols` for all four names,
which passes for `updated_at` today. Keep that assertion — it is still right for the three statute attributes — but
it must not be read as progress on `updated_at`; the upsert test below is the only check that distinguishes the two
states, and it does not exist yet (`tests/unit/documents/test_chunk_upsert_columns.py` is absent).

**One question this step should settle rather than inherit.** The two models disagree about who owns `updated_at`:
the document's is Python-side, the chunk's is server-side. This step's own `server_default` assertion encodes the
principle that "the ORM owns the application-side defaults and change 0 must not duplicate them in DDL" — but
`updated_at` is not in its `appdefaults` list, so the assertion does not reach the one column where the two models
actually diverge. Decide which side owns it, make both models agree, and add `updated_at` to the list if the answer
is the ORM.

- [ ] Add the **three statute attributes** to the model. `updated_at` is already there.
- [ ] Add all four to the upsert conflict-resolution set — including `updated_at`, which is the live defect.
- [ ] Ensure the row builder carries `updated_at`; it currently carries only what the caller's chunk dict holds.
- [ ] Reconcile `updated_at`'s default ownership across `UnifiedDocument` and `UnifiedChunk`.

**Proof**

```bash
uv run python -c "
from app.features.documents.model import UnifiedChunk, UnifiedDocument
cols = set(UnifiedChunk.__table__.c.keys())
for c in ('updated_at','instrument_name','section_ref','instrument_year'):
    assert c in cols, f'missing {c}'
appdefaults = {'chunk_kind','document_kind','status','parties','metadata_','custom_metadata',
               'quality_warnings','page_no','graphiti_verified','preamble'}
bad = [c.name for t in (UnifiedDocument, UnifiedChunk) for c in t.__table__.c
       if c.name in appdefaults and c.server_default is not None]
assert not bad, f'server_default added where the ORM is authoritative: {bad}'
print('columns present; no server_default drift')"
uv run pytest tests/unit/documents/test_chunk_upsert_columns.py -q 2>&1 | tail -3
```

The test compiles the upsert against the PostgreSQL dialect and asserts all four names appear **after**
`DO UPDATE SET` in the rendered SQL, and that `build_chunk_rows` output carries `updated_at`. Both halves are
required and this is the D16 trap stated as a check: the ORM's update hook does not fire for a conflict-resolving
insert, which is the only way chunks are written, so a column present in the payload but absent from the conflict set
exists, is non-nullable, and never changes — it looks maintained and is not. The `server_default` assertion enforces
the recorded division of authority: the ORM owns the application-side defaults and change 0 must not duplicate them
in DDL.

## 16. Final verification, and the change-0-dependent migration assertions

- [ ] Run the full check set against the step 1 baselines, then the openspec validators.

**Proof**

```bash
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/ 2>&1 | tail -2     # <= /tmp/c2-baseline/ruff.txt
uv run ty check src/              2>&1 | tail -2       # <= /tmp/c2-baseline/ty.txt
uv run pytest -q                  2>&1 | tail -3       # vs /tmp/c2-baseline/pytest.txt: failures must not increase
openspec validate documents-unified-schema --type change --strict
openspec validate --all 2>&1 | tail -1                 # must stay 21 passed / 6 failed — never a 7th failure
```

**Depends on change 0.** The two migration-facing assertions cannot run until the authoritative create-schema
revision exists. Substitute `<rev>` and `<down>` when it does:

```bash
rg -n -A1 'op\.create_table\(' src/alembic/versions/<rev>_*.py       # exactly "documents" and "chunks"
rg -n 'search_documents|search_chunks|"clauses"' src/alembic/versions/<rev>_*.py; test $? -eq 1
rg -c 'CREATE EXTENSION IF NOT EXISTS (vector|vectorscale|pg_textsearch|pg_trgm)' src/alembic/versions/<rev>_*.py  # 4
rg -n 'chunks_bm25_idx|chunks_embedding_idx|chunks_search_text_trgm_idx|ix_chunks_instrument_section' \
   src/alembic/versions/<rev>_*.py
uv run alembic upgrade <down>:<rev> --sql | rg -c 'CREATE TABLE'     # scoped range: renders only this revision
```

The last command uses an explicit `<down>:<rev>` range for a reason worth restating, because it is the single
easiest thing to get wrong in this change: **`alembic upgrade head --sql` renders from base**, so it emits
`8a7d9b1c2e3f`'s superseded `CREATE TABLE`s no matter what this change does, and it cannot complete at all because
`9f4a1b7c6d2e:103` alters a relation nothing creates. The from-base form is not a stricter proof; it is a proof of a
different, false claim. The assertion this change makes — and the only one it can make — is about the authoritative
revision's **own** rendering.
