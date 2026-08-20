# Tasks

Sixteen ordered steps. Each leaves the repository importable and the suite collecting, and each carries a **Proof**
that is a command you can run — no step is complete because someone read it and agreed.

**Three rules every Proof below obeys.**

1. **Compare against a captured baseline, never an absolute number.** Deleting roughly eleven hundred lines moves
   every lint, type and test count for reasons unrelated to correctness. Step 1 captures the baselines; later steps
   assert `<=` or `identical` against those files.
2. **Read the printed summary line, never the process exit code, for `pytest`.** `--cov-fail-under=80` in
   `pyproject.toml:759` makes a fully green suite exit non-zero. `0 failed` in the summary is the signal; `$?` is
   noise. (Exit codes *are* meaningful for `rg`, `alembic` and the gate's own CLI, and are used there.)
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

**Depends on change 1 by ownership, not by ordering.** `shared/langgraph_layer/ingestion_kb/nodes.py:751` runs
`SELECT bm25_force_merge('clauses_bm25_idx')` — the second source reader of the only identifier the drift gate finds
red. This change touches **exactly one string literal** in change 1's module, because the gate at step 9 cannot go
green while that reader survives, and a gate that ships red is an expected-fail list waiting to be written. Change 1
must not revert it; the Coordination points record the handoff.

- [ ] Change that literal to `chunks_bm25_idx`. It stays a literal — Decision 10 forbids interpolating the constant.

**Proof**

```bash
rg -n 'clauses_bm25_idx' src/ --glob '!src/alembic/**'; test $? -eq 1 && echo "zero readers outside frozen history"
rg -n "bm25_force_merge\('chunks_bm25_idx'\)" src/app/shared/langgraph_layer/ingestion_kb/nodes.py
```

The `--glob` exclusion is deliberate and not a loophole: `9f4a1b7c6d2e` is frozen, editing it is rejected by
decision, and the gate's rule is about *query text in source*, not about migration bodies.

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

**Proof**

```bash
ls src/app/features/search/     # expect: __init__.py and the embedding client only
uv run pytest --collect-only -q 2>&1 | rg -ci 'error'   # expect 0
uv run pytest -q 2>&1 | tail -3                          # printed summary: 0 failed, 0 errors
uv run python -c "
from app.shared.langgraph_layer.retrieval_kb.graph import build_retrieval_graph
from app.features.documents.service import DocumentQueryService
assert any('ask' in n for n in dir(DocumentQueryService)), 'ask path did not land'
print('graph builder still has a caller')"
```

`rg -ci 'error'` returning `0` on collection is the specific guard for this step; a green `pytest` summary alone
would not distinguish "tests pass" from "tests were never collected".

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

- [ ] Delete the task module, its re-export, its entry in the worker's include list, and the conftest stub line.

**Proof**

```bash
rg -n 'search_tasks' src/ tests/; test $? -eq 1 && echo "no references"
uv run python -c "
import importlib
from app.connections.celery import celery_app
for m in celery_app.conf.include: importlib.import_module(m)
print(f'all {len(celery_app.conf.include)} include targets import cleanly')"
```

The second command is the one that matters: a dangling entry in the include list kills every migration and worker
command with an import error, and no grep catches a typo'd replacement.

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

This proves the mounted-owner requirement's **first** scenario. Its **second** scenario — that an unauthenticated
call is refused as an authorization failure rather than an unhandled internal error — is **not** provable here and
must not be claimed: `documents/dependencies.py:61-62` reads `request.state.user_id` unguarded and no middleware
assigns it, so it raises `AttributeError` today. **That fix is change 0's `UserIdDep` work (D5.2)** and is an
explicit Non-Goal of this change.

## 14. Wire the static gate into the default suite and add the real-database gate behind a marker

`--strict-markers` is enabled (`pyproject.toml:753`), so an unregistered marker is a hard error rather than a
warning — register `requires_db` in `pyproject.toml`'s `markers` list in the same step.

- [ ] Run the static gate in the default suite. Add the real-database gate marked `requires_db`, deselected by
      default so the default suite stays offline. **Autogenerate comparison is deliberately not added**: it cannot
      distinguish a drifted model from migrations that never ran, and stays unusable until change 0 rebuilds the
      database by upgrade rather than stamp.

**Proof**

```bash
rg -n 'requires_db' pyproject.toml                      # marker registered, or --strict-markers fails
uv run pytest -q -m 'not requires_db' 2>&1 | tail -3    # default suite: 0 failed, and it opens no connection
uv run pytest -q -m 'requires_db' --collect-only 2>&1 | tail -2   # the db gate exists and is collected only here
```

**Depends on change 0** for the marked gate to *pass* — it needs a database where the two tables and four extensions
actually exist, which has never been true in any environment. Until then the marked gate is expected to be
deselected, and `design.md`'s Risks section is the standing record that this change's SQL ships unexecuted.

## 15. Align the ORM with what change 0 creates: chunk `updated_at` and the three statute attributes

Four columns, all ORM-side only — **no `ALTER` accompanies any of them**; they ship in change 0's `CREATE TABLE`.
`updated_at` is D16. The statute attributes (`instrument_name`, `section_ref`, `instrument_year`) are nullable and
write-through: nothing in this change populates them, and change 3's `legal-corpus-retrieval` is their reader.

- [ ] Add all four to the model, the upsert conflict-resolution set and the row builder.

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
uv run pytest -q                  2>&1 | tail -3       # printed summary: 0 failed, 0 errors
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
