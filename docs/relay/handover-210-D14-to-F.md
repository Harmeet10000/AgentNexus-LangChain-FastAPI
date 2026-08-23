# Handover — todo 210, bands D(14–16), E, F

Written 2026-08-23. Branch `refactor/todo-210-sequence`, tip `1ce52ec`, **pushed**.
Everything below `## Instructions` is what remains. Everything above it is ground truth, measured.

---

## 1. Git state

**No PR exists for this branch.** `gh pr list --state all` returns one row only: `#28
refactor/phase-0-green` — MERGED on 2026-08-20. A PR for `refactor/todo-210-sequence` has to be
created when all bands are done. The user's instruction, verbatim: *"first do a conventional commit
of every task done first and push them to the branch and after everything is done merge every PR
into main"* — so **do not merge to main until D, E and F are all complete.**

### Commits from earlier bands (already on the branch)

| SHA | Subject | Band |
|---|---|---|
| `319c698` | fix(utils): import from leaf modules to sever app.utils cycle | pre-work |
| `6525c6f` | fix(tools): repoint shadow-package importers at survivor modules | **A complete** |
| `79a1d95` | fix(foundation): repair stamped schema, consolidate db access, delete dead trees | **B** |
| `383d6ec` | (ingestion-pipeline-unification, 24/24) | **C complete** |

### Commits from the most recent session — pushed as `ad6d34d..1ce52ec`

| SHA | Subject | Covers |
|---|---|---|
| `974c740` | `test(outbox): remove the app.utils stub that aborted collection` | the collection-abort fix |
| `f35cf4b` | `fix(result): give the execution_path ContextVar read a default` | `shared/result/logging.py` |
| `fdbad83` | `test(schema): add a static gate against schema identifier drift` | D step 8 |
| `db46a6b` | `refactor(documents): retire the search feature onto the unified schema` | D steps 3, 10, 11, 12 |
| `7a9a1ed` | `docs(openspec): record measured amendments for documents-unified-schema 1-13` | tasks.md |
| `1ce52ec` | `docs(todo): add items 225-228` | user's own todo notes |

`db46a6b` is deliberately one large commit. `src/tasks/search_tasks.py` imports
`search/service.py`, which imports `search/{rag,fusion,chunking}.py`, so **any** split leaves an
intermediate commit where `src/tasks/__init__.py` will not import — breaking alembic, Celery
discovery and pytest collection at once. Git still records the three relocations as `R` renames.

---

## 2. Measured gate state at `1ce52ec` — these are your baselines

| Gate | Value |
|---|---|
| `uv run pytest -q` | **317 passed, 36 deselected, 5 warnings** — 0 failed, 0 errors |
| `uv run ruff check --no-cache src/` | All checks passed |
| `uv run ruff check --no-cache tests/` | **262 errors**, 26 fixable — pre-existing, see §5 |
| `uv run ruff format --check src/ tests/` | **6 files would reformat** — see §5 |
| `uv run ty check src/` | All checks passed |
| `uv run alembic heads` | `b3e7c41d92af (head)` |
| `openspec validate documents-unified-schema --strict` | valid |
| `openspec validate --all` | **21 passed, 6 failed (27 items)** — never a 7th failure |
| mounted routes | **87**, identical to the pre-change baseline |

Working tree is **clean**.

**`ruff` cache lies after concurrent edits.** `ruff check src/` once reported 2 errors in a file
that had none; `--no-cache` reported clean. Always pass `--no-cache` when a count surprises you.

**The 6 format-drift files**, so you know which are yours: `src/app/config/settings.py`,
`tests/conftest.py`, `tests/integration/conftest.py`, `tests/integration/test_auth.py`,
`tests/property/test_credit_properties.py`,
`tests/unit/shared/langchain_layer/test_embeddings_unified.py`. Of these, band D touched only the
two conftests. The other four belong to other work.

---

## 3. Band status

| Band | Change | Status |
|---|---|---|
| A | `agent-tools-unification` — importer rewrite only | **DONE** `6525c6f` |
| B | `cleanup-foundation` — all 12 groups | **DONE**, migration head `b3e7c41d92af` |
| C | `ingestion-pipeline-unification` — 24/24 | **DONE** `383d6ec` |
| D | `documents-unified-schema` — 16 steps | **steps 1–13 done**; 14, 15, 16 remain |
| E | `agent-tools-unification` remainder (2.1–2.4 + extras) | not started |
| F | `cognee-agent-memory` | not started |

`cognee-saul-memory-migration` is **NOT** part of todo 210. Do not touch it.

**F must follow D** — both edit `src/alembic/env.py`.

In `openspec/changes/documents-unified-schema/tasks.md` the only unchecked boxes left are at lines
816 (step 14), 875–878 (step 15) and 907 (step 16). All 17 boxes for steps 1–13 are `[x]` and each
step carries an `**Amended 2026-08-23 — …**` block recording what measuring it actually returned.

---

## 4. Partially done — read this before starting anything

### D13 second scenario — proven in half, and the unproven half is a finding

`/tmp/probe_unauth.py` exists and runs. Result:

| Route | Unauthenticated | Why |
|---|---|---|
| `POST /api/v1/documents/upload` | **401** ✅ | `DocumentCommandServiceDep` |
| `GET /api/v1/documents/{id}/status` | **401** ✅ | `DocumentCommandServiceDep` |
| `POST /api/v1/search`, `/search/rag`, `/search/ask`, `/legal/ask` | **500** | `DocumentQueryServiceDep` |

`get_document_query_service` resolves `get_redis` **and** calls `_get_document_llm()`;
`get_document_command_service` does neither. With no local Redis the first raises. With Redis
overridden by `fakeredis` the second raises `ImportError: Initializing ChatVertexAI requires the
langchain-google-vertexai package`. Both are **environment absences**, not authorization defects,
and this is recorded in tasks.md as *not provable in this environment* rather than claimed.

**The live finding, not yet fixed:** FastAPI resolves a path operation's dependencies as a set and
surfaces whichever raises first — a sibling's failure is not deferred until the auth dependency has
had its say. So `get_document_query_service` **constructs the model-provider client while the caller
is still unauthenticated**, and when that raises, the 500 masks the 401 the request had already
earned. That is an ordering defect independent of installed packages. The fix is a lazily
constructed LLM in `documents/dependencies.py`. **Out of scope for a proof step — decide whether it
belongs in E or in a follow-up, but do not silently drop it.**

### D14 — measured, nothing written

Nothing has been edited for step 14. What is already known:

* `pyproject.toml` `[tool.pytest.ini_options]` starts at **line 756**. `addopts` is at **758–763**
  and is `["--strict-markers", "--strict-config", "-m", "not integration"]`. The `markers` list is
  at **769–773** and holds `slow`, `integration`, `unit`. **`requires_db` is not registered.**
* `--strict-markers` is on, so an unregistered marker is a hard error, not a warning.
* `tests/unit/test_schema_identifier_gate.py` is 164 lines and holds 9 tests, **all of them over
  synthetic `tmp_path` trees**. Not one audits the real repository. So the gate module exists, is
  tested, and **does not yet guard this codebase** — which is exactly what step 14 is for.
* `src/app/utils/schema_identifier_gate.py` exposes `audit(source_root, migrations_root) ->
  list[Finding]` and a `python -m` entry point. Standard library only, opens no connection.

---

## 5. Instructions

### D14 — wire the static gate into the default suite, add the DB gate behind a marker

1. Register the marker in `pyproject.toml`'s `markers` list (~769–773):
   `"requires_db: marks tests that need a live database (deselect with -m 'not requires_db')"`.

2. **Trap — this is the one thing to get right.** `pytest -m 'not requires_db'` does **not** AND with
   `addopts`; a CLI `-m` **replaces** the `-m` already in `addopts`, so that command silently drops
   `not integration` and starts collecting integration tests. The proof in tasks.md step 14 is
   written in exactly that broken form. Deselect by default by editing `addopts` itself to
   `"-m", "not integration and not requires_db"`, then verify with a plain `uv run pytest -q` —
   the count must stay **317 passed** and `deselected` must rise by however many tests you mark.

3. Add a default-suite test that runs the gate over the **real** tree:
   `audit(Path("src"), Path("src/alembic/versions"))` must return zero findings. Put it in
   `tests/unit/test_schema_identifier_gate.py`. Resolve the paths from `__file__`, not from the
   process CWD — pytest can be invoked from anywhere and a gate that silently scans an empty
   directory reports clean.

4. Add the `requires_db` gate: read `information_schema.tables`, `information_schema.columns` and
   `pg_indexes` and assert the identifiers the ORM declares actually exist. **Read-only only.**
   **Print host/port/database and nothing else** — never a credential, never a connection string.
   It is expected to be deselected and is not expected to pass in any current environment; that is
   recorded in `design.md`'s Risks section.

5. Do **not** add an autogenerate comparison. It cannot distinguish a drifted model from migrations
   that never ran, and stays unusable until the database is rebuilt by upgrade rather than stamp.

### D15 — align the ORM: chunk `updated_at` and the three statute attributes

**Verify this first, before writing a line.** Step 15 adds three columns to the ORM with **no
`ALTER`**, on the stated basis that they ship in change 0's `CREATE TABLE`. Change 0 is *done*.
So go read `src/alembic/versions/a71f0d7d9c12_add_unified_documents_and_chunks.py` and
`b3e7c41d92af_create_the_five_phantom_relations.py` and confirm they create `instrument_name`,
`section_ref` and `instrument_year` on `chunks`. **If they do not, adding them to the ORM creates
ORM/DB divergence, and the static gate will not catch it — it checks index and constraint names,
not columns.** In that case the honest move is to add the columns *and* a migration, or to stop and
report. Eight times in this refactor a plan asserted something false about the ground; this is the
ninth candidate.

Then, per the step's own amendment (which is accurate — it was measured):

* `UnifiedChunk.updated_at` **already exists**. It is `nullable=False`, has
  `server_default=func.now()`, has `onupdate`, and has **no** Python-side default.
  `UnifiedDocument.updated_at` is the mirror image: Python-side default, no `server_default`.
* **`onupdate` is why this looks fine and is not.** SQLAlchemy applies `onupdate` to generated
  `UPDATE` statements. `insert().on_conflict_do_update(set_={…})` carries an **explicit** SET clause
  and SQLAlchemy does not merge `onupdate` defaults into it. Chunks have exactly one write path and
  it is that upsert — so the hook is present and dead, `server_default` fills the column on first
  insert, and every chunk's `updated_at` equals its creation time forever. Non-nullable, populated,
  never changing, indistinguishable from a working column until someone diffs it against
  `created_at`.
* Add the **three statute attributes** to the model (nullable, write-through; nothing in this change
  populates them, change 3's `legal-corpus-retrieval` is their reader).
* Add all four names to the conflict-resolution `set_` at `documents/repository.py:292-304`.
* `build_chunk_rows` at `repository.py:676-679` is
  `[{**chunk, "document_id": …, "user_id": …} for chunk in chunks]` — it passes the caller's dict
  through and adds nothing. Make it carry `updated_at`.
* **Settle the ownership question rather than inheriting it.** Decide whether `updated_at` is
  ORM-owned or server-owned, make both models agree, and if the answer is the ORM, add `updated_at`
  to the `appdefaults` set in step 15's Proof — as written, that assertion does not reach the one
  column where the two models actually diverge.
* Write `tests/unit/documents/test_chunk_upsert_columns.py` — it does not exist. Compile the upsert
  against the PostgreSQL dialect and assert all four names appear **after** `DO UPDATE SET` in the
  rendered SQL, and that `build_chunk_rows` output carries `updated_at`. Both halves are required.
* Step 15's first Proof assertion (`assert c in cols` for all four) **passes today** for
  `updated_at`. Keep it — it is right for the statute attributes — but do not read it as progress on
  `updated_at`. The upsert test is the only check that distinguishes the two states.

### D16 — final verification, audited

**Step 16 was never audited and its Proof has two defects.** Fix them rather than running it as
written.

1. `uv run ruff format src/ tests/` and `ruff check --fix src/ tests/` reach **files this change does
   not own** — 262 lint errors and 6 format-drift files, of which band D touched only
   `tests/conftest.py` and `tests/integration/conftest.py`. Scope both commands to the owned surface:
   `src/`, `tests/conftest.py`, `tests/integration/conftest.py`, `tests/unit/documents/`,
   `tests/unit/test_schema_identifier_gate.py`, `tests/unit/test_outbox.py`,
   `tests/unit/celery/test_task_registration.py`. Note the project's documented gate is
   `ruff check src/`, which is already clean; `tests/` has never been in it.

2. The **22 items in owned files**, already enumerated so you don't have to re-measure:

   | File | Items |
   |---|---|
   | `tests/conftest.py` | 8 unused imports — `AsyncGenerator` (:2), `fakeredis.aioredis` (:6), `UserResponse` (:70), `SessionData` (:73), `ConflictException`/`NotFoundException`/`ServiceUnavailableException`/`UnauthorizedException` (:78–81) |
   | `tests/conftest.py` | `unnecessary-assign` :88, :100 · `dict-get-with-none-default` :113 |
   | `tests/conftest.py` | 4× `E402` and 1× `INP001` — **structural and intentional, leave them**: the `sys.modules` stubs must precede the app imports, and adding `tests/__init__.py` would change collection |
   | `tests/unit/documents/test_vector_width_configured.py` | `yoda-conditions` :75 |
   | `tests/unit/test_outbox.py` | 2× `no-self-use` (:22, :61) · `assert-false` + `PT015` + `PT017` at :75, :77 — convert the `try/assert False/except` to `pytest.raises` |

   The 8 dead conftest imports are the **same defect class** documented in that file's own comment
   block: the 21 `app.features.search.*` imports removed in `db46a6b` were dead too. Removing these
   makes the comment true of the whole file instead of only its history. Low risk — importing an
   exception class has no side effect — but run the suite after, because `app.features.auth.dto` and
   `app.features.auth.service` are Beanie-adjacent.

3. `openspec validate --all` must report **21 passed, 6 failed**. A 7th failure is a regression you
   introduced. `openspec validate documents-unified-schema --type change --strict` must pass.

4. The two migration-facing assertions in step 16 are no longer blocked — change 0 shipped.
   Substitute `<rev>` = `a71f0d7d9c12` and `<down>` = its `down_revision`.
   **Never run `alembic upgrade head --sql`.** It renders from base, so it emits the superseded
   `CREATE TABLE`s regardless of this change, and it cannot complete at all because
   `9f4a1b7c6d2e:103` alters a relation nothing creates. Use the explicit `<down>:<rev>` range.

5. Baseline files under `/tmp/c2-baseline/` may be gone. If so, recapture with `/tmp/enum_routes.py`
   for routes (see §7) and take the numbers in §2 as the baseline.

### E — `agent-tools-unification` remainder

* Checkboxes **2.1–2.4**. 2.3's expectation needs amending (see the change's own notes).
* Relocate `src/app/shared/rag/rag_agent_advanced.py` → `src/app/examples/` per decision Q-A.
* Repoint its `embedder.embed_query` call sites to `embed_text(..., task_type=QUERY)`, then delete
  `embedder.embed_query` and `_Embedder.embed_query`.
* **Count discrepancy to settle:** the plan and a test docstring say "four" call sites; ripgrep finds
  **five**, at `rag_agent_advanced.py:128,198,265,374,438`. `strategies.py`'s ~dozen matches are all
  inside comments. Resolve by reading, and amend whichever document is wrong.
* Fail-closed fix at `agent_saul/dependencies.py:49` per D17.
* Narrow the `except Exception` at `ingestion_kb/nodes.py:791`.
* Note while you are in that file: `search_tasks` is a **local variable** at
  `rag_agent_advanced.py:124,138,141` with no relation to the deleted Celery module. It is why D12's
  original grep produced false positives.

### F — `cognee-agent-memory`

Touches `src/alembic/env.py`, `config/settings.py`, `features/health/service.py`,
`lifecycle/lifespan.py`, `middleware/health_check.py`, `shared/langchain_layer/messages.py`,
`src/tasks/__init__.py`. **Must run after D**, because D also edits `env.py`.

### After F — the PR

Create the PR for `refactor/todo-210-sequence` → `main` and merge it. Only then. There is currently
no open PR to merge.

---

## 6. Still owed, and known-but-not-fixed

**Owed document:** `docs/relay/envelope-registration.md` does not exist. It should record the
BREAKING surface per feature and status code, the 6/6 mutation results, the harness-flaw lesson, and
the C10 inversion note.

**To report, not fix** (accumulated across bands; none are regressions from this work):

* Doubled route prefixes: `/api/v1/auth/auth/…`, `/api/v1/users/users/…`,
  `/api/v1/agent-saul/agent-saul/…`, and the same under `/api/v2`.
* `retrieval_kb/nodes.py:118,155` still need C6's conversion.
* `retrieval_kb/nodes.py:255,298` — `except Exception` would swallow a LangGraph pause.
* `documents/service.py` half-live branches; `legal_metadata.py:76`.
* Residual dispatch literals: `documents/service.py:184`, `auth/service.py:271,298`.
* The auth producer/consumer `idempotency_key` gap.
* `tasks.auth_email_tasks_typed` is a deletion candidate.
* `README.md:263` spells the app `src.app.main:app`.
* `search()`'s setnx lock is not released on the new raise path — bounded by its 15s expiry, and
  pre-existing for every other raise in that method.
* `retrieval_kb/graph.py:31 repo: Any` can now be narrowed to `DocumentRepository` — `search/service.py`
  is gone, so nothing else passes through it.
* **`documents/service.py` holds two expressions of one behaviour** — the compiled graph
  (`ask_via_retrieval_graph`) and `ask`'s inline node sequence, which is what the mounted router
  serves. The graph is the intended survivor. Collapsing them changes a live endpoint.
* **Coverage gap created by D10:** the unified feature has no integration test (`test_search.py` was
  deleted, not rewritten — it was stale against the Result pattern and `integration`-deselected, so
  it never ran), and `DocumentCommandService.upload_document` plus the three Redis caching helpers
  have no direct test of any kind.

---

## 7. Hard-won mechanics — read once, save hours

* **A bare `types.ModuleType` in `sys.modules` has no `__path__`**, so it makes *every* submodule of
  that name unimportable for the rest of the process (`'<pkg>' is not a package`). At module scope it
  fires during collection, and **collection errors are total**: one `ImportError` produced
  `Interrupted: 1 error during collection` and **zero** tests ran, not "all but one file". pytest
  collects alphabetically, so the blast radius is every file sorted after it.
* **A grep for a string cannot see `from mod import NAME`.** A `python -c "import ..."` cannot see
  `if TYPE_CHECKING:` — only `ty check src/` does. Match the probe to the edge kind.
* **Never put a proof's own grep literal into a comment.** Four proofs in band D went green or red
  falsely because documentation used the word the proof searched for. `schema_identifier_gate.py`
  must never contain the words `sqlalchemy`, `asyncpg`, `psycopg`, `create_engine`, `get_session`.
  Likewise never spell a suppression directive out in prose — `ty` reads the mention as a live
  declaration.
* **`rg -r` is ripgrep's *replace* flag, not a line-number flag.** `rg -rn <pat>` rewrites every
  match to the literal `n` and prints a plausible, useless list.
* **FastAPI 0.140 does not flatten included routers.** `{r.path for r in app.routes}` reports **8**
  paths for this 87-route app, and non-emptily, so it passes its own check and the downstream `diff`
  compares two truncated files. Use `/tmp/enum_routes.py`, which recurses through `_IncludedRouter`
  and carries an `assert paths` line.
* **`diff` against `pytest --collect-only -q` output can never pass** — the last lines carry elapsed
  time. Compare the count, or use `rg -ci error`.
* **Dead imports do not constrain a move.** Whether imports are *used* is a different question from
  whether they exist, and only the first one constrains anything. This cost band D three unnecessary
  re-export shims.
* **`git rm` fails on a file staged as a rename's source.** If a relocation is staged as `R`, a shim
  left at the old path is *untracked* — plain `rm` for it, `git rm` for tracked modules.
* **`uv sync` prunes the test toolchain.** A bare `uv sync` uninstalls `pytest-asyncio` because the
  test deps sit outside `default-groups = ["dev"]`. Run `uv lock --check` first; it usually means no
  sync is needed.
* **`alembic heads` is the cheapest command that forces `src/alembic/env.py` to import** and touches
  no database. Use it after any edit to `env.py`. Note `env.py` breaks every *other* migration
  command in this environment.
* **A workaround justified by a comment rather than a test outlives the problem it solved.** The
  outbox stub cited a cycle severed two commits earlier and guarded a module that imports nothing
  from `app` at all.

---

## 8. Standing constraints — verbatim, still in force

* **Never print a credential.** Any probe touching a connection prints host/port/database only.
  No credential, connection string or password has been emitted at any point in this work. Keep it
  that way.
* Every migration proof runs against a **local scratch database**. Applying DDL to the deployed
  instance is a **separately authorized act**. That authorization was granted once, exercised, and
  is **closed**. `critical-path-210.md:120`: *"The single authorization for `CREATE EXTENSION IF NOT
  EXISTS pg_textsearch` is **spent**; any further DDL needs fresh authorization."*
* Every probe against the managed instance in this work has been **read-only**. Keep that invariant.
* The checkpointer "never logs the string or its credentials… Never print the value in a Proof's own
  output."
* **The `[:24]` slice in change 0 step 9's proof reaches into URL userinfo and leaks the username.
  Do not run it as written.**
* The configured Celery broker is a **live managed RabbitMQ instance**. **Do not start a consuming
  worker against it** — the registered task set includes `billing.*`, `credits.*` and
  `auth.send_password_reset_email`, and it could send mail to real recipients.
* The database is **TigerData/Timescale Cloud, live**. Ignore the docker Postgres image.
* **Do not spawn subagents and do not invoke Workflow.** The user rejected a Workflow call and said
  explicitly: *"dont [spawn] three agents … do it yourself and do it fast."*

## 9. Document handling

The five `openspec/changes/<name>/` directories are self-contained working documents. Four
exceptions carry evidence not duplicated inside them:

* `critical-path-210.md`
* `decisions.md` — *"Your eight locked decisions — these override anything a design says."*
* `open-questions.md` — change 1 has none; its open questions live in `design.md:766-795`
* `findings-database.md` §10–§11 — change 0 only

**Do not read or hand over the `plan-change*.md` files (68–134 KB each) or `GRAPH_REPORT.md`** —
they are superseded scaffolding and will eat context for nothing.

**Openspec semantics to remember:** a `MODIFIED` delta block replaces its requirement wholesale on
archive, so an omitted scenario is silently deleted and `validate --strict` cannot detect it.
