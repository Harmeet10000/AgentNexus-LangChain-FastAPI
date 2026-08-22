# Tasks

> Ordered. Every task carries an executable **Proof**: a shell command, or a check that resolves at import time or
> type-check time. No task is complete on a read-through.

## How to read the Proofs

Five rules, each of which exists because ignoring it has already cost this project a wrong conclusion.

1. **The coverage gate makes a green suite exit non-zero.** The gate demands 80% against current coverage near 18%.
   Every `pytest` Proof below compares the **summary counts** and never the exit code. Re-measure the count
   immediately before the task and assert the stated *rise*; do not trust a number written here weeks earlier.
2. **No Proof may render migrations offline.** `alembic upgrade head --sql` always renders from base, never
   incrementally, so it cannot show what a single revision adds (D14.3). Any schema fact is proven by inspecting the
   live catalogue after change 0's migration ran, never by offline rendering.
3. **No Proof may depend on a durable outbound event firing.** Those tables do not exist and change 0 creates them.
   Worker readiness is proven by interrogating the worker; dispatch validation is proven by invoking the dispatch
   helper directly.
4. **Checkpointer Proofs are import-level, type-level, or unit tests over a construction the test itself owns.** D17
   forbids re-enabling the deliberately commented lifespan wiring, and commented code cannot be linted, type-checked,
   or executed. Where a Proof genuinely needs a database, it uses a **local scratch Postgres the task brings up
   itself** — never the managed instance, because the checkpointer's setup issues DDL and every probe against the
   managed instance in this work has been read-only.
5. **The single upload-to-chunks acceptance check does not exist inside this change** and must not be written as a
   task. The shared wiring stays commented by decision, so there is no path from a mounted route through a
   provisioned graph to a persisted row here. Acceptance is decomposed instead.

Baselines to re-measure before starting, so a lost test shows up as a count that did not rise: `uv run pytest -q`
summary counts; `uv run ruff check src/ | tail -1`; `uv run ty check src/ 2>&1 | tail -1`.

---

## Band 0 — preconditions and named cross-change dependencies (no code change)

- [x] **0.1 — Record the closed lexical precondition and the index dependency it exposed.**
  F8 is closed: the lexical extension's index access method is `bm25`, `to_bm25query` has a two-argument overload
  taking the index name, and the repository's existing lexical SQL is **already correct** — it is a harvest, not a
  rewrite. The remaining break is that **no `bm25` index exists anywhere**, and because the index name is a literal
  argument inside the query, an index of the right shape under a different name does not satisfy the SQL. This task
  is the recording, not a probe; the probe already ran under the user's scoped authorisation.
  **Dependency (change 0):** its migration must create the lexical indexes under exactly the pinned names.
  - **Proof:** `rg -n "F8" openspec/changes/ingestion-pipeline-unification/design.md` → hits in Decision 6 and in
    `## Open Questions → Closed since the first draft`, and `rg -n "Coordination point 5" …/design.md` → a hit.
  - **Proof:** `rg -n "SEARCH_CHUNKS_BM25_INDEX_NAME" src/app/features/search/constants.py` → the pinned name is
    present and is the name task E2 references and change 0's migration must create.
  - **Proof (negative, and it must stay red until change 0 lands):** a read-only catalogue query for indexes whose
    access method is the lexical one returns zero rows. Record the output. This is the gate for E2's live check.

  **Amendment (measured 2026-08-23, change 0 landed at head `b3e7c41d92af`).** Proof 3's polarity has inverted by
  design: change 0 has landed, so the negative proof is now expected to be **green**, and it is. A read-only
  catalogue query over the chunk relations reports `chunks_bm25_idx [bm25]`, `chunks_embedding_idx [diskann]`, and
  `chunks_search_text_trgm_idx [gin]` — all three retrieval branches present.

  Proof 2 is green but **materially incomplete as a dependency statement**, and the gap is the point of this task.
  Three distinct lexical index names are referenced by live SQL, and change 0 created exactly one of them:

  | Index name | Referenced from | In the database |
  |---|---|---|
  | `chunks_bm25_idx` | `features/documents/repository.py` ×6 | **yes** |
  | `search_chunks_bm25_idx` | `features/search/repository.py` ×6, hard-coded | **no** |
  | `clauses_bm25_idx` | `features/search/repository.py` ×3 and `ingestion_kb/nodes.py:765` | **no**, and neither is its table |

  Change 0's claim that `SEARCH_CHUNKS_BM25_INDEX_NAME` has "zero readers anywhere in `src/`" is **literally true and
  misleading**. `features/search/repository.py` never imports `features/search/constants.py`; it hard-codes the same
  string six times inside `_build_bm25_statement`. So the *constant* is dead while the *name* is live — the constant
  cannot be deleted as dead code without leaving six unattributed literals pointed at a missing index.

  Neither missing-index path is reachable over HTTP: `api/v1.py:12-17` mounts auth, health, users, profile, documents,
  and agent_saul; there is no search router. So these are latent, not live, faults.

  **One is not latent.** `ingestion_kb/nodes.py:765` `_force_merge_bm25` issues
  `SELECT bm25_force_merge('clauses_bm25_idx')` from inside the pipeline this change promotes (D1), called at `:355`,
  wrapped in `except Exception` with a `BLE001` waiver and a warning. It cannot ever succeed — the index does not
  exist and D15 forbids creating its table — so it degrades silently on every ingestion run. That is precisely the
  invisible-failure shape this change exists to remove, and it is recorded here rather than fixed here because
  retargeting it is one line of the persistence work the missing Band D owns (see 0.3).

- [x] **0.2 — Establish the runnable-database gate for every table-touching task.**
  No document, chunk, search, clause, durable-event, dead-letter, or memory table exists; the schema was stamped, not
  migrated. Every Proof below that reads or writes a table is blocked until change 0's single migration runs on the
  merged head.
  **Dependency (change 0):** revision-head merge, then one migration creating the target schema, the extensions, and
  the lexical indexes by exact name.
  - **Proof:** against the migrated database, a read-only catalogue query lists the document and chunk tables and the
    lexical index by name. Until it does, tasks A3, D3, E6, and E7's persistence Proofs are **blocked, not skipped**.
  - **Proof:** `uv run alembic heads` reports a single head. (Not `--sql`; see rule 2.)

  **Amendment (measured 2026-08-23).** Both proofs are green. `uv run alembic heads` → `b3e7c41d92af (head)`, a
  single head. The catalogue reports 27 public tables, including all nine of the target set this change and its
  siblings need: `documents`, `chunks`, `search_documents`, `search_chunks`, `document_vectors`, `chat_messages`,
  `chat_sessions`, `outbox_events`, `dead_letter_events`. `clauses` is absent, which is **correct**: D15 rules that
  change 1 writes `chunks` and never `clauses`, so its absence is a decision honoured, not a gap.

  **The gate opens for A3 and stays shut for the persistence work, for a reason this task did not anticipate.** The
  blocker is no longer the schema — it is that the pipeline being promoted writes to four relations that do not exist
  and are not going to:

  | `nodes.py` line | Statement target | In the database |
  |---|---|---|
  | `:513` | `INSERT INTO parent_documents` | no |
  | `:567` | `INSERT INTO entities` | no |
  | `:613` | `INSERT INTO relationships` | no |
  | `:676` | `INSERT INTO clauses` | no |

  Every persistence statement in the promoted pipeline targets a missing relation, so the pipeline cannot persist
  anything at all. A3's live-catalogue Proof is unblocked and green (see A3); D3's, E6's, and E7's are **not blocked
  on the database** as this task assumed — they are blocked on tasks that do not exist in this file. See 0.3.

- [x] **0.3 — Report that the Bands this file's own gates depend on are missing (NEW; added 2026-08-23).**
  0.2's first Proof defers persistence verification to tasks **`D3`, `E6`, and `E7`**, and 0.1's third Proof defers the
  live lexical check to **`E2`**. **None of those four tasks exists.** This file contains Band 0 and Bands A, B, and C
  only; there is no Band D and no Band E, and `design.md` never mentions them either. `openspec validate
  ingestion-pipeline-unification --type change --strict` reports the change valid regardless, because it validates
  requirement/scenario structure and does not resolve cross-references between task identifiers.

  This is not a documentation defect. The work those Bands were to contain is the **core of the change**: retargeting
  the promoted pipeline's persistence from `parent_documents`/`clauses` onto `documents`/`chunks` (D15's arrow read as
  a work order, per D3 of `decisions.md`), settling whether `entities`/`relationships` persist relationally at all or
  belong solely to Graphiti, and retargeting `clauses_bm25_idx` → `chunks_bm25_idx`. Bands A–C do not touch a single
  `INSERT`; A fixes cold correctness, B unifies embedding and parsing, C fixes the runtime substrate.
  - **Proof:** `rg -n "^- \[.\] \*\*[DE][0-9]" openspec/changes/ingestion-pipeline-unification/tasks.md` → **no
    matches**, while `rg -n "D3|E2|E6|E7" …/tasks.md` → hits inside Band 0's Proofs. The dangling references are
    demonstrable from the file alone.
  - **Proof:** `rg -n "Band D|Band E" openspec/changes/ingestion-pipeline-unification/design.md` → **no matches**.
  - **Assumption recorded, not a decision taken:** the retargeting is treated as belonging to **this change** on the
    strength of D15 ("change 1 writes `chunks`, never `clauses`") and because the statements live inside the pipeline
    D1 promotes; change 2 (`documents-unified-schema`) is read as owning the **read/search** side — consolidating
    `search_documents`/`search_chunks` and the six hard-coded `search_chunks_bm25_idx` literals onto the same
    relations. If that split is wrong, it is wrong in one direction only: the work is enumerated here and can be moved
    wholesale rather than rediscovered.

- [x] **0.4 — Remove the stale test stub that made every Band C Proof unwritable (NEW; added 2026-08-23).**
  `tests/conftest.py` replaced `app.shared.langgraph_layer` and four of its submodules with
  `MagicMock()` at module scope, before any app module loaded. A `MagicMock` has no `__path__`, so **every**
  `app.shared.langgraph_layer.<anything>` import inside the suite raised
  `ModuleNotFoundError: … 'app.shared.langgraph_layer' is not a package`. The stubbed set included
  `checkpointer` and `kb_retry` by name.

  This is a Band 0 precondition, not a tidy-up. Rule 4 of this file requires checkpointer Proofs to be
  "import-level, type-level, or unit tests over a construction the test itself owns" — none of the three is possible
  against a module that cannot be imported. **C2, C4, C5, and C6 each require a unit test over `checkpointer.py` or
  `kb_retry.py`, and all four were unwritable.** That was invisible because no test had tried.

  The stubs were introduced to "break circular/broken imports before any app module loads." Those cycles are gone,
  severed by `319c698` (the `app.utils` cycle) and `6525c6f`. Removing all five entries leaves the suite at **exactly**
  its prior counts — `3 failed, 138 passed, 48 deselected, 9 errors` before and after — and runs faster (7.4s against
  13.0s). The comment left in their place records all of this so they are not restored.

  A second drift found in passing, recorded but not acted on: the stub list no longer matches what `src/` imports. The
  file's own regeneration command reports `tasks.example` as an importer, which is **not** stubbed, while
  `mcp_core.server.middleware` **is** stubbed and is imported by nothing.
  - **Proof:** `uv run pytest -q 2>&1 | tail -3` before and after the removal reports identical counts. A stub whose
    removal changes nothing was load-bearing for nothing.
  - **Proof:** `rg -n "app.shared.langgraph_layer" tests/conftest.py` → hits only inside the explanatory comment, never
    as a `sys.modules` assignment.
  - **Proof:** a unit test in `tests/unit/shared/langgraph_layer/` imports from the package and passes, which was
    impossible before. This is the gate for C2, C4, C5, and C6.

  **It was also concealing a live defect** — see A6, which was found on the first attempt to construct a state object
  once the package became reachable. A stub that makes a package unimportable does not isolate the suite from that
  package; it removes the package from the suite's reach and takes its defects out of view with it.

---

## Band A — cold correctness fixes (no database, no graph, no network)

Each is independently committable and each fixes something already wrong today.

- [x] **A1 — Fix the logger submodule shadowing in `src/app/utils/embedding.py`.**
  `:5`'s `from app.utils import logger` binds the **submodule**, not the loguru object, because that import runs while
  `app/utils/__init__.py` is still initialising — `from .embedding import normalize_embedding` at `:35` precedes the
  logger export at `:59`, so Python's circular-import fallback resolves the attribute to
  `sys.modules["app.utils.logger"]`. `logger.warning(...)` at `:22` therefore raises `AttributeError` on every
  dimension mismatch: the diagnostic path destroys the diagnostic. Change to
  `from app.utils.logger import logger`, which is already house style elsewhere. **Only** modules imported from
  inside `app/utils/__init__.py` are affected — `kb_retry.py`, `checkpointer.py`, and `retrieval_kb/reranker.py` use
  the same idiom from outside the package and are correct. Do not "fix" them.
  This lands **first**: until it does, every dimension mismatch is an error rather than a warning and six tests stay
  red, so no later task has a clean baseline.
  - **Proof:** `uv run pytest -q 2>&1 | tail -3` — failed count drops to **0** and passed count rises by exactly 6.
  - **Proof:** `uv run pytest -q 2>&1 | grep -c "AttributeError: module 'app.utils.logger'"` → `0`.
  - **Proof:** `uv run rg -n "^from app.utils import logger" src/app/utils/` → **no matches**.

  **Amendment (measured 2026-08-23).** Already landed, ahead of this file, in commit `52baccb` — the same cycle-severing
  work as `319c698`, which moved this module and its siblings onto leaf-module imports. Proof 3 is green: no match in
  `src/app/utils/`. The carve-out holds as written — `kb_retry.py:9`, `checkpointer.py`, and `retrieval_kb/reranker.py`
  still use `from app.utils import logger` and are still correct, because they import from **outside** the package and
  so never observe it mid-initialisation. They were not touched.

  **Proof 1 is unexecutable as written and must not be re-attempted.** It asserts the failed count drops to **0** and
  the passed count rises by exactly 6. The failed count cannot reach 0: twelve pre-existing failures live in
  `tests/unit/test_websocket_security_bug_conditions.py` and `tests/unit/test_websocket_security_preservation.py`
  (3 failed + 9 collection errors) from websocket fixture drift that no task in any of the five changes owns. The
  correct acceptance is D12's — "no new failures beyond the recorded baseline" — and the baseline for this change is
  **103 passing** with those 12 red. Every later Proof in this file that names an absolute failed count inherits the
  same correction.

- [x] **A2 — Resolve the dimension conflict and delete every placeholder vector in the batch embedder.**
  `src/app/shared/rag/document_processing/embedder.py` declares `{"dimensions": 1536, …}` for every model key against
  vector columns declared at 768, and returns `[0.0] * config["dimensions"]` at `:167`, `:177`, and `:228` when a
  provider call fails. A zero vector is a **valid row that ranks against nothing**, so a failed embedding becomes an
  invisible hole in the corpus. Read the dimension from the single configured value; raise a typed project exception
  with a note naming the model, task type, and text count instead of substituting.
  This module stays a **batch-only carve-out** — it must not become reachable from a request or an ingestion stage.
  - **Proof:** `uv run rg -n "0\.0\] \* |\[0\.0\]" src/app/shared/rag/document_processing/embedder.py` → **no
    matches**.
  - **Proof:** `uv run rg -n "1536" src/app/` → no occurrence outside historical migration revisions.
  - **Proof:** a new unit test drives a provider failure through the batch path and asserts a typed exception is
    raised, that its `__cause__` is the provider's own exception, and that its notes name model, task type, and text
    count. `uv run pytest tests/unit -q 2>&1 | tail -3` shows the count risen.

  **Amendment (measured 2026-08-23).** Proofs 1 and 3 are green as written. Proof 1: no match in `embedder.py`. Proof 3:
  `tests/unit/shared/rag/test_embedder_no_substitution.py`, 13 tests, all passing — the named assertions plus the
  no-partial-list, wrong-width-refused, blank-input, and `embed_chunks` batch-position cases.

  **Proof 2 is amended from elimination to enumeration.** As written — `rg -n "1536" src/app/` → "no occurrence outside
  historical migration revisions" — it cannot pass, and should not. Re-measured word-bounded (`\b1536\b`, since the
  bare pattern also matches substrings of unrelated numbers) there are exactly **three** survivors in `src/app/`, and
  all three are correct:

  | Site | Why it stays |
  |---|---|
  | `config/settings.py:55` | `"text-embedding-3-small": 1536` is a **true** entry in the validator's cross-check map. That model is not the configured one; the entry exists so that selecting it would be validated, not to assert the corpus width. Deleting it would remove a correct fact and weaken the validator. |
  | `embedder.py:17` | Prose recording that the deleted table read 1536 against columns declared at 768. |
  | `embedder.py:20` | Prose recording that the configured model was **absent** from that table, so the `.get(model, default)` lookup fell through to the wrong width for the deployed configuration regardless of its entries. |

  The two prose sites are the reason the deletion is auditable at all, and removing them to satisfy a grep would delete
  the explanation and keep the pattern satisfiable only by silence. The revised Proof is: `rg -n '\b1536\b' src/app/`
  → exactly these three, each accounted for above. The **substantive** guarantee — that no code path derives a width
  from a model-keyed table — is proven instead by
  `test_model_keyed_dimension_table_is_gone` and `test_dimension_accessor_takes_no_model_argument`, which assert the
  accessor is gone and that the surviving one refuses a model argument. Those cannot be satisfied by prose.

- [x] **A3 — Make the persisted vector width derive from the configured dimension, not a literal.**
  The ORM models hard-code `Vector(768)`. The declared width must equal the single configured value. Because there is
  no data anywhere, this is a **column definition**, not a type migration — nothing to widen, no index to drop,
  nothing to preserve. This is the cheapest moment in the project's life to settle it.
  **Dependency (change 0):** the migration that creates these columns is change 0's; this task ships **no revision**.
  - **Proof:** `uv run rg -n "Vector\(768\)|Vector\(1536\)|Vector\(3072\)" src/app/` → **no matches**.
  - **Proof:** `uv run ty check src/ 2>&1 | tail -1` → count not risen.
  - **Proof (blocked on 0.2):** after change 0's migration, a read-only catalogue query reports each vector column's
    declared width equal to the configured dimension.
  - **Proof (N6 — no data exists, so this is a stub test, never a data check):** a unit test stubs a *stored* width
    differing from the configured one and asserts new writes are refused with a diagnostic reporting that re-embedding
    is required. There are zero stored vectors; a data check here would be a Proof that cannot run.

  **Amendment (measured 2026-08-23).** All four Proofs green. Proofs 1 and 2: no `Vector(768|1536|3072)` match in
  `src/app/`; `ty check src/` → `All checks passed!`. Proof 4:
  `tests/unit/documents/test_vector_width_configured.py`, 14 tests, all passing.

  **Proof 3 is now unblocked and green, with the column set derived rather than listed.** Querying every
  `vector`-typed column in the `public` schema — rather than naming the ones expected — reports exactly two, both
  equal to the configured `EMBEDDING_DIMENSION` of 768: `chunks.embedding = vector(768)` and
  `search_chunks.embedding = vector(768)`. Deriving the set is what makes this a proof rather than a spot check; had a
  third relation carried a vector column at another width, a listed query would have missed it. `document_vectors` has
  **no** vector column — its nine columns are `id, user_id, document_id, title, content, vector_id, metadata,
  created_at, updated_at` — so it is a *pointer* table into an external store, not a vector-bearing relation, and is
  correctly outside this task's scope.

  **Extended twice beyond the task text, both necessary rather than opportunistic:**

  1. **The producer side escaped the grep.** `features/search/embeddings.py` passed a literal to the provider's
     `output_dimensionality` argument. Proof 1's pattern only matches the `Vector(...)` **column** declaration, so a
     hard-coded width on the *request* would have survived every Proof in this task while producing vectors of the
     wrong shape for a correctly-declared column. Fixed to read the configured value.
  2. **N6 required a refusal mechanism that did not exist.** The task says writes must be "refused"; nothing refused
     them. Built as this project's documented dual-method pattern, because `upsert_chunks` returns `AppResult` and must
     not raise, while the offline batch paths have no `Result` to put a failure into:
     `stored_width_mismatch` (pure predicate, returns the pair or `None`), `width_mismatch_detail` (the one shared
     diagnostic, so the two halves cannot drift), and `assert_stored_width_matches_configured` (raising half), all in
     `utils/embedding.py`; consumed by `DocumentRepository._reject_width_mismatch`, called from `upsert_chunks` before
     any statement is issued.

  Two findings from building it, both worth keeping:

  - **`retryable` defaults to `True` on `InfrastructureAppError`.** Left unset, a width disagreement would tell Celery
    to retry to its ceiling against a condition only a re-embedding run can change. Both raise sites now set
    `retryable=False` explicitly. The N6 test asserts it, so the default cannot silently reassert itself.
  - **The declared-vs-configured comparison is tautological in production**, precisely *because* A3 made the column
    derive from configuration — the class body reads the setting, so the two agree by construction at import. It stops
    being tautological the moment configuration is reloaded after import, and that is exactly what the N6 stub test
    drives: it moves the configured value out from under an already-built column. The check that fires in **practice**
    is the second one, row-width against declared width, which turns an opaque psycopg error deep inside a batch into a
    diagnostic naming the relation, both widths, and the remedy. Both are implemented; only the second is reachable
    without a configuration reload.

  A named module constant `CHUNK_EMBEDDING_DIM` (`features/documents/model.py:48`) carries the width the column was
  built from. It exists because reading it back off the column — `__table__.c.embedding.type.dim` — is typed as the
  base `TypeEngine`, which declares no `dim`, so every consumer would otherwise need a narrowing dance or a
  suppression. `test_the_named_constant_matches_the_column_it_built` guards it against drifting from the column it
  built, which would leave the guard validating writes against a width the database does not have — this task's own
  failure mode, reintroduced one layer up.

- [x] **A4 — Delete the phantom `ingestion.embedder` import in `rag_agent_advanced.py`.**
  `:119`, `:198`, `:267`, and `:373` each do `from ingestion.embedder import create_embedder`, a module that does not
  exist. These are function-local imports, so the failure is deferred to first call rather than surfacing at import.
  Retarget them at the single embedding path.
  - **Proof:** `uv run rg -n "from ingestion.embedder|ingestion\.embedder" src/app/` → **no matches**.
  - **Proof:** `uv run python -c "import importlib; importlib.import_module('app.shared.rag.rag_agent_advanced')"`
    exits 0.
  - **Proof:** a unit test exercises each former call site's embedding branch and asserts it resolves without a
    module-resolution error. A deferred import failure is not acceptable — the requirement says so explicitly.

  **Amendment (measured 2026-08-23).** The named defect is fixed and Proof 1 is green: no match in `src/app/`. The four
  function-local imports collapsed to **one module-level** import of
  `app.shared.rag.document_processing.embedder`, which is the property the requirement is actually about — a future
  breakage is now an import error rather than a first-call error.

  **Proof 2 is unsatisfiable, and A4 is not the reason.** `uv run python -c "…import_module('app.shared.rag.rag_agent_advanced')"`
  cannot exit 0. That module has **nine undefined names** across four identifiers — `RunContext` ×6 (`:53, :98, :179,
  :249, :291, :348`), `Agent` (`:482`, module scope, invoked), `List` (`:53`), and `itemgetter` (`:223`, imported
  locally only at `:157` inside a different function) — and `pydantic-ai`, which supplies `Agent` and `RunContext`, is
  declared in **neither** `pyproject.toml` nor `uv.lock` (`importlib.util.find_spec("pydantic_ai")` → `None`). Because
  the module carries no `from __future__ import annotations`, those annotations are evaluated when each `def` executes,
  so the failure lands at **import**, not at call.

  None of it is this task's doing: `git show HEAD` carries the same six `RunContext` occurrences, and this task's diff
  is one added import line against four removed ones.

  **Why `ruff check src/` was green over an unimportable module** — worth recording, because it is the mechanism that
  let this survive: `pyproject.toml:474-481` lists `F821` in this file's per-file-ignores. One entry in an ignore list
  suppressed the only rule that would have named all nine. The true count came from `ruff check --isolated --select F821`.

  **Disposition: do not add the dependency.** Adding a package to make a CLI that **nothing imports** importable is a
  dependency decision this task never asked for, and leg E already owns this file — Q-A decided it relocates to
  `src/app/examples/`, and E may rewrite or delete it. `scout-tools-schema.md:30` had already recorded "module imported
  by nothing."

  **Substitute Proof, delivered:** `tests/unit/shared/rag/test_rag_agent_embedder_import.py`, 10 tests, all passing.
  It proves statically what A4 protects, reading the source text rather than importing it: the phantom reference is
  gone; the package `ingestion` does not exist; there is **exactly one** embedder import and it is at **module level**
  (compared against `tree.body`, not by pattern); the target resolves; and the target exposes every member the former
  call sites invoke — with the invoked set **derived from the AST** rather than listed, so a new call site is covered
  without editing the test. The last point matters: retargeting without `embed_query` present would have traded
  `ModuleNotFoundError` for `AttributeError` at the same moment, which is not what the requirement asks for.

  **Plus a tripwire:** `test_module_remains_unimportable_pending_leg_e` asserts `pydantic_ai` is absent *and* that
  `Agent`/`RunContext` are used-but-never-imported. It **fails the moment the module becomes importable**, which is the
  intent — it forces leg E to revisit this Proof rather than inherit it stale. It is a tripwire, not an endorsement.

- [x] **A5 — Fix the degraded-branch handler that destroys the diagnostic it exists to preserve.**
  `ingestion_kb/nodes.py:212-256` calls `exc.add_note(f"doc_id={state.doc_id}, …")` inside the degraded branch, but
  the state is a mapping at that point, so the attribute access raises inside the handler and replaces the original
  failure with a secondary error. Build the note from the values the branch already holds; the handler must not raise.
  - **Proof:** `uv run rg -n "state\.doc_id" src/app/shared/langgraph_layer/ingestion_kb/nodes.py` → no occurrence
    inside an exception handler (verify by reading each remaining hit, not by pattern alone).
  - **Proof:** a new unit test induces the degraded branch and asserts (a) it returns a degraded result rather than
    raising, (b) the recorded diagnostic names the **original** cause, and (c) the degradation record carries the
    document and chunk identity. `uv run pytest tests/unit/shared -q 2>&1 | tail -3`.

  **Amendment (measured 2026-08-23) — the premise is refuted, and the code was already right about the exact hazard
  this task describes.** The defect as stated does not exist, for two independent reasons:

  1. **The named range contains no such call.** `nodes.py:212-254` is `make_contextualize_chunk_node`. Its handler
     (`:236-240` before this task's edit) built its note from `segment.clause_id` — a **local**, `model_validate`-d
     `ClauseSegment` bound at `:216` — and never touched `state.doc_id`. There is no `state.doc_id` occurrence anywhere
     in `212-256`.
  2. **Where `state.doc_id` *is* used in a handler, it cannot raise.** The two such sites are `:183`
     (`make_segment_document_node`) and `:288` (`make_classify_extract_node`). Both receive `state: IngestionState`,
     and `IngestionState` is a Pydantic **`BaseModel`** (`state.py:166`) — not a `TypedDict`, not a mapping — whose
     `doc_id: str = ""` (`:169`) is a **defaulted** field. So the access is valid attribute access that cannot even
     fail on an unset value.

  The pattern is worth naming, because it is the opposite of a bug: `contextualize_chunk_node` is the **one** node
  annotated `dict[str, Any]` (`:215`), and it is the **one** node that does not use `state.doc_id`. That is correct and
  deliberate — `Send("contextualize_chunks", {...})` *replaces* the state for the fanned-out invocation, so that node
  receives the dispatcher's dict literal and not `IngestionState`. Whoever wrote it had already handled precisely the
  hazard A5 warns about. Proof 1's remaining hits inside handlers are therefore **expected**, and the Proof is amended
  to say so: what must hold is that no handler reads an attribute off a value that is a mapping at that point, not that
  no handler mentions `state.doc_id`.

  **Proof 2(c) does, however, name a real and smaller defect, and that is what was fixed.**
  `dispatch_contextualize_chunks` (`:199-209`) built its `Send` payload from three keys — `segment`,
  `contract_metadata`, `source` — and **no `doc_id`**. Since `Send` replaces the state, there was no document identity
  available to that node at all, so a degraded contextualization was attributable to a clause but never to a document,
  and `clause_id` alone does not disambiguate under concurrent ingestion. Proof 2(c) was therefore not merely unproven
  but **unsatisfiable**. Changed:

  - `dispatch_contextualize_chunks` now carries `doc_id` in the payload, with a comment recording why anything a
    fanned-out node needs must be put there rather than reached from graph state.
  - `contextualize_chunk_node` binds `doc_id: str = state.get("doc_id", "")` **before** the `try`, and with `.get`
    rather than `[...]` — for this task's own reason: a handler must not introduce a new raise site of its own, and a
    missing key would turn a recoverable failure into a `KeyError` that replaces the original diagnostic. That is the
    shape A5 set out to remove, and it would have been reintroduced by the naive fix.
  - Both the note and the log bind now carry `doc_id`, `clause_id`, and `chunk_index` — document **and** chunk identity,
    which is what 2(c) asks for.

  **Proof 2(a) and 2(b) are blocked on C6, and this is an ordering defect in this file.** They require the degraded
  branch to *execute*. It cannot. `retry_immediate` (`kb_retry.py`) catches `Exception` and raises
  `TransientExternalError(msg) from exc`; `TransientExternalError` derives from `Exception`; and every degraded branch
  in `nodes.py` catches `LangChainException`. That `except` can never match, however it is chained — which is C6's
  finding, here confirmed live and found to be **broader than C6 states**: it is not one branch but **all three**
  (`:183` segmentation, `:240` contextualize, `:288` entity extraction). Every fallback path in this pipeline is
  currently dead code, and the pipeline propagates a `TransientExternalError` where it appears to degrade.

  A5 sits in Band A and C6 in Band C, so as ordered this task's Proof 2 cannot pass when it is reached. The identity
  fix above is complete and independently correct; **2(a) and 2(b) are deferred to C6**, which is where the test that
  drives a degraded branch belongs — C6's own third Proof already asks for exactly that test ("drives an exhausted
  retry through a **converted caller** and asserts that caller's degradation branch **executed**"). Writing a second
  one here would duplicate it and would fail until C6 lands either way.

  A curiosity in `kb_retry.py` worth recording for C5/C6: `AsyncRetrying(..., reraise=True)` exists specifically to
  re-raise the *original* exception instead of Tenacity's `RetryError`, so the author intended type preservation — and
  the hand-written `except Exception: raise TransientExternalError(...) from exc` immediately below discards it. The two
  cancel out; the only surviving effect of `reraise=True` is that `__cause__` is the original rather than a `RetryError`.

- [x] **A6 — Make `IngestionState` constructible at runtime (NEW; added 2026-08-23).**
  `IngestionState` could not be constructed **at all**. Every `IngestionState(...)` raised
  `PydanticUserError: 'IngestionState' is not fully defined; you should define 'Annotated'`. The state model of the
  pipeline this whole change promotes was unusable.

  `state.py` carries `from __future__ import annotations`, so every annotation in it is a **string**, and Pydantic
  **evaluates** those strings when it builds the model. `Annotated` was imported inside an `if TYPE_CHECKING:` block, so
  the name was absent from the module namespace at runtime and
  `IngestionState.contextualized_chunks: Annotated[list[ContextualizedChunk], operator.add]` could never resolve.

  Fixed by importing `Annotated` at runtime with `# noqa: TC003` and a comment recording that the suppression is
  **load-bearing, not cosmetic**. The neighbouring `AppError` import at `state.py:21` already carried the identical
  suppression for the identical reason; this one was the one that got away.

  Three layers had hidden it, and each is worth carrying forward:
  1. **Ruff was right and its advice was wrong.** Under `from __future__ import annotations` the import genuinely is
     typing-only by the language's rules, so `TC003` correctly asked for it to be moved. The rule does not know that
     Pydantic resolves annotations at runtime. `ruff check src/` was green across the defect.
  2. **`ty` cannot see it.** Nothing is mis-typed — the name resolves fine under type checking. The failure exists only
     in Pydantic's runtime namespace lookup. `ty check src/` was green across the defect too.
  3. **The suite could not reach it.** See 0.4 — the package was a `MagicMock`, so nothing had ever constructed this
     model, or could have. The defect surfaced on the first attempt once that stub was gone.

  The field that could not resolve is the load-bearing one. `Annotated[..., operator.add]` is the LangGraph **reducer**
  that makes `Send` fan-out results accumulate instead of overwrite, so a regression here does not raise — it silently
  keeps one chunk out of every N a document produces. That makes the annotation worth pinning behaviourally, not just
  structurally.
  - **Proof:** `tests/unit/shared/langgraph_layer/test_ingestion_state_runtime.py` — 7 tests, all passing.
    `test_the_state_model_can_be_constructed` is the defect in one line; `test_the_state_model_is_fully_defined`
    asserts `__pydantic_complete__` directly, because Pydantic defers this failure to first use and a model with an
    unresolvable annotation on an untouched field looks healthy until something touches it.
  - **Proof (tripwire):** `test_annotated_is_available_at_runtime_not_only_to_the_type_checker` asserts
    `state_module.Annotated is typing.Annotated`. A future `ruff check --fix` that moves the import back into a
    type-checking block fails **here**, at the mechanism, instead of in production far from the import.
  - **Proof (behavioural):** `test_the_fan_out_reducer_accumulates_rather_than_overwrites` asserts `operator.add` is in
    the field metadata, and `test_the_reducer_actually_concatenates` calls the reducer and checks both chunks survive —
    metadata being present is not the same as it working.
  - **Gates:** `ruff check src/`, `ruff format --check`, and `ty check src/` all pass. Full suite
    `3 failed, 154 passed, 48 deselected, 9 errors`; 154 − 138 = 16, exactly the two new files added under A5 and A6.
    The 12 red are the pre-existing websocket fixture drift owned by no task, per D12's baseline.

  This task did not exist in the plan. It is recorded here rather than in a later Band because it is Band A by
  definition — cold correctness, no database, no graph, no network — and because **every** Band B and Band C task that
  builds or drives the ingestion graph depends on its state model being constructible.

---

## Band B — the seams: unify what both pipelines duplicate, while both still exist

- [ ] **B1 — One embedding path replacing four, with task type declared on both sides.**
  Four paths exist with two mutually incompatible dimensions: one builds a fresh provider client per call with a
  hard-coded width and no cache; one imports that same client (so two features are already one path); one duck-types
  the embedding callable through three candidate method names, embeds one text per call, and passes **no task type**,
  making its stored vectors asymmetric with the query side — a silent relevance defect, not a style issue; the fourth
  is the batch carve-out fixed in A2. Collapse to one path: one client per process, dimension from configuration,
  explicit task type on query and document sides, batched form available with a documented batch size and a
  single-text form that does not construct a batch of one.
  - **Proof:** `uv run rg -n "_call_embedding_fn|embed_documents\(|embed_query\(" src/app/` → every live call site
    resolves to the one path; the duck-typed resolver is gone.
  - **Proof:** a unit test asserts the provider client is constructed **once** across two embedding calls in one
    process (assert on a construction spy), and that a request omitting task type is rejected.
  - **Proof:** a unit test asserts the same text embedded once as a query and once as a document occupies **distinct**
    cache entries.
  - **Proof:** `uv run pytest tests/unit -q 2>&1 | tail -3` — count risen, failed count `0`.

- [ ] **B2 — Collapse the two digest-keyed embedding caches into one, and record the rejection of the framework wrapper.**
  The shared-cache mechanism already exists in this exact shape twice in the codebase; those two collapse into one,
  keyed by a digest of text together with model and task type, with a documented expiry. The framework's
  cache-backed embeddings wrapper is **rejected** on two independent grounds recorded in Decision 4 and ADR-1
  alternative (b): it is importable only from the version-zero compatibility shim, which this project's import rules
  forbid, and its prescribed backing store is per-container, so each replica would silently keep its own cache.
  - **Proof:** `uv run rg -n "CacheBackedEmbeddings|langchain_classic" src/app/` → **no matches**.
  - **Proof:** exactly one cache implementation remains: `uv run rg -n "embedding.*cache|cache.*embedding" src/app/ -i`
    resolves to one module (verify by reading the hits).
  - **Proof:** a unit test asserts a repeated text calls the provider once and is served from cache the second time,
    and that the entry is visible to a second process (assert against the shared cache backend, not an in-process
    dict).

- [ ] **B3 — Stop blocking the event loop during parse, and stop discarding parsed tables.**
  `src/app/features/documents/parser.py` calls the synchronous converter at `:25` inside `async def parse_document`
  (`:19`), blocking the loop for the whole parse, and returns `tables=[]` at `:34`, discarding structure the parser
  already extracted. Offload the synchronous call; carry the tables through.
  - **Proof:** `uv run rg -n "converter\.convert\(" src/app/features/documents/parser.py` → the call is inside an
    offload, not directly awaited in the coroutine body.
  - **Proof:** `uv run rg -n "tables=\[\]" src/app/features/documents/parser.py` → **no matches**.
  - **Proof:** a unit test asserts the event loop remains responsive across a parse (schedule a second coroutine and
    assert it runs before the parse completes) and that a fixture document with a table yields a non-empty table
    collection. **Mandatory** — this defect is invisible to lint and types.

- [ ] **B4 — Cache the token counter, and record why the transformer dependency cannot be dropped.**
  The counter is acquired uncached and synchronously, with a first-use disk or network load, on **every** call. Cache
  it. Separately, the counter in force is **not the embedding model's counter**: chunks are budgeted by one model's
  token count and embedded by a different provider, so the token bound is enforced against the wrong counter. Either
  match the counter to the embedding model, or state the divergence and its safety margin.
  The "drop the transformer dependencies" item is **unachievable as stated** (Decision 3), and one half is now
  **settled rather than merely unachievable**: the cross-encoder re-ranker genuinely needs the sentence-transformer
  package, so it **stays** (Decision 19). Only the tokenizer half was ever in scope.
  - **Proof:** a unit test asserts the counter is constructed **once** across two chunking calls in one process.
  - **Proof:** `uv run rg -n "AutoTokenizer|from_pretrained" src/app/` → every hit is behind the cached accessor.
  - **Proof:** `rg -n "Decision 19|sentence" openspec/changes/ingestion-pipeline-unification/design.md` → the
    dependency decision is on the record with its reason, not silently dropped.

---

## Band C — the runtime substrate: checkpointer, retries, worker

Band C changes whether a crash is recoverable and whether the queue can be consumed at all. Every checkpointer Proof
here is import-level, type-level, or a unit test over a construction the test owns — see rule 4.

- [ ] **C1 — Install the client-library binary binding and delete the placeholder alias, in one commit.**
  `src/app/shared/langgraph_layer/checkpointer.py:26-29` catches `ImportError` and assigns
  `AsyncPostgresSaver = Any`. That fallback is the **live** path, not dead code: `psycopg` 3.3.3 is installed with no
  libpq binding, so `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver` raises
  `ImportError: no pq wrapper available`. It is currently the only reason the application boots on this machine.
  Consequently the binding install and the alias deletion must land **together** — splitting them produces a commit
  that does not boot. Add the binary binding as a declared dependency; delete the alias and let an import error be an
  import error.
  - **Proof:** `uv run python -c "from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver; print(AsyncPostgresSaver)"`
    prints the real class, not `typing.Any`.
  - **Proof:** `uv run rg -n "AsyncPostgresSaver = Any|except ImportError" src/app/shared/langgraph_layer/checkpointer.py`
    → **no matches**.
  - **Proof:** `uv run rg -n "psycopg" pyproject.toml` → the binary-binding extra is a declared dependency, and
    `uv lock --check` exits 0.
  - **Proof:** `uv run ty check src/app/shared/langgraph_layer/checkpointer.py` → the `ty:ignore[invalid-return-type]`
    suppression at the old `:67` is **gone**. A typed-ignore that hid a real defect is the tell; its removal is the
    type-level proof that the return type is now honest.

- [ ] **C2 — Fix the constructor call: setup must yield a usable saver, never an unentered resource manager.**
  `from_conn_string` is decorated `@classmethod @asynccontextmanager`
  (`.venv/…/langgraph/checkpoint/postgres/aio.py:55-57`), so `checkpointer.py:56-57` binds a context manager and then
  calls `.setup()` on it — an uncaught `AttributeError`, since the handler at `:58` catches only
  `(ConnectionError, TimeoutError, OSError)`. This defect is currently **unreachable** because C1's alias fallback
  short-circuits before it; it becomes live the moment C1 lands, which is why C2 follows C1 immediately. Setup must
  return a saver that can read and write checkpoints, or raise — never an absent value from a function typed to
  return one.
  - **Proof:** `uv run rg -n "from_conn_string" src/app/` → **no matches**.
  - **Proof:** `uv run rg -n "return None" src/app/shared/langgraph_layer/checkpointer.py` → **no matches** in
    setup's body; `uv run ty check src/app/shared/langgraph_layer/checkpointer.py` reports no
    `invalid-return-type`.
  - **Proof (unit, no database):** a test patches the driver connection layer and asserts setup returns an instance
    of the real saver class, and that a construction failure **raises** rather than returning `None`.

- [ ] **C3 — Consume the shared accessor for the plain client-library URL flavour; repair nothing at the call site.**
  The raw configured URL carries **no password**; the relational engine's accessor injects it but returns the
  engine's **dialect alias**, which this driver cannot parse. One is unauthenticated, the other unparseable. The
  checkpointer takes its string from the accessor for its own flavour, retains the transport-security parameters that
  driver requires, and never logs the string or its credentials. The module docstring currently names the
  dialect-aliased scheme and is **wrong** — the saver is client-library-based. Fix the docstring; do not follow it.
  **Dependency (change 0):** `infrastructure-client-access` owns the accessor set. There are **two** flavours, not
  three, and this checkpointer is the reason the plain flavour exists. Change 1 is purely a consumer.
  - **Proof:** `uv run rg -n "replace\(|split\(|\+asyncpg" src/app/shared/langgraph_layer/checkpointer.py` → **no
    matches**: no scheme repair or credential injection at this call site.
  - **Proof:** a unit test asserts the string the checkpointer is constructed with does **not** contain the
    relational engine's dialect alias, **does** contain the configured credentials, and **retains** the
    transport-security parameter.
  - **Proof:** a unit test captures log records across a successful and a failed setup and asserts no record contains
    the connection string or any credential substring. Never print the value in a Proof's own output.

- [ ] **C4 — Make teardown report which of its three outcomes occurred.**
  This is the one live defect in this area fixable **without uncommenting anything**, and it is import- and
  type-provable. Three problems compound: `lifespan.py:317` calls teardown on shutdown while the setup it pairs with
  at `:294-305` is commented out; teardown's `if checkpointer is None: return` is **silent**, indistinguishable from
  a successful close; and its guard tests `hasattr(checkpointer, "pool")` against a value that — because of C2's
  defect — is an async context manager, and in any case the saver class sets only `conn`, `pipe`, `lock`, `loop`, and
  `supports_pipeline`, never `pool`. So the pool would go unclosed **silently** even when one existed. Teardown must
  distinguish: closed a pool / nothing was provisioned / was handed something with no pool to close.
  Ownership follows the constructing process, which in this change is the queue worker. The commented lifespan
  construction **stays commented** (D17).
  - **Proof:** `uv run rg -n 'hasattr\(checkpointer, "pool"\)' src/app/` → **no matches**.
  - **Proof (unit, three cases, no database):** teardown given `None` completes without raising and reports
    "not provisioned"; teardown given an object with no pool reports that it could not close one rather than
    completing as though it had; teardown given a double with a closable pool closes it and reports the close.
  - **Proof (D17 compliance, and it must stay red forever):**
    `rg -n "^\s*#\s*saul_checkpointer = await setup_langgraph_checkpointer" src/app/lifecycle/lifespan.py` → still a
    **commented** line. Also `uv run rg -n "app.state.langgraph_checkpointer\s*=" src/app/` → no *uncommented*
    assignment introduced by this change.

- [ ] **C5 — Correct the retry policy: named transient types, growing wait, no catch-all.**
  `kb_retry.py` uses `retry=retry_if_exception_type(Exception)` at `:29` and `wait=wait_none()` at `:28` — three
  immediate attempts against a rate-limited endpoint produce three refusals in about zero milliseconds, and a
  catch-all around node-internal code will swallow a framework control-flow pause, which pauses **by raising**.
  Ingestion has no such pause today; change 3 adds one, so fixing it now is cheaper than debugging it then. Retry
  wrappers stay at input/output client boundaries and must not wrap a whole graph node.
  - **Proof:** `uv run rg -n "retry_if_exception_type\(Exception\)|wait_none" src/app/` → **no matches**.
  - **Proof:** a unit test asserts (a) a type outside the retryable set propagates on the **first** attempt with no
    further attempt, (b) a named transient type retries to the configured count with a **growing** wait between
    attempts, and (c) a control-flow pause exception propagates immediately with no retry. **Mandatory** — this
    function wraps every input/output call in the pipeline being promoted, so it is the highest-fan-in untested
    function in the change.
  - **Proof:** `uv run rg -n "retry_immediate" src/app/` → every call site wraps a client call, not a node body
    (verify by reading each hit).

- [ ] **C6 — Raise one typed transient failure at the boundary, and convert the callers that catch around it.**
  **This task replaces a remedy that could not work, and the reason must not be re-lost.** The earlier contract said
  chain via `raise … from exc` "so a caller's existing degradation branch still matches". Chaining sets `__cause__`;
  it does **not** change the type raised. `kb_retry.py:41-43` raises `TransientExternalError(msg) from exc`, and
  `TransientExternalError` (`:15`) derives from `Exception`, while `nodes.py:236` catches `LangChainException` — that
  `except` cannot match, however it is chained. Chosen contract: the boundary raises **one typed transient failure**
  chained to the original, and **every caller with a degradation branch around a retried operation is converted** to
  catch it. A caller that is missed is a degradation branch that silently stops firing, which is why the caller
  inspection is a Proof in its own right and not a side effect of another task.
  - **Proof:** `uv run rg -n "except LangChainException" src/app/shared/langgraph_layer/` → every remaining hit also
    catches the transient-failure type (verify by reading each hit; the known sites are `nodes.py:182`, `:236`, and
    `:289`).
  - **Proof:** a unit test asserts the raised transient failure's `__cause__` is the original exception and that the
    original type and message are recoverable from it.
  - **Proof:** a unit test drives an exhausted retry through a **converted caller** and asserts that caller's
    degradation branch **executed**, and that the recorded diagnostic names the original failure reached through the
    cause. This is the scenario the old contract failed; it must pass.
  - **Proof:** a unit test asserts authentication, quota, and malformed-response failures remain distinguishable by
    their chained causes rather than collapsing into one opaque failure.

- [ ] **C7 — Add a worker process and a scheduler process to the deployment.**
  Nothing consumes the queue today, so every dispatched ingestion task enqueues forever. This is the actual blocker
  and it ranks ahead of the registration work: the queue item cannot be verified by any code-level check, so without
  the process the requirement has no proof at all.
  **Dependency — OPEN QUESTION, do not guess:** whether ingestion gets a **dedicated queue** or shares the default
  one is unanswered (`design.md` Open Question 1). The configuration forbids creating queues implicitly, so the queue
  set is fixed and this is a deliberate operational decision with a cost. Write the services with the queue list left
  as the single point of change, and record which topology was chosen when it is answered. **Do not select a topology
  by default.**
  - **Proof:** `docker compose config --services` lists a worker service and a scheduler service.
  - **Proof:** with the stack up, interrogating the running worker reports its registered tasks and the queues it
    consumes, and the ingestion task name appears among the former. This requires **no** durable outbound event
    (rule 3).
  - **Proof (topology-gated, executable either way):** the queues the worker reports consuming equal the queues the
    routing configuration routes ingestion tasks to. This Proof holds under either answer to Open Question 1; only
    the expected list changes.
  - **Proof:** a latency check — with several long-running tasks executing, a newly dispatched short task begins
    executing without waiting for them. Under a shared-queue answer this Proof is expected to **fail**, which is
    exactly the cost the open question is about; record the result rather than adjusting the check.

- [ ] **C8 — Fix the documented worker start command so it matches the deployed one exactly.**
  The documented command names an application module that does not exist. Fix it to name the real task application,
  and make the documented command and the command the deployed service runs the same string, so they cannot drift.
  - **Proof:** `uv run rg -n "\-A app|celery -A" Makefile docker-compose.yml README.md docs/` → every occurrence
    names a module that exists, and `uv run python -c "import importlib; importlib.import_module('<that module>')"`
    exits 0 for each distinct name found.
  - **Proof:** the documented command, run verbatim, starts a worker that reports its registered tasks and does not
    fail to load the application.
  - **Proof:** the string in the documentation and the string in the compose service definition are identical
    (`diff <(…) <(…)` or an equality assertion in a check script).

- [ ] **C9 — Make task registration explicit and typed, harvesting the archived registry contract.**
  The ingestion task **is** registered today, but only transitively: one package initialiser imports it, and
  importing any listed sibling imports that initialiser first. That is a **latent fragility**, and it becomes live
  precisely when change 0 tidies that initialiser — a genuine cross-change hazard where one change edits the file
  that silently guarantees another's dispatch. List every task module explicitly. Then harvest the archived typed
  task-registry contract (`openspec/changes/archive/2026-06-22-quality-fixes-batch-2/specs/celery-task-registry/spec.md`,
  absent from the live capability directory, so harvest rather than delta — the Decision 16 situation): validate a
  dispatched payload against its registered model **at dispatch time**. One deliberate tightening over the archived
  text: an unregistered name must be **reported as a failure**, not fall through permissively behind a warning, which
  is the invisible-failure shape this change exists to remove. Define each task name once and reference it from both
  the dispatching side and the declaration.
  **Dependency (change 0):** run the registration Proof **after** change 0's tidy of the task package initialiser,
  not only before. There is a window where dispatch is broken; nothing consumes the queue in that window, so it has
  no observable effect.
  - **Proof:** `uv run rg -n "include=|imports\s*=" src/app/connections/celery*.py src/tasks/` → every module
    containing a dispatched task is listed explicitly.
  - **Proof:** removing the unrelated imports from the task package's initialiser leaves the dispatched task names
    still registered (interrogate the task application in a subprocess after the removal, then restore).
  - **Proof (unit, no outbox — this is N3's fix):** invoking the dispatch helper **directly** with an unregistered
    name reports a failure naming the task rather than discarding the dispatch; invoking it for a registered name
    with a payload that does not match the declared payload raises a validation failure naming the task at dispatch
    time. Neither Proof records or relays a durable event.
  - **Proof:** `uv run rg -n '"tasks\.' src/app/ src/tasks/` → no task-name string literal outside the single
    definition module.
  - **Proof:** a declared-but-unimplemented task is registered, and invoking it fails with an explicit
    not-implemented error rather than an unknown-task error.

- [ ] **C10 — Do NOT put the pipeline graph or the checkpointer on shared application state; prove the block stays disabled.**
  The plan's original step here was to wire both onto shared application state. **That step cannot be performed.**
  The user confirmed both commented blocks are deliberate and D17 forbids re-enabling them, so this task is the
  recorded non-goal plus the check that keeps it true. Ingestion runs in the **queue worker process**, which never
  executes the application lifespan and therefore never had access to shared application state in the first place;
  the build-once requirement applies per worker process, not to application state. The synchronous ingestion surface
  that reads the shared graph stays unprovisioned and must **fail closed** with a typed service-unavailable error in
  the standard envelope — and since its router is not mounted, no service-unavailable surface ships.
  **Dependency (change 3):** the equivalent fail-closed contract for the shared **checkpointer** read site is change
  3's step 1, per D17. Do not fix that read site here.
  - **Proof:** `uv run rg -n "app.state.ingestion_graph|app.state.langgraph_checkpointer" src/app/` → every hit is
    either a read guarded to fail closed, or a commented line. No new uncommented assignment.
  - **Proof:** a unit test drives the synchronous ingestion surface with the shared graph absent and asserts a
    service-unavailable response in the standard error envelope naming the missing capability — **not** an attribute
    error, and **not** a success status.
  - **Proof:** `uv run rg -n "INGESTION_GRAPH_ENABLED|CHECKPOINTER_ENABLED" src/app/` → **no matches**. A flag
    defaulting to enabled is the forbidden thing with extra steps.
