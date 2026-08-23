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

- [x] **B1 — One embedding path replacing four, with task type declared on both sides.**
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

  **Done.** `src/app/shared/langchain_layer/embeddings.py` is the one path: `get_embedding_client()` under
  `lru_cache(maxsize=1)`, `EmbeddingTaskType.QUERY`/`.DOCUMENT` required keyword-only on both `embed_text` and
  `embed_texts`, width from `settings.EMBEDDING_DIMENSION`, `DOCUMENT_BATCH_SIZE = 100` pinned rather than inherited.
  Both `_call_embedding_fn` copies deleted; `features/search/embeddings.py` deleted via `git rm`; `embedding_fn` removed
  from `build_ingestion_graph`, `build_retrieval_graph`, `make_embed_store_node`, `make_hybrid_retrieval_node`, and the
  commented lifespan block (with a note there not to restore it). Repointed: `features/search/service.py`,
  `features/documents/service.py`, both `nodes.py`.

  **The count was six, not four, and then seven.** The design named four. Two more were found by reading:
  `langchain_layer/models.aembed_text`/`aembed_batch` — dead, and offloading the *synchronous* provider method to a
  thread while the client has native async methods. Deleting that pair orphaned a seventh,
  `_build_embedding_model_gemini_full`, which was the worst-shaped of all: it bound `task_type` at **construction**
  (`models.py:216`) and set no `output_dimensionality`, so it produced provider-default-width vectors against a
  configured-width column and could never have served a query. All three deleted. The module docstring enumerates all
  six prior paths so the count is on the record rather than in this file alone.

  **Two extensions beyond Decision 4, both deliberate, both pinned by tests.** Decision 4 names text, model, and task
  type in the cache key. Added: (a) the **configured width**, because the same model serves several widths, so without
  it a deployment that changes `EMBEDDING_DIMENSION` reads back previous-width vectors from a warm cache — the one
  failure this cache could cause that re-embedding would not fix, because a wrong-width vector looks valid; (b) **NUL
  separators**, because bare concatenation collides across field boundaries (`model="m"`+`dim=7`+`text="68:x"` and
  `model="m"`+`dim=768`+`text=":x"` both yield `mRETRIEVAL_QUERY768:x`).

  **A defect the tests found in my own module.** `from_json_float_list(str(cached))` is correct only because
  `connections/redis.py:48` sets `decode_responses=True`. Against a client without it, `str()` on `bytes` yields the
  *repr* `"b'[0.1]'"`, and the failure surfaces as a pydantic `Invalid JSON` error naming neither Redis nor the
  encoding. Fixed with `_decode_cached`; pinned by `test_a_client_that_returns_bytes_is_read_correctly`, verified
  load-bearing by reverting the guard and watching that test alone go red.

  **Retry placement unchanged, deliberately.** The unified module embeds no retry: `retry_immediate` lives in
  `shared/langgraph_layer/kb_retry`, so importing it into `langchain_layer` would invert the layer dependency, and
  C5/C6 rewrite it. Callers keep their existing `retry_immediate` wrapper, so this change alters no retry semantics.

  **`bypass_cache` reaches the embedding cache, not only the response cache.** `ask_legal` already passed
  `redis=None if payload.bypass_cache else self.redis`; the three new call sites follow that convention rather than
  inventing a second policy, so a debugger chasing a stale ranking is not served a cached query vector.

  - **Evidence — first Proof does not come back empty, and cannot.** Residual hits, each classified:
    `embeddings.py:8,10` are the docstring's own historical record of the deleted resolvers; `embeddings.py:209,261`
    are the one path's own provider calls; `rag/strategies.py` hits are all inside a commented-out block;
    `rag/document_processing/embedder.py:365` is `embed_query` in the Decision 15 carve-out; and
    `rag/rag_agent_advanced.py` has five `embedder.embed_query(...)` sites that resolve to it.
  - **Evidence — the residual is a real query/document asymmetry, scoped to E.** `embedder.embed_query` delegates to
    `generate_embedding`, which sends the module-level `GEMINI_TASK_TYPE` (`embedder.py:46`), pinned to
    `retrieval_document`. So it is a query-side call producing a document-side vector — precisely what B1 removes
    elsewhere. It is **not** fixed here: Decision 15 keeps this module out of the unified path, and A4's import guard
    (`tests/unit/shared/rag/test_rag_agent_embedder_import.py`) means deleting `embed_query` trades
    `ModuleNotFoundError` for `AttributeError` at first call. `rag_agent_advanced.py` has **no production importer** —
    its only reference is that guard — so nothing in a request or ingestion path reaches it. Step E relocates the file
    to `src/app/examples/`; that is where its five sites repoint to `embed_text(..., task_type=QUERY)` and where
    `embedder.embed_query` and `_Embedder.embed_query` are deleted. Recorded in the `embed_query` docstring so the
    finding travels with the code, not only with this file.
  - **Evidence:** `tests/unit/shared/langchain_layer/test_embeddings_unified.py` — 25 tests, all passing.
    Construction-once (`test_the_client_is_constructed_once_per_process`) plus its complement
    (`test_the_first_call_in_a_cold_process_does_construct`), so the assertion is not vacuous against a spy that never
    fires. Omission rejected two ways: by signature (`test_the_task_type_has_no_default_so_it_cannot_be_omitted`
    asserts `default is inspect.Parameter.empty` and `kind is KEYWORD_ONLY`) and at runtime
    (`test_omitting_the_task_type_raises_rather_than_guessing`).
  - **Evidence:** `test_query_and_document_vectors_for_one_text_occupy_distinct_cache_entries`, plus three key-level
    tests separating task types, widths, and models, and `test_the_separators_stop_two_different_requests_from_colliding`.
  - **Evidence:** `uv run pytest -q` → **187 passed**, up from 154 at Band A close (+25 here, +8 from B4). The 3
    failures and 9 errors are the pre-existing websocket fixture-drift set, owned by no change in this band.
  - **Evidence:** `uv run ruff check src/` and `uv run ty check src/` → **All checks passed** on both.

- [x] **B2 — Collapse the two digest-keyed embedding caches into one, and record the rejection of the framework wrapper.**
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

  **Done, and it had to land with B1.** The two byte-identical `_cached_embedding` copies shared the
  `kb:embedding:` prefix with a **text-only** SHA-256 key. That collision was harmless *only* because neither declared
  a task type — the moment one side declares one, a text-only key serves the query side a document-side vector. So B1
  alone would have been **worse than the status quo**, and the two shipped together. The new namespace
  `embedding:v1` is deliberately distinct from the prefixes it replaces, so entries written under the old keying are
  orphaned rather than read back under the new contract; they expire within `CACHE_TTL_SECONDS` and nothing deletes
  them.

  **Normalisation moved to before the cache write.** `documents/service` wrote the raw vector and normalised only the
  value it returned, so a miss produced a width-corrected vector and a hit produced the raw one — divergent exactly
  when A3's width guard matters, and dependent on cache warmth. `normalize_embedding` is idempotent
  (`utils/embedding.py:77-100` returns its input unchanged at the expected width), so it is also applied on read: a
  no-op for anything this module wrote, and a logged warning for an entry written by something else.

  - **Evidence — a third cache existed and is deleted.** The design named two. `embedder.py` held a third:
    `EmbeddingCache` (in-memory, LRU by access time, **MD5 on text alone**) plus `create_embedding_cache` and
    `create_cached_embedder`. All three had **zero references** anywhere — not in that module, not re-exported from
    `document_processing/__init__.py`, not in any test. Deleted rather than left: a dead cache of exactly the shape
    B2 forbids, in the module a future caller reaches for first, is how it comes back — the same argument applied to
    `models.aembed_text` under B1. Per-process on top of text-only keying, so it also failed the replica test that
    rejects the framework wrapper. `uv run pytest -q` → **187 passed, unchanged**, confirming it was dead.
  - **Evidence — second Proof now resolves to one implementation.** Hits remain in three files; read, they are
    `embeddings.py` (the implementation), `features/search/service.py:169` (a comment about `bypass_cache`), and
    `embedder.py:413-421` (the comment recording this deletion). One implementation.
  - **Evidence:** `test_a_repeated_text_is_served_from_the_cache_not_the_provider`, and
    `test_the_cache_is_visible_to_a_second_process` — asserted across **two `FakeRedis` clients sharing one
    `FakeServer`**, which is what makes it a shared-backend claim rather than an in-process-dict claim.
  - **Evidence:** `test_the_vector_is_normalised_before_it_is_written_not_only_on_return` pins the write-side
    normalisation; `test_batch_cache_granularity_is_per_text_not_per_batch` pins per-text granularity, so one edited
    clause does not invalidate a document's whole batch.
  - **Evidence:** `test_the_namespace_is_distinct_from_the_prefixes_it_replaces` pins the namespace change, so a
    future edit cannot quietly reuse `kb:embedding:` and start reading text-only entries.


- [x] **B3 — Stop blocking the event loop during parse, and stop discarding parsed tables.**
  `src/app/features/documents/parser.py` calls the synchronous converter at `:25` inside `async def parse_document`
  (`:19`), blocking the loop for the whole parse, and returns `tables=[]` at `:34`, discarding structure the parser
  already extracted. Offload the synchronous call; carry the tables through.
  - **Proof:** `uv run rg -n "converter\.convert\(" src/app/features/documents/parser.py` → the call is inside an
    offload, not directly awaited in the coroutine body.
  - **Proof:** `uv run rg -n "tables=\[\]" src/app/features/documents/parser.py` → **no matches**.
  - **Proof:** a unit test asserts the event loop remains responsive across a parse (schedule a second coroutine and
    assert it runs before the parse completes) and that a fixture document with a table yields a non-empty table
    collection. **Mandatory** — this defect is invisible to lint and types.

  **Done.** The conversion body moved into a nested `_sync_parse` returned through
  `asyncer.asyncify`. That idiom rather than `asyncio.to_thread` deliberately: `ingestion_kb/nodes.py:420-452` already
  offloads the same converter the same way, and change 2 consolidates these two paths — arriving there with two
  offload idioms would make that a reconciliation instead of a deletion.

  **The converter is still built per call, and that is a decision, not an omission.** Making it a cached singleton is
  the obvious next optimisation and is wrong to do in the same step, because the offload is what first makes
  concurrent parses possible and `DocumentConverter` holds mutable pipeline state docling does not document as
  thread-safe. Recorded in the `parse_document` docstring so the next reader does not "fix" it.

  - **Evidence — the table discard was in three places, not one, and B3 named the least harmful.** All three called
    `table.to_markdown()`. `TableItem` has no such method; it is `export_to_markdown`. Verified directly:
    `hasattr(TableItem, "to_markdown")` is `False`, and `inspect.signature` gives
    `export_to_markdown(self, doc: DoclingDocument | None = None)`.
    1. `parser.py:34` — an empty literal. Discarded openly; the one B3 describes.
    2. `ingestion_kb/nodes.py:433-437` — the comprehension was guarded by `hasattr(table, "to_markdown")`, so the
       guard was **false for every table** and the KB ingestion path returned an empty list for every document it has
       ever parsed. Worse than `parser.py`'s version, because the guard made a wrong method name read as defensive
       handling of an optional one. This is the path B3 would have been told to copy.
    3. `docling_enhanced.py:84` (`extract_tables`) — called it bare. `AttributeError` is **not** a subclass of
       `docling.exceptions.BaseError` (verified: `issubclass(AttributeError, BaseError)` is `False`), which is what
       `except DoclingError` binds, so this one **raises uncaught** on any document with a table.
       `DoclingEnhancementConfig.extract_tables` defaults to `True` (`models.py:226`), so it is on by default. Latent
       rather than live: `convert_document` and `extract_tables` have no caller outside their own package.
  - **Evidence — one expression now, in one place.** `docling_enhanced.table_markdown(doc)` passes `doc=` rather than
    omitting it: without it the call logs a deprecation warning and falls back to walking `self.data.grid`, which
    cannot resolve what a cell refers to (read from `export_to_markdown`'s source, not inferred). Imported from the
    leaf module in both consumers, matching the leaf-import idiom `319c698` established.
  - **Evidence — `extract_tables`'s except clause deliberately still does not catch `AttributeError`.** Widening it
    was the tempting fix and would have been the wrong one: it converts the crash into `nodes.py`'s lie, logging
    "Failed to extract table 3" for every table of every document. `(DoclingError, ValueError, IndexError, KeyError)`
    is what a malformed cell grid raises — a per-table problem worth skipping. A missing attribute is a per-build
    problem worth crashing on.
  - **Evidence:** `tests/unit/features/documents/test_parser_offload_and_tables.py` — 7 tests, all passing.
    Responsiveness is asserted on **interleaving, not duration**: the fake converter blocks on `time.sleep` (which
    `asyncio.sleep` is not), a competing coroutine sets an `asyncio.Event` with no prior await, and the assertion is
    that the flag is set by the time the parse returns. A wall-clock threshold would measure the runner instead of the
    code. `test_two_parses_overlap_rather_than_serialising` adds the claim the first test cannot make: an offload to a
    single serialising worker would pass it and still queue uploads.
  - **Evidence — the fake `TableItem` mirrors the real class's asymmetry.** It defines `export_to_markdown` and
    deliberately does *not* define `to_markdown`. A fake offering both names would have passed against the broken
    code and proved nothing.
  - **Evidence — both mandatory claims mutation-tested.** Replacing `await asyncer.asyncify(_sync_parse)()` with
    `_sync_parse()` fails exactly the two responsiveness tests and no others; reverting `tables=` to an empty literal
    fails exactly the two table tests and no others. Each guard is load-bearing, and each fails for its own reason.
  - **Evidence:** `uv run pytest -q` → **194 passed** (187 + 7). `uv run ruff check src/` and `uv run ty check src/` →
    **All checks passed**.
  - **Note — the second Proof needed the codebase's own convention applied to my comment.** The replacement comment
    first quoted the literal the proof greps for, which defeated it. Described instead, the way
    `embedder._provider_failure` already does for A2's zero-vector grep.
  - **Note — no `docs/relay/b3-*.md`.** B3's first subagent died on the same API error class as B4's, leaving **no**
    file changes at all (verified by `git status` at the time). Redone here directly; this block is the record.


- [x] **B4 — Cache the token counter, and record why the transformer dependency cannot be dropped.**
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

  **Done.** `chunker.py` gains `DEFAULT_TOKENIZER_MODEL_ID`, `_TOKENIZER_CACHE_SIZE = 4`, an
  `@lru_cache(maxsize=_TOKENIZER_CACHE_SIZE) def _load_tokenizer(model_id)`, and a signature-compatible
  `get_tokenizer(model_id=DEFAULT_TOKENIZER_MODEL_ID)` wrapper. The wrapper is **not** decoration: a default argument
  is not part of an `lru_cache` key, so `get_tokenizer()` and `get_tokenizer(DEFAULT)` would hash to two entries and
  load the tokenizer twice. Normalising in the wrapper and memoising the inner function is what makes "once" true for
  both call shapes.

  **Chose Decision 3's second option: state the divergence.** Matching the counter to the embedding model was
  rejected because the real bound on the embedding side is not a token bound at all. `embedder.py:146,210` applies
  `_MAX_INPUT_TOKENS * 4` as a **character** limit, by silent truncation. Against a 512-token chunk budget that leaves
  roughly 4x headroom, and the divergence degrades in only one direction: WordPiece maps an unsegmentable run of up to
  100 characters onto a single unknown token, so the local counter can *under*-count a pathological input, never
  over-count a normal one. Recorded in the module docstring with the computed margins, because a safety margin nobody
  can find is not a safety margin.

  - **Evidence:** `tests/unit/shared/rag/test_chunker_tokenizer_cache.py` — 8 tests, all passing. Construction is
    counted by a `_ConstructionSpy` substituted for the module-global `AutoTokenizer`, with an autouse fixture
    clearing the cache on **both** entry and exit — entry because a prior test could have warmed it, exit because a
    later one would inherit a spy that no longer exists.
  - **Evidence:** the fake tokenizer subclasses `PreTrainedTokenizerBase` and overrides `__len__`, which is not
    incidental: Docling's legacy tokenizer-coercion path evaluates the tokenizer for **truthiness**, and a subclass
    with an empty vocabulary is falsy, so without the override the fake is silently rejected and the test measures
    nothing.
  - **Evidence — beyond the task, correctness fix.** Six annotations named `AutoTokenizer` as the *type* of a
    tokenizer. `AutoTokenizer` is a **factory class**: `from_pretrained` returns a `PreTrainedTokenizerBase` subclass
    instance, never an `AutoTokenizer`. All six corrected to `PreTrainedTokenizerBase`. `ty` accepted both, because
    the wrong one was never contradicted by a call — it was wrong documentation that typechecked.
  - **Evidence:** `uv run rg -n "AutoTokenizer|from_pretrained" src/app/` → two hits, both in `chunker.py`: the import
    (`:55`) and the single call inside `_load_tokenizer` (`:87`). No uncached acquisition remains.
  - **Note — no `docs/relay/b4-tokenizer-cache.md`.** The B4 subagent died on an API error
    (`StreamNoEventsError`, HTTP 200 with an empty body) after its work was green but before writing its report, the
    same failure class that killed B3's. Its diff and test file were verified directly rather than taken on faith;
    this block is the record.


---

## Band C — the runtime substrate: checkpointer, retries, worker

Band C changes whether a crash is recoverable and whether the queue can be consumed at all. Every checkpointer Proof
here is import-level, type-level, or a unit test over a construction the test owns — see rule 4.

- [x] **C1 — Install the client-library binary binding and delete the placeholder alias, in one commit.**
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

  **Done.** `psycopg[binary,pool]>=3.3.3` declared; the alias assignment and its `except ImportError` deleted. All four
  Proofs pass — the import prints the real class, both greps are empty, `uv lock --check` exits 0, and no type
  suppression of any kind remains in the file.

  - **Evidence — the premise was verified before acting rather than assumed.** With the binding absent, `import psycopg`
    raises `ImportError: no pq wrapper available`: no `psycopg_c`, no `psycopg_binary`, and no system libpq on this
    machine. So the fallback genuinely was the live path and genuinely was the only reason the application booted, which
    is what makes the one-commit constraint real rather than stylistic. `psycopg.pq.__impl__` now reports the binary
    implementation.
  - **Evidence — `pool` was declared too, beyond what C1 asks for.** The rewritten module imports `psycopg_pool`
    directly (it owns its own `AsyncConnectionPool`), and that package reached the environment only as a transitive
    dependency of `langgraph-checkpoint-postgres`. A direct import behind someone else's declaration breaks silently the
    day that dependency reorganises — the same hazard class B4 recorded for `transformers`, which is also declared in
    this commit, closing it.
  - **Evidence — the lock diff adds one distribution, not a re-resolution.** `psycopg-pool` was already in `uv.lock`
    transitively, so the `pool` extra records an existing fact; `psycopg-binary==3.3.3` is the only genuinely new
    package. `uv lock --check` exits 0 both before and after.
  - **Warning for anyone re-running this — a bare `uv sync` uninstalls the test toolchain in this repo.** `pytest-asyncio`
    is declared only in the `test` extra and the `test` dependency group, while `[tool.uv] default-groups = ["dev"]`, and
    the `dev` group holds `ruff`/`ty`/`hypothesis` but no pytest. A plain `uv sync` therefore prunes it, and pytest then
    reports `ERROR: Unknown config option: asyncio_mode` and collects **zero** tests — which reads as "the change broke
    the suite" and is not. Restore with
    `uv sync --extra dev --extra test --group dev --group test`. Nothing here needed a sync at all: `uv lock --check`
    passing already meant the resolution was unchanged.

- [x] **C2 — Fix the constructor call: setup must yield a usable saver, never an unentered resource manager.**
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

  **Done.** Setup now owns an `AsyncConnectionPool`, hands it to the saver constructor directly, runs the migrations, and
  either returns the saver or raises with the pool already closed. All three Proofs pass.

  - **Evidence — the defect is worse than C2 describes: that classmethod cannot produce a long-lived saver at all.**
    C2 diagnoses the call as returning an unentered context manager, which is true and is the visible bug. Reading the
    source shows the deeper one: the body is `async with await AsyncConnection.connect(…) as conn: … yield cls(conn=conn,
    …)`, so the connection is closed when the block exits. Entering the manager *correctly* would still hand back a saver
    whose connection dies at the end of the `with`. There is no arrangement of `async with` that yields a saver outliving
    the block, which is why this is a rewrite that owns the pool rather than a two-line fix. The classmethod's name is
    described rather than written throughout the module because C2's first Proof is a grep for it.
  - **Evidence — the old exception handler was unreachable for every failure it named.** It caught
    `(ConnectionError, TimeoutError, OSError)`; `psycopg.Error.__mro__` is `(Error, Exception, BaseException)` and it
    derives from none of the three. So no driver failure was ever handled there, and the `AttributeError` C2 predicts was
    not the only thing falling through — the clause was decorative. The replacement catches `psycopg.Error`.
    `PoolTimeout` deliberately gets no clause of its own: it derives from `psycopg.OperationalError`, so naming both
    would imply they were siblings and invite someone to "complete" the tuple.
  - **Evidence — owning the pool means inheriting three connection settings, and one of them is load-bearing.** The
    library's own helper connected with `autocommit=True`, `prepare_threshold=0`, and `row_factory=dict_row`. That last
    one is not stylistic: `setup()` reads `row["v"]` off its migration query, which a tuple row cannot answer. A pool
    built without them yields a saver that imports, constructs, and then fails *inside* its migration. Not one of C2's
    stated Proofs — added as `test_the_connection_settings_the_saver_depends_on_are_passed_on`, because it is precisely
    the defect this fix is most likely to introduce.
  - **Evidence — the failure paths close the pool, and the guard is a `finally`, not the `except`.** A `pool_handed_off`
    flag is set only on the successful return; the `finally` closes otherwise. This covers what the `except` cannot: an
    unexpected exception type, and cancellation inside `open()` — a shutdown racing startup would otherwise leave a pool
    holding live server connections with nothing referencing it. Two tests cover the two distinct leak paths (open
    failed; open succeeded and the DDL failed), because a `try` around only the constructor passes the first and fails
    the second.
  - **Evidence — the test never patches the saver class.** C2's claim is that setup returns an instance of the *real*
    class; a test that substituted the class could not tell that apart from returning a stand-in. Only the pool class and
    the migration step are replaced.

- [x] **C3 — Consume the shared accessor for the plain client-library URL flavour; repair nothing at the call site.**
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

  **Done.** The module calls the accessor's plain flavour and does nothing to the result. All three Proofs pass; no
  credential, host, or port was printed by any of them.

  - **Evidence — the docstring C3 says is wrong was already fixed, by change 0.** C3 states the module docstring "names
    the dialect-aliased scheme and is **wrong**". It does not. `git log -p` shows change 0's commit `79a1d95` already
    replaced that wording with the correct plain/libpq description. This is the **eighth** instance in this refactor of a
    plan step describing work that was already done — the pattern is now reliable enough to budget for. The docstring was
    rewritten anyway, but to record the two-strings-neither-will-do reasoning and the D17 warning, not to fix an error.
  - **Evidence — C3's grep Proof conflicted with the first implementation, and the code was improved rather than the
    Proof bent.** The scrubber initially used `urlsplit(dsn).password` and `.replace(…)`; the Proof greps for `split(`
    and would have failed on it. Switching to `conninfo_to_dict(dsn).get("password")` plus
    `re.sub(re.escape(form), …)` satisfies the Proof *and* fixes a real latent gap: a DSN may legally be keyword-value
    form rather than a URL, and against `host=… password=…` a URL parser finds no credential and the scrubber silently
    scrubs nothing. The switch then forced a second finding — the driver's parser returns the secret **percent-decoded**
    while the DSN carries the encoded form the accessor wrote, so an error echoing either one has to be matched. Both
    encodings are scrubbed, and the test's fixture secret is chosen to percent-encode so that a scrubber knowing only one
    of the two fails.
  - **Evidence — the project's log redaction does not cover this, and looks like it does.** `redact_sensitive_data`
    (`src/app/utils/logger.py:82`) iterates `record["extra"]` and blanks entries whose **key name** contains
    password/token/credit_card/secret. It never inspects a value and never touches the message. So `error=str(e)` passes
    through entirely untouched — and the driver reports a connection failure by quoting the connection info it failed on.
    A redaction mechanism that appears to cover a case and does not is worse than none, because it stops anyone from
    looking. Hence a local scrub at the two log sites, with the reason recorded in its docstring.
  - **Evidence — the credential test needed a loguru sink, not `caplog`.** `caplog` does not see loguru at all, and the
    interesting failure is a credential arriving as a *value* under an innocuous field name. The fixture adds a sink with
    `format="{message} | {extra}"`; rendering `{extra}` is what makes that visible to an assertion.
  - **Evidence — the assertion reports a line index and never the line.** A plain `assert secret not in line` prints the
    offending content on failure, which would write the credential into the test output and turn a leak into a leak that
    is also archived in CI. Reported as "log line *N* carries the decoded credential" instead. This is rule 4 applied to
    a test's own failure message, which is a place it is easy to forget.

- [x] **C4 — Make teardown report which of its three outcomes occurred.**
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

  **Done, with four outcomes rather than three.** Teardown returns a `CheckpointerTeardown` `StrEnum` — returned rather
  than only logged, so a caller can act on it and a test can assert on it without scraping log records. All three Proofs
  pass; the D17 grep still matches a commented line, and both `app.state.langgraph_checkpointer =` hits in the tree
  (`lifespan.py:322`, `:328`) remain commented.

  - **Evidence — the fourth outcome, `CLOSE_FAILED`, is deliberate and not scope creep.** C4 names three: closed a pool /
    nothing was provisioned / handed something with no pool to close. Reporting a *failed close* as the third would
    rebuild the exact conflation this task exists to remove — one of those says no action was needed, the other says an
    action was attempted and did not work, and only the second is worth waking anyone for.
  - **Evidence — the old guard was betting on an attribute no saver the library can build has.**
    `AsyncPostgresSaver.__init__` sets `conn`, `pipe`, `lock`, `loop`, `supports_pipeline` — and `serde`, which C4's list
    omits — and nothing named `pool` exists on it or on either of its bases. So the pool went unclosed on **every**
    shutdown, and silently, because a false `if` fell straight through to a successful return. `conn` is the attribute
    that holds it, and it is a union: `AsyncConnection | AsyncConnectionPool`. The replacement is an `isinstance` against
    the pool class, and the bare-connection case reports rather than closing, because a connection someone else opened is
    not ours to close. `test_the_saver_has_no_pool_attribute` pins that fact as a regression test, so a future library
    version adding the attribute surfaces as a decision to revisit rather than as behaviour that quietly changes.
  - **Evidence — C4's line numbers have drifted; the Proof survives it.** Teardown is called at `lifespan.py:339-340`,
    not `:317`, and the commented setup block is `:312-328`, not `:294-305`. The D17 grep is anchored on the line's text
    rather than its number and still matches, at `:319` — which is why it was written that way.
  - **Evidence — the test's pool double is a real `AsyncConnectionPool` subclass, and it has to be.** Teardown decides
    what it may close with an `isinstance` check, so an unrelated look-alike double would be reported as "nothing to
    close" and the test would pass for the wrong reason. `__init__` deliberately does not call `super().__init__`, which
    would start background workers and try to reach a server.

  - **Evidence — `tests/unit/shared/langgraph_layer/test_checkpointer_lifecycle.py`, 14 tests, all passing, and every
    guard mutation-tested.** A passing test is not evidence a guard works. Each mutation was applied to the working copy,
    the file restored afterwards, and each killed exactly the tests it should and no others:

    | Mutation | Tests killed | Which |
    |---|---|---|
    | `_scrub(str(e), dsn)` → raw `str(e)` | 1 | the failed-setup credential test; the captured log visibly carried the DSN |
    | `isinstance(held, AsyncConnectionPool)` → the old attribute guard | 3 | both close tests and the teardown credential test |
    | drop the `finally` cleanup | 2 | both pool-leak tests |
    | `CLOSE_FAILED` → `NO_POOL_TO_CLOSE` | 2 | both close-failure tests |
    | drop `_CONNECTION_KWARGS` | 1 | the connection-settings test |

    Two tests exist for what looks like one claim because the defect they cover was hidden by a **type suppression**
    rather than by a missing branch: the old function was annotated as returning a saver and returned `None` down two
    paths, each with the type checker silenced on the line. "Does it ever return `None`" is therefore asserted here rather
    than delegated to `ty`, which had already been told not to look.

  - **Gate at the close of C1–C4:** `uv run ruff format src/` → 360 files unchanged · `uv run ruff check --fix src/` →
    All checks passed · `uv run ty check src/` → All checks passed · `uv run pytest -q` → **208 passed** (194 + 14), with
    the same 12 pre-existing websocket fixture failures (3 failed + 9 errors) owned by no change in this band. The tree
    also carried in-flight C5/C6 and C9 work when this was measured, so the four test files belonging to those tasks were
    excluded to make the count attributable; their `src/` edits were left in place and break no baseline test.

- [x] **C5 — Correct the retry policy: named transient types, growing wait, no catch-all.**
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

  **Done.** `kb_retry.py` rewritten, 46 → 279 lines. All three Proofs pass. Proof 1 returns no matches. Proof 3 was
  performed by reading all 22 sites: none wraps a node body, and the restriction is now recorded in the function's own
  docstring, because it is what makes the pause exclusion *sufficient* rather than merely correct.

  - **Evidence — the Proof 3 count reconciles to 22, not the 27 a raw grep reports.** `rg -c "retry_immediate"`
    returns 27 lines: 1 definition, 4 imports, 22 calls. Recorded so the next reader does not conclude four sites went
    missing.
  - **Evidence — the pause hierarchy was verified against the installed package, not assumed.**
    `GraphBubbleUp.__mro__` is `(GraphBubbleUp, Exception, BaseException, object)` and
    `issubclass(GraphInterrupt, GraphBubbleUp)` is `True`. Two consequences follow from that single fact, and they pull
    in opposite directions: a predicate keyed on `Exception` **does** match a pause, so the old catch-all retried it —
    re-running side effects that had already landed, then relabelling the pause as an external error, at which point
    the graph never pauses and the resume value is lost. `GraphBubbleUp` is now excluded from the retryable set **and**
    from the wrapping, in that order, ahead of every positive test, so widening a later test cannot re-capture it.
  - **Evidence — `jitter < initial` is load-bearing for the test, and the bound was checked arithmetically rather than
    empirically.** `wait_exponential_jitter(initial=0.5, max=8.0, jitter=0.25)` draws attempt *n* from
    `[0.5·2ⁿ, 0.5·2ⁿ + 0.25]`. Attempt *n*'s **maximum** is strictly below attempt *n+1*'s **minimum** at every step
    (0.750 < 1.000, 1.250 < 2.000, 2.250 < 4.000, 4.250 < 8.000), so the sequence increases for *every possible draw*.
    The monotonicity assertion is non-flaky by construction, not by tolerance.
  - **Evidence — no test measures elapsed time, and the interception seam is not the obvious one.** tenacity assigns
    `self.sleep` as an **instance** attribute in `AsyncRetrying.__init__` and `__anext__` awaits `self.sleep(do)`, so
    patching a module-level sleep would intercept nothing. The test patches the *class* in `kb_retry`'s own namespace
    with a subclass that swaps its sleep for a recorder; the real wait strategy, stop condition and predicate all still
    run, so the recorded durations are the ones production would have slept for rather than durations the test chose.
    Monotonicity is asserted across three gaps (4 attempts), because a single comparison can pass by coincidence in a
    way a chain cannot.
  - **Evidence — the retryable set is named rather than inherited, and the status check exists to avoid an import.**
    `TimeoutError`, `ConnectionError`, `httpx.TransportError`, redis `ConnectionError`/`TimeoutError`, SQLAlchemy
    `OperationalError`/`InterfaceError`, `OutputParserException`, plus a structural status check against
    `{408, 425, 429, 500, 502, 503, 504}` so a quota refusal is retried **without importing a provider SDK into a
    shared boundary**. `_status_code_of` reads `status_code`, then `code`, then `response.status_code`, and accepts
    only `int` excluding `bool` — several libraries use `code` for a *string* identifier, and a string compared against
    a set of integers silently never matches, which is indistinguishable from "not retryable". 401/403/404/422 are
    pointedly absent: credentials do not become valid by waiting.
  - **Evidence — `TransientExternalError` is itself non-retryable (`kb_retry.py:192`).** Without that, nesting two
    boundaries would multiply the attempt budget by the nesting depth.
  - **Evidence — the suppression was removed, not relocated.** The original module carried
    `# ty: ignore[unresolved-attribute]`; the lambda became `_log_before_sleep(label, attempts)`, which checks
    `state.outcome is not None` instead of asserting it. This is the stale-baseline hazard this work has already been
    bitten by once: a moved suppression keeps a gate green while the thing it hid moves house.
  - **Evidence — every guard mutation-tested, each killing exactly its intended tests.**

    | Mutation | Newly red | What it proves |
    |---|---|---|
    | M1 `_is_transient` → `return True` (the old catch-all) | 7 | every "does not retry" claim, including all three pause tests |
    | M2 `initial=0.0, jitter=0.0` (the old no-wait) | 2 | the wait defect is invisible to every other assertion — which is why it survived |
    | M3 drop `if not _is_transient(exc): raise` | 8 | every "must not be relabelled" claim, across both test files |
    | M4 un-convert the segmentation caller | 2 | the caller conversion is load-bearing independently of the boundary fix |
    | M5 `describe_failure(exc)` → `str(exc)` | 1 | the mutation's own output is the argument for the helper — see C6 |

- [x] **C6 — Raise one typed transient failure at the boundary, and convert the callers that catch around it.**
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

  **Done, and the conversion is a *tuple*, not a replacement.** Three callers converted in
  `ingestion_kb/nodes.py`; 9 tests in `test_kb_transient_boundary.py`. Proofs 2, 3 and 4 pass as written. **Proof 1's
  stated expectation cannot hold and is amended below.**

  - **Evidence — Proof 1 asks for something a correct fix makes impossible; the command survives, the expectation
    inverts.** The Proof asks that every remaining `except LangChainException` hit *also* catch the transient type. A
    correct conversion writes `except (LangChainException, TransientExternalError)`, which **deletes the exact string
    the Proof greps for**. A fully converted codebase returns *fewer* hits, never annotated ones — so as written the
    Proof can only ever surface **unconverted** sites. **Amendment:** keep the command, invert the expectation to
    "every hit is a site that has *not* been converted; classify each", and add the positive command
    `rg -n "except \(LangChainException, TransientExternalError\)" src/app/shared/langgraph_layer/`, expecting **3**
    hits in `ingestion_kb/nodes.py`. Both were run: the positive command returns exactly 3, and the original returns 2
    residuals, classified below.
  - **Evidence — the tuple, not a replacement, because two routes now reach one branch and both had to stay open.** A
    **deterministic** framework failure is outside the named retryable set, so it arrives unretried and as its own type
    — the original `except` still matches it. A **transient** failure is retried and, once the budget is spent, arrives
    as `TransientExternalError` with the original recoverable through `__cause__`. Replacing rather than extending
    would have traded one silently-dead degradation branch for another. Written literally at each site rather than
    hoisted into a named constant, so `LangChainException` stays lexically visible where a future reader decides what a
    branch catches.
  - **Evidence — the reason the old remedy could not work is recorded in three places, deliberately.** Chaining
    populates `__cause__`; it does not change the type of the object raised, so `TransientExternalError` is not an
    instance of `LangChainException` however carefully the chain is built. That reasoning sits in `kb_retry.py`'s module
    docstring, in the C6 test module's docstring, and in an 11-line comment at the first converted site — because it is
    the kind of "obvious" fix that gets re-proposed.
  - **Evidence — the task's own line numbers were already stale, and the Proof's read-each-hit instruction is what
    caught it.** The task names `nodes.py:182`, `:236`, `:289`. Measured **pre-edit**: 182, **248**, **305**.
    **Post-edit**: **197**, **265**, **324**. Verified independently after the fact — `rg -c` on the converted pattern
    returns 3, at exactly those lines.
  - **Evidence — two residual `except LangChainException` hits, one a genuine instance of the same defect.**
    `retrieval_kb/nodes.py:118` wraps a `retry_immediate` call and degrades on `LangChainException`, so its degradation
    branch **cannot fire for a wrapped transient failure** — the identical defect, unfixed, needing the identical
    one-line change. It sits outside C6's file scope; it needs either an explicit extension of that scope or its own
    task. `open_deep_search/graph.py:332` is out of scope by **D7** and was confirmed harmless by reading it: no
    `retry_immediate` anywhere in that module, so no wrapped failure can reach that `except`.
  - **Evidence — seven further unconverted callers exist beyond the task's list, found by sweeping every
    `retry_immediate` site for an enclosing degradation branch.** `retrieval_kb/nodes.py:118` and `:155` (the latter
    catches `GraphitiError`, same defect against the retried `graphiti.search`); `documents/service.py:733`, `:767`,
    `:797`, `:816` and `documents/legal_metadata.py:76` catch `(ValueError, TypeError)`. The four in `service.py` are
    the subtle ones: they **do** still catch a pydantic `ValidationError` raised by `model_validate` *outside* the
    retry, so those branches are not wholly dead — but they can never catch a retry-exhausted provider failure. **A
    half-live branch is harder to notice than a dead one.** None was fixed; all sit outside the exclusive file list.
  - **Evidence — `describe_failure(exc)` was added because the degraded record was a diagnosis with no diagnosis in
    it.** Without it a degraded record reads `"gemini_segment_document failed after 3 attempts"`. It walks `__cause__`
    joining with `" <- caused by "`, carrying an `id()` seen-set, because `__cause__` is caller-assignable and a cycle
    would hang the logging path — a degradation handler is the last place that can afford to be the thing that fails.
    Mutation M5 replaced it with `str(exc)` and produced exactly the uninformative string above.
  - **Evidence — an edit outside the named exclusive files: A5's tripwire was retired, on its own instructions.**
    `test_ingestion_degraded_identity.py`'s `test_the_boundary_still_converts_the_type_the_handler_catches` asserted the
    *defect* and its own docstring designated C6 as owner. Fixing C6 turned it red, correctly — a bare
    `LangChainException` is no longer in the transient set, so it propagates unwrapped and
    `pytest.raises(TransientExternalError)` stops matching. **Considered and rejected:** adding `LangChainException` to
    the transient set to keep it green — that base also covers deterministic configuration errors ("model not
    configured for structured output" is not a thing waiting fixes), so it would re-create the catch-all one level down.
    `OutputParserException` is named individually instead, and `test_the_framework_base_exception_alone_is_not_retryable`
    pins that distinction so the tempting widening cannot land silently. Replaced with
    `test_the_boundary_now_reaches_the_handler_by_both_routes`, plus two now-false paragraphs of that file's module
    docstring corrected.
  - **Evidence — three `except Exception` handlers would swallow a pause one frame after the boundary protects it.**
    `retrieval_kb/nodes.py:255`, `:298`, and `ingestion_kb/nodes.py:791` (`_graphiti_add_episode`). These are not
    *broken* — `except Exception` catches `TransientExternalError` fine — so they need no conversion. But
    `ingestion_kb/nodes.py:791` sits directly around a retried call whose boundary now guarantees a pause escapes
    intact, and then catches it anyway. Deliberately not fixed: narrowing a blanket handler is a behaviour change with
    its own blast radius and no C5/C6 Proof covers it. **Belongs with change 3**, which is when a pause first exists to
    be swallowed.

- [x] **C7 — Add a worker process and a scheduler process to the deployment.**
  Nothing consumes the queue today, so every dispatched ingestion task enqueues forever. This is the actual blocker
  and it ranks ahead of the registration work: the queue item cannot be verified by any code-level check, so without
  the process the requirement has no proof at all.
  **Dependency — OPEN QUESTION, now ANSWERED (2026-08-23):** whether ingestion gets a **dedicated queue** or shares
  the default one was unanswered (`design.md` Open Question 1). The configuration forbids creating queues implicitly,
  so the queue set is fixed and this was a deliberate operational decision with a cost. **The answer: a dedicated
  ingestion queue with its own concurrency, and its own worker service.** `design.md` records it under "Closed since
  the first draft" with the consequences. The topology chosen is therefore the dedicated-queue one, and every Proof
  below is evaluated against it.
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

  **Done, with two Proofs amended in place and the reasons recorded.** Three services now exist — `celery-worker`
  (`-Q default`), `celery-worker-ingestion` (`-Q ingestion`), `celery-beat` — all three commands derived from the
  single `Makefile` definition site C8 established, and `tests/unit/celery/test_queue_topology.py` (11 tests) pins the
  topology. C8's armed Proof 3 is now satisfied rather than skipped. The whole directory is `69 passed`.

  - **Evidence — Proof 1 satisfied.** `docker compose config --services` lists seven services, three of them Celery:
    `celery-worker`, `celery-worker-ingestion`, `celery-beat` — a worker service and a scheduler service, as required.
    `docker compose config --format json` resolves their commands verbatim, so the strings are what Docker would run
    and not merely what the file reads as.
  - **Evidence — the answered topology, and why prefetch is not a substitute for it.** Ingestion is minutes of model
    work per message; the default queue carries sub-second billing, credit and transactional-email tasks. One shared
    pool makes those wait behind ingestion whenever every slot is busy, and `worker_prefetch_multiplier=1` does **not**
    prevent it: prefetch stops one worker hoarding messages off the broker and says nothing about head-of-line blocking
    once every slot is already occupied. Two queues with two disjoint consumer sets is what removes the coupling. That
    reasoning is recorded at each site it constrains — `settings.py`, `celery.py`, the `Makefile`, `docker-compose.yml`
    and the two test modules — because a future reader deleting one worker service to save a container needs the cost
    in front of them.
  - **Evidence — a live defect found while writing the services: a worker with no `-Q` drains the dead-letter queue.**
    Measured, not reasoned: with no queue selection, `celery_app.amqp.queues.consume_from` equals the **entire**
    declared set — `['default', 'default.dlq', 'ingestion']`. The command documented before this task carried no `-Q`,
    so it turned the dead-letter queue into a second inbox and re-ran precisely the messages that had been parked for a
    human to look at, while reporting itself healthy. `-Q` on every worker command is therefore a fix, not tidiness,
    and `test_no_deployed_worker_consumes_the_dead_letter_queue` asserts it **with the hazard as its own positive
    control** in the same test body, so it cannot be read as defending against something that does not happen.
  - **Evidence — the routing table was rewritten, because 11 of 16 names were never explicitly routed.** The single
    route was a `tasks.*` glob, which matches only the five `tasks.*` names; every `auth.*`, `billing.*`, `credits.*`
    and `document_extraction.*` name reached the default queue through `task_default_queue` instead. Two mechanisms
    delivering to one queue read as one mechanism right up until a name needs a different queue — which is exactly what
    C7 does. This change's own spec requires it (`specs/celery-worker-deployment/spec.md`, "Routing is explicit for
    every dispatched task": every dispatched name SHALL resolve to an explicitly configured destination rather than an
    implicit default), and C7 is the last task in the change, so nothing else would have implemented it. `task_routes`
    is now derived from `TASK_DECLARING_MODULES` with membership taken from `INGESTION_TASK_NAMES`; all **16** names
    resolve with `router.lookup_route(...) is not None`, and behaviour for the eleven is unchanged — they reached the
    default queue before and after.
  - **Evidence — a library-precedence fact checked rather than assumed, and then designed around anyway.**
    `celery.app.routes.MapRoute.__init__` partitions its mapping into exact keys (`self.map`) and globs
    (`self.patterns`, via `fnmatch.translate`), and `__call__` consults `self.map` **first** — so exact task names beat
    a glob regardless of dict insertion order. Verified by constructing both orderings and resolving. The glob was
    deleted regardless: that is a fact about one library version, and a config whose correctness depends on it is a
    config that breaks on upgrade with no test to say why. Each name also gets its **own** route dict, because
    `Router.expand_destination` **pops** `queue` out of the dict it is handed.
  - **Evidence — Proof 2 AMENDED: registration and consumed queues proven without a running worker.** As written the
    Proof requires interrogating a running worker, i.e. a **consuming** worker, and the configured broker is a live
    managed instance whose registered task set includes `billing.*`, `credits.*` and `auth.send_password_reset_email` —
    such a worker could execute real queued work including sending mail to real recipients. C8 recorded that
    non-execution deliberately and C7 inherits it. **Amended form, which exercises the same machinery:**
    `app.loader.import_default_modules()` — precisely what a worker performs at boot — registers all **16** declared
    names, `tasks.documents_ingest` among them; and the queues each deployed worker consumes are read from the `-Q`
    text of the commands Docker resolves, which is the same string the worker would parse. No broker connection is
    opened anywhere in this task. Both halves are asserted in tests rather than left as a one-time observation.
  - **Evidence — Proof 3 satisfied, as set equality in both directions.** The queues tasks route to and the queues the
    deployed workers consume are compared as sets: `{'default', 'ingestion'}` both sides. Equality rather than
    containment because both failure directions are silent — a routed queue nobody consumes accepts messages forever
    and nothing runs them, which is the state this whole change began from; a consumed queue nothing routes to gives a
    worker that reports itself healthy and processes nothing. The two sides have **no mechanical link** (compose cannot
    read the application's settings), so this is the drift guard for a hand-maintained agreement, and mutations M7 and
    M8 break it from each side in turn.
  - **Evidence — Proof 4 AMENDED: the starvation property is asserted structurally, which is strictly stronger than the
    latency check.** As written the Proof needs a broker and two *consuming* workers — the same safety bar as Proof 2.
    The property that check would observe is that the two worker pools' queue sets are **disjoint**: neither pool's
    slots can ever be occupied by the other pool's messages, so there is no ordering, timing or load under which one
    delays the other. `test_the_two_worker_pools_share_no_queue` asserts exactly that, plus that the ingestion queue and
    the latency-sensitive queue are different queues and that only one pool consumes ingestion. A single latency
    measurement would have been one observation of a property that now holds universally. The Proof's own note — "under
    a shared-queue answer this Proof is expected to fail, which is exactly the cost the open question is about" — is
    what the answer resolved: the dedicated-queue topology is the one under which it holds.
  - **Evidence — C8's `len(documented) == 1` had to be generalised, and was strengthened rather than relaxed.** C7
    introduces three deployed commands, so `uv run celery ` now appears three times in `README.md`. Weakening the
    assertion to `>= 1` would have let an undocumented or drifted command through. Instead: the set of documented
    commands must **equal** the set the `Makefile` defines (count still pinned — to the definition site's own count
    rather than to the literal 1, and now catching a documented command that no longer exists), every celery command in
    any of the three files must resolve to the same `-A` value, every deployed command must be exactly one of the
    defined ones, exactly one defined command is the scheduler, and every worker command carries `-Q`. Three exact
    strings across three files where there was one across two. `CELERY_WORKER_CMD` itself is left byte-identical so
    `test_the_makefile_command_needs_exactly_one_substitution` keeps meaning what it says; the three deployed commands
    are derived from it, and a bounded fixed-point expander with its own guard
    (`test_the_derived_commands_expand_completely`) resolves them.
  - **Evidence — eighteen mutations, and one of them found a real hole in a test rather than confirming it.**

    | Mutation | Site | Newly red | What it proves |
    |---|---|---|---|
    | M1 delete the ingestion `Queue` declaration | `celery.py` | 1 | the queue set is closed, so the declaration is load-bearing |
    | M2 drop the ingestion queue's dead-letter arguments | `celery.py` | 1 | a rejected ingestion message parks rather than vanishing |
    | M3 restore the `tasks.*` glob | `celery.py` | 1 | explicit routing is enforced, and names 11 offenders |
    | M4 route every name to `default` | `celery.py` | 3 | **found a hole** — see below |
    | M5 route every name to `ingestion` | `celery.py` | 1 | the split cannot pass by moving everything |
    | M6 `task_create_missing_queues=True` | `celery.py` | 1 | the positive control really can fail |
    | M7 typo the compose `-Q` value | `docker-compose.yml` | 1 | drift caught from the deployment side |
    | M8 rename the queue in settings | `settings.py` | 1 | drift caught from the configuration side |
    | M9 add `default.dlq` to a worker's `-Q` | `docker-compose.yml` | 1 | the dead-letter queue stays a parking space |
    | M10 give the ingestion worker `default` too | `docker-compose.yml` | 1 | the disjointness the anti-starvation claim rests on |
    | M11 rewrite the beat command so the parse misses it | `docker-compose.yml` | 1 | the regex fails loudly instead of comparing empty sets |
    | M12 remove one worker's `-Q` entirely | `docker-compose.yml` | 1 | "two worker pools" is counted, not assumed |
    | M13 drop `-Q default` from the definition site | `Makefile` | 1 | the `-Q` requirement is enforced where it is defined |
    | M14 make the scheduler command a worker | `Makefile` | 1 | the worker/scheduler split the `-A`-only check relies on |
    | M15 make a Makefile variable self-referential | `Makefile` | 1 | the expander refuses to compare unexpanded strings |
    | M16 drift the README by one concurrency figure | `README.md` | 1 | documentation equality is on the whole string |
    | M17 drift a compose concurrency figure | `docker-compose.yml` | 1 | deployed commands are pinned to the definitions |
    | M18 use the `src.` spelling in compose | `docker-compose.yml` | 1 | one `-A` identity across every file |

    Every touched file restored byte-identically (`sha256` compared before and after each mutation), and the harness
    restores from an in-memory copy rather than `git checkout`, because the working tree is deliberately dirty and HEAD
    does not contain this change.
  - **Evidence — M4 survived on the first run, and the test was wrong, not the mutation.**
    `test_the_ingest_names_route_to_the_ingestion_queue` originally asserted
    `routed_queue(name) == routed_queue(DOCUMENTS_INGEST)` — the ingest names compared to **one another**. Routing all
    16 names back to the default queue left the three still equal, so the test passed while the ingestion queue had no
    producers at all: the exact defect C7 exists to prevent, invisible to its own guard. Rewritten to name the
    configured queue and to assert the two queue names differ. M4 now kills it with `assert 'default' == 'ingestion'`.
    This is the value of mutating a guard rather than trusting a green run.
  - **Evidence — a credential was leaked into a traceback by this task's own test design, and the fix is structural.**
    An early version read `real_celery.app.conf.CELERY_DEAD_LETTER_QUEUE` — Celery's own config object, which does not
    hold project settings. The `AttributeError` came from inside the library, and pytest rendered that frame's locals,
    **which include the broker URL with its credentials**. Reading the value from the project's settings object instead
    removes the whole class of exposure, because its secret fields are `SecretStr` and mask on repr. The reason is
    recorded beside `_settings` in `test_queue_topology.py` so the next person does not reintroduce it.
  - **Evidence — the Makefile gained two targets and the docs stopped lying.** `make celery` (default queue),
    `make celery-ingestion`, `make celery-beat`, and `make celery-command` prints all three without running them.
    C8's single-definition-site discipline is kept exactly: one `CELERY_APP`, one `CELERY_WORKER_CMD`, queue selection
    and concurrency appended — never a second copy of the `-A` string. `src/app/examples/CELERY.md` documented two
    queues and worker commands without `-Q`; it now names three queues, explains the split, and points at the `make`
    targets rather than holding a fourth uncontrolled copy of the command.
  - **Evidence — the concurrency figures are a decision, not a default.** `--concurrency=8` on the default queue: its
    tasks are short and mostly waiting on other services. `--concurrency=2` on ingestion: each slot holds a
    document-conversion and embedding pipeline, so raising it multiplies peak memory by whatever the largest document
    costs. The comment at both sites says to scale that service's **replicas** rather than its concurrency. Exactly one
    `celery-beat` replica may run — a second would publish every scheduled task twice, and the billing and credit tasks
    it emits are not all idempotent.
  - **Evidence — `RABBITMQ_URL` is overridden on all three services as a safety property, not a convenience.**
    `.env.development` points at the managed broker carrying real billing, credit and password-reset work; a worker that
    inherited it would consume and execute that work. Compose's `environment` beats `env_file`, which is what makes the
    override effective, and that is recorded in the file because deleting the line looks harmless.
  - **Evidence — two findings outside C7's scope, with locations, deliberately unfixed.**
    - `src/app/features/documents/service.py:184` and `src/app/features/search/service.py:109` pass raw task-name
      string literals as `event_type` (`"tasks.documents_ingest"`, `"tasks.search_ingest"`), and
      `src/app/shared/outbox/relay.py:136` hands `event_type` straight to `CeleryTaskRegistry.typed_send(...)` — so
      those literals **are** task names. A C9 Proof-4 residual; fixing it is an edit to two feature services, which
      C7 does not own.

      **Corrected 2026-08-23 — the hazard is real, the mechanism described was not.** This entry said a rename would
      "silently misroute them, and C7 makes that worse, because a misrouted name now also lands on the wrong queue."
      Measured: both literals currently **match** registered names, so nothing is mis-dispatched today; and a rename
      would not misroute at all. `typed_send` calls `ensure_declared_module_imported` then `validate` **before**
      `send_task`, so an unregistered name raises `UnregisteredTaskError` and, as its own docstring puts it, "a
      refused dispatch never reaches a broker" — it cannot land on the wrong queue because it lands on no queue.
      `UnregisteredTaskError` and `TaskPayloadValidationError` both subclass `CeleryError` through
      `TaskDispatchError`, which the relay's `except (CeleryError, PostgresError)` at `:139` therefore catches: the
      row is marked **failed**, the `event_type` is attached with `add_note`, `outbox_publish_failed` is logged, and
      the loop continues to the next row. C9 rooting its exceptions under `CeleryError` is what makes that work, and
      it is load-bearing — a registry error outside that hierarchy would escape `_publish` into the caller at `:73`
      and `:125`.

      So the true consequence of a rename is **lost events, loudly and per-row**, pending a fix: the affected rows
      sit in a failed state, nothing is misdelivered, and no other event type is affected. Worth fixing (one import
      and one substitution each) because a durable store holding literals that must match a constant elsewhere will
      also strand rows written *before* any rename. Not worth fixing as an urgent misrouting bug, because it is not
      one.
    - `LEGAL_BATCH_EXTRACTION` (`document_extraction.legal_batch`, `src/tasks/document_extraction_tasks.py`) is also
      minutes of model work per message via langextract, and stays on the **default** queue. The answer that closed
      Open Question 1 named the three ingest names; a third queue needs a third consumer or it silently accumulates, so
      moving it is a topology decision to be **asked for**, not inferred from the fact that it looks similar. The
      reason is recorded beside `INGESTION_TASK_NAMES` so the omission reads as deliberate.

- [x] **C8 — Fix the documented worker start command so it matches the deployed one exactly.**
  The documented command names an application module that does not exist. Fix it to name the real task application,
  and make the documented command and the command the deployed service runs the same string, so they cannot drift.
  - **Proof:** `uv run rg -n "\-A app|celery -A" Makefile docker-compose.yml README.md docs/` → every occurrence
    names a module that exists, and `uv run python -c "import importlib; importlib.import_module('<that module>')"`
    exits 0 for each distinct name found.
  - **Proof:** the documented command, run verbatim, starts a worker that reports its registered tasks and does not
    fail to load the application.
  - **Proof:** the string in the documentation and the string in the compose service definition are identical
    (`diff <(…) <(…)` or an equality assertion in a check script).

  **Done, with one Proof armed rather than satisfied and one Proof deliberately not executed.** The command is now
  defined **once**, in the `Makefile` (`CELERY_APP` + `CELERY_WORKER_CMD`), and
  `tests/unit/celery/test_documented_worker_command.py` — 8 tests, 1 skip — asserts every other copy equals it.
  Depends on C9, which settled the task-application module; that dependency is now discharged.
  **Superseded in one detail by C7:** the armed Proof 3 below is now satisfied, not skipped, and the module is 14 tests
  with no skip. The counts here record the state at C8's completion and are left as written; C7's evidence records the
  change and why the assertion was generalised rather than relaxed.

  - **Evidence — the defect was worse than "a wrong name": the documented command could not start a worker at all.**
    `Makefile:52` and `README.md:279` both named a Celery configuration module that **has never existed** in this
    repository — confirmed three ways: `find` over the tree returns nothing, the import raises `ModuleNotFoundError`,
    and `src/` contains only `alembic app database lynk mcp_core tasks`. The correct target is
    `app.connections.celery:celery_app`. Nothing in the suite noticed, because nothing compared the documentation to
    anything.
  - **Evidence — fixing both literals would have left the actual defect in place.** Two copies of one command are free
    to diverge again, which is why the task says "so they cannot drift". The command has one definition site; the README
    keeps a copy-pasteable literal (a README's job) and a test asserts equality. C8's third Proof explicitly permits
    "an equality assertion in a check script" — a test was chosen over a script because it runs on every commit rather
    than when someone remembers.
  - **Evidence — the `-A` value names an attribute, and the `app.` prefix is not cosmetic.** Written
    `module:attribute` rather than the bare module: Celery *would* find the instance by probing, but the probe takes
    whatever it finds first, so naming the attribute means adding a second `Celery` instance to that module cannot
    silently re-target the worker. The prefix matters more. **Both** `app.connections.celery` and
    `src.app.connections.celery` import successfully — and they are **two different module objects**, because Python
    keys `sys.modules` by the import string, so each carries its own `Celery` instance and its own task registry. A
    worker started under one and a producer importing the other would agree about every task name and share none of
    them. Measured from `/tmp`: the installed spelling imports, the source-path spelling **fails** — `src/` reaches the
    path through the editable install, while the source-path spelling resolves only when the working directory happens
    to be the repository root. A container with any other `WORKDIR` would fail to boot with the `src.` spelling.
    Mutation M3 confirms the identity claim empirically rather than by argument: switching the command to the `src.`
    spelling turns the `resolved is real_celery.app` assertion red, because it really is a different object.
  - **Evidence — Proof 1's scope needs amending: it names all of `docs/`, and `docs/relay/` quotes the broken command
    as evidence.** Every `celery -A` occurrence under `docs/` is in `docs/relay/` (6 files: `plan-change1.md`,
    `c9-task-registration.md`, `scout-ingestion-graphs.md`, `plan-change4.md`, `findings-deployment.md`,
    `dispositions.md`) and each is a working note recording the defect. **Rewriting evidence to satisfy a grep is the
    wrong direction.** Amended scope: the files that are executed or copy-pasted — `Makefile`, `README.md`,
    `docker-compose.yml`. Rescoped, Proof 1 returns 2 hits, both naming a module that imports.
  - **Evidence — Proof 2 was NOT executed as written, for a safety reason that overrides it.** The Proof asks that the
    documented command be run verbatim so a worker reports its registered tasks. **The configured broker is a live
    managed instance** (checked by opening a socket to it; scheme, host and port only, no credential printed), and the
    registered task set includes `billing.*`, `credits.*` and `auth.send_password_reset_email`. Starting a *consuming*
    worker against it could execute real queued work, up to and including sending mail to real recipients — an
    irreversible outward-facing side effect that no Proof in this change authorises. **Substitute evidence, which
    exercises the same machinery without consuming anything:** `celery.app.utils.find_app("app.connections.celery:celery_app")`
    — the function Celery's own `-A` handling uses — returns a `Celery` instance, and `app.loader.import_default_modules()`,
    which is precisely what a worker performs at boot, registers **all 16** declared task names. No broker connection is
    opened. Registration is separately proven without a broker by C9's `test_task_registration.py`.
  - **Evidence — Proof 3 is armed, not deferred, and the arming was mutation-proven.** No worker service exists in
    `docker-compose.yml` (services are `rabbitmq`, `timescale`, `caddy`), because that service is **C7**, blocked on
    Open Question 1. `test_the_compose_worker_service_runs_the_documented_command` skips while the compose file does not
    mention the worker and begins asserting the moment it does — so **C7 cannot land a worker whose command has drifted
    from the documentation.** A skipped test can hide a broken assertion, so mutation M6 added a worker service with a
    drifted command: the suite went to `1 failed, 8 passed` with **no skip**, proving the arming fires. **C7 has since
    landed the services and the arming held** — the base command appears verbatim in `docker-compose.yml` as the shared
    prefix of both worker commands, and the test now asserts instead of skipping.
  - **Evidence — the regression guard initially failed on the text documenting the fix.** The first version of the
    `Makefile` comment and the README note *quoted* the phantom module name, so the test greping those files for it went
    red on the explanation. The constant is now assembled from two fragments and the prose describes the name without
    spelling it. **Third occurrence of this pattern in this work: a comment containing a proof's grep literal defeats
    the proof.**
  - **Evidence — `make` is parsed, not invoked, and the parser's honesty is itself asserted.** Invoking `make` needs
    `subprocess`, and bandit's rules are **enabled** here (`"S"` appears under `unfixable`, not `ignore`; the tests'
    per-file-ignores lift only `S101`), so it would need two suppressions — against CLAUDE.md's preference for
    satisfying a check over silencing it — and would make the suite depend on `make` being installed. Parsing risks
    testing the parser instead of the Makefile, so `test_the_makefile_command_needs_exactly_one_substitution` pins that
    the definition stays simple enough for one textual substitution to be complete. Add a second variable to that line
    and it fails **with instructions**, rather than the comparison silently going vacuous.
  - **Evidence — six mutations, each killing its intended guard.**

    | Mutation | Newly red | What it proves |
    |---|---|---|
    | M1 Makefile names the phantom module again | 4 | the original defect is caught from four independent directions |
    | M2 README drifts by one flag (`--loglevel=debug`) | 1 | equality is on the whole string, not just the module name |
    | M3 Makefile uses the `src.` spelling | 4 | the two spellings really are different objects — the identity assertion fails |
    | M4 a second documented copy appears in README | 1 | "exactly one documented command" is enforced, not assumed |
    | M5 the command grows a second variable | 1 | the expander refuses to keep comparing once it is no longer faithful |
    | M6 a drifted worker service appears in compose | 1 | the skipped Proof-3 test genuinely arms itself |

    All three touched files restored byte-identically (`sha256`), and `docker-compose.yml` is absent from
    `git status` afterwards.
  - **Evidence — one finding outside C8's scope.** `README.md:263` starts uvicorn with `src.app.main:app`, the
    source-path spelling this task rejected for the worker. It works today only because the documented invocation runs
    from the repository root. Not changed here: it is the API process's command, C8 owns the worker's, and a
    behaviour-affecting edit to the documented boot command of the running service deserves its own task.

- [x] **C9 — Make task registration explicit and typed, harvesting the archived registry contract.**
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

  **Done, with one repair the task did not ask for and could not have known to ask for.** New
  `src/app/connections/celery_task_names.py` is the single definition site for all **16** dispatchable names plus
  `TASK_DECLARING_MODULES` (name → declaring module). `include=` went from 5 entries to 8 explicit literals.
  `celery_registry.py` harvested with one tightening and one repair. 18 new test items across
  `test_task_registration.py` and `test_typed_dispatch.py`, plus `tests/unit/celery/conftest.py`. The change-0
  dependency was honoured: Proof 2 was run **after** reducing the task package initialiser to what change 0's tidy
  leaves.

  - **Evidence — the harvested contract validated nothing in the only process that used it.** Registration is a side
    effect of importing the declaring module, and **nothing under `src/` imports the task package at all** — verified,
    not assumed: `rg -o 'from tasks(\.[\w.]+)? import' src/` returns exactly one hit, inside a Markdown example. So the
    API process that runs the outbox relay held an **empty** registry, and every payload was validated against the
    archived text's permissive fallback. `ensure_declared_module_imported()` is the repair; the fallback
    (`LegacyTaskPayload`) is deleted.
  - **Evidence — the tightening the task asked for, and why the *base class* of the refusal is load-bearing.**
    `validate()` now raises `UnregisteredTaskError` for a name with no registered model, where the archived text
    substituted a permissive model, logged a warning and sent anyway — producing a well-formed message addressed to
    nobody, which Celery discards in silence. Both refusals derive from `TaskDispatchError(CeleryError)`, and that is
    not decoration: `OutboxRelay._publish` catches `(CeleryError, PostgresError)` to mark an event failed and retry it
    toward the dead-letter table — **verified by reading `src/app/shared/outbox/relay.py`**. A bare pydantic
    `ValidationError` would escape that catch into the relay's outer blanket handler, which logs a warning and drops the
    event, putting the invisibility back one layer up. The original is preserved as `__cause__` and on
    `.validation_error`. `NoKwargsPayload` was added for scheduler-dispatched jobs, so "nothing, and nothing extra" is
    stated as a contract rather than left unregistered.
  - **Evidence — registration verified independently of the agent's own report.** `find_app` on the settled module,
    then `loader.import_default_modules()` — what a worker does at boot — registers all 16 names: the five `tasks.*`
    (`add`, `documents_ingest`, `pageindex_ingest`, `process_document`, `search_ingest`), two `auth.*`, six `billing.*`,
    two `credits.*`, and `document_extraction.legal_batch`. Note that a *bare* import of the application registers
    **zero** — `include` is resolved lazily at worker boot, not at construction — which is the trap a registration test
    can silently fall into. This directory's conftest resolves it through `importlib` rather than trusting a bare
    import.
  - **Evidence — Proof 1 does not display what it checks; amended.** As written, `rg -n "include=|imports\s*="` prints
    only the `include=` line, so the list itself is invisible, and the `imports` half of the alternation matches nothing
    (that Celery setting is unused here). **Amendment:** `rg -n -A 10 "include=" src/app/connections/celery*.py`, which
    returns the eight module literals. Eight is every module under `src/tasks/` that declares a task **except**
    `auth_email_tasks_typed`, excluded on purpose — see the finding below.
  - **Evidence — Proof 4's stated expectation cannot be met by its pattern, and the pattern is blind to most of what it
    polices; amended.** `rg -n '"tasks\.' src/app/ src/tasks/` returns 31 hits, and it *must*: 8 are the `include`
    module paths Proof 1 requires, 1 is a routing glob `"tasks.*"`, 1 is the dead-letter **exchange** `"tasks.dlx"`, 13
    are the definition site itself, and 6 are Markdown examples. Worse, **only 5 of the 16 declared names begin with
    `tasks.`** — independently confirmed by the 16-name registry dump above — so the `auth.*`, `billing.*`, `credits.*`
    and `document_extraction.*` families are invisible to it, including two live dispatch-side literals.
    **Amendment — a name-agnostic form that cannot go stale, because it builds its alternation from the definition
    module:**

    ```python
    alt = "|".join(re.escape(n) for n in sorted(TASK_DECLARING_MODULES))
    rg -n f'"({alt})"' src/ -g '!*.md' -g '!celery_task_names.py'
    ```

    15 hits: **four genuine residual dispatch-side literals** — `documents/service.py:184`, `search/service.py:109`,
    `auth/service.py:271`, `auth/service.py:298` — and eleven `logger.bind(operation="...")` labels in
    `billing_tasks.py`/`credit_tasks.py`. The eleven were **deliberately left as literals**: the same files also bind
    `operation="billing.invoice_backfill"` and `"billing.receipt_backfill"`, which are not task names at all, so that
    taxonomy is independent of the registry and merely coincides with it. Coupling it would make a task rename silently
    re-label existing dashboards and alerts — the opposite of what stability means for a log field. The four dispatch
    sites are not C9's files and were left untouched; each needs one import and one substitution from
    `app.connections.celery_task_names`. The documents and search payloads were checked against their registered models
    and match, so those two are name-only; the two auth payloads do **not** — see the next item.
  - **Evidence — a producer/consumer gap on the auth emails, now a loud refusal instead of a silent one.**
    `auth/service.py:272` and `:299` emit `{user_id, email, token}`; both task bodies require `idempotency_key` as well
    (`auth_email_tasks.py:91`, `:130`) — verified directly. The payload models were registered faithful to the
    **declarations**, so this gap now surfaces as a dispatch refusal rather than a `None` lock key inside the worker. No
    live behaviour change today, because the outbox tables do not exist; **it will fire the moment change 0's tables are
    created.** Fix belongs to whoever owns `auth/service.py`.
  - **Evidence — two live defects fixed on the way.** `document_extraction_tasks.py` declared `bind=True` while its body
    takes no `self`, so Celery would have passed the `Task` instance as the first positional argument (`urls`) —
    removed, with a comment recording why (the file now carries the explanation at `:26` and no `bind=True`). And
    `tasks.auth_email_tasks_typed` declares the **same two task names** as the live email module, so listing both in
    `include` would let import order pick the winner; it is excluded and documented in place, and is a deletion
    candidate.
  - **Evidence — a new conftest rather than an edit to a file three concurrent tasks share.** `tests/conftest.py` puts
    `MagicMock()` into `sys.modules` for the task application and the task package. A `MagicMock` has no `__path__`, so
    under those entries the declaring modules are not merely mocked but **unimportable** ("`tasks` is not a package"),
    and every C9 proof is unwritable while they stand. `tests/unit/celery/conftest.py` provides an opt-in,
    module-scoped `real_celery` fixture that lifts them and restores at teardown. It is **not** autouse, so the sibling
    module in that directory that wants the mocks keeps them. It also lifts `app.utils` at its top level only, because
    another unit test in the suite replaces that module at import time with a two-attribute proxy whose logger is an
    `AsyncMock` and never restores it — a module imported into that state binds a logger whose `.bind()` returns a
    coroutine, so the first diagnostic a refusal writes raises `AttributeError` instead. Verified non-leaky: the
    pre-existing red set is byte-identical before and after, across 7 mutation runs.
  - **Evidence — the three `tasks*` stubs in `tests/conftest.py` are dead.** `rg -o 'from tasks(\.[\w.]+)? import' src/`
    — the command that conftest's own comment says regenerates the list — now returns nothing under `src/` but a
    Markdown example. Removing them would let `real_celery` shrink to nothing. **Not done: not this task's file.**
  - **Evidence — seven mutations, each killing exactly the intended tests, with no pre-existing red item turned green.**

    | Mutation | Newly red | What it proves |
    |---|---|---|
    | M1 `include` drops `tasks.document_tasks` | 3 | explicit listing is enforced in both directions |
    | M2 unregistered name falls through permissively | 2 | the pre-tightening behaviour was a **silent send** — the dispatch reaches the spy |
    | M3 payload mismatch warns and passes through | 4 | it reaches the spy with a payload the consumer cannot accept |
    | M4 helper stops importing the declaring module | 5 | **the production defect, reproduced**: the registry is empty, so every name looks unregistered |
    | M5 refusals stop deriving from `CeleryError` | 2 | the relay's retry path would stop catching them |
    | M6 declaration drifts from the definition | 2 | the single definition site is load-bearing, not stylistic |
    | M7 unimplemented task returns instead of raising | 1 | a declared-but-unimplemented name fails explicitly |

  - **Evidence — two document-path inaccuracies in C9's own dispatch, recorded so they are not re-followed.**
    `openspec/changes/decisions.md` and `openspec/changes/critical-path-210.md` do not exist; both live under
    `docs/relay/`. And **"Decision 16" is ambiguous**: `docs/relay/decisions.md`'s Decision 16 concerns `UnifiedChunk`
    gaining `updated_at`, while the harvest-vs-delta ruling C9's text invokes is in `design.md` (~line 592). Two
    numbering schemes share one label.
  - **Evidence — `document_extraction.legal_batch` and `tasks.pageindex_ingest` have zero dispatchers.** Both are
    registered and bound, so a future dispatch gets a real diagnostic rather than an unknown-task error, but nothing
    dispatches either today.

- [x] **C10 — Do NOT put the pipeline graph or the checkpointer on shared application state; prove the block stays disabled.**
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

  **Done, and it found a live defect the task did not name plus a much larger one it could not have.** Nothing was
  provisioned and nothing should be; `tests/unit/features/ingestion/test_unprovisioned_graph_fails_closed.py` is 7 tests
  holding the non-goal true. Proof 1 and Proof 3 pass; **Proof 2 passes on its substance but not on its wording**, for a
  reason recorded below and pinned by a tripwire.

  - **Evidence — the guard that looked careful was unreachable, which is the third instance of that pattern in this
    change.** `dependencies.py` read `request.app.state.ingestion_graph` directly and then tested `if graph is None`.
    Starlette's `State` **raises** on an unknown attribute, and the attribute is never *set* — not set to `None` — so
    the read raised `AttributeError` before the test could run: the guard that looks like it produces a typed 503
    produced an unhandled attribute error and a **500**. Now read through `getattr(..., None)`, which collapses "never
    provisioned" and "explicitly provisioned as absent" into one branch **deliberately** — from a caller's position they
    are the same condition, and keeping them apart is what produced two status codes for one situation. (The two earlier
    instances: B3's `hasattr(t, "to_markdown")` filter and C4's `hasattr(checkpointer, "pool")`.)
  - **Evidence — the fix removes the read site from Proof 1's literal grep, so Proof 1 needs a second command.**
    `getattr(request.app.state, "ingestion_graph", None)` contains no `app.state.ingestion_graph` substring, so the
    Proof's pattern **cannot see the very site it exists to check**. Amendment: run it alongside
    `rg -n '"ingestion_graph"|"langgraph_checkpointer"' src/app/`, a by-name check that catches the `getattr` form.
    Both were run. The literal form returns 6 hits: 3 commented (`lifespan.py:257`, `:322`, `:328`), 1 docstring line,
    1 teardown call at `lifespan.py:340` already guarded by `hasattr` at `:339`, and **1 genuinely unguarded read at
    `agent_saul/dependencies.py:49`** — which is **change 3's step 1 per D17** and correctly not fixed here. Proof 3
    returns no matches: no enabling flag was introduced.
  - **Evidence — Proof 2 asks for the standard error envelope, and no exception in the `APIException` family can produce
    one. This is a live application-wide defect.** Starlette splits `add_exception_handler` across two middlewares:
    `Exception` goes to `ServerErrorMiddleware` as a **500-only** net, while every other class is resolved by
    `ExceptionMiddleware` walking the raised exception's **MRO** against its registry. FastAPI pre-registers
    `HTTPException`, and `APIException` inherits from it, so the walk short-circuits three classes before `Exception` is
    ever considered. **Consequence: the entire `isinstance(exc, APIException)` branch of
    `global_exception_handler` is dead in the deployed app** — verified against `create_app()`, which selects
    `('HTTPException', 'http_exception_handler')`. Every `APIException` returns `{"detail": …}`, never the documented
    `{success, statusCode, request, message, data, error}` envelope. The structural tell: a handler branch inspecting
    `isinstance(exc, APIException)` can only run if `APIException` or a subclass is a registry **key**.
  - **Evidence — recorded and pinned rather than fixed, because the fix is not an ingestion change's to make.**
    Registering the family would change the body of **every** error response in the application — auth, users, billing,
    documents. `test_the_project_envelope_is_still_unreachable_tripwire` asserts the *defect* and **must fail when it is
    fixed**, carrying the instruction to delete the tripwire and restore C10's envelope Proof rather than adjust the
    assertion to keep it green. The capability is still asserted structurally in the meantime:
    `APIException.__init__` folds `data` into `rich_detail["data"]`, so `{"capability": "ingestion_graph"}` is reachable
    at `body["detail"]["data"]["capability"]` even through FastAPI's default handler — a caller can branch on it without
    parsing a sentence.
  - **Evidence — the test app is hand-built, and that choice is itself asserted rather than trusted.** `create_app()`
    imports the full stack and its lifespan opens connections. The registry that matters is reproducible exactly: the
    real app holds precisely `HTTPException`, `RequestValidationError`, `WebSocketRequestValidationError` and
    `Exception` — the first three are FastAPI's defaults on any `FastAPI()`, the fourth is the single line `main.py:110`
    adds. `test_the_handler_registry_matches_the_real_application` asserts that 4-entry registry, so a bare app plus one
    registration is not an approximation of the real registry but the same registry — and if FastAPI's defaults change,
    the claim fails loudly instead of the file quietly testing something else.
  - **Evidence — `raise_server_exceptions=False` is load-bearing, and there is a positive control.** Without it an
    unhandled exception is re-raised into the test as a traceback and the assertion cannot distinguish "failed closed"
    from "fell over"; with it the 500 is observable as a status code, which is exactly the distinction this task exists
    to make. `test_a_provisioned_graph_is_handed_through` prevents the other failure mode: without it every test here
    would still pass against a dependency hard-coded to refuse, and the file would prove only that a constant is a
    constant.
  - **Evidence — three mutations, each killing exactly its intended tests.** M1 revert to the direct attribute read → 3;
    M2 drop the `capability` data → 2; M3 return `None` instead of raising → 4. File restored from a pristine copy
    afterwards; 7 passing.
  - **Evidence — ruff on the new test file, and one rule that would have broken the app.** `TC002` demanded
    `IngestionGraphDep` move into a `TYPE_CHECKING` block. **Obeying it breaks the application at import time**: FastAPI
    calls `get_type_hints()` on every endpoint to build its dependency graph, so an `Annotated[..., Depends(...)]` alias
    must exist at runtime. Resolved by omitting `from __future__ import annotations`, which is what
    `src/app/features/ingestion/dependencies.py` and `documents/router.py` already do — matching the module under test
    rather than suppressing a rule. `TC003` on `Any` still fired afterwards, correcting an earlier belief that it only
    fires under the future import: it keys on annotation **position**, and PEP 526 local-variable annotations are never
    evaluated at runtime, so `Any` used only there is genuinely type-only either way (a function *return* annotation is
    evaluated at def time, which is why `Iterator` stays a runtime import). `implicit-namespace-package` was left as-is:
    it is endemic across `tests/` (9 pre-existing, including the sibling `documents` package) and the documented gate is
    **`src/` only**.
  - **Evidence — CLAUDE.md's "Key files" table has a stale path.** It lists `src/app/shared/response_type.py`, which
    does not exist. The real paths are `src/app/utils/response_type.py` and `src/app/utils/http_response.py`.
