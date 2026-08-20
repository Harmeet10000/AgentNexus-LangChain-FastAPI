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

- [ ] **0.1 — Record the closed lexical precondition and the index dependency it exposed.**
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

- [ ] **0.2 — Establish the runnable-database gate for every table-touching task.**
  No document, chunk, search, clause, durable-event, dead-letter, or memory table exists; the schema was stamped, not
  migrated. Every Proof below that reads or writes a table is blocked until change 0's single migration runs on the
  merged head.
  **Dependency (change 0):** revision-head merge, then one migration creating the target schema, the extensions, and
  the lexical indexes by exact name.
  - **Proof:** against the migrated database, a read-only catalogue query lists the document and chunk tables and the
    lexical index by name. Until it does, tasks A3, D3, E6, and E7's persistence Proofs are **blocked, not skipped**.
  - **Proof:** `uv run alembic heads` reports a single head. (Not `--sql`; see rule 2.)

---

## Band A — cold correctness fixes (no database, no graph, no network)

Each is independently committable and each fixes something already wrong today.

- [ ] **A1 — Fix the logger submodule shadowing in `src/app/utils/embedding.py`.**
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

- [ ] **A2 — Resolve the dimension conflict and delete every placeholder vector in the batch embedder.**
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

- [ ] **A3 — Make the persisted vector width derive from the configured dimension, not a literal.**
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

- [ ] **A4 — Delete the phantom `ingestion.embedder` import in `rag_agent_advanced.py`.**
  `:119`, `:198`, `:267`, and `:373` each do `from ingestion.embedder import create_embedder`, a module that does not
  exist. These are function-local imports, so the failure is deferred to first call rather than surfacing at import.
  Retarget them at the single embedding path.
  - **Proof:** `uv run rg -n "from ingestion.embedder|ingestion\.embedder" src/app/` → **no matches**.
  - **Proof:** `uv run python -c "import importlib; importlib.import_module('app.shared.rag.rag_agent_advanced')"`
    exits 0.
  - **Proof:** a unit test exercises each former call site's embedding branch and asserts it resolves without a
    module-resolution error. A deferred import failure is not acceptable — the requirement says so explicitly.

- [ ] **A5 — Fix the degraded-branch handler that destroys the diagnostic it exists to preserve.**
  `ingestion_kb/nodes.py:212-256` calls `exc.add_note(f"doc_id={state.doc_id}, …")` inside the degraded branch, but
  the state is a mapping at that point, so the attribute access raises inside the handler and replaces the original
  failure with a secondary error. Build the note from the values the branch already holds; the handler must not raise.
  - **Proof:** `uv run rg -n "state\.doc_id" src/app/shared/langgraph_layer/ingestion_kb/nodes.py` → no occurrence
    inside an exception handler (verify by reading each remaining hit, not by pattern alone).
  - **Proof:** a new unit test induces the degraded branch and asserts (a) it returns a degraded result rather than
    raising, (b) the recorded diagnostic names the **original** cause, and (c) the degradation record carries the
    document and chunk identity. `uv run pytest tests/unit/shared -q 2>&1 | tail -3`.

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
