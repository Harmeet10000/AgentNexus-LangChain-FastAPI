# Plan — openspec change 1: ingestion

Planner leg, 2026-08-17. Read-only on `src/`. Gates: D1, D5.1, D8, D9, D10 (`docs/relay/decisions.md`).

## Prior art found (kb_retry.py / checkpointer.py) — the highest-value unknown, now closed

Both files exist, both are substantive, and **both are broken in ways that change the shape of the work.**
This was flagged as the single highest-value unknown in `brief-langgraph-practices.md` Fog §4/§5. Verdict:
prior art exists for both sub-todos, so neither is greenfield — but neither is usable as written.

### `src/app/shared/langgraph_layer/kb_retry.py` (46 lines) — retries: prior art EXISTS, and it is the
### exact anti-pattern the brief condemns

Public symbols:
- `TransientExternalError(Exception)` — `kb_retry.py:15`
- `async def retry_immediate[T](operation, *, label: str, attempts: int = 3) -> T` — `kb_retry.py:19`
  (PEP-695 generic; takes a zero-arg async thunk)

Callers: every I/O call in `ingestion_kb/nodes.py` (per `scout-ingestion-graphs.md` §1: "`retry_immediate`
wraps every I/O call"), including `nodes.py:729` `retry_immediate(label="gemini_embedding")`. It is listed in
`dispositions.md` (item 172 row) as one of the three places tenacity already lives at an I/O boundary.

**It already uses tenacity** — `from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt,
wait_none` (`:7`), tenacity 9.1.4 installed (`uv.lock:8445`). So sub-todo (j) "add tenacity" is **already done**,
badly. Four defects, each a planned step below:

1. **`retry=retry_if_exception_type(Exception)` (`:29`) is the literal catch-all the docs forbid.**
   `brief:ref:1633` — "Do not wrap interrupt calls in try/except… you will catch this exception and the
   interrupt will not be passed back to the graph." `retry_immediate` wraps node-internal I/O, so any future
   `interrupt` inside a wrapped operation is swallowed and retried 3x. This is the tenacity-vs-middleware
   conflict made concrete in existing code, not hypothetical.
2. **`wait=wait_none()` (`:28`) — zero backoff.** Three immediate attempts against a rate-limited Gemini
   endpoint is three 429s in a row. The docs' middleware capability list says "exponential backoff"
   (`brief:ref:1117-1118`). `wait_none` is the opposite.
3. **`reraise=True` (`:30`) is dead code.** The `async for` loop is wrapped in `except Exception` (`:41`) which
   re-wraps into `TransientExternalError` (`:43`). So the original exception type never escapes; `reraise=True`
   has no observable effect. Every distinct upstream failure (auth, quota, malformed response) collapses to one
   opaque type, which is why `nodes.py`'s `except LangChainException` branches cannot ever fire for a wrapped
   call — `TransientExternalError` is not a `LangChainException`. **This silently defeats the graph's own
   degradation paths at `nodes.py:182`, `:236`, `:289`.**
4. **No idempotency coupling.** `brief:ref:1612-1614` requires per-side-effect task decomposition + idempotency
   keys because "the checkpointer's recovery unit is the node, not the statement". A retry counter in a local
   variable inside a node is not a checkpointed channel: on replay the node re-enters at line 1 with the counter
   reset, so the retry budget is silently multiplied. Adding a checkpointer (step group C below) makes this
   live — it is latent only because no checkpointer exists today.

Note `kb_retry.py:9` does `from app.utils import logger`. That is **safe here** (see the embedding.py analysis
below) but it is the same fragile idiom.

### `src/app/shared/langgraph_layer/checkpointer.py` (92 lines) — checkpointing: prior art EXISTS and
### **crashes on the first call**

Public symbols:
- `async def setup_langgraph_checkpointer(conn_string: str) -> AsyncPostgresSaver` — `checkpointer.py:32`
- `async def teardown_langgraph_checkpointer(checkpointer: AsyncPostgresSaver | None) -> None` — `:70`

Callers: `setup_` has **no live caller** — `lifespan.py:295-305` is commented out (`scout-ingestion-graphs.md`
§4). `teardown_` **is** live at `lifespan.py:317`, guarded by `hasattr` at `:316`. Readers of the state slot it
would populate: `features/agent_saul/dependencies.py:45` (**unguarded**), `lifespan.py:317` (guarded).

**Defect 1 — `from_conn_string` is an async context manager, not a factory. Verified against installed
`langgraph-checkpoint-postgres` 3.0.4.** `.venv/.../langgraph/checkpoint/postgres/aio.py:55-80`:

```
@classmethod
@asynccontextmanager
async def from_conn_string(cls, conn_string, *, pipeline=False, serde=None) -> AsyncIterator[AsyncPostgresSaver]:
    async with await AsyncConnection.connect(conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row) as conn:
        ...
        yield cls(conn=conn, serde=serde)
```

`checkpointer.py:56-57` does `checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)` then
`await checkpointer.setup()`. The bound name is an `_AsyncGeneratorContextManager`, which has **no `.setup`
attribute** → `AttributeError`. The `except (ConnectionError, TimeoutError, OSError)` at `:58` does **not**
catch `AttributeError`, so it propagates out of the lifespan. **Consequence: uncommenting `lifespan.py:295-305`
as-is takes the app from "boots without a checkpointer" to "does not boot at all."** Item 138 residue (a) is
therefore *not* an uncomment task — it is a rewrite of `checkpointer.py`. This is the single most important
ordering fact in this change.
Even if `.setup()` resolved, the connection is opened by an `async with` inside the generator that is never
entered, so no connection would ever exist.

**Defect 2 — `teardown_langgraph_checkpointer` is a permanent no-op.** `:83` tests
`hasattr(checkpointer, "pool")`. `AsyncPostgresSaver.__init__` (`aio.py:37-53`) sets exactly `self.conn`,
`self.pipe`, `self.lock`, `self.loop`, `self.supports_pipeline` — **there is no `pool` attribute anywhere in
`aio.py`** (grep for `self.pool`/`pool:` in that file returns zero hits). So the `if` is always False and
teardown returns silently. The pool leak it claims to prevent is unprevented.
The correct production shape is `AsyncConnectionPool(conninfo=..., kwargs={"autocommit": True,
"prepare_threshold": 0, "row_factory": dict_row})` → `AsyncPostgresSaver(pool)` → `await saver.setup()`, with
teardown closing the pool the app owns. That makes `.pool` real because *we* hold it, not because the saver
exposes one.

**Defect 3 — the module docstring contradicts the settings.** `checkpointer.py:9` documents
`conn_string: postgresql+asyncpg://`. `AsyncPostgresSaver` is **psycopg**-based (`AsyncConnection.connect`,
`aio.py:73`) and rejects a SQLAlchemy `+asyncpg` dialect string. `settings.py:140` is
`POSTGRES_URL: str = Field(default="postgresql://user:pass@host/db")` — psycopg-compatible, so the *setting* is
right and the *docstring* is wrong. Any step that copies the docstring's advice breaks it.

**Defect 4 — `AsyncPostgresSaver = Any` fallback (`:26-29`) swallows a real dependency error.** On
`ImportError` the module aliases the class to `typing.Any` and `setup_` returns `None` with a `warning`
(`:51-53`), typed as returning `AsyncPostgresSaver`. `langgraph-checkpoint-postgres` **is** installed (3.0.4),
so this branch is unreachable today; it exists only to defer a hard failure into an
`AttributeError`-on-`None` at `agent_saul/dependencies.py:45`. Delete the fallback; let the import fail loudly.

### What this changes about the plan

| Sub-todo | Was assumed | Actually |
|---|---|---|
| (j) tenacity retries | add tenacity | tenacity is already there (`kb_retry.py`); **fix its policy**, do not add a second one |
| 138(a) checkpointer on `app.state` | uncomment `lifespan.py:295-305` | **rewrite `checkpointer.py` first** — uncommenting as-is is a boot crash |
| teardown | works | never fired; no-op since written |

Neither file is deletable and neither is reusable unmodified. Both land **before** the graph is promoted,
because promoting `ingestion_kb` multiplies `retry_immediate`'s blast radius and the checkpointer is what makes
`Send` fan-out replay-safe.

---

## Shape

Change 1 is **not** "build the new ingestion pipeline". It is **a retarget plus seven correctness fixes**, and
the retarget is the *last* thing that happens, not the first. `ingestion_kb/` already contains the better
implementation (7 real nodes, `Send` fan-out at `nodes.py:200`, one `operator.add` reducer at `state.py:181`,
`retry_immediate` on every I/O call, an LLM-degradation fallback at `nodes.py:442`), and `documents/` already
contains the three things `ingestion_kb` lacks (S3 fetch at `service.py:477`, the `documents.status` column
transitions at `:490/:528/:570`, and per-chunk Graphiti verification at `:673`). So the shape is: **make each
of the two pipelines correct where it is, unify the pieces they duplicate into one place each, then move
`documents/`'s three unique concerns into `ingestion_kb`'s graph and delete the decorative one-node graph.**

The change decomposes into five bands, and the ordering between bands is forced by failure mode, not by taste:

- **Band A — cold fixes (steps 1-5).** Defects that are already wrong today, each independently committable,
  each provable by a command, none needing a running database or a graph. `embedding.py:5`'s logger shadowing,
  `embedder.py:26-29`'s 1536-vs-768 dimension conflict, the `Vector(768)` un-hardcode, the
  `rag_agent_advanced.py` phantom import, the `nodes.py:238` `state.doc_id` `AttributeError`. Band A moves the
  test baseline (6 failures → passes) so every later band has a trustworthy signal.
- **Band B — the seams (steps 6-9).** One embedder, one chunker path, one parse call that does not block the
  event loop, one tokenizer. These are the pieces both pipelines duplicate; unify them **while both pipelines
  still exist**, so each unification is verified twice by two callers rather than once by a new one.
- **Band C — the runtime substrate (steps 10-13).** Rewrite `checkpointer.py`, fix `kb_retry.py`'s policy,
  register `tasks.document_tasks` with Celery, put the graph and the checkpointer on `app.state`. Nothing in
  Band C changes what the pipeline computes; it changes whether a crash is recoverable and whether the worker
  can dispatch at all. **Band C is where "promote `ingestion_kb`" becomes physically possible** — before it,
  the graph has no checkpointer parameter (`graph.py:37-43`) and the live task is unregistered
  (`celery.py:191-196`).
- **Band D — state and identity (steps 14-16).** Pointer State (Up#5) and entity canonicalisation (Trap1).
  Both are cheap now and unfixable later: Pointer State must land **before** the first checkpoint is written or
  every checkpoint row carries full legal-document text through `JsonPlusSerializer`; canonicalisation must land
  **before** the first Graphiti write or the graph accumulates duplicate party nodes that no later pass can
  separate (`dispositions.md` Trap1).
- **Band E — the retarget (steps 17-21).** Hierarchical chunking for legal docs, langextract moved upstream
  (D9), re-ranking wired into the retrieval path (item 195's genuinely-missing third), the `failure`
  short-circuit edge, and finally the fold: `documents/`'s three unique concerns move into the graph, the
  router mounts, `ingestion_graph.py` dies.

Two structural facts govern every band. First, **the modules being promoted have zero test coverage** —
codegraph reports no covering tests for all 7 node factories, `build_document_ingestion_graph`,
`process_document_ingestion`, and `run_document_ingestion_task`, and no test references `ingestion_kb`,
`documents_ingest`, or `ingestion_graph` at all (`scout-ingestion-graphs.md`, Prior art §). There is therefore
**no regression net to promote against**, which is why Band A/B carry mandatory new unit tests (they are the
first evidence that ever existed) while Band E carries mandatory graph-level tests (a promoted graph with no
test is an unverifiable checkbox, which `schema.yaml:406-409` forbids as a task). Second, **the alembic merge
in change 0 gates any step that touches a column type or runs a migration** (D8): on today's two-head tree
with `9f4a1b7c6d2e` un-runnable on a clean DB, `alembic upgrade head` cannot be a Proof line for anything.

What this change deliberately does **not** do: it does not mount the search router (D5.1's explicit carve-out,
gated on D5.2), does not collapse `search_*` into `UnifiedDocument`/`UnifiedChunk` (change 2), does not touch
`MessagesState`/tool registry/prompt adoption (change 3), and does not add a `vector_store` singleton to
`app.state` (DROPped in `dispositions.md` — it would be a third retrieval path).

---

## Ordering constraints

### Cross-change (what must land elsewhere first)

| Constraint | Evidence | Which steps it gates |
|---|---|---|
| **Change 0's alembic head merge must land first.** Two heads (`0004`, `a71f0d7d9c12`) and `9f4a1b7c6d2e` is not runnable on a clean DB — `batch_alter_table("clauses")` at `:63` and `UPDATE clauses` at `:101` presuppose a table no revision creates (`scout-persistence-docling.md` §2). | D8: "the alembic merge gates change 1" | **Steps 3, 12, 13, 20** — anything whose Proof is `alembic upgrade head` or which writes to `clauses`/`chunks`. Band A steps 1, 2, 4, 5 do **not** need it (pure-Python proofs). |
| **Change 0 must delete `todo_temp.py` first.** Both `invalid-syntax` errors in the ruff-125 baseline are that file (`:406`, `:773`) (D11). | D11 | **Every step with a `ruff check` Proof** — until it is gone, the expected error count is 125, after it 123. Each Proof below states which baseline it assumes. |
| **Change 0 must fix `tasks/__init__.py:6-9`.** It imports the reconciliation helpers and re-exports at `:18-20`; deleting the module without editing it breaks every Celery worker at import (`dispositions.md`, change 0 note). | dispositions.md | **Step 12** — you cannot prove Celery registration while the worker cannot import `tasks`. |
| **Change 0 must fix `UserIdDep` (D5.2).** `documents/dependencies.py:61-62` reads `request.state.user_id`, never assigned; the documents router **is** mounted. | D5.2 | **Step 21** (mount + end-to-end upload Proof). Not a blocker for anything earlier, because Band A-D never goes through HTTP. |
| **Change 0's health probe (item 198.2) is the acceptance test for step 13.** `features/health/service.py:160` probes neo4j but not graphiti/cognee; lifespan degrades silently (`lifespan.py:220-223`). | dispositions.md 198.2 | **Step 13** — extend the same probe to report checkpointer + graph presence rather than inventing a second mechanism. |
| **Change 2 owns the schema collapse; change 1 must not pre-empt it.** `UnifiedChunk` has no `updated_at`; `chunks.user_id`/`documents.object_uri` are NOT NULL with no source value. | D5.2, `scout-search.md` §2 | **Step 3** un-hardcodes the dimension *in place* on all three models; it does not merge tables. |
| **Change 3 owns `MessagesState`/middleware retries.** | D8, `dispositions.md` item 172 | **Steps 11, 14** — ingestion state shapes and the retry policy must not contradict change 3's TypedDict + `@wrap_model_call` direction. Recorded in step notes. |

### Within change 1 (the forced edges)

1. **Step 1 (logger fix) precedes everything.** `normalize_embedding` has 15 callers and `logger.warning` at
   `embedding.py:22` raises `AttributeError` on **every** dimension mismatch. Until it is fixed, any step that
   changes a dimension turns a diagnostic warning into a crash, and 6 tests stay red — so no later step has a
   clean baseline to compare against.
2. **Step 2 (embedder dimension conflict) precedes step 3 (un-hardcode).** If you parameterise the ORM column on
   `settings.EMBEDDING_DIMENSION` while `embedder.py:26-29` still claims 1536, you have moved a `DataError` from
   a literal to a config value without removing it.
3. **Step 6 (unified embedder) precedes step 17 (hierarchical chunking).** Chunking changes the *number and
   boundaries* of texts embedded; doing it before the embedder is unified means two embedding paths must both be
   re-validated against new chunk shapes.
4. **Step 8 (unblock docling) precedes step 17.** `HybridChunker` runs on the `DoclingDocument` that
   `parser.py:25` currently discards (`tables=[]` at `:34`); the parse call must return the structure before the
   chunker can consume it, and must not block the loop while doing it.
5. **Step 10 (rewrite `checkpointer.py`) precedes step 13 (`app.state` wiring).** Uncommenting
   `lifespan.py:295-305` against today's `checkpointer.py` is an `AttributeError` at startup — a boot crash, not
   a degraded boot. Proven above.
6. **Step 14 (Pointer State) precedes step 13's checkpointer going live on the *ingestion* graph.**
   `IngestionState.raw_bytes: bytes` (`state.py:172`) and `AppError` in `failure` (`:194`,
   `arbitrary_types_allowed=True` at `:167`) are `JsonPlusSerializer` hazards: the first checkpoint write either
   fails or persists whole documents. Order: rewrite the saver (10) → shrink the state (14) → attach the saver
   to the ingestion graph (13/16).
7. **Step 15 (canonicalisation) precedes step 20 (graph Graphiti writes) absolutely.** Duplicate party nodes
   are unrecoverable after the fact.
8. **Step 11 (`kb_retry` policy) precedes step 13.** Once a checkpointer exists, `retry_immediate`'s
   node-local counter is silently multiplied on replay (`brief:ref:1612-1614`); fixing the policy while there is
   still no checkpointer keeps the two failure modes separable.
9. **Step 12 (Celery registration) precedes step 21 (mount + e2e).** The outbox event
   `event_type="tasks.documents_ingest"` (`documents/service.py:188`) resolves against Celery's registry in
   `OutboxRelay._publish` (`shared/outbox/relay.py:118`); unregistered means `NotRegistered` at dispatch.
10. **Step 19 (`failure` short-circuit edge) precedes step 21.** `graph.py:78` has no conditional edge on the
    `failure` channel, so every guard-clause failure still walks all downstream nodes and each hits its own
    guard. Mounting a router in front of that turns one bad upload into 6 wasted LLM calls.
11. **Step 21 (delete `ingestion_graph.py`) is last, and only after step 20 proves the fold.** Deleting the live
    path before its replacement is proven leaves the repo unbootable-in-effect (uploads accepted, nothing
    processes them).

---

## Steps

**How to read a Proof line.** `pyproject.toml:752-759` sets `addopts` including `--cov-fail-under=80`, and
current coverage is 18.38%. **A fully green suite still exits 1.** Every pytest Proof below therefore names the
*summary line* to compare, never `$?`. Baseline to beat: **55 passed**, **ruff 125 errors** (123 after change 0
deletes `todo_temp.py` — D11), **ty 46 errors**. Where a step's Proof is a ruff/ty count, it states the assumed
baseline explicitly. Commands are always `uv run` (CLAUDE.md).

### Band A — cold correctness fixes (no DB, no graph, no network)

#### Step 1 — Fix the logger submodule shadowing in `src/app/utils/embedding.py`

Inbound dependency: **none.** This is the first step in the change.

`embedding.py:5` is `from app.utils import logger`. This binds the **submodule**, not the loguru object, and
`logger.warning(...)` at `:22` raises `AttributeError` on every dimension mismatch. `normalize_embedding` has
**15 callers**.

The mechanism, stated precisely because it determines the blast radius: `app/utils/__init__.py` *does* export the
loguru object (`from .logger import execution_path, logger, request_state, trace_layer`), so
`from app.utils import logger` is normally correct. But that line sits **after** `from .embedding import
normalize_embedding` in `__init__.py`. When `embedding.py` runs its import, `app.utils` is only partially
initialised — the name `logger` is not yet in its namespace — so Python's circular-import fallback resolves the
attribute to `sys.modules["app.utils.logger"]`, i.e. the module. **Only modules imported *from inside*
`app/utils/__init__.py` are affected.** A grep of `src/app/utils/` for this idiom returns exactly one hit:
`embedding.py:5`. `kb_retry.py:9`, `checkpointer.py:24`, and `retrieval_kb/reranker.py:10` use the same idiom but
live outside the package, resolve after `__init__` completes, and are **correct** — do not "fix" them.

Change: `from app.utils import logger` → `from app.utils.logger import logger`. The repo already uses this exact
form at `shared/rag/document_processing/chunker.py:18` (`from app.utils.logger import logger as loguru_logger`),
so this is house style, not a new convention.

- **Proof:** `uv run pytest -q 2>&1 | tail -3` — the summary line reports **61 passed** (baseline 55 + the 6
  currently-failing tests this converts) and **0 failed**. Coverage still fails the 80% gate; ignore the exit
  code, read the counts.
- **Proof:** `uv run pytest -q 2>&1 | grep -c "AttributeError: module 'app.utils.logger'"` → **0**.
- **Proof:** `uv run ruff check src/app/utils/embedding.py` → `All checks passed!`.

#### Step 2 — Resolve the 1536-vs-768 dimension conflict in `document_processing/embedder.py`

Inbound dependency: **step 1** (a mismatch must be able to log a warning before you change what mismatches).

`shared/rag/document_processing/embedder.py:26-29` returns `{"dimensions": 1536}` for `gemini-embedding-001` and
defaults to 1536, while **every live vector column is `Vector(768)`** (`documents/model.py:94`,
`search/model.py:73`, `memory_schema.py:218`). A 1536-vector insert raises `DataError`. The zero-vector fallbacks
at `:167`, `:177`, `:228` (`[0.0] * config["dimensions"]`) propagate the same wrong width, and they do it
*silently* — a failed embedding currently becomes 1536 zeros rather than an error.

Two sub-changes, both required:
1. The dimension map must read `settings.EMBEDDING_DIMENSION` (`settings.py:212`, value 768, with a
   model/dimension consistency validator at `settings.py:48-60`) instead of returning literals. `settings.py:212`
   is authoritative per `scout-persistence-docling.md` §3.
2. The zero-vector fallback must become an explicit failure, not a silent substitution. A zero vector is a
   *valid* insert that ranks against nothing — the worst possible failure mode for a legal retrieval product.
   Per `EXCEPTION-RULES.md`, raise; do not return a sentinel.

Note `embedder.py` is a **D4 carve-out** — it stays because `ingest_v2.py:17` imports `embed_chunks` from it. So
this is a fix, not a deletion, and `ingest_v2.py` is the caller whose behaviour must not break.

- **Proof:** `uv run rg -n "1536" src/app/shared/rag/document_processing/embedder.py` → **no matches**.
- **Proof:** `uv run rg -n "\[0\.0\] \* " src/app/shared/rag/document_processing/embedder.py` → **no matches**
  (zero-vector fallbacks gone).
- **Proof:** a new unit test in `tests/unit/shared/` asserts `create_embedder()`'s reported dimension equals
  `get_settings().EMBEDDING_DIMENSION`; `uv run pytest tests/unit/shared -q 2>&1 | tail -3` shows it passing and
  the total passed count rising by the number of tests added. **This test is mandatory, not optional** — the
  module has no covering test today and the defect it guards is a silent `DataError`.

#### Step 3 — Un-hardcode `Vector(768)` on the three ORM models (item 198.3)

Inbound dependency: **step 2** (settings must be the single source before models read it) and **change 0's
alembic merge** (see Ordering constraints).

Three sites become `Vector(settings.EMBEDDING_DIMENSION)`: `features/documents/model.py:94`,
`features/search/model.py:73`, `src/database/schemas/memory_schema.py:218`. Plus
`features/search/embeddings.py:16`'s `output_dimensionality=768` → `settings.EMBEDDING_DIMENSION` (this file is
in scope under D5.1 and is the client `documents/service.py:24` already imports).

**Three constraints that make this narrower than it looks, and must be written into `design.md`:**
- **Alembic versions stay frozen literals.** `8a7d9b1c2e3f:50`, `a71f0d7d9c12:63`, `9f4a1b7c6d2e:105`,
  `c0c17c6eb1cc:70` are history. A migration that reads a live setting is not reproducible.
- **The ORM column default is evaluated at import time**, so `EMBEDDING_DIMENSION` must be settled before the app
  boots. This is a boot-order fact, not a runtime knob.
- **Changing the *value* is a re-embedding job, not a DDL migration.** pgvector's `vector(n)` typmod is not
  widenable in place; every HNSW/IVFFlat/diskann index on the column must be dropped first, and
  `ALTER … TYPE vector(1024)` fails while any row holds a different width
  (`scout-persistence-docling.md` §3 Trap). This step makes the model *track* the setting; it does not make the
  setting *changeable*. Say so explicitly or a later reader will try.
- `c0c17c6eb1cc:70`'s `Vector(1536)` on `document_vectors` is inconsistent with every other table and is **out of
  scope** — record it as a known divergence, do not migrate it here.
- `settings.py:208` `PINECONE_DIMENSION=768` has **zero readers**; flag for change 0's deletion sweep, do not
  wire it.

- **Proof:** `uv run rg -n "Vector\(768\)|Vector\(dim=768\)|output_dimensionality=768" src/app/ src/database/` →
  **no matches** (the only remaining hits are under `src/alembic/versions/`, which is correct and expected).
- **Proof:** `uv run python -c "from app.features.documents.model import UnifiedChunk; print(UnifiedChunk.__table__.c.embedding.type.dim)"` → `768`.
- **Proof:** `uv run ty check src/` → **46 errors or fewer** (no new type errors from the settings indirection).

#### Step 4 — Delete the phantom `ingestion.embedder` import in `rag_agent_advanced.py`

Inbound dependency: **step 2** (the replacement is the corrected embedder).

`shared/rag/rag_agent_advanced.py:119,198,267,373` all do `from ingestion.embedder import create_embedder`. There
is no `ingestion` top-level package — `src/` roots are `alembic, app, database, lynk, mcp_core, tasks`. All four
are **function-local** imports, so the module imports cleanly and the failure is deferred to call time as
`ModuleNotFoundError`; that is precisely why ruff and ty never see it
(`scout-persistence-docling.md` §6, confirmed by a repo-wide sweep that found only these four lines).

Retarget to the real module (`app.shared.rag.document_processing.embedder.create_embedder`) **or** delete the
call sites if `graphify affected` proves them unreachable. Prefer retarget: the file also uses
`embedder.embed_query` at `:129,201,270,380,444`, so deletion is a larger decision than this step should make.
Note this file also depends on `match_chunks()`, a **phantom Postgres function** in no migration
(`scout-persistence-docling.md` §1) — record that as a Risk, do not fix it here.

- **Proof:** `uv run rg -n "from ingestion" src/` → **no matches**.
- **Proof:** `uv run python -c "import ast,sys; [ast.parse(open(f).read()) for f in ['src/app/shared/rag/rag_agent_advanced.py']]"` exits 0, and
  `uv run rg -n "^\s+from app.shared.rag.document_processing.embedder import" src/app/shared/rag/rag_agent_advanced.py`
  returns **4 lines**.

#### Step 5 — Fix the `AttributeError` that masks errors in `contextualize_chunk_node`

Inbound dependency: **none** (independent of 1-4; ordered here because it belongs to Band A's cold-fix band).

`shared/langgraph_layer/ingestion_kb/nodes.py:215` types the state as `dict[str, Any]` (correct — it receives a
plain `Send` payload dict constructed at `nodes.py:200`), but the `except LangChainException` branch at `:238`
calls `state.doc_id`. On a dict that is an `AttributeError`, which **replaces** the original `LangChainException`
and destroys the diagnostic. Fix: `state["doc_id"]` (or `state.get("doc_id")` if the `Send` payload does not
guarantee the key — check `nodes.py:194-200`'s constructed dict and match it exactly).

This is the one node whose state is deliberately a `dict` rather than `IngestionState`, which is why the bug
survived: the signature is right and the body is wrong.

- **Proof:** a new unit test in `tests/unit/shared/` invokes the factory's node with a `Send`-shaped dict and an
  `embedding_fn`/LLM double that raises `LangChainException`, and asserts the returned state carries the
  degraded-preamble result **and** that no `AttributeError` is raised. `uv run pytest tests/unit/shared -q 2>&1 | tail -3`
  shows it passing. **Mandatory new test** — this node factory has no covering test (codegraph: "no covering
  tests found" for all 7 factories), and the bug is invisible to lint and types.
- **Proof:** `uv run rg -n "state\.doc_id" src/app/shared/langgraph_layer/ingestion_kb/nodes.py` → **no matches**.

---

## Corrections adopted mid-plan (from `docs/relay/findings-deployment.md`)

Two facts established after this plan was dispatched. Both change steps, so they are recorded here **before**
the affected bands rather than patched into them silently.

### Correction 1 — sub-todo (e) is not a code task. There is no consumer.

The framing I was briefed with — "`tasks.document_tasks` is absent from `celery.py:191-196` `include`, so the
live ingestion task is never registered" — is **wrong in its mechanism**. `src/tasks/__init__.py:4` does
`from .document_tasks import ingest_document`; Celery's `include` names `tasks.example`, `tasks.search_tasks`,
`tasks.billing_tasks`, `tasks.auth_email_tasks`, and importing **any** of them imports the `tasks` *package*
first, which runs `tasks/__init__.py`, which imports `document_tasks`. The `@celery_app.task` decorators execute.
**The task IS in the registry.**

So the `include` omission is a **latent fragility, not a live break** — it holds only by side effect, and it
breaks the moment `tasks/__init__.py` is tidied, which **change 0 is about to do** (`:6-9` imports the
reconciliation module being deleted, re-exported at `:18-20`). That makes this a genuine cross-change hazard:
change 0 edits the file that silently guarantees change 1's dispatch path.

The real breaks rank **ahead** of it and are worse. Ranked causes of "ingestion does not run":

1. **No worker and no beat service exist in the deployment.** `docker-compose.yml` services are exactly
   `rabbitmq`, `timescale`, `caddy`, `ai-service-1`, and `ai-service-1` declares no `command:` so it runs the API
   image CMD. **Nothing consumes the queue.** Every task dispatched from `features/documents/service.py:188`
   enqueues to rabbitmq and is never executed by anything. `beat_schedule`'s 4 billing entries
   (`celery.py:259-276`) have never fired either.
2. **The documented way to start a worker is broken.** `Makefile:52` runs
   `uv run celery -A celery_config worker --loglevel=info`, and **`celery_config` does not exist anywhere in the
   repo** — the real app is `src/app/connections/celery.py`. `make celery` fails at application load, before any
   registration question arises.
3. **Then, and only then, the `include` list.**

Consequence for the plan: **step 12 is replaced by steps 12a/12b/12c in that order** (worker exists → command
works → guarantee made explicit). Sub-todo (e) "Celery for offloading to a queue" cannot be proven by any
code-level check; its Proof must show a process consuming the queue. Also note `docker-compose.yml` mounts
`./scripts/init-db.sql` and **that file does not exist**, so Docker materialises a *directory* there and the
postgres entrypoint finds no `init.sql` — there is no pre-alembic bootstrap, and nothing outside the three
migrations creates any extension. That is change 0 terrain but it is why step 12c's Proof cannot assume a
migrated database.

### Correction 2 — the BM25 extension is `pg_textsearch`, and its availability is a PRECONDITION

Not VectorChord. `pg_textsearch` (Timescale/TigerData) provides the `bm25` index access method and
`to_bm25query()`, and is required by three migrations (`a71f0d7d9c12:26`, `8a7d9b1c2e3f:27`, `9f4a1b7c6d2e:26`).
Full required set: `vector`, `vectorscale` (DiskANN), `pg_textsearch`, `pg_trgm`, `unaccent`, `uuid-ossp`.
Compose image is `timescale/timescaledb-ha:pg18`.

**Whether that image ships `pg_textsearch` is UNVERIFIED.** If it does not, every BM25 path is dead on arrival
regardless of code quality and `CREATE EXTENSION IF NOT EXISTS pg_textsearch` fails the migration outright. This
becomes **step 0**, a precondition with its own Proof, gating steps 18 and 20 (the BM25/RRF harvest and the
re-ranking wiring), because harvesting BM25 into the unified path is worthless if the operator does not exist.

Two further consequences already visible:
- **`clauses_bm25_idx` is queried at `features/search/repository.py:356,361,362` and created at
  `9f4a1b7c6d2e:132` on the `clauses` table that no migration creates.** The `clauses` hole from a third angle:
  the migration that indexes it cannot run and the code that queries it cannot succeed. `legal_rrf_search`
  (`repository.py:308-405`) is therefore **dead against a clean database**, which downgrades my confidence that
  the "existing BM25 + RRF to harvest" (item 195, D5.1) is *running* code rather than merely *written* code. It
  is written and tuned; it has never demonstrably executed. Recorded in Risks and Fog.
- **`features/search/constants.py:15` defines `SEARCH_CHUNKS_BM25_INDEX_NAME` and the SQL at
  `repository.py:415,417,419,430` hardcodes the literal anyway.** `pg_textsearch` requires the *index name*
  inside the query (it reads that index's corpus statistics), so a rename breaks these queries at runtime with
  no lint or type warning. The unused constant is a live hazard, not a style nit. Folded into step 18.

### Step 0 — Precondition: prove the required Postgres extensions exist in the deployed image

Inbound dependency: **none.** This gates steps 18 and 20 and should be run before any of Band E is planned in
detail. It is a check, not a change — if it fails, the answer is a platform decision, not a code edit.

Six extensions are required by the three migrations: `vector`, `vectorscale`, `pg_textsearch`, `pg_trgm`,
`unaccent`, `uuid-ossp`. `pg_textsearch` is the one in doubt: it provides the `bm25` index access method and
`to_bm25query()`, and **nothing else in the repo creates any extension** — `scripts/init-db.sql` is referenced by
`docker-compose.yml` and does not exist, so there is no pre-alembic bootstrap.

- **Proof:**
  `docker run --rm timescale/timescaledb-ha:pg18 ls /usr/share/postgresql/18/extension/ | grep -E 'textsearch|vectorscale|vchord|vector|trgm|unaccent|uuid'`
  → the output **must** contain a `pg_textsearch*` control file, plus `vectorscale`, `vector`, `pg_trgm`,
  `unaccent`, `uuid-ossp`. Record the literal output in `design.md`.
- **Proof (failure branch, and it must be written down before it is needed):** if `pg_textsearch` is absent, the
  step's deliverable becomes a recorded blocker in `design.md` Open Questions naming the two options — change the
  image, or replace `to_bm25query` with a `tsvector`/`ts_rank` implementation — and **steps 18 and 20 are cut
  from change 1** rather than built on a non-existent operator. Do not silently proceed; D5.1's whole
  justification for pulling `search/` into scope is that BM25 "already exists and works", and this Proof is what
  makes that claim true or false.

### Band B — the seams: unify what both pipelines duplicate, while both still exist

#### Step 6 — One embedder in `langchain_layer`, replacing four paths

Inbound dependency: **steps 1-3** (logger must log, dimensions must agree, models must read settings).

Today there are four embedding paths and two mutually incompatible dimensions
(`scout-persistence-docling.md` §5). After D5.1 the target is **one**, not two:

| Path | Today | Fate |
|---|---|---|
| `features/search/embeddings.py:10` `build_embedding_client()` | LangChain `GoogleGenerativeAIEmbeddings`, `output_dimensionality=768` hardcoded at `:16`, **new client per call**, no cache | becomes the seed of the unified embedder; the hardcode dies in step 3 |
| `features/documents/service.py:24` | **imports search's client** — documents and search are already one path (D5.1) | follows the survivor |
| `ingestion_kb/nodes.py:738` `_call_embedding_fn` | duck-typed `embedding_fn` (tries `aembed_query`, then `ainvoke`, then `__call__`), **no `task_type`**, one text per call, redis-cached at `:716-736`, `normalize_embedding` at `:733`, `retry_immediate` at `:729` | **this is the genuine second path**; it is replaced, but its cache/normalize/retry behaviour is what the survivor must keep |
| `document_processing/embedder.py:51` | raw `google.genai` SDK, in-memory cache at `:299`, `batch_size=100` at `:76` | stays (D4 carve-out for `ingest_v2.py`) but must not be a *second live path* — it is batch/offline only. Say this in `design.md`. |

Acceptance criteria for the survivor, taken from `scout-search.md` §6 and `scout-persistence-docling.md` §5:
single-text **and** batched embedding; explicit `task_type` on both sides (`RETRIEVAL_QUERY` for queries,
`RETRIEVAL_DOCUMENT` for documents — `ingestion_kb` passes **none** today, so its stored vectors are asymmetric
with the query side and that is a silent relevance bug, not a style issue); dimension from
`settings.EMBEDDING_DIMENSION`; redis cache with a documented key and TTL (there are **two** independent
`_cached_embedding` implementations today — `documents/service.py:813` and `ingestion_kb/nodes.py:716` — collapse
to one); `normalize_embedding` applied **consistently or never** so one `chunks.embedding` column never mixes
conventions (search stores raw, `ingestion_kb` stores normalized — cosine distance is scale-invariant so ranking
survives today, but any switch to inner-product ops silently mis-ranks a mixed column); process-lifetime client
reuse, not per-call construction (`brief:01-…:86` — "the model object should be a module-level singleton");
`retry_immediate` wrapping (via step 11's corrected policy); and batch size preserved at 200
(`search/service.py:314`, `documents/service.py:635`) rather than `ingestion_kb`'s one-at-a-time.

- **Proof:** `uv run rg -n "GoogleGenerativeAIEmbeddings|genai.Client\(" src/app/ | grep -v document_processing/embedder.py`
  → exactly **one** construction site, inside `shared/langchain_layer/`.
- **Proof:** `uv run rg -n "task_type" src/app/shared/langchain_layer/ src/app/features/documents/ src/app/features/search/`
  → every `aembed_query`/`aembed_documents` call site has an explicit `task_type`; zero call sites without one.
- **Proof:** `uv run rg -n "_cached_embedding" src/app/` → **one** definition (was two).
- **Proof:** new unit tests in `tests/unit/shared/` assert (a) a cache hit does not call the underlying client
  twice for identical text, (b) `task_type` reaches the client, (c) the reported dimension equals
  `settings.EMBEDDING_DIMENSION`. `uv run pytest tests/unit/shared -q 2>&1 | tail -3` shows them passing.
  **Mandatory** — the unified embedder becomes the single point of failure for every retrieval path in the repo,
  and no embedding path has a test today.

#### Step 7 — `CacheBackedEmbeddings` decision, and resolve the `langchain_classic` collision (item 171)

Inbound dependency: **step 6** (there must be one embedder to wrap).

Zero hits for `CacheBackedEmbeddings` in `src/`. The docs name the exact defect —
*"Embeddings aren't cached. `aembed_batch` calls the API every time… a simple LRU cache keyed on SHA256(text)
would eliminate redundant API calls entirely"* (`brief:ref:2049`) — and prescribe
`CacheBackedEmbeddings.from_bytes_store(underlying, store, namespace=underlying.model)`
(`brief:13-…:30-54`).

**The collision is real and must be decided in `design.md`, not in code.** The import is
`from langchain_classic.embeddings import CacheBackedEmbeddings` (verified present:
`.venv/.../langchain_classic/embeddings/__init__.py:14`, `langchain-classic` 1.0.3 installed, and it is **not**
in `langchain_core` or `langchain` — grep for `class CacheBackedEmbeddings` in those trees returns nothing). But
`brief:ref:60-63` (note 9) forbids legacy import paths, and `langchain_classic` is the v0-compat shim.
Recommendation: **do not adopt `CacheBackedEmbeddings`.** Step 6 already delivers what it provides (SHA256-keyed
caching) via **redis**, which is (a) shared across processes — a `LocalFileStore("./cache/")` is per-container and
useless behind `ai-service-1` scaling, (b) already the repo's cache substrate
(`app.state.redis`, `lifespan.py:190`), and (c) already implemented twice in this exact shape at
`documents/service.py:813` and `ingestion_kb/nodes.py:716`. Adopting the classic shim would add a legacy import,
a second cache tier, and a `.model`-attribute requirement on the embeddings object, to duplicate a working
mechanism. Record it as a **considered-and-rejected alternative** in `design.md` Decisions (the schema requires
alternatives per decision) so item 171 is answered on the record rather than dropped.

- **Proof:** `uv run rg -n "langchain_classic" src/` → **no matches**.
- **Proof:** `design.md` contains a Decisions entry naming `CacheBackedEmbeddings`, the `langchain_classic`
  import-rule collision, and redis as the chosen mechanism. `uv run rg -n "CacheBackedEmbeddings" openspec/changes/<change-1-slug>/design.md`
  → at least one match.

#### Step 8 — Unblock the docling event loop and stop discarding parsed tables

Inbound dependency: **none** technically, but it must precede step 17 (hierarchical chunking consumes the
`DoclingDocument` this step stops throwing away).

`features/documents/parser.py` has three defects in 16 lines:
- **`:25` calls synchronous `converter.convert()` inside `async def`** with no offload — it blocks the event loop
  for an entire OCR/layout pass. `:29` `export_to_markdown()` is also sync/CPU. This is a **documented convention
  violation**, not merely slow: `config.yaml` (per `conventions-openspec-skeleton.md`) states *"Async-first: all
  I/O through async clients… no blocking calls in async functions."*
- **`:24` rebuilds a fresh `DocumentConverter` per call** via `create_document_converter(gpu_available=False)`.
  Docling converters load models; this is the same anti-pattern as `build_chat_model()` that `brief:01-…:86`
  condemns.
- **`:34` hardcodes `tables=[]`** — docling parses table structures and the parser then discards them. For a
  legal-contract product, schedules and payment tables are among the highest-value content in the document.

Fix shape: offload with `asyncer.asyncify` — the repo's established pattern, already used correctly by
`ingestion_kb`'s `_parse_document_with_docling` (`nodes.py:407`, `_sync_parse` at `:412`, `asyncer.asyncify` at
`:439`) and by `retrieval_kb/reranker.py:53`. So **the correct implementation already exists inside the module
being promoted**; the live path is the one that is wrong. That makes this step partly a deletion: `documents/`'s
parser converges on `ingestion_kb`'s. Hoist the converter to a process-lifetime singleton, and populate `tables`
from `result.document` instead of `[]`.

Note `:20` does `del content_type` — the parameter is accepted and ignored. Either use it to select a converter
pipeline or remove it from the signature; a silently-ignored argument is how callers come to believe
content-type is honoured.

- **Proof:** `uv run rg -n "converter.convert\(|export_to_markdown\(" src/app/features/documents/parser.py`
  → every hit is inside a function passed to `asyncer.asyncify` (or the file is deleted in favour of
  `ingestion_kb`'s parse path).
- **Proof:** `uv run rg -n "tables=\[\]" src/app/features/documents/` → **no matches**.
- **Proof:** `ast-grep scan src/` reports no new findings versus baseline; and a new unit test asserts
  `parse_document` on a small fixture PDF/DOCX returns `len(parsed.tables) > 0` for a document containing a
  table, and that the coroutine yields control (e.g. it completes while a concurrent `asyncio.sleep(0)` ticker
  advances). `uv run pytest tests/unit/documents -q 2>&1 | tail -3` shows both passing. **Mandatory** — a
  blocking-call regression is invisible to ruff, ty, and every existing test.

#### Step 9 — Tokenizer: cache it, and record why `transformers` cannot be dropped (item 176)

Inbound dependency: **none.** Ordered here because step 17 depends on the tokenizer being cheap to obtain.

Item 176 as worded is *"check `sentence_transformers`/`AutoTokenizer`, or a langchain replacement"*, scoped by
`dispositions.md` to *"if the only use is token counting, drop the direct `transformers` dependency"*.
**That premise is false, and the evidence defeats the item — see Conflicts surfaced.** Both packages are
load-bearing for work that is IN scope:
- `shared/rag/document_processing/chunker.py:16` `from transformers import AutoTokenizer`, used at `:23-26`
  (`get_tokenizer`) and passed to `HybridChunker(tokenizer=…)` at `:29-36`. Docling's `HybridChunker` **is** the
  chosen splitter (item 162, and step 17), so `AutoTokenizer` is a dependency of the thing this change adopts.
- `shared/langgraph_layer/retrieval_kb/reranker.py:8` `from sentence_transformers import CrossEncoder` —
  `CrossEncoderReranker` is the re-ranking implementation item 195 needs (step 20).
- `pyproject.toml:51` declares `sentence-transformers>=5.1.2` as a direct dependency; `:74` declares `docling`.

What remains actionable, and it is real:
- **`get_tokenizer` is uncached** (`chunker.py:23-26`): every call runs `AutoTokenizer.from_pretrained`, which
  hits disk or the network on first use and is fully sync. Cache it at process scope (and offload the first load)
  so step 17 does not reintroduce a blocking call.
- **The tokenizer is not the embedding model's tokenizer.** Default is
  `"sentence-transformers/all-MiniLM-L6-v2"` (`chunker.py:23`) while embeddings are Gemini. So `max_tokens=512`
  is enforced against the *wrong* tokenizer — chunks are budgeted by a MiniLM count and then embedded by Gemini.
  This is a correctness gap in the chunk-size guarantee. Either switch to a Gemini-consistent counter or record
  the divergence explicitly with the safety margin it implies.

- **Proof:** `uv run rg -n "from_pretrained" src/app/` → every hit is behind a cache (e.g. `functools.lru_cache`
  or a module-level singleton), verified by a test that calls `get_tokenizer()` twice and asserts the underlying
  loader ran once.
- **Proof:** `uv run rg -n "sentence-transformers|transformers" pyproject.toml` → dependencies **still present**,
  and `design.md` carries the recorded reason (item 176 is answered "cannot drop", not left silent).
- **Proof:** `uv run pytest tests/unit/shared -q 2>&1 | tail -3` shows the caching test passing.

---

### What "the repo still boots" means, per band

Every step must leave the repo bootable. That claim is only meaningful if it has a command, and it differs by band:

| Band | "Still boots" means | Proof of it |
|---|---|---|
| **A, B** | The app imports and the ASGI app constructs. No DB, worker, or graph involved. | `uv run python -c "from app.main import app; print(type(app).__name__)"` → `FastAPI`. Plus `uv run pytest -q 2>&1 \| tail -3` passed-count ≥ the band's stated floor. |
| **C** | Lifespan startup completes **and** shutdown completes, with each newly-wired `app.state` attribute present. Degradation must stay degradation, never a crash. | `uv run python -c` driving the lifespan context manager directly (`async with app.router.lifespan_context(app)`) and asserting the attribute set; plus the change-0 health endpoint reporting each client. |
| **D** | Same as C, **plus** a checkpoint round-trip: write a checkpoint and read it back, proving every state channel is `JsonPlusSerializer`-encodable. | `graph.aget_state(config)` returns a `StateSnapshot` after one `ainvoke` on a stub state. |
| **E** | Same as D, **plus** an end-to-end upload produces a terminal document status and non-zero chunk rows. | HTTP `POST` through the mounted router, then `SELECT status, count(*)` against the DB. |

Bands A and B do **not** require a database, a worker, or `pg_textsearch`; that is deliberate, and it is why the
seven correctness bugs are front-loaded rather than bundled with the retarget.

### Band C — the runtime substrate (checkpointer, retries, worker, `app.state`)

#### Step 10 — Rewrite `shared/langgraph_layer/checkpointer.py` (item 138 residue a)

Inbound dependency: **none** in code, but its Proof needs a reachable Postgres, so it follows change 0's alembic
merge in practice.

This is a rewrite, not an uncomment. Four defects, all established above in the prior-art section and all
verified against installed `langgraph-checkpoint-postgres` 3.0.4:

1. `from_conn_string` is `@classmethod @asynccontextmanager` (`aio.py:55-80`). `checkpointer.py:56-57` binds the
   context manager and calls `.setup()` on it → **`AttributeError`, uncaught** by the
   `except (ConnectionError, TimeoutError, OSError)` at `:58`. **Uncommenting `lifespan.py:295-305` as-is turns a
   degraded boot into no boot at all.**
2. `teardown_langgraph_checkpointer` tests `hasattr(checkpointer, "pool")` at `:83`; `AsyncPostgresSaver` sets
   only `conn`, `pipe`, `lock`, `loop`, `supports_pipeline` (`aio.py:37-53`) and **has no `pool` attribute** — the
   teardown has been a silent no-op since it was written, while `lifespan.py:317` calls it on every shutdown.
3. The module docstring at `:9` says `postgresql+asyncpg://`; the saver is **psycopg**-based
   (`AsyncConnection.connect`, `aio.py:73`) and `settings.py:140` is `postgresql://…`. The setting is right, the
   docstring is wrong; do not follow the docstring.
4. `AsyncPostgresSaver = Any` on `ImportError` (`:26-29`) makes `setup_` return `None` while typed as returning
   the saver, deferring a hard dependency failure into an `AttributeError` at
   `agent_saul/dependencies.py:45`. The package **is** installed, so delete the fallback and let an import error
   be an import error (`EXCEPTION-RULES.md`: raise, do not sentinel).

Target shape: the **application** owns an `AsyncConnectionPool(conninfo=settings.POSTGRES_URL,
kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})`, constructs
`AsyncPostgresSaver(pool)`, awaits `saver.setup()` once, and stores **both** on `app.state` so teardown closes the
pool it actually owns. `brief:ref:1465` mandates the async saver in production ("always use the async version to
avoid blocking your DB connection pool"); a pool we hold is the only way `teardown` can be non-vacuous.

- **Proof:** `uv run rg -n "from_conn_string" src/app/` → **no matches**.
- **Proof:** `uv run rg -n 'hasattr\(checkpointer, "pool"\)|AsyncPostgresSaver = Any' src/app/` → **no matches**.
- **Proof:** against a reachable Postgres,
  `uv run python -c "<await setup, assert isinstance(saver, AsyncPostgresSaver), await teardown>"` exits 0 and
  `psql -c "\dt checkpoint*"` lists `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`,
  `checkpoint_migrations` — the tables `saver.setup()` creates. Their existence is the only positive proof
  `setup()` ran; today they cannot exist because `setup()` has never been reached.
- **Proof:** `uv run ty check src/app/shared/langgraph_layer/checkpointer.py` → the two existing suppression
  comments (`ty:ignore[unresolved-attribute]` at `:57`, `ty:ignore[invalid-return-type]` at `:67`) are **gone**,
  and the total `ty` count is ≤ 46. Those suppressions were masking exactly this bug — a typed-ignore that hides
  a real `AttributeError` is the tell.

#### Step 11 — Correct `kb_retry.py`'s retry policy (sub-todo (j), reshaped)

Inbound dependency: **step 10** conceptually (the replay trap only becomes live once checkpoints exist), but
committable independently and **must** land before step 13.

`retry_immediate` (`kb_retry.py:19`) already uses tenacity, so sub-todo (j) is **already implemented — badly**.
Four fixes:
1. **`retry=retry_if_exception_type(Exception)` (`:29`) → a named, specific exception tuple.** A catch-all around
   node-internal code is the pattern `brief:ref:1633` forbids, because `interrupt` pauses by *raising*, and a
   catch-all swallows it and retries three times. Ingestion has no `interrupt` today; change 3 adds HITL, so
   fixing it now is cheaper than debugging it then.
2. **`wait=wait_none()` (`:28`) → exponential backoff with jitter.** Three immediate attempts against a
   rate-limited Gemini endpoint produce three 429s in ~0ms. The docs' own capability list says "exponential
   backoff" (`brief:ref:1117-1118`).
3. **Stop collapsing every failure into `TransientExternalError`.** `reraise=True` at `:30` is dead code: the
   `async for` is wrapped in `except Exception` at `:41` which re-wraps at `:43`. Because
   `TransientExternalError` is **not** a `LangChainException`, this silently defeats the graph's own degradation
   branches at `nodes.py:182`, `:236`, `:289` — they can never fire for a wrapped call. Preserve the original
   type (that is what `reraise=True` is for) and let `TransientExternalError` be reserved for genuine
   retries-exhausted, chained with `from exc`.
4. **Add an idempotency note at each call site.** `brief:ref:1612-1614`: the checkpointer's recovery unit is the
   **node**, not the statement, so a node-local attempt counter resets on replay and the retry budget is silently
   multiplied. The remedy the docs give is task decomposition plus idempotency keys, not a bigger decorator.

`design.md` must state the layering, because it is the resolution of a live conflict (see Conflicts surfaced):
**tenacity stays at I/O-client boundaries only** — `kb_retry.py`, `connections/redis.py`,
`razorpay_client.py` — and **middleware (`@wrap_model_call`) owns model/tool retries in change 3.** No `tenacity`
retry may wrap a whole graph node.

- **Proof:** `uv run rg -n "retry_if_exception_type\(Exception\)|wait_none" src/app/` → **no matches**.
- **Proof:** a new unit test asserts (a) a non-listed exception type propagates on the **first** attempt with no
  retry, (b) a listed transient type retries exactly `attempts` times and the final raise chains the original via
  `__cause__`, (c) the original exception type is observable to a caller catching `LangChainException`.
  `uv run pytest tests/unit/shared -q 2>&1 | tail -3` shows them passing. **Mandatory** — `retry_immediate`
  wraps every I/O call in the pipeline being promoted, so it is the highest-fan-in untested function in change 1.

#### Step 12a — Add a Celery worker (and beat) service to the deployment (sub-todo (e), the actual blocker)

Inbound dependency: **none.** Ranked first among the Celery causes per `findings-deployment.md` §2.

`docker-compose.yml` services are exactly `rabbitmq`, `timescale`, `caddy`, `ai-service-1`, and `ai-service-1`
declares **no `command:`** so it runs the API. **Nothing consumes the queue.** Every task dispatched from
`features/documents/service.py:188` enqueues to rabbitmq and is never executed by anything; the four
`beat_schedule` billing entries (`celery.py:259-276`) have never fired either. **No amount of code refactoring
makes sub-todo (e) true** — it needs a process.

Add a `worker` service (`celery -A app.connections.celery:celery_app worker`) and a `beat` service, both on the
API image, both depending on `rabbitmq` and `timescale`. `task_create_missing_queues=False` (`celery.py:229`)
means the queue set is fixed at `task_queues` (`:233-252`) — one queue plus one DLQ — so decide here whether
ingestion gets a dedicated queue. Recommendation: **yes**, one `ingestion` queue with its own concurrency, because
ingestion tasks are minutes-long CPU/LLM work and will otherwise starve the sub-second billing and auth-email
tasks sharing the default queue. Note `billing.*` and `document_extraction.legal_batch` do not match the
`tasks.*` pattern in `task_routes` (`:253-258`) and fall to `task_default_queue` (`:225`) by default — that is
pre-existing and should be recorded, not fixed here.

- **Proof:** `docker compose config --services` lists `worker` (and `beat`) alongside the existing four.
- **Proof:** `docker compose up -d && docker compose exec worker celery -A app.connections.celery:celery_app inspect registered | grep tasks.documents_ingest`
  → the task name appears. This is the **only** check that proves a consumer exists; a code-level grep cannot.
- **Proof:** `docker compose exec worker celery -A app.connections.celery:celery_app inspect active_queues`
  lists the ingestion queue.

#### Step 12b — Fix the documented worker command

Inbound dependency: **step 12a** (fix the command to match the service you just added, so the two cannot drift).

`Makefile:52` runs `uv run celery -A celery_config worker --loglevel=info`, and **`celery_config` does not exist
anywhere in the repo** — the real app is `src/app/connections/celery.py`. `make celery` fails at application load
with "Unable to load celery application", before any registration question arises. Point it at
`app.connections.celery:celery_app`, matching step 12a exactly.

- **Proof:** `make celery` starts and logs the Celery banner including `[tasks]` with `tasks.documents_ingest`
  listed; it does **not** print "Unable to load celery application".
- **Proof:** `uv run rg -n "celery_config" . --glob '!.venv'` → **no matches**.

#### Step 12c — Make task registration explicit and typed (item 198.4, reframed as latent-not-live)

Inbound dependency: **step 12b**, and **coupled to change 0** — see below.

**Corrected mechanism:** `tasks.document_tasks` is absent from `include` (`celery.py:191-196`) but the task **IS**
registered, transitively: `src/tasks/__init__.py:4` does `from .document_tasks import ingest_document`, and
importing any listed `tasks.*` module imports the `tasks` package first, running `__init__.py`. So this is a
**latent fragility, not a live break** — and it breaks the moment `tasks/__init__.py` is tidied, which
**change 0 is about to do** (`:6-9` imports the reconciliation module being deleted, re-exported at `:18-20`).
That is the cross-change hazard: change 0 edits the one file that silently guarantees change 1's dispatch path.

Work: add `tasks.document_tasks`, `tasks.pageindex_tasks`, `tasks.document_extraction_tasks`, and
`tasks.auth_email_tasks_typed` to `include` so registration no longer depends on a side effect; then add typed
signatures. `pageindex_tasks` currently raises `NotImplementedError` and is a **D4 carve-out** (todo (b) defers
pageindex) — include it so it is *registered and explicit*, do not implement it.

The "string dispatch, no type safety" half of 198.4 is real: `documents/service.py:188` passes
`event_type="tasks.documents_ingest"` as a string to `with_outbox`, resolved against Celery's registry in
`shared/outbox/relay.py:118`. A rename breaks dispatch at runtime with no lint or type warning. Minimum fix: a
single module of task-name constants that both the producer and the `@celery_app.task(name=…)` decorator read, so
the two cannot drift. Do **not** attempt a full typed-signature framework here — that is a subsystem, and
`auth_email_tasks_typed` already hints at a competing approach that change 3 should reconcile.

- **Proof:** `uv run rg -n "include=" -A 10 src/app/connections/celery.py` shows all eight task modules listed.
- **Proof:** the ordering guarantee is now independent of `tasks/__init__.py`:
  `uv run python -c "import app.connections.celery as c; ks=[k for k in c.celery_app.tasks if 'document' in k]; print(sorted(ks))"`
  → includes `tasks.documents_ingest`, **and still does after** `tasks/__init__.py`'s reconciliation imports are
  removed. Run it both before and after change 0's edit; the second run is the real Proof.
- **Proof:** `uv run rg -n '"tasks\.documents_ingest"' src/` → exactly **one** definition site (the constant), and
  `documents/service.py:188` references the constant rather than a literal.

#### Step 13 — Put the ingestion graph and the checkpointer on `app.state` (sub-todo (f))

Inbound dependency: **steps 10, 11** (a saver that constructs, a retry policy that does not swallow), and
change 0's health probe (item 198.2) which is this step's acceptance test.

`config.yaml` states the convention: *"shared clients live in lifespan and are read from `app.state`."* Every row
below is therefore a **convention violation**, not merely a bug (`scout-ingestion-graphs.md` §3):

| Attribute | Read at | Set in lifespan? |
|---|---|---|
| `ingestion_graph` | `features/ingestion/dependencies.py:8` → `IngestionGraphDep` | **NO** — commented `lifespan.py:241` |
| `langgraph_checkpointer` | `agent_saul/dependencies.py:45` (**unguarded**), `lifespan.py:317` (guarded by `hasattr` at `:316`) | **NO** — commented `:295-305` |

Uncomment and correct `lifespan.py:235-248`: build the Gemini `ingestion_llm` once (`:236-240`,
`temperature=0.1`, `retries=0` — note `retries=0` is deliberate and correct now that step 11 owns retries), then
`app.state.ingestion_graph = build_ingestion_graph(...)` with the step-6 embedder in place of
`build_embedding_client()` at `:244`. All five dependencies are already closure-captured by the node factories
(`graph.py:46-73`) and `build_ingestion_graph`'s own docstring says "once during application startup"
(`graph.py:44`) — this matches `brief:01-…:43-70`'s build-once shape exactly, so there is no design question
here, only wiring.

Also wire `langgraph_checkpointer` from step 10, since `lifespan.py:317` already tries to tear it down and
`agent_saul/dependencies.py:45` reads it unguarded. Leave `pageindex_client` (`:249`) commented — no reader
exists in `src/` (`scout-ingestion-graphs.md` §4), so wiring it would create an unused client.
**Do not** add `app.state.vector_store` — DROPped in `dispositions.md`: zero read sites exist, and under D5.1
retrieval is raw asyncpg + `pg_textsearch`, so a LangChain `VectorStore` object would be a **third** retrieval
path. Record the gap.

- **Proof:** `uv run python -c "<async with app.router.lifespan_context(app): assert app.state.ingestion_graph is not None; assert app.state.langgraph_checkpointer is not None>"`
  exits 0 — and the same script exits 0 through **shutdown**, proving step 10's teardown no longer no-ops.
- **Proof:** the change-0 health endpoint (`features/health/service.py`) reports the checkpointer and graph as
  present; `curl -s localhost:8000/api/v1/health | jq '.data.dependencies'` shows both. Lifespan already degrades
  silently (`lifespan.py:220-223` sets `graphiti = None` and continues), so the probe is the **only** observable
  signal that degradation happened.
- **Proof:** `uv run rg -n "app\.state\.ingestion_graph|app\.state\.langgraph_checkpointer" src/app/lifecycle/lifespan.py`
  → both present and **not** inside a comment (`uv run rg -n "^\s*#.*app\.state\.(ingestion_graph|langgraph_checkpointer)"`
  → no matches).

### Band D — state and identity (the two "cheap now, unfixable later" steps)

#### Step 14 — Pointer State: stop putting documents in checkpointed channels (Up#5)

Inbound dependency: **step 10** (there must be a saver), and it **must precede step 16** (the saver going live on
the ingestion graph). This is the ordering that matters most in Band D: once a checkpoint is written, the shape of
what was written is history.

`IngestionState` (`ingestion_kb/state.py:166`) is a Pydantic `BaseModel` with `extra="forbid"` and
`arbitrary_types_allowed=True` (`:167`). Two channels are `JsonPlusSerializer` hazards:
- **`raw_bytes: bytes` (`:172`)** — the entire uploaded document. `JsonPlusSerializer` (ormsgpack + JSON,
  `brief:ref:1604-1609`) will encode `bytes`, so this does not *fail*; it **succeeds and writes the whole legal
  document into every checkpoint row**, at `sync`/`async` durability, once per superstep. That is the exact defect
  Up#5 names.
- **`AppError` in `failure` (`:194`)** — an arbitrary object permitted only by `arbitrary_types_allowed`. The docs
  are explicit that unsupported objects need `pickle_fallback` (`brief:ref:1604-1609`); relying on that is a
  silent-corruption path across schema versions.

Target: state carries **UUIDs and small scalars**; the bytes live in S3 and are fetched inside the node that needs
them (which is also the seam that lets step 22 fold `documents/`'s S3 fetch at `service.py:477` into the graph —
Pointer State and the fold are the same refactor seen from two angles). `failure` becomes a serialisable
structure (code + message + note), not an exception instance. The
`contextualized_chunks: Annotated[list[ContextualizedChunk], operator.add]` reducer (`:181`) stays — it is the
one correct reducer in the repo — but each accumulated item must itself be small; if a contextualised chunk
carries full text, the reducer multiplies the payload across the `Send` fan-out.

**Do not convert `IngestionState` to a `TypedDict` in this change.** That is change 3's decision and it is
genuinely unresolved for bare `StateGraph`: `brief:ref:1341-1345` ("custom state schemas must be TypedDict…
Pydantic models and dataclasses are no longer supported") is scoped to `create_agent`'s `state_schema`, and
`brief` Fog §1 says explicitly *"do not treat note 67 as settled for bare `StateGraph`"*. `ingestion_kb` uses bare
`StateGraph(IngestionState)` (`graph.py:45`). Keeping Pydantic here is defensible; what is **not** defensible is
adding new Pydantic-only affordances (validators, computed fields, arbitrary types) that a later TypedDict
conversion would have to unpick. Constrain this step to *shrinking* the channels. Record the direction in
`design.md` so change 3 inherits a state that is already TypedDict-convertible.

- **Proof:** `uv run rg -n "raw_bytes|arbitrary_types_allowed" src/app/shared/langgraph_layer/ingestion_kb/state.py`
  → **no matches**.
- **Proof:** a serialisation test: construct a populated `IngestionState`, round-trip it through
  `langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer().dumps_typed(...)`/`loads_typed(...)`, assert equality
  **and** assert the serialised payload is under a stated byte budget (e.g. 8 KB) so a regression that reintroduces
  document text fails loudly. `uv run pytest tests/unit/shared -q 2>&1 | tail -3` shows it passing. **Mandatory** —
  a size regression here is invisible to every other check and only shows up as checkpoint-table bloat in
  production.
- **Proof:** `uv run rg -n "AppError" src/app/shared/langgraph_layer/ingestion_kb/state.py` → **no matches**.

#### Step 15 — Canonicalise entities before any Graphiti write (Trap1)

Inbound dependency: **none** in code. It **must** precede step 22 (the graph's Graphiti writes going live).
Absolute ordering: duplicate party nodes are unrecoverable after the fact.

Zero `canonical*` hits in `src/` (`dispositions.md` Trap1). Today `make_classify_extract_node`
(`ingestion_kb/nodes.py:259`) extracts entities and `_store_entities`/`_store_relationships` write them, then
`make_graphiti_upsert_node` (`:354`) writes episodes — all keyed on **raw extracted text**. "Acme Corp.",
"Acme Corporation", and "ACME CORP" become three party nodes in the same contract graph. Once written, no later
pass can separate a genuine second party from a spelling variant, because the evidence that they were the same
was the extraction context that is now gone.

Work: a canonicalisation function that maps extracted party text to a stable `party_id`, and every entity and
relationship write keyed on `party_id` with the raw surface form retained as an attribute (never discarded — it is
the audit trail a legal product needs). Note this rhymes with change 3's Trap2 ("hash structural IDs, never
content") and with `brief:ref:1614`'s idempotency-key requirement: `party_id` **is** an idempotency key, so
canonicalisation and replay-safety are one mechanism, not two. Say so in `design.md` so change 3 does not build a
second one.

There are three Graphiti call sites today (`scout-ingestion-graphs.md` §1): `documents/service.py:596`, `:601`,
and `:673` → `write_and_verify_chunk`. All three must go through canonicalisation, or the fold in step 22 will
carry uncanonicalised writes into the promoted graph.

- **Proof:** `uv run rg -n "canonical" src/app/shared/langgraph_layer/ingestion_kb/ src/app/features/documents/`
  → the canonicaliser is defined once and referenced at every entity/relationship/episode write site.
- **Proof:** a unit test asserts `canonicalise("Acme Corp.") == canonicalise("ACME CORPORATION")` and that two
  genuinely different parties do **not** collide; plus a test that `_store_entities` called twice with variant
  surface forms produces **one** row. `uv run pytest tests/unit/shared -q 2>&1 | tail -3` shows them passing.
  **Mandatory** — this is the only step in the change whose failure is permanent, so its test is the most
  load-bearing new test in change 1.
- **Proof:** `uv run rg -n "add_episode|write_and_verify_chunk" src/app/` → every call site is downstream of
  canonicalisation (verified by reading the call, not by grep alone — record the audit in the task).

#### Step 16 — Attach the checkpointer to the ingestion graph, with `thread_id` and a durability choice

Inbound dependency: **steps 10, 13, 14** — a working saver, on `app.state`, and a state small enough to persist.

`build_ingestion_graph` (`graph.py:37-43`) has **no `checkpointer` parameter** and `graph.py:84` calls
`graph.compile()` with no argument. So even with step 13's saver on `app.state`, the ingestion graph gets no
persistence. Add the parameter and pass it through.

Three things must be decided here, not left implicit:
- **`thread_id` is not optional** (`brief:08-…:14-18`, `ref:1409`). `features/ingestion/service.py` already puts
  `thread_id` in the state dict (`scout-ingestion-graphs.md` §5), but state is not config — it must go in
  `config={"configurable": {"thread_id": …}}` on `ainvoke`. Choose the document ID as the thread key so a retried
  upload resumes rather than duplicating.
- **Durability mode.** `brief:ref:1616-1619`: `exit` gives best performance but "you cannot recover from system
  failures that occur mid-execution" — which is precisely the failure this change exists to fix (today a crash at
  stage 9 replays stages 1-8, `scout-ingestion-graphs.md` §1). Recommend **`async`** as the default: it persists
  while the next step executes, with a small crash-window risk, and ingestion supersteps are long enough
  (LLM calls) that the write is fully hidden. `sync` is defensible for the Graphiti node specifically since its
  writes are the non-idempotent ones. State the choice and its reason.
- **Schema-version hazard.** `brief:ref:105` — "always normalise agent state after fetching from checkpointer so
  that there is no version mismatch." Step 14 **changes the state schema**, so any checkpoint written before it is
  unreadable after. Because no checkpointer has ever run (step 10 proves `setup()` was never reached, so the
  checkpoint tables cannot exist), there is **no legacy data** — this is the one free window to change the schema,
  and it closes permanently at step 16. Note it in `design.md` Migration Plan as "no backfill required, by
  accident of the checkpointer never having worked", because that reasoning is not obvious to a later reader.

- **Proof:** `uv run rg -n "def build_ingestion_graph" -A 10 src/app/shared/langgraph_layer/ingestion_kb/graph.py`
  shows a `checkpointer` parameter, and `graph.compile(checkpointer=...)`.
- **Proof:** one `ainvoke` with `config={"configurable": {"thread_id": "<uuid>"}}` on a stub state, then
  `await graph.aget_state(config)` returns a `StateSnapshot` whose `.values` contains the expected channels, and
  `psql -c "select count(*) from checkpoints"` → **> 0**. This is the band's boot proof.
- **Proof:** re-invoking with the **same** `thread_id` after an induced failure in a late node does **not** re-run
  the early nodes — assert by counting calls on a spy injected into the parse node. This is the single check that
  proves the change's headline benefit (per-stage recovery) actually works; without it, "resumable" is a claim.

### Band E — the retarget (chunking, item 195, the fold)

#### Step 17 — Hierarchical chunking for legal documents via docling `HybridChunker` (todo (a), item 162)

Inbound dependency: **steps 6, 8, 9** (one embedder, a parse that returns structure without blocking, a cached
tokenizer).

The gap is precise: `HybridChunker` **already exists and is correctly configured** —
`shared/rag/document_processing/chunker.py:29-36` builds `HybridChunker(tokenizer=…, max_tokens=config.max_tokens,
merge_peers=True)`, giving heading-path-contextualised, token-bounded, peer-merged chunks keyed to the
`DoclingDocument` structure. **It is reachable only for `document_kind == "generic"`.** For legal documents the
live path is `classification.py:141` `_segment_legal_chunks`, whose actual implementation at `:146` is
`re.split(r"\n\s*\n", parsed.markdown)` — a **blank-line regex**. It is fully synchronous, bypasses
`HybridChunker` entirely, and therefore discards the heading hierarchy docling extracted, the `max_tokens=512`
budget (`classification.py:125`), and `merge_peers=True`.

Worse, `:158` truncates to `blocks[:200]`, **silently dropping paragraph 201 onward with no warning** — the only
`QualityWarning` in the function (`:148-156`) fires on the *opposite* condition (`len(blocks) <= 1`). For a
50-page contract that is most of the document, and nothing observable says so. This is a data-loss bug, not a
quality issue, and it should be called out as such in the proposal's Why.

Work: route legal documents through `HybridChunker` plus clause-boundary awareness (the one thing `HybridChunker`
does not know about), delete the regex path and the `[:200]` truncation, and emit a `QualityWarning` on the
*truncation* condition if any cap is retained at all. `design.md` answers item 162's two questions on the record:
**splitter = docling `HybridChunker`**, and **PGVector-vs-PGVectorStore = neither** — the repo uses raw asyncpg +
`pg_textsearch`/pgvector directly, and introducing a LangChain `VectorStore` would be a third retrieval path
(`dispositions.md` item 162).

- **Proof:** `uv run rg -n "re.split|blocks\[:200\]" src/app/features/documents/classification.py` → **no matches**.
- **Proof:** a test on a fixture contract with >200 paragraphs and a heading hierarchy asserts (a) chunk count
  reflects the whole document, not 200 blocks, (b) every chunk's token count ≤ `max_tokens`, (c) chunks carry
  their heading path. `uv run pytest tests/unit/documents -q 2>&1 | tail -3` shows them passing. **Mandatory** —
  the truncation bug is the clearest example in this change of a defect that no lint, type, or existing test can
  see.
- **Proof:** `uv run rg -n "HybridChunker" src/app/` → reached from the legal branch, not only the generic one.

#### Step 18 — Harvest the existing BM25 + RRF into the unified path, and kill the hardcoded index names (item 195, part 1)

Inbound dependency: **step 0** (if `pg_textsearch` is absent this step is cut), **step 6** (one embedder feeds the
vector branch).

**Not greenfield** (D5.1). What exists and is tuned: BM25 at `features/search/repository.py:415,417,419` —
`c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')`, scores negated to make higher-better; RRF at
`features/search/fusion.py:28` with `k=RRF_K=60` (`constants.py:8`); three fused branches assembled by
`_run_parallel_search` (`service.py:367-396`: bm25 + vector + trigram). A **second, in-database weighted RRF**
exists at `repository.py:308-405` over the `clauses` table. Under D5.0 we would have rebuilt this in a second
place while the working copy stayed unreachable.

Two hazards to fix while harvesting:
- **`features/search/constants.py:15` defines `SEARCH_CHUNKS_BM25_INDEX_NAME` and the SQL hardcodes the literal
  anyway** (`repository.py:415,417,419,430`). `pg_textsearch` requires the *index name inside the query* because
  it reads that index's corpus statistics — so a rename breaks these queries at runtime with **no lint or type
  warning**. The unused constant is a live hazard, not a style nit.
- **`trigram_search` (`repository.py:236`) and its index `ix_search_chunks_content_trgm` have no target
  equivalent** on `UnifiedChunk` (D5.1). It is 1 of 3 RRF branches, so dropping it silently degrades fusion
  quality. Either carry the index forward or **drop the branch on the record** — D5.1 requires this be a
  deliberate decision, not an omission.

**Do not mount the search router** — D5.1 is explicit that mounting is gated on D5.2 and stays out of scope.
"In scope" means refactor and unify only.

- **Proof:** `uv run rg -n "'search_chunks_bm25_idx'|'clauses_bm25_idx'" src/app/` → **no string literals**; every
  site references a constant.
- **Proof:** `uv run rg -n "reciprocal_rank_fusion|to_bm25query" src/app/` → one RRF implementation and one BM25
  query builder, both in the unified path (the second in-database RRF at `repository.py:308-405` is either the
  survivor or removed — state which).
- **Proof:** `uv run pytest tests/unit/search -q 2>&1 | tail -3` — the existing pure-function tests
  (`test_fusion.py`, `test_chunking.py`, `test_rag.py`) still pass, proving the harvest preserved behaviour. These
  are the **only** pre-existing tests in the whole change that constitute a real regression net; do not let them
  break.
- **Proof:** `design.md` records the trigram decision explicitly.

#### Step 19 — Add the `failure` short-circuit edge to the graph

Inbound dependency: **none**; must precede step 22 (mounting).

`ingestion_kb` has two disjoint error styles (`scout-ingestion-graphs.md` §1): guard clauses return
`_state_failure` (`nodes.py:70`) which sets the `failure` channel, and LLM nodes catch `LangChainException` and
degrade (`nodes.py:182`, `:236`, `:289`). **But `graph.py:78` has no conditional edge on `failure`**, so after a
guard failure every downstream node still runs and hits its own guard. On a bad upload that is 6 wasted node
invocations including LLM calls; behind a mounted router it is 6 wasted LLM calls per bad request.

Add a conditional edge from each node (or a single router node) that terminates on a populated `failure` channel.
Note the archived spec `openspec/changes/archive/2026-06-14-result-adoption-phases-2-5/specs/langgraph-node-result-pattern/spec.md`
**already binds how graph nodes signal failure** and explains the existing `_state_failure`/`Failure` split at
`nodes.py:70-87` — check it before inventing a convention, and target that capability in the spec delta rather
than a new one if it fits.

- **Proof:** an induced parse failure results in **exactly one** node invocation after the failing node (the
  terminator), asserted by spies on the downstream node factories; `uv run pytest tests/unit/shared -q 2>&1 | tail -3`
  shows it passing. **Mandatory new test.**
- **Proof:** `uv run rg -n "add_conditional_edges" src/app/shared/langgraph_layer/ingestion_kb/graph.py` → more
  than the single `dispatch_contextualize_chunks` edge at `:78`.

#### Step 20 — Move LangExtract upstream and wire re-ranking (item 195, parts 2 and 3; D9)

Inbound dependency: **steps 0, 9, 18**.

**D9 settles three contradictory positions on the record:** item 136 marked LangExtract `ABANDONDED`, item 43
shipped it, item 195 makes it a prerequisite stage. **195 wins** — LangExtract is repositioned **upstream** of the
postgres and Graphiti writes. Concretely that means extraction runs before `embed_store` (`nodes.py:303`) and
`graphiti_upsert` (`nodes.py:354`), so both write already-extracted structure rather than re-deriving it. Item
136's abandonment is superseded; say so in `design.md` or a reader will re-abandon it.

**Correction to the disposition on re-ranking — it is narrower than "genuinely missing".** `CrossEncoderReranker`
**already exists** at `shared/langgraph_layer/retrieval_kb/reranker.py:20`: lazy `CrossEncoder` load with a
fallback model (`BAAI/bge-reranker-v2-m3` → `cross-encoder/ms-marco-MiniLM-L-6-v2`), correctly offloaded via
`asyncer.asyncify` at `:53`, and **already imported by `documents/service.py:32`**. What is missing is that
**search's own `hybrid_search` never re-ranks** — `service.py:161-211` goes straight from RRF to hydration
(`scout-search.md` §8). So the work is *wiring an existing component into one more path*, not building one. That
materially shrinks item 195 and should be stated, because "add re-ranking" reads as a new subsystem.

Two notes on the existing reranker to carry forward: its docstring says *"CPU-bound: move this behind Celery in V2
if query latency becomes visible"* — with step 12a there is now a worker to move it behind, so record that as the
follow-up rather than doing it here; and its `except (OSError, ValueError, RuntimeError)` at `:54` returns
`chunks[:limit]` unranked on failure, which is a **silent** quality degradation. It logs a warning, so it is
acceptable, but the health probe should surface reranker-model availability.

- **Proof:** `uv run rg -n "CrossEncoderReranker|\.rerank\(" src/app/` → referenced from the unified retrieval
  path, not only `documents/service.py:32`.
- **Proof:** a test asserts that for a fixed query and a fixed candidate set, the re-ranked order differs from the
  RRF order and the top result matches an expected fixture; and that a reranker load failure returns the
  RRF order rather than raising. `uv run pytest -q 2>&1 | tail -3`.
- **Proof:** `uv run rg -n "langextract" src/app/` → its call site is upstream of `_store_chunks`
  (`nodes.py:629`) and `_graphiti_add_episode`, verified by reading the node order in `graph.py`.
- **Proof:** `design.md` records D9's resolution and that item 136 is superseded.

#### Step 21 — Eliminate the duplicate chunk write and the serial per-chunk Graphiti round-trips

Inbound dependency: **step 15** (canonicalisation must precede any change to Graphiti write paths).

Confirmed defect in `features/documents/service.py`: `:520` upserts **all** chunk rows; `:553` then calls
`_verify_legal_chunks` (`:663`), which loops every chunk through `write_and_verify_chunk` (`:673` →
`graphiti_verifier.py:28`) doing **one `graphiti.add_episode` at `:50` plus one `graphiti.search` at `:68` per
chunk — 2 Graphiti round-trips per chunk, serial, no `gather`** — mutates the dicts in place (`:682-683`), and
**re-upserts the entire set at `:686`**. `build_chunk_rows` (`repository.py:601-604`) is a pure dict-spread, so
this is not a type error; it is a genuine second full write of every embedding payload
(`scout-persistence-docling.md` §4).

Work: write chunks **once**, after verification, or update only the verification columns
(`graphiti_verified`, `graphiti_episode_id`) in the second pass rather than the whole row. Bound the per-chunk
Graphiti calls with `asyncio.gather` — `config.yaml`'s stated convention is *"bounded fan-out via
`asyncio.gather`"*, so the serial loop is a convention violation as well as a latency bug. Note the natural target
is `ingestion_kb`'s `graphiti_upsert` node (`nodes.py:354`), which is where step 22 lands this logic anyway —
so decide whether to fix it in `documents/` first (safer, independently committable, provable now) or fold and fix
together (fewer commits, but the Proof needs the whole graph). **Recommend fixing in place first**: it is provable
without the graph and it makes step 22 a move rather than a rewrite.

- **Proof:** `uv run rg -n "upsert_chunks" src/app/features/documents/service.py` → **one** call for the full-row
  payload (a second call, if kept, updates only verification columns — assert by reading the argument).
- **Proof:** a test with N=10 chunks counts calls on a `repo.upsert_chunks` spy → **1** full-row call (was 2), and
  counts Graphiti round-trips → they are issued concurrently (assert via a gather-detecting double or elapsed-time
  bound). `uv run pytest tests/unit/documents -q 2>&1 | tail -3`.
- **Proof:** `uv run rg -n "for chunk in" src/app/features/documents/service.py` → the verification loop is gone
  or is a `gather` comprehension.

#### Step 22 — The fold: promote `ingestion_kb`, absorb `documents/`'s three unique concerns, mount, delete (D1, item 190)

Inbound dependency: **every preceding step.** This is the last step and it is the only one that is not
independently revertible without reverting the change.

Item 190 *"can `documents/` move into the ingestion pipeline"* **is** the structural question behind D1
(`dispositions.md`: "This *is* the structural question… Folding is the change"). The overlap is settled
(`scout-ingestion-graphs.md` §6): **11 shared concerns**, 3 unique to `documents/`, 2 unique to `ingestion_kb/`.

Unique to `documents/`, and therefore what must move **into** the graph:
1. **S3 fetch** — `service.py:477` `object_store.get_object(key_from_s3_uri(object_uri))`. Step 14's Pointer State
   makes this natural: the graph receives an `object_uri`, not `raw_bytes`.
2. **Status transitions** — `repo.update_document_status` at `:490`, `:528`, `:570`
   (`parsed` → `stored_postgres` → `completed*`). `ingestion_kb` has only `ingestion_complete: bool`
   (`state.py:193`). The status column is the **only** recovery signal `documents/` has today; with a checkpointer
   it becomes an *observability* surface rather than the recovery mechanism, but it must not be dropped — it is
   what the API reports to the user.
3. **Graphiti verification** — `write_and_verify_chunk` (`:673`), reshaped by step 21.

Unique to `ingestion_kb/`, and therefore what is *gained*: entity/relationship extraction
(`make_classify_extract_node`, `nodes.py:259`) and per-I/O retry (step 11's corrected `retry_immediate`).

Also in this step: mount `ingestion_router` in `src/app/api/v1.py` (7 imports / 6 mounts today, no
`ingestion_router`), and delete `features/documents/ingestion_graph.py` — the **decorative one-node graph**
(`build_document_ingestion_graph:39` adds exactly one node at `:50`, a pure pass-through at `:65` forwarding 5
state fields to `ingest_document_fn`, with 4 of its 9 `DocumentIngestionState` channels write-only and **zero
`Annotated` reducers**). Also delete or retarget `run_document_ingestion_task` (`service.py:580`), which rebuilds
`init_db()`, `StorageService`, `_build_chat_model`, `setup_graphiti`, **and the graph itself** per invocation
(`:588-605`) — the exact anti-pattern `brief:01-…:41,86` condemns, now unnecessary because step 13 builds
everything once at startup.

`features/ingestion/service.py:69` logs `log.exception("ingestion_graph_failed")` and the scout could not
determine whether it re-raises or swallows (`scout-ingestion-graphs.md` Fog). **Read it in this step** — a
swallowing except here means a failed ingestion returns 200.

- **Proof:** `uv run rg -n "ingestion_router" src/app/api/v1.py` → import **and** `include_router` present.
- **Proof:** `uv run rg -n "build_document_ingestion_graph|DocumentIngestionState" src/` → **no matches** (file
  deleted, no dangling importer). Then `uv run python -c "from app.main import app"` exits 0, proving no import
  was orphaned.
- **Proof (the band's boot proof, and the change's acceptance test):** with the compose stack up (step 12a),
  `POST` a fixture contract to the mounted upload endpoint, then poll until
  `SELECT status FROM documents WHERE id=…` reaches a terminal status, and assert
  `SELECT count(*) FROM chunks WHERE document_id=…` > 0 **and** `SELECT count(*) FROM checkpoints` > 0. This single
  check proves: worker consumes (12a), task dispatches (12c), graph runs from `app.state` (13), checkpoints persist
  (16), chunks embed at the right width (2, 3, 6), and the router is mounted (22).
- **Proof:** `uv run pytest -q 2>&1 | tail -3` — passed count ≥ 61 plus every mandatory test added by steps 2, 5,
  6, 8, 9, 11, 14, 15, 17, 19, 20, 21. State the final expected count in `tasks.md` so a silent test loss is
  visible.
- **Proof:** `uv run ruff check src/` ≤ **123** errors (post-`todo_temp.py` baseline) and `uv run ty check src/`
  ≤ **46**; `ast-grep scan src/` no new findings.

---

## Conflicts surfaced

Each conflict states the evidence on **both** sides and my recommendation. None is resolved silently, and none
overrides a locked decision.

### C1 — Sub-todo (j) asks for `tenacity`; the docs condemn what tenacity does inside a node. **And tenacity is already there.**

**Side A (the sub-todo):** add `tenacity` for retries. `tenacity` 9.1.4 is installed (`uv.lock:8445`).

**Side B (the reference docs):** `tenacity`, `RetryPolicy`, and `.with_retry()` are named **zero times** in the
entire repo doc corpus — verified by grep over the 2189-line organized reference, all 14 reference files, and both
`SKILL.md` copies (`brief` §(j)). What the docs *do* prescribe is a manual retry loop inside `@wrap_model_call`
middleware (`brief:05-…:93-105`), `ToolNode(handle_tool_errors=…)` (`brief:ref:38`), and
`brief:ref:1633`'s explicit prohibition: *"Do not wrap interrupt calls in try/except… you will catch this
exception and the interrupt will not be passed back to the graph."* Because `interrupt` pauses by **raising**, a
catch-all retry wrapper around node code silently swallows pauses and retries them.

**The decisive new evidence (mine, from opening the file the brief flagged as unread):** the conflict is not
hypothetical and not future — **it is already in the code.** `shared/langgraph_layer/kb_retry.py:29` is literally
`retry=retry_if_exception_type(Exception)`, wrapping every I/O call inside `ingestion_kb`'s nodes, with
`wait=wait_none()` (`:28`, zero backoff) and a re-wrap at `:41-43` that collapses every distinct failure into
`TransientExternalError` — which, because it is not a `LangChainException`, **silently defeats the graph's own
degradation branches** at `nodes.py:182`, `:236`, `:289`.

**Recommendation (matches `dispositions.md` item 172, now with direct evidence):** sub-todo (j) is **already
satisfied in location and defeated in policy**. Do not add tenacity; **fix the tenacity that exists** (step 11).
Layering, to be written into `design.md` as a Decision with alternatives:
- **tenacity stays at I/O-client boundaries only** — `kb_retry.py`, `connections/redis.py`, `razorpay_client.py` —
  with *specific* exception types and exponential backoff, never a bare `Exception` catch-all, and never wrapping a
  whole node.
- **middleware (`@wrap_model_call`) owns model and tool retries**, in change 3.
- **idempotency keys, not retry budgets, own replay safety** (`brief:ref:1612-1614`): the checkpointer's recovery
  unit is the node, so a node-local attempt counter resets on replay and the budget is silently multiplied. Step
  15's `party_id` is the first such key.

### C2 — Sub-todo (i) names `MessagesState`; the docs never use it and forbid Pydantic state. This is change 3's, but change 1 must not contradict it.

**Side A:** sub-todo (i) names `MessagesState` for Agent A → Agent B.

**Side B:** `MessagesState` appears **once** in the whole corpus, descriptively (`brief:ref:1479`), never imported
or subclassed. The docs prescribe a `TypedDict` with `Annotated[list, add_messages]` plus sibling channels
(`brief:07-…:26-36`), and `brief:ref:1341-1345` states *"As of langchain 1.0, custom state schemas must be
TypedDict types. Pydantic models and dataclasses are no longer supported."*

**The nuance that keeps this from being settled, and it is load-bearing for change 1:** that passage is scoped to
`create_agent`'s `state_schema`. `brief` Fog §1 is explicit — *"do not treat note 67 as settled for bare
`StateGraph`."* `ingestion_kb` uses **bare** `StateGraph(IngestionState)` (`graph.py:45`) with a Pydantic
`IngestionState` (`state.py:166`, `extra="forbid"`, `arbitrary_types_allowed=True`), and a superseded repo plan
(`docs/superpowers/plans/2026-04-13-…:32`) prescribed Pydantic state as the house pattern.

**Recommendation:** change 1 **does not convert `IngestionState` to a TypedDict** — that is change 3's call and the
evidence for bare `StateGraph` is genuinely absent. Change 1 constrains itself to **shrinking** the channels (step
14) and adds **no new Pydantic-only affordances** (validators, computed fields, arbitrary types) that a later
conversion would have to unpick. `arbitrary_types_allowed=True` is removed by step 14 anyway, which is the single
biggest obstacle to conversion. Record in `design.md` that change 1 leaves the state TypedDict-convertible, and
flag for change 3 that the bare-`StateGraph` question needs the langgraph 1.1.2 `StateGraph` docstring to settle —
not another pass over the repo docs.

### C3 — Item 176 asks to drop `transformers`; two IN-scope items depend on it.

**Side A:** item 176, scoped by `dispositions.md` to *"if the only use is token counting, drop the direct
`transformers` dependency. Verification = import no longer present."*

**Side B (mine, verified):** the premise is false. `sentence_transformers` is imported at
`retrieval_kb/reranker.py:8` for `CrossEncoder`, which **is** the re-ranking implementation item 195 needs (step
20); `transformers.AutoTokenizer` is imported at `document_processing/chunker.py:16` and passed to
`HybridChunker(tokenizer=…)` at `:29-36`, which **is** the splitter item 162 and step 17 adopt. Both are declared
direct dependencies (`pyproject.toml:51`, `:74`).

**Recommendation:** item 176's stated deliverable is **unachievable and should be recorded as such**, not quietly
dropped — dropping either package would break two other IN-scope items in the same change. What remains actionable
is real and worth doing (step 9): cache `get_tokenizer` (currently uncached, sync, network-on-first-use at
`chunker.py:23-26`), and record that the chunker's tokenizer
(`"sentence-transformers/all-MiniLM-L6-v2"`) is **not** the embedding model's tokenizer, so `max_tokens=512` is
enforced against the wrong counter. That mismatch is a genuine finding item 176 would have surfaced by accident.

### C4 — Item 171 prescribes `CacheBackedEmbeddings`, which requires a forbidden import path.

**Side A:** `brief:ref:2049` names the exact defect (*"`aembed_batch` calls the API every time"*) and
`brief:13-…:30-54` prescribes `CacheBackedEmbeddings.from_bytes_store(...)` with `LocalFileStore`.
`dispositions.md` marks item 171 **IN 1**.

**Side B:** the import is `from langchain_classic.embeddings import CacheBackedEmbeddings` — verified at
`.venv/.../langchain_classic/embeddings/__init__.py:14`, and **not** available from `langchain_core` or
`langchain`. `brief:ref:60-63` (note 9) forbids legacy import paths; `langchain_classic` is the v0-compat shim.
`brief` Contradictions §4 already flags this tension. Additionally `LocalFileStore("./cache/")` is per-container,
so it is useless behind a scaled `ai-service-1`, and `namespace=underlying.model` requires the embeddings object to
expose `.model`.

**Recommendation:** **satisfy item 171's intent, reject its named vehicle** (step 7). Step 6's redis-backed
SHA256-keyed cache is the same mechanism, is shared across processes, and already exists twice in the repo
(`documents/service.py:813`, `ingestion_kb/nodes.py:716`). Record `CacheBackedEmbeddings` as a
considered-and-rejected alternative in `design.md` Decisions — the schema requires alternatives per decision, so
this is the correct place for it, and it closes item 171 on the record rather than leaving it looking unaddressed.

### C5 — Sub-todo (e) "Celery for offloading" is not a code task, and the brief's mechanism was wrong.

**Side A (as briefed):** `tasks.document_tasks` is absent from `include` (`celery.py:191-196`), so the live
ingestion task is never registered.

**Side B (`findings-deployment.md` §2-§3):** the task **is** registered, transitively via
`src/tasks/__init__.py:4`. The real reason ingestion does not run is that **no worker or beat service exists in
`docker-compose.yml` at all**, and the documented command (`Makefile:52`, `-A celery_config`) references a module
that does not exist.

**Recommendation:** re-rank as steps 12a → 12b → 12c (worker exists → command works → guarantee made explicit).
The `include` gap is a **latent fragility**, and it becomes live precisely when **change 0** tidies
`tasks/__init__.py:6-9` — so step 12c carries a cross-change Proof: run the registry check **after** change 0's
edit, not only before. This is the one conflict where the correction makes the work *larger* (a deployment change,
not a config line) rather than smaller.

### C6 — D5.1 justifies pulling `search/` into scope because BM25/RRF "already work". They have never demonstrably run.

**Side A (D5.1):** BM25 (`repository.py:415-419`) and RRF (`fusion.py:28`, k=60) exist and are tuned, so item 195
is not greenfield and rebuilding them elsewhere would duplicate working code.

**Side B:** the search router is unmounted, so no HTTP path reaches them; `legal_rrf_search`
(`repository.py:308-405`) queries the **`clauses`** table, which **no migration creates**, and its index
`clauses_bm25_idx` is created at `9f4a1b7c6d2e:132` on that same non-existent table — so that migration cannot run
on a clean DB (`findings-deployment.md` §6). Whether `timescale/timescaledb-ha:pg18` even ships `pg_textsearch`
is **unverified** (§5). Tests never touch a database (`tests/integration/test_search.py` patches the repository and
uses an `AsyncMock` session).

**Recommendation:** keep D5.1 — the code is still the best available foundation and rebuilding it would be worse.
But **downgrade the claim from "working" to "written and tuned, never executed"**, and make step 0 a hard
precondition. `CREATE EXTENSION IF NOT EXISTS pg_textsearch` does **not** protect against a missing extension — it
only suppresses "already exists" — so a missing extension aborts the migration outright. If step 0 fails, steps 18
and 20 are cut from change 1 and the blocker is recorded in `design.md` Open Questions with its two options
(change the image, or reimplement on `tsvector`/`ts_rank`). Do not let a green `ruff`/`ty`/`pytest` run stand in
for "BM25 works" — nothing in the test suite touches Postgres.

---

## Openspec mapping

### The existing namespace, enumerated

`openspec/specs/` holds **20** capability directories (I listed them; the brief's "~22" was approximate):

```
cognee-v1-api            datetime-utc-cleanup       llm-injection            mcp-context-di
mcp-directory-restructure  mcp-server-codemode      mcp-server-composition   mcp-server-pagination
mcp-server-prompts       mcp-server-resources       mcp-server-codemode      mcp-telemetry
mcp-testing              noqa-documentation         outbox-helper-extraction pattern-matching-standard
session-required         settings-validation        test-mock-isolation      transactional-outbox
typed-exception-handling
```

Eleven of the twenty are MCP-topic capabilities; the rest are cross-cutting standards. **None covers ingestion,
document processing, embeddings, chunking, retrieval, checkpointing, or Celery workers.** So change 1 is almost
entirely **NEW capabilities**, which the proposal rules permit provided you check first (`config.yaml:39-43`) —
this section is that check, written down.

### Proposed capability deltas

| Capability | New / Modified | Covers (steps) | Why this boundary |
|---|---|---|---|
| `document-ingestion-pipeline` | **NEW** | 5, 13, 16, 19, 22 | The observable behaviour of "an uploaded document becomes retrievable chunks with a reported status". The graph promotion, `app.state` wiring, failure short-circuit, and the fold are all one externally-visible contract: upload → terminal status → chunks exist. |
| `hierarchical-document-chunking` | **NEW** | 8, 9, 17 | Separate from the pipeline because it is a *quality* contract with its own scenarios (token bound respected, heading path retained, **no silent truncation**) that hold regardless of which pipeline invokes it. The `blocks[:200]` data loss is its headline requirement. |
| `unified-embedding` | **NEW** | 2, 3, 6, 7 | A contract others build on: one provider, one dimension sourced from settings, explicit `task_type` on both query and document sides, cached, normalised consistently. Change 2 and change 4 both consume it (change 4's Cognee dimension bug is the same class of defect). |
| `langgraph-checkpointing` | **NEW** | 10, 14, 16 | Persistence and resumability, with `thread_id` and durability as observable behaviour ("a crash mid-pipeline resumes at the failed stage, not the first"). Consumed by change 3's HITL work, so it must outlive change 1. |
| `celery-worker-deployment` | **NEW** | 12a, 12b, 12c | The deployment contract: a process consumes the queue, the documented command starts it, and task registration does not depend on an import side effect. Note `openspec/changes/archive/2026-06-22-quality-fixes-batch-2/specs/celery-task-registry/spec.md` exists — **harvest its requirement text; do not duplicate it.** If that capability was archived into `openspec/specs/` under another name, target it as MODIFIED instead. |
| `graph-entity-canonicalisation` | **NEW** | 15 | Deliberately its own capability, not folded into the pipeline, because it is the change's only irreversible requirement and it is consumed by change 3 (Trap2 idempotency keys) and change 4 (Cognee entity writes). A separate capability makes the dependency legible. |
| `hybrid-retrieval-ranking` | **NEW** | 0, 18, 20 | BM25 + RRF + re-ranking as one retrieval-quality contract, with `pg_textsearch` availability as a stated precondition. Kept separate from `unified-embedding` because the embedder is an input to it, and separate from the pipeline because retrieval is read-path. |
| `langgraph-node-result-pattern` | **MODIFIED** (check first) | 11, 19 | This capability **already exists as an archived change spec** — `openspec/changes/archive/2026-06-14-result-adoption-phases-2-5/specs/langgraph-node-result-pattern/spec.md` — and it *already binds how graph nodes signal failure*, which is exactly the `_state_failure`/`Failure` split at `nodes.py:70-87` and the retry-policy question. **Check whether it landed in `openspec/specs/` under a name I did not match** before writing a NEW capability here. If it is live, MODIFY it (copying the entire requirement block, per the delta rules); if it is only archived, harvest its text into the pipeline capability. |
| `typed-exception-handling` | **MODIFIED** | 2, 11 | **Existing** capability. Step 2 replaces silent zero-vector fallbacks with raises and step 11 stops collapsing every failure into one opaque type — both are exception-taxonomy changes on an existing contract. |
| `transactional-outbox` | possibly **MODIFIED** | 12c | **Existing** capability. Step 12c changes how `event_type` is resolved (constant instead of literal) — the outbox's dispatch contract. **Note `spec/transactional-outbox` is one of the 6 pre-existing validation failures**, so touching it needs care: a delta on an already-failing spec makes attribution of a new failure harder. Prefer leaving it alone and putting the constant requirement in `celery-worker-deployment` unless the outbox contract genuinely changes. |
| `settings-validation` | possibly **MODIFIED** | 3 | **Existing** capability. Step 3 makes ORM columns track `EMBEDDING_DIMENSION` and records that the value is not runtime-changeable. If that reads as a settings contract, MODIFY; otherwise it belongs in `unified-embedding`. Decide once; do not delta both. |

### Formatting traps that will bite this change specifically

- **Scenario headers take exactly four hashtags.** `schema.yaml:164-165`: three hashtags or bullets **fail
  silently** — the scenario is dropped with no error. With ~11 capabilities this is the highest-frequency risk in
  the change. Grep the finished deltas: `rg -n "^#{1,3} Scenario:" openspec/changes/<slug>/specs/` must return
  **zero** matches.
- **`MODIFIED` must copy the entire existing requirement block**, `### Requirement:` through all scenarios, header
  text matching whitespace-insensitively. Partial content silently loses detail at archive time. Three of the rows
  above are MODIFIED, so this applies three times.
- **`REMOVED` needs both Reason and Migration.** Step 22 removes the one-node graph and step 17 removes the regex
  splitter — if either is expressed as a REMOVED requirement rather than a MODIFIED one, both fields are mandatory.
- **No internal class or function names in requirement text.** House style tolerates *graph node names* (the live
  cognee change references `persist_memory`), so `parse_document`/`embed_store` as node names are acceptable;
  `retry_immediate`, `normalize_embedding`, `HybridChunker` are **not** — they go in `design.md`.
- **`.openspec.yaml` must say `schema: spec-gated`** to match `openspec/config.yaml:1`. The in-flight cognee change
  says `spec-driven` and is stale — do not copy it as a template.
- **Change ID is a bare slug** (D12); the `YYYY-MM-DD-` prefix is added at archive time.

### Artifact obligations

Change 1 is class **L** (multi-module, data migration, new external behaviour), so all six artifacts apply with the
dependency order `proposal → specs → design → review → tasks` (`schema.yaml`). `design.md` is **mandatory** and must
carry, at minimum: item 162's two answers (splitter = `HybridChunker`; PGVector-vs-PGVectorStore = neither), the
tenacity/middleware layering (C1), the `CacheBackedEmbeddings` rejection with alternatives (C4), the durability-mode
choice (step 16), the trigram-branch decision (step 18), and the Non-Goals below.

**`review.md` must exist with a `VERDICT:` line before `tasks.md` is legitimate** (`schema.yaml:394-396`), written
**as a reviewer, not the author** (`schema.yaml:321`) — i.e. by a fresh subagent, per D12.

**`adrs.md`:** change 1 has one genuine ADR candidate — **the unified embedder contract** (provider, dimension
source, `task_type` policy, normalisation policy, cache key). It outlives the change because change 2 and change 4
both build on it, and change 4's Cognee dimension defect (3072 vs 768) is the same decision re-litigated. Entity
canonicalisation is the second candidate. Everything else is change-local; write `No durable architectural decision`
for the rest rather than padding (`schema.yaml:369-371`).

**Non-Goals that must be stated explicitly** (each is a DROP/DEFER from `dispositions.md` and D13 requires it be
surfaced, not silently omitted): `app.state.vector_store` (DROP — would be a third retrieval path); item 164
"refactor RAG code" (DROP as an unverifiable umbrella); Up#4 `markitdown` (DROP — second parser); items 165/Up#3
Uber-style agentic RAG (DEFER); mounting the search router (out per D5.1, gated on D5.2); the `search_*` →
`UnifiedDocument`/`UnifiedChunk` collapse (change 2); `MessagesState` and middleware retries (change 3);
`document_vectors`' inconsistent `Vector(1536)` (recorded divergence, not migrated).

### Validation acceptance criterion

**Baseline is 16 passed / 6 failed of 22 items** (D12). Pre-existing failures: `spec/cognee-v1-api`,
`change/mintlify-documentation`, `spec/noqa-documentation`, `spec/pattern-matching-standard`,
`spec/transactional-outbox`, `spec/typed-exception-handling`.

- **Proof:** `openspec validate --all` reports **no failures beyond those six**, and the passed count rises by the
  number of artifacts added. Never "validate --all passes" — it cannot.
- **Caution:** two of the six pre-existing failures (`typed-exception-handling`, `transactional-outbox`) are
  capabilities this change proposes to MODIFY. If either is delta'd, its failure becomes ambiguous — is it the old
  failure or a new one? **Record each one's failure output verbatim before editing** so attribution stays possible.

---

## Risks

Format follows `design.md`'s required `[Risk] → Mitigation`.

**[The promoted modules have zero test coverage, so "it still works" is unfalsifiable]** → codegraph reports *no
covering tests found* for all 7 `ingestion_kb` node factories, `build_document_ingestion_graph`,
`process_document_ingestion`, and `run_document_ingestion_task`, and no test references `ingestion_kb`,
`documents_ingest`, or `ingestion_graph` at all. **The evidence that stands in today is: ruff (125→123), ty (46),
`ast-grep scan`, and 55 passing tests that touch none of this code.** That is enough to prove the repo *imports*,
and nothing more. Mitigation: the 12 tests marked **Mandatory** in steps 2, 5, 6, 8, 9, 11, 14, 15, 17, 19, 20, 21
are the regression net being built, and step 22's end-to-end Proof is the only integration-level check. Tests are
mandatory in exactly the places where a defect is invisible to lint and types: a blocking call, a silent
truncation, a zero-vector substitution, a swallowed exception, a serialisation-size regression, an unrecoverable
entity duplicate. Do not let any of those steps ship on a lint-only Proof.

**[`--cov-fail-under=80` against 18.38% coverage means a green suite exits 1, so `$?` is a lie]** → every Proof in
this plan names the summary line to compare. Mitigation: state the expected passed-count in `tasks.md` per task, so
a silently-lost test is visible as a count that did not rise. The risk is specifically that an implementer wires
CI or a hook on exit code and concludes the suite is broken (or "fixes" it by lowering the gate).

**[`pg_textsearch` may not exist in the deployment image, which would void D5.1's entire justification]** → step 0
is a hard precondition with an explicit failure branch. `CREATE EXTENSION IF NOT EXISTS` does **not** protect
against a missing extension — it only suppresses "already exists" — so a missing extension **aborts the
migration**, taking outbox and billing with it. Mitigation: run step 0 before any of Band E is planned in detail;
if it fails, cut steps 18 and 20 and record the blocker rather than building on a non-existent operator.

**[The `clauses` table has no DDL origin, and three migrations depend on it]** → `9f4a1b7c6d2e` only ALTERs
`clauses` (`:63-99`), backfills (`:101-102`), re-types `embedding` (`:105`), and creates `clauses_bm25_idx`
(`:132`) — **no revision anywhere creates the table.** So it is in the same category as `statutes` and
`match_chunks()`, but worse, because a migration *depends* on it and therefore blocks the whole `0001→0004` chain.
`ingestion_kb`'s `_store_chunks` (`nodes.py:629`) and `_upsert_parent_document` (`:488`) write to this schema, and
`legal_rrf_search` reads it. Mitigation: this is **change 0's** problem (the merge revision alone is insufficient
— it needs a `create_table`), but change 1 **cannot be proven without it**, so it is a named inbound dependency on
steps 3, 12, 13, 20. Do not let change 1 start Band E until a clean-DB `alembic upgrade head` succeeds.

**[Change 0 edits the file that silently guarantees change 1's task dispatch]** → `tasks/__init__.py:4` is the only
reason `tasks.documents_ingest` is registered; change 0 must edit `:6-9`/`:18-20` to delete reconciliation.
Mitigation: step 12c's Proof runs the registry check **after** change 0's edit, and adding the explicit `include`
entries removes the dependency entirely. Sequencing note: if change 0 lands first (as D8 requires), there is a
window where ingestion dispatch is broken until step 12c. Accept it — nothing consumes the queue anyway
(`findings-deployment.md` §2), so the window has no observable effect. **Say this in `design.md`** or a reviewer
will flag it as a regression.

**[Changing `EMBEDDING_DIMENSION` later is a re-embedding job, and step 3 makes it look like a config knob]** →
pgvector's `vector(n)` typmod is not widenable in place; every HNSW/IVFFlat/diskann index must be dropped first,
and `ALTER … TYPE` fails while any row holds a different width. Mitigation: step 3 states this in `design.md`
explicitly, and the ORM default is documented as read at **import time**. The risk is a future engineer flipping
the setting and getting a boot-time mismatch against live data.

**[Step 14 changes the state schema; a checkpoint written under the old schema is unreadable]** →
`brief:ref:105` warns to normalise state after fetching from a checkpointer. Mitigation: there is **no legacy
checkpoint data**, because step 10 proves `setup()` was never reached and therefore the checkpoint tables cannot
exist. This is a free window that **closes permanently at step 16**. If the ordering slips and step 16 precedes
step 14, the mitigation evaporates and a migration becomes necessary.

**[Entity canonicalisation is the change's only irreversible step]** → duplicate party nodes cannot be separated
after the fact, because the disambiguating context is the extraction that has already been discarded. Mitigation:
step 15 precedes step 22 absolutely, and its unit test is the most load-bearing new test in the change. Secondary
mitigation: there are three Graphiti call sites today (`documents/service.py:596`, `:601`, `:673`) — audit all
three by reading, not by grep, because a missed site poisons the graph exactly as thoroughly as no
canonicalisation at all.

**[The fold (step 22) is the one step that is not independently revertible]** → it deletes the live path.
Mitigation: it is last, every prerequisite is separately proven, and its Proof is an end-to-end check that
simultaneously exercises steps 2, 3, 6, 12a, 12c, 13, 16, and 22. If that Proof cannot be run (no compose stack, no
migrated DB), **step 22 must not be attempted** — the change ships at step 21 with `ingestion_graph.py` still live
and the promotion deferred. State that fallback in `tasks.md`.

**[`features/ingestion/service.py:69` may swallow failures, turning a failed ingestion into a 200]** → the scout
saw `log.exception("ingestion_graph_failed")` but not the full except clause. Mitigation: step 22 reads it. This
matters more once the router is mounted: an unmounted swallowing handler is invisible; a mounted one is a
correctness bug on shipped surface.

**[`rag_agent_advanced.py` depends on `match_chunks()`, a Postgres function in no migration]** → step 4 fixes the
phantom *import* but the phantom *function* remains, so the file's call paths still cannot succeed. Mitigation:
record it; do not fix it in change 1. It is a candidate for change 0's deletion sweep if `graphify affected` proves
the call paths unreachable — but that determination was not made, so the file stays.

**[Scope creep via D5.1: "search is in scope" can be read as "collapse the schema now"]** → it cannot. D5.1 means
refactor and unify; the schema collapse is change 2 and mounting is gated on D5.2. Mitigation: step 18's Proof is
about index-name constants and one RRF implementation, **not** about table names. If a task starts editing
`search/model.py` columns, it has left change 1.

**[Eleven new openspec capabilities is a large namespace addition, and the four-hashtag trap fails silently]** →
mitigation: the grep in the Openspec mapping section (`rg -n "^#{1,3} Scenario:"` must return zero), plus recording
the two pre-existing failures' output verbatim before delta'ing the capabilities they belong to.

---

## Fog

What I could not establish, and what it would take. Each is assigned rather than left floating.

1. **Whether `timescale/timescaledb-ha:pg18` ships `pg_textsearch`.** The single highest-consequence unknown left:
   if it does not, steps 18 and 20 are void and D5.1's justification collapses. **Resolve by:** step 0's command,
   `docker run --rm timescale/timescaledb-ha:pg18 ls /usr/share/postgresql/18/extension/ | grep -E 'textsearch|vectorscale'`.
   I did not run it — no Docker call was in my remit.

2. **Whether `openspec/specs/` contains a live `langgraph-node-result-pattern` capability under a name I did not
   match.** I enumerated the 20 directory names but did not read their `spec.md` contents. The archived change
   `2026-06-14-result-adoption-phases-2-5` shipped that capability, and archived specs normally land in
   `openspec/specs/`; none of the 20 names matches, which is itself suspicious. **Resolve by:**
   `rg -l "langgraph|graph node" openspec/specs/*/spec.md`. Consequence if live: one row of my mapping table flips
   from NEW to MODIFIED, and the entire requirement block must be copied verbatim.

3. **Whether `graph.compile()` accepts the `Send` payload dict against `IngestionState`'s `extra="forbid"`.** The
   fan-out at `nodes.py:200` constructs a plain dict containing `segment`/`contract_metadata`, and
   `contextualize_chunk_node` types state as `dict[str, Any]` (`:215`), while the graph is compiled with
   `StateGraph(IngestionState)` (`graph.py:45`, `extra="forbid"` at `state.py:167`). Whether LangGraph validates the
   `Send` payload against the state schema — which would **reject** those keys — is unresolved and inherited from
   `scout-ingestion-graphs.md` Fog. If it does validate, step 22's fold hits it immediately. **Resolve by:** running
   the graph once with a stub state (which step 16's Proof does anyway), or reading langgraph 1.1.2's `Send`
   handling. I chose not to run code from a read-only planning remit.

4. **Whether bare `StateGraph` still accepts Pydantic state on langgraph 1.1.2.** Inherited from `brief` Fog §1 and
   deliberately left open (C2). It does not block change 1 — step 14 only shrinks channels — but it blocks change 3.
   **Resolve by:** the langgraph 1.1.2 `StateGraph` docstring, not another pass over the repo docs, which are
   `create_agent`-scoped.

5. **Whether any `search_chunks` / `chunks` / `clauses` rows exist in a deployed environment.** All three scouts
   proved no *code path* populates `search_*` and no seed/fixture/factory exists, but none could inspect a running
   database. This determines whether step 3's dimension work and change 2's collapse are `DROP TABLE` or a
   re-embedding job. **Resolve by:** `SELECT count(*)` against staging. Given `9f4a1b7c6d2e` cannot run on a clean
   DB yet tables are believed to exist, some schema was almost certainly created out-of-band — which is itself
   unresolved and matters for every migration Proof in this plan.

6. **Whether `chunks.embedding` has a diskann index and `chunks.content` a trigram index.**
   `documents/model.py:73-79` declares four btree/GIN indexes and no vector index, while
   `documents/repository.py:15-19` imports DiskANN `SET LOCAL` tuning and `TRIGRAM_SIMILARITY_THRESHOLD` — implying
   indexes created in a migration none of us opened. If absent, both branches seq-scan and step 18's harvest is
   fast in code and slow in production. **Resolve by:** `rg -n "diskann|gin_trgm_ops" src/alembic/versions/`.

7. **Whether `features/ingestion/service.py:69`'s except clause re-raises or swallows.** Assigned to step 22, which
   must read it. A swallowing handler behind a newly mounted router returns 200 on a failed ingestion.

8. **The exact number of `documents/` stages.** I use the scout's counted **10** discrete operations, not the
   brief's asserted 7. If a canonical named list of 7 exists somewhere, neither the scout nor I found it. Low
   consequence — it affects prose in `proposal.md`, not any step.

9. **Whether `retry_immediate`'s `before_sleep` callback has ever fired.** With `wait=wait_none()` the sleep is 0,
   and tenacity does still invoke `before_sleep`, so `logger.bind(...).warning(...)` at `kb_retry.py:31-35` should
   execute on every retry. `from app.utils import logger` at `:9` resolves to the loguru object there (outside the
   `app.utils` package), so I believe it works — but I did not execute it, and the analogous line inside
   `app/utils/` is the step-1 bug. Also `logger.warning("kb_retry_immediate_retry", error=...)` passes a kwarg
   loguru will treat as a format argument for a message containing no placeholders, so **the `error` value is
   probably discarded**. If so, every retry in the pipeline has been logging a bare event name with no cause for as
   long as the file has existed. **Resolve by:** one unit test asserting the emitted record carries the error —
   folded into step 11's test, which is the cheapest place to settle it.

10. **Whether `documents/service.py`'s `_verify_legal_chunks` second upsert actually rewrites embeddings or only
    metadata.** `build_chunk_rows` (`repository.py:601-604`) is a pure dict-spread over `Sequence[dict]`, and the
    dicts are mutated in place at `:682-683`, so the second `upsert_chunks` at `:686` almost certainly carries the
    full embedding payload again — but I did not trace which columns the upsert's `on_conflict_do_update` actually
    touches. If it excludes `embedding`, step 21 is a latency fix rather than a write-amplification fix. **Resolve
    by:** reading `repository.py`'s `upsert_chunks` `set_` clause. Assigned to step 21.

11. **Whether a dedicated ingestion Celery queue is wanted.** I recommend yes (step 12a) on the reasoning that
    minutes-long LLM work will starve sub-second billing and auth tasks on the shared default queue, but
    `task_create_missing_queues=False` (`celery.py:229`) means this is a deliberate configuration decision with an
    operational cost, and no user decision covers it. **This is a genuine `design.md` Open Question** — and per
    `schema.yaml:297-301`, because it would change the task breakdown, it should be **asked, not guessed**.

---

## Corrections adopted post-plan (from docs/relay/findings-database.md, live DB probe 2026-08-17)

The orchestrator connected to the **actual** database and to the venv. Five things in the sections above are now
wrong or need restating. This section is authoritative over anything earlier in this file that contradicts it.

### CX-1 — Defect 4 is INVERTED. The `= Any` fallback is the live path, and it is why the app boots.

I wrote that `checkpointer.py:26-29`'s `AsyncPostgresSaver = Any` ImportError fallback is "unreachable today
because `langgraph-checkpoint-postgres` 3.0.4 is installed". The **package** is installed; its **driver cannot
load**:

```
langgraph-checkpoint-postgres 3.0.4  installed
psycopg                       3.3.3  installed
psycopg-pool                  3.3.0  installed
psycopg-binary                       NOT INSTALLED
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
  -> ImportError: no pq wrapper available.  (no psycopg_c, no psycopg_binary, libpq not found)
```

So `except ImportError` **fires**, `AsyncPostgresSaver` **is** `typing.Any`, and `setup_langgraph_checkpointer`
short-circuits at `:51-53` — `logger.warning(...)`, `return None`.

Consequences, and they reorder the work:

1. **My Defect 1 is currently unreachable.** The `is Any` guard returns before `:56` ever runs, so the
   `from_conn_string`-is-an-`@asynccontextmanager` bug cannot fire today. It becomes live **the moment the driver
   is fixed** — which is to say, the moment step 10 starts. Both defects are real; I had the order backwards.
2. **Restate the Band C boot claim.** Earlier text says uncommenting `lifespan.py:295-305` "takes the app from
   boots to does-not-boot". Correct statement: as things stand it logs **one warning**, sets
   `app.state.langgraph_checkpointer = None`, and then raises `AttributeError` at
   `features/agent_saul/dependencies.py:45`, which reads the slot unguarded. Not a boot crash — a **silent
   degradation that becomes a crash at first agent request**. Same invisible-failure register, later blast point,
   and materially worse than a boot crash because CI that only checks startup will pass.
3. **The `= Any` fallback is currently the only reason the app boots on this machine.** Therefore deleting it
   **must land in the same commit** as the dependency fix. Splitting them produces a commit that does not boot,
   which violates this plan's own per-step rule.

### CX-2 — NEW Step 0b (hard precondition for step 10): install a working psycopg driver.

**Depends on:** nothing. **Blocks:** step 10 (checkpointer rewrite) and step 13 (checkpointer on `app.state`).
Nothing else in the plan depends on it, so Bands A, B, D and most of E are unaffected and can proceed in parallel.

Add `psycopg[binary]` to the dependency set (`psycopg-pool` already being installed is strong evidence the
`AsyncConnectionPool` shape was the original intent, which is what step 10 prescribes anyway). A system libpq is
the alternative but is not reproducible across the compose image and this working copy, so prefer the wheel.

- **Proof:** `uv run python -c "from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver; from psycopg_pool import AsyncConnectionPool; print('driver OK', AsyncPostgresSaver.__name__)"`
  → expects exactly `driver OK AsyncPostgresSaver`. Today this command raises
  `ImportError: no pq wrapper available`. **The import of `AsyncConnectionPool` is part of the Proof** — step 10
  cannot be written against a pool it cannot import.
- **Second Proof (that the guard is now dead code):**
  `uv run python -c "from app.shared.langgraph_layer.checkpointer import AsyncPostgresSaver; from typing import Any; print('is Any:', AsyncPostgresSaver is Any)"`
  → expects `is Any: False`. Today it prints `True`. Run this **before** deleting the fallback; it is the evidence
  that deleting it is safe.
- **Commit boundary:** step 0b and the fallback deletion are **one commit**. Step 10's rewrite may be a second.

### CX-3 — Step 0's precondition CLOSES FAVOURABLY, but I aimed it at the wrong server.

`pg_textsearch` is **available at 1.3.0** and not yet installed, so `CREATE EXTENSION IF NOT EXISTS
pg_textsearch` will succeed. **Steps 18 and 20 are not cut.** D5.1's justification holds. Vendor confirmed as
TigerData/Timescale, closing the VectorChord mis-attribution for good (`vchord` 0.5.3 is available but unused;
`vchord_bm25` and `pg_search` are **NOT AVAILABLE**, so there is no fallback if `pg_textsearch` were ever pulled).

But the target was wrong. `.env.development`'s `POSTGRES_URL` points at **Timescale Cloud**
(`qbid1qrc75.nnro3dh8tf.tsdb.cloud.timescale.com:39662/tsdb`, PG 18.0.4), **not** the compose `timescale`
service — and nothing is listening on `localhost:5432`, so the compose Postgres has never been up in this working
copy. My step 0 Proof (`docker run --rm timescale/timescaledb-ha:pg18 ls .../extension/`) therefore answers a
question nobody asked.

**Replace step 0's Proof with a query against the instance the app actually opens a connection to**, and keep it
as a **recurring per-environment precondition** rather than a one-time check, because the managed instance's
extension set is controlled by the vendor, not by this repo:

- **Proof:** `SELECT name, default_version, installed_version FROM pg_available_extensions WHERE name IN ('pg_textsearch','vector','vectorscale','pg_trgm','unaccent','uuid-ossp');`
  → every row present with a non-null `default_version`. Expected today: `pg_textsearch` 1.3.0 available/not
  installed; `vector` **installed 0.8.2** (older than the available 0.8.6); `vectorscale` installed 0.9.0;
  `pg_trgm`, `unaccent`, `uuid-ossp` available and **not installed**.
- **Second Proof (the access method, which is what the query syntax actually needs):**
  `SELECT amname FROM pg_am WHERE amname IN ('bm25','diskann','hnsw','ivfflat');`
  → today returns `diskann`, `hnsw`, `ivfflat` and **no `bm25`**, because `pg_textsearch` is not installed yet.
  After step 18's migration it must return `bm25`. Likewise `to_bm25query()` currently has **0 `pg_proc` rows**.
- **Failure branch unchanged in shape, only in trigger:** if a future environment lacks `pg_textsearch`,
  steps 18 and 20 are void there and BM25 must degrade, not error.

### CX-4 — THE BIG ONE: there are no tables. The document/vector/search schema is greenfield.

`alembic_version` holds exactly one row: **`0004`**. The billing lineage is genuinely applied (15 billing/audit
tables exist). Everything upstream of it on the same lineage — `c0c17c6eb1cc`, `2bc7726317f6`, `8a7d9b1c2e3f`,
`9f4a1b7c6d2e` — is **stamped but never applied**. Confirmed absent: `documents`, `chunks`, `search_documents`,
`search_chunks`, `clauses`, `parent_documents`, `events`, `memory_versions`. **Zero rows anywhere.**

What this changes in this plan:

1. **Step 3 (`Vector(768)` un-hardcode) stops being a migration risk and becomes a column definition.** There is
   no `ALTER TYPE`, no typmod widening problem, no data to preserve. The pgvector non-widenability constraint I
   flagged is still true in general and still worth encoding as a design note — but it does not bite here.
   Correspondingly, **the `## Risks` entry "changing `EMBEDDING_DIMENSION` is a re-embedding job" is downgraded to
   a forward-looking risk only**: today it is free. This is the single cheapest moment in the project's life to
   settle the dimension, which strengthens step 2's case rather than weakening it.
2. **Any step whose Proof reads or writes a table must state that it creates the table first.** Every
   `SELECT count(*)`-shaped Proof I wrote against `chunks` / `search_chunks` is currently a
   `relation does not exist` error, not a `0`. Where such a Proof appears above, read it as gated on change 0.
3. **`alembic upgrade head` cannot repair this** — the revisions are marked applied, so upgrade skips them; and
   `alembic downgrade` **fails**, because the downgrade bodies drop tables that do not exist. Do **not** attempt
   `stamp base` + re-upgrade: the billing revisions sit downstream in the same lineage and would try to recreate
   the 15 tables that genuinely exist. **Change 0 owns this**, and its shape is now: merge the two heads, then add
   **one new migration that creates the target schema outright**. My D8 gate is unchanged in force but larger in
   scope than "a merge revision".
4. **D5.1's `DROP TABLE` + retarget shape is unnecessary** — there is nothing to drop. This is also the strongest
   possible confirmation that no write path is live, which is the premise every band of this plan rests on.
5. **The `clauses` hole was never a `clauses` problem.** `findings-deployment.md` §6's three angles were all
   pointing at one much larger hole.
6. **`pg_trgm` is not installed**, so `ix_search_chunks_content_trgm` has never existed and the trigram branch of
   RRF **has never run**. Step 18 must therefore treat trigram as a **new** branch to bring up, not an existing
   one to harvest — and its Proof must assert the extension and the index, not just the SQL.

### CX-5 — The checkpointer needs its own URL accessor. `get_database_url()` is the wrong one.

Not in my plan at all, and step 10 cannot be written without it. `checkpointer.py:9` reads
**raw `settings.POSTGRES_URL`**, and that value carries **no password** — auth for the main engine works only
because `connections/postgres.py:30-71` `get_database_url()` repairs the URL: rewrites `postgres://` /
`postgresql://` → `postgresql+asyncpg://` (`:36-47`), strips `?sslmode=require` / `&channel_binding=require`
which asyncpg rejects as query args (`:51-54`), and **injects the password** from `settings.POSTGRES_PASSWORD`
(`:57-70`).

So step 10 faces two URL defects at once, and they pull in opposite directions:

- Raw `POSTGRES_URL` is **passwordless** → psycopg cannot authenticate.
- `get_database_url()` returns **`+asyncpg`** → psycopg cannot parse it. It is a SQLAlchemy dialect alias, not a
  libpq scheme.

**Neither existing option works.** Step 10 must add a **psycopg-flavoured accessor** (`postgresql://` scheme,
password injected, `sslmode=require` **retained** — psycopg wants that parameter, unlike asyncpg). This also
finally explains `checkpointer.py:9`'s docstring recommending `postgresql+asyncpg://`: it is wrong in **both**
directions, and I had only diagnosed one of them.

- **Proof for step 10:** `uv run python -c "from app.connections.postgres import get_psycopg_url; u = get_psycopg_url(); assert u.startswith('postgresql://'), u; assert '@' in u.split('//')[1].split('/')[0], 'no credentials'; print('psycopg url shape OK')"`
  → expects `psycopg url shape OK`. Never print the URL itself; it carries a secret.
- **Note for change 0:** the durable fix is that no caller can obtain an unusable URL — the repair belongs in
  `settings` or a single accessor pair, not in three call sites.
  `shared/langchain_layer/agents/memory/cognee_client.py:111` has the **same** raw-URL bug (change 4's territory),
  and `features/auth/service.py:512` uses the right helper but builds a **second engine outside the lifespan**.

### CX-6 — Fog items now CLOSED by the probe

- **Fog 1** (`pg_textsearch` availability) → **CLOSED, favourable**, but re-aimed at Timescale Cloud (CX-3).
- **Fog 5** (rows in `search_chunks` / `chunks` / `clauses`) → **CLOSED: the tables do not exist.** My suspicion
  that "some schema was created out-of-band" is **disproved** — the opposite happened: someone ran
  `alembic stamp` instead of `alembic upgrade`.
- **Fog 6** (diskann index on `chunks.embedding`, trigram on `chunks.content`) → **CLOSED: neither exists**,
  because `chunks` does not exist and `pg_trgm` is not installed. `documents/repository.py:15-19`'s DiskANN
  `SET LOCAL` tuning and `TRIGRAM_SIMILARITY_THRESHOLD` are therefore tuning for indexes that have never been
  built. The `diskann` access method **is** present (vectorscale 0.9.0 installed), so the DDL will work once the
  table exists — but step 18 must **create** both indexes, not assume them.
- **Fog 11** (dedicated ingestion Celery queue) → still open, still a `design.md` Open Question to be asked.
- Fog 2, 3, 4, 7, 8, 9, 10 → **unchanged and still open.**
