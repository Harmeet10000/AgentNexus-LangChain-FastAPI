> Change class: **L** (full checklist + verification matrix).
> Role: reviewer, not author. Read `proposal.md`, `specs/` (8 files), `design.md`, `adrs.md` before completing anything.

Reviewed against `docs/relay/decisions.md` (D1–D17, D14.1–D14.4), `docs/relay/dispositions.md` `## change 1 — ingestion`,
`docs/relay/conventions-openspec-skeleton.md`, and the four sibling changes' spec deltas. All eight prescribed checks
are reported below, including the ones that pass.

| # | Check | Result |
|---|---|---|
| 1 | Coverage of plan steps (Bands A–E) | pass — ~26 steps sampled, no unrecorded omission that changes work; 1 minor gap (N1) |
| 2 | Decision contradictions (D14 / D15 / D17 / no-outbox) | **1 blocking** (F) + 1 open-question defect (E) |
| 3 | Non-Goals completeness vs dispositions | pass — all six required DROP/DEFERs present |
| 4 | Implementation leakage into `specs/` | pass — 2 hits, both house style; non-blocking |
| 5 | Delta mechanics (`ADDED` vs change 0's `MODIFIED`) | pass — no requirement collision, `Purpose` rules correct |
| 6 | Falsifiability | **1 blocking** (I) + 4 weak scenarios (N2) |
| 7 | Factual accuracy of code claims | 22 claims checked, **21 confirmed, 1 wrong** (G) |
| 8 | Scope encroachment / double ownership | **3 blocking** (C, A, B) |

## Completeness

**Check 1 — plan-step coverage.** I sampled ~26 step identifiers across all five bands of `docs/relay/plan-change1.md`
by targeted `rg` (never a whole-file read), including the `## Corrections adopted post-plan` block at ~line 1503, which
I treated as authoritative over earlier text. Every sampled step lands either as a requirement in `specs/` or as an
explicit `design.md` Decision with a stated reason it cannot be a spec. Specifically verified as landed: the pipeline
fold (`document-ingestion-pipeline:9`), paragraph truncation and the async-parse block (`hierarchical-document-chunking`),
the four-path embedding collapse and dimension unification (`unified-embedding` + ADR-1), the placeholder-vector ban
(`typed-exception-handling:3`), the retry-policy repair (`typed-exception-handling:21,38,56`), the driver/alias fix
(`langgraph-checkpointing:96`), state shrinkage (`langgraph-checkpointing:51`), worker deployment
(`celery-worker-deployment`), canonicalisation (`graph-entity-canonicalisation` + ADR-2), the index-name literal
(`hybrid-retrieval-ranking:79`), and the fuzzy branch (`hybrid-retrieval-ranking:91`). Verified as recorded-not-specced
with a reason: the transformer-dependency item (Decision 3, recorded **unachievable**, which is what disposition item
176 requires), the `CacheBackedEmbeddings` wrapper (Decision 4), the batch embedder carve-out (Decision 15), the
node-failure-pattern harvest (Decision 16), and the durable-event capability left alone (Decision 18). One gap, minor,
is recorded as N1 below.

**Check 3 — Non-Goals completeness.** Cross-checked `design.md:33-80` against every DROP/DEFER change 1 owns in
`docs/relay/dispositions.md`. All present: item 165 + Up#3 (Uber agentic-RAG pattern), Up#4 (`markitdown`), item 164
(the RAG umbrella), 138-residue-b (the `vector_store` singleton), item 176 (present as *unachievable as stated*, not a
silent drop — Decision 3 carries the reasoning), and the `CacheBackedEmbeddings` rejection (Non-Goal plus Decision 4
plus ADR-1 alternative (b)). No required disposition is missing. This check is clean.

## Correctness

**Check 2 — decision contradictions.**

- **D15 (write `chunks`, never `clauses`)** — honoured, and honoured well. `document-ingestion-pipeline:103` ("Chunk
  records are the sole persisted retrieval unit", with a scenario forbidding legacy clause records) plus Decision 10
  plus `hybrid-retrieval-ranking:50,58`. Nothing in the change writes `clauses`, `parent_documents`, `entities`, or
  `relationships` as retrieval truth. Clean.
- **D14 (all DDL belongs to change 0)** — honoured. `document-ingestion-pipeline:150` forbids the change from defining
  or applying schema, and Decision 11 plus the Migration Plan gate every table-touching requirement behind change 0's
  single migration. Clean. Note that `unified-embedding`'s "persisted column width equals the configured value" and
  `hybrid-retrieval-ranking:91`'s extension/index requirements are *satisfiable only after* change 0 lands; design
  states that dependency, so this is a sequencing fact rather than a defect.
- **D17 (Saul wiring stays commented; no flag defaulting on; import/type-level proofs)** — honoured in Decision 12 and
  in the Non-Goals, and no spec requires re-enabling the commented wiring or introduces a flag. **But one requirement
  contradicts it — finding F.**
- **No Proof may depend on an outbox event firing** — honoured explicitly: `proposal.md:93-94` forbids it, and
  `celery-worker-deployment` carries a dedicated requirement that worker readiness be verifiable without the dispatch
  chain. Clean, with one residual noted as N3.
- **F8 (is the `pg_textsearch` access method literally named `bm25`?)** left open by D14.2/D14.4 — **finding E**.

**Check 6 — falsifiability.** One scenario asserts something the codebase makes impossible — **finding I**. Four
scenarios are weak ("a reader inspects…") but are checkable against an inspectable artifact, so they are non-blocking
(N2). `unified-embedding`'s "A changed dimension against stored vectors is refused" cannot be exercised today (there
are zero stored vectors and no vector columns), but design's Risks section says so and the contract is still the right
one — not a defect, recorded here so the task author does not promise a proof that cannot exist.

**Check 7 — factual accuracy.** I spot-checked 22 code claims in `proposal.md` and `design.md` against source with
targeted `rg`/`sed -n`. **21 confirmed, 1 wrong (finding G).** Confirmed, with the source I checked:

- Paragraph truncation — `src/app/features/documents/classification.py:146,148,158` (`re.split(r"\n\s*\n", …)`,
  `blocks[:200]`). Confirmed.
- Blocking parse in an async function — `src/app/features/documents/parser.py:19,25,29,34` (sync `converter.convert(`
  inside `async def parse_document`, `tables=[]`). Confirmed.
- Dimension mismatch — `src/app/shared/rag/document_processing/embedder.py` declares `{"dimensions": 1536, …}` for every
  key and returns `[0.0] * config["dimensions"]` at `:167,:177,:228`. Confirmed, including the placeholder-vector claim.
- Phantom import — `src/app/shared/rag/rag_agent_advanced.py:119,198,267,373` `from ingestion.embedder import
  create_embedder`. Confirmed.
- The `= Any` fallback is the **live** path, not dead code — `src/app/shared/langgraph_layer/checkpointer.py` catches
  `ImportError` and assigns `AsyncPostgresSaver = Any`, then `if AsyncPostgresSaver is Any: … return None` from a
  function typed to return a saver, and calls `await checkpointer.setup()` on an unentered context manager. `psycopg`
  3.3.3 is installed with no libpq binding, so the import does raise. The change treats the fallback as live
  throughout (Decision 13, `langgraph-checkpointing:96-111`) — **correct, not inverted**. Confirmed.
- Retry policy — `src/app/shared/langgraph_layer/kb_retry.py` uses `wait=wait_none()`,
  `retry=retry_if_exception_type(Exception)`, `reraise=True`, and re-raises `TransientExternalError … from exc`.
  Confirmed (and it is the source of finding D).
- `add_note()` on a `dict` state — `src/app/shared/langgraph_layer/ingestion_kb/nodes.py:212-256`. Confirmed.
- Duck-typed embedding callable and the literal BM25 index name — `nodes.py:738-746 _call_embedding_fn`,
  `_force_merge_bm25` with `'clauses_bm25_idx'`. Confirmed.
- Literal index name in retrieval — `src/app/features/search/repository.py:410-422` (three occurrences of
  `'search_chunks_bm25_idx'`), `:337 FROM clauses`, `:383 JOIN clauses c`, `:236 trigram_search`. Confirmed.
- Fusion exists with a stated constant — `src/app/features/search/fusion.py`, `constants.py:8 RRF_K = 60`. Confirmed.
- Single-stage ingestion graph — `src/app/features/documents/ingestion_graph.py:49-61` (one `add_node`, one
  `add_edge(..., END)`). Confirmed.
- Logger shadowing produces a live `AttributeError` — `src/app/utils/embedding.py:5,22` against
  `src/app/utils/__init__.py:35` (embedding) preceding `:59` (logger). An interactive `from app.utils import logger`
  yields the loguru `Logger` and looks fine; running the actual test file is what proves it:
  `uv run pytest tests/unit/documents/test_normalize_embedding.py -q` → `6 failed, 1 passed` with
  `AttributeError: module 'app.utils.logger' has no attribute 'warning'`. **The proposal's claim and design's "six
  tests stay red" are both correct** — I nearly filed a false finding here and record the method so it is not
  re-litigated.
- Commented wiring / lifespan state — `src/app/lifecycle/lifespan.py:31,124,241,294-305,316-317`,
  `src/app/shared/outbox/relay.py:72`, `src/app/connections/postgres.py:31-71`,
  `src/app/features/agent_saul/dependencies.py:40-49`, `src/app/features/documents/service.py:180-188,533`,
  `Makefile:51-52`, `docker-compose.yml`. Confirmed.

## Standards

**Check 4 — implementation leakage into specs.** A backtick scan across all eight spec files returns exactly two hits,
both in `specs/typed-exception-handling/spec.md`: `:5` `exc.add_note()` and `:40` `raise ... from exc`. Both are Python
*language* idioms rather than project class or library names, and the live capability at
`openspec/specs/typed-exception-handling/spec.md` is written in the same register throughout, so naming them keeps the
delta consistent with the capability it extends. No class name, no function name, no library name leaks into any of the
seven new capabilities. Graph node names, where they appear, are the tolerated house style. This check is clean;
`:11`'s "the code catches…" phrasing is noted as N4.

**Check 5 — delta mechanics.** `specs/typed-exception-handling/spec.md` is the change's only delta on an existing
deployed capability, and its use of `## ADDED Requirements` is correct:

- The live capability `openspec/specs/typed-exception-handling/spec.md` carries 11 requirement headers (Redis, HTTP,
  Graphiti, Cognee, LLM provider, Database asyncpg, Document processing, Agent tools, Celery, Degradation boundaries,
  Import aliasing). **None** of change 1's four requirement names matches any of them, so this genuinely adds rather
  than alters. `ADDED` is the right operation; a `MODIFIED` here would have been wrong.
- **No collision with change 0.** `openspec/changes/cleanup-foundation/specs/typed-exception-handling/spec.md` carries
  a `## MODIFIED Requirements` block over exactly one requirement — *"Database operations SHALL catch
  asyncpg.exceptions.PostgresError"* — which appears in neither of change 1's four. Two changes delta one capability;
  they do not both touch one requirement. Correct.
- `## Purpose` handling is correct in both directions: absent on the delta against the existing capability (which is
  itself why that capability has a pre-existing validation failure — `design.md` Decision 17 records those failures
  verbatim rather than pretending they are new), and present and substantive — not a stub — on all seven new
  capabilities.
- Four-hashtag scenario trap: scanned all eight files for `### Scenario`. Zero hits; every scenario uses `####`.
- Counts: 61 requirements / 153 scenarios, matching the proposal's claim exactly.
- `.openspec.yaml` declares `schema: spec-gated`, `created: 2026-08-17`. Correct for this project's `config.yaml`.
- `~/.bun/bin/openspec validate --all` → `✓ change/ingestion-pipeline-unification`; `Totals: 21 passed, 6 failed
  (27 items)`. The six failures are exactly the pre-existing D12 set (`spec/cognee-v1-api`,
  `change/mintlify-documentation`, `spec/noqa-documentation`, `spec/pattern-matching-standard`,
  `spec/transactional-outbox`, `spec/typed-exception-handling`). **This change introduces no new validation failure.**

Artifact structure matches the conventions skeleton: `proposal.md` carries the class-`L` blockquote and the prescribed
sections; `design.md` carries Context / Goals-Non-Goals / Decisions / Risks-Trade-offs / Migration Plan / Open
Questions with risks in `[Risk] → Mitigation` form; `adrs.md` carries two ADRs, each with Status / Context / Decision /
Rationale-Alternatives / Consequences, and correctly declines to author the following change's schema contract.

## Risk

**Check 8 — scope encroachment.** Three requirements in this change claim work another change already owns. This is the
most consequential category in this review, because the artifacts are accepted independently and the second implementer
either duplicates the work or overwrites it. Findings C, A, and B below.

Change 0's `dependency-health-probe` was checked for a collision with `hybrid-retrieval-ranking:38-48` (re-ranker
degradation on the health surface): change 0's *"Extending the health report SHALL be additive"* makes change 1's
addition compatible rather than colliding. **Not a finding.** Change 3's `MessagesState`/TypedDict state conversion and
tool registry, and change 4's Cognee work, are not touched by any change-1 requirement: Decision 2 explicitly limits
this change to *shrinking* channels and hands the state-type conversion to change 3. Clean.

## Verdict

Eight blocking items, ranked most severe first. Each names the file, what is wrong, and what would fix it.

### C — `hybrid-retrieval-ranking` gives the same code the opposite instruction to change 2 (most severe)

`specs/hybrid-retrieval-ranking/spec.md:71-73` requires that when the lexical extension is unavailable *"the lexical
branch SHALL be omitted, fusion SHALL continue with the remaining branches, and the omission SHALL be reported"*, and
`:100-102` requires the same degrade-and-continue behaviour for the fuzzy branch.
`openspec/changes/documents-unified-schema/specs/document-retrieval-schema/spec.md:83` ("Three rank-fused retrieval
modes") requires the exact opposite: when *"a mode's required database capability is absent … provisioning SHALL fail
loudly"* and the system *"SHALL NOT silently serve a fused result from fewer modes than it declares."*

These are contradictory contracts over one code path (`src/app/features/search/service.py:161` → `fusion.py`). Whichever
change is implemented second violates the other, and a reviewer of the second change will read a passing test as a
regression. Compounding it, change 1's `:9` ("one fusion rule", "exactly one fusion implementation") and `:50` ("reads
the same chunk records the pipeline writes", "no alternative retrieval table") restate change 2's *"Single retrieval
source of truth"* (`document-retrieval-schema:10`) and its fusion requirement. **Neither design records the other**, so
this is invisible from inside either change.

**Fix:** decide the boundary once and write it into both designs. The coherent split is that change 2 owns
*provisioning-time* behaviour (a declared mode whose extension is missing is a loud provisioning failure) and change 1
owns *query-time* behaviour only in the degraded state change 2 permits — or, cleaner, change 1 drops `:62-77` and
`:91-102` entirely and depends on change 2 for extension preconditions, keeping only re-ranking and the index-name
literal. Either way, delete the duplicated fusion and single-source requirements from one side and add a
cross-reference in `design.md`.

### A — the unprovisioned-checkpointer fail-closed requirement is change 3's

`specs/langgraph-checkpointing/spec.md:131-142` ("Consumers of a deliberately unprovisioned checkpointer fail closed",
scenarios "An agent request without a checkpointer returns service unavailable" and "The absent checkpointer never
surfaces as an internal error") is the same behaviour, at the same code site
(`src/app/features/agent_saul/dependencies.py:40-49`), as
`openspec/changes/agent-tools-unification/specs/agent-runtime-resilience/spec.md`'s *"An unavailable agent dependency
yields a service-unavailable response"* with its scenario *"The persistence layer attribute was never assigned."* D17
assigns this to change 3, step 1.

**Fix:** remove requirement `:131-142` from `specs/langgraph-checkpointing/spec.md` and note in `design.md` Decision 12
that the consumer-side fail-closed contract is change 3's, this change supplying only the honest `None` return.

### F — "The application owns the checkpointer connection pool and closes it on shutdown" contradicts D17 and this change's own Decision 12

`specs/langgraph-checkpointing/spec.md:72-83` requires that *"The application SHALL create and own the connection pool
the checkpointer uses, and shutdown SHALL close that pool"*, with the observable-closure scenario *"WHEN the
application shuts down with a checkpointer active."* But `design.md` Decision 12 keeps the lifespan wiring commented and
places ingestion in the **queue worker process**, and D17 forbids re-enabling that wiring in this change. As written,
the only way to satisfy `:72-83` is to write the forbidden lifespan wiring — which is precisely the wrong work a
requirement can cause.

**Fix:** rewrite the requirement so "the application" is the *owning process* (the worker), and the observable closure
is the worker's shutdown hook, not the FastAPI lifespan; or scope it to "whichever process constructs the checkpointer
owns and closes its pool" and state in the requirement body that in this change that process is the queue worker. Add
the commented-lifespan constraint to the requirement so an implementer cannot read it as licence.

### B — "Each database URL flavour has exactly one accessor" is change 0's

`specs/langgraph-checkpointing/spec.md:144-159` ("Every consumer of a flavour uses its accessor", "No consumer can
obtain an unusable URL", "Scheme repair is not duplicated") duplicates change 0's
`openspec/changes/cleanup-foundation/specs/infrastructure-client-access/spec.md:43` (*"Every database consumer SHALL
obtain its connection URL from the shared accessor"*), `:61` (*"The accessor SHALL serve every driver flavour its
consumers need"*), and `:85` (*"No flavour is derived at the call site"*). This change's own Decision 14 concedes the
durable fix is change 0's — so the spec asserts ownership the design disclaims.

**Fix:** drop `:144-159` and keep only what is genuinely local: `:113-129` (the checkpointer consumes a credentialed
driver-scheme URL and never the SQLAlchemy dialect alias, and never logs it). Reference
`infrastructure-client-access` from `design.md` Decision 14 as the owner of the accessor itself.

### I — "Missing tables produce a clear terminal failure" cannot be satisfied

`specs/document-ingestion-pipeline/spec.md:155-157`: *"WHEN ingestion runs against a database where the document or
chunk tables do not exist / THEN the document SHALL reach a terminal failure status whose diagnostic names the missing
schema."* The document status lives in the `documents` table, which `design.md` Context confirms is one of the absent
tables. If the document table does not exist there is nowhere to record a terminal status, so the scenario is impossible
in the exact condition it describes — and the requirement body's *"SHALL NOT leave a document in a non-terminal status"*
inherits the same problem.

**Fix:** split it. One scenario for the chunk table absent while the document table exists (there the terminal status is
recordable and the requirement is meaningful); one scenario for the document table absent, where the contract must be
that ingestion fails with a diagnostic naming the missing relation through the task result and log, with no document
row implied. This also removes a Proof the task author would otherwise be unable to execute.

### D — the retry-chaining requirement's remedy cannot satisfy its own scenario

`specs/typed-exception-handling/spec.md:38-50`. The requirement body prescribes chaining via `raise ... from exc` so
that a caller's *"existing degradation branch"* still matches; the scenario at `:48-50` then asserts *"WHEN a caller
catches the provider's or framework's own base exception type around a retried operation / THEN that catch SHALL still
match the failure the retried operation produced."* Chaining sets `__cause__`; it does not preserve the raised **type**.
`src/app/shared/langgraph_layer/kb_retry.py:41-43` raises `TransientExternalError(msg) from exc`, and
`src/app/shared/langgraph_layer/ingestion_kb/nodes.py:236` catches `LangChainException` — that `except` will not match a
`TransientExternalError` however it is chained. As written, an implementer who follows the body will produce code that
fails the scenario, and the honest reading is that the two cannot both hold.

**Fix:** choose one contract and say which. Either (a) the retry boundary **re-raises the original exception type** when
attempts are exhausted and attaches the retry context as a note, in which case the caller's existing branch matches and
`from exc` is not the mechanism; or (b) the boundary raises the typed transient exception and the **callers are updated**
to catch it, in which case scenario `:48-50` must be replaced by a scenario asserting that the original type is
reachable via `__cause__` and that named callers were converted. (b) matches Decision 1's direction; either way `:48-50`
as phrased must go.

### E — F8 is treated as closed when the locked decisions leave it open

`design.md` Decision 6 correctly downgrades *"lexical and fusion already work"* to *"written and tuned, never
executed"*, but reads the lexical extension question as settled. D14.2 leaves F8 open — is the `pg_textsearch` access
method **literally named `bm25`**? — and records that answering it needs the user's authorisation for a scratch-database
check; D14.4 does not close it. F8 appears in none of the three Open Questions, so the task list will be written as if
`USING bm25` is known-correct. If the access method has another name, `hybrid-retrieval-ranking:62-69` and change 0's
index DDL are both wrong, and the failure surfaces at migration time.

**Fix:** add F8 as a fourth Open Question, phrased as needing user authorisation for a scratch-DB check, and mark the
requirement at `:62-69` as gated on its answer. Do not restate Decision 6 as closed.

### G — `proposal.md:33` inverts harvest-versus-build on re-ranking

`proposal.md:33` calls re-ranking *"the one genuinely missing third of the hybrid contract."* It is not missing. The
re-ranker exists and is already wired: `src/app/shared/langgraph_layer/retrieval_kb/graph.py:35,49,60-61` takes it as a
parameter, adds it as a node, and edges `hybrid_postgres → reranker → context_grader`;
`retrieval_kb/nodes.py:200-213` is `make_reranker_node`; `reranker.py:22` even carries the CPU-bound note. What is
actually missing is re-ranking on **one** path — `src/app/features/search/service.py:161 hybrid_search` fuses at `:184`
and returns without re-ranking, while `:259 ask_legal` goes through `build_retrieval_graph(...)` and does re-rank. This
is the one code claim of 22 that is wrong, and it matters because it turns a harvest-and-wire task into a build task and
misstates the size of `hybrid-retrieval-ranking:26-36`.

**Fix:** correct `proposal.md:33` to say the re-ranker exists and is wired in the retrieval graph but is absent from the
direct `hybrid_search` path, and add a `design.md` decision that the work is to route that path through the existing
re-ranker rather than to build one.

## Non-blocking

These do not block. Listed so they are not mistaken for oversights.

- **N1 — the archived `celery-task-registry` harvest is unrecorded.** The plan instructs checking and harvesting an
  archived task-registry capability, parallel to the `langgraph-node-result-pattern` case that `design.md` Decision 16
  records explicitly. Decision 18 covers the durable-event capability and the task-name constant, but not the archived
  registry harvest. Add a sentence to Decision 18 or a Decision 19 so the omission is on the record either way.
- **N2 — four documentation-shaped scenarios.** `unified-embedding` "The convention is recorded",
  `hierarchical-document-chunking` "The counter in force is discoverable", `hybrid-retrieval-ranking:22-24` "The fusion
  constant is stated", `langgraph-checkpointing:47-49` "The mode is stated, not implied". Each is satisfied by an
  inspectable artifact rather than system behaviour, so they are checkable but weak; `:22-24` is already satisfied by
  `src/app/features/search/constants.py:8`.
- **N3 — one scenario reaches through the outbox.** `celery-worker-deployment`'s *"WHEN an event dispatches a task name
  that is not registered"* exercises the relay, whose tables change 0 creates. Sequencing makes it satisfiable, but
  `proposal.md` says this change must not assume the dispatch chain works. Restating it as a unit-level check on the
  dispatch helper would make it independent.
- **N4 — two mechanism-shaped phrasings.** `specs/typed-exception-handling/spec.md:5` and `:40` name `exc.add_note()`
  and `raise ... from exc`, and `:11` says *"the code catches…"* rather than describing observable behaviour. Consistent
  with the live capability's register, so keep or normalise as a matter of taste.
- **N5 — two vacuous-in-practice scenarios.** `document-ingestion-pipeline:22-24` ("identical regardless of entry
  point") cannot fail once the fold leaves a single entry point; `:18-20`'s *"continue or restart"* disjunction makes
  the interesting half unfalsifiable. Tighten to "SHALL resume at the first incomplete stage" if the intent is
  resumption.
- **N6 — a forward-only contract with no executable proof.** `unified-embedding`'s "A changed dimension against stored
  vectors is refused" has no stored vectors to test against. The Risks section says so; the task author should express
  its verification as a unit test over a stubbed width, not as a data check.

## Verified clean

Examined and found correct, so a later reader can tell "verified good" from "not examined":

- **Delta mechanics** — `ADDED` is the right operation against the live `typed-exception-handling` (11 requirement names
  checked, zero overlap with the four new ones); no collision with change 0's single `MODIFIED` requirement
  (*"Database operations SHALL catch asyncpg.exceptions.PostgresError"*); `## Purpose` correctly absent on the existing-
  capability delta and substantive on all seven new capabilities.
- **Scenario hashtag depth** — zero `### Scenario` across all eight files; the silent-drop trap is not present.
- **Counts** — 61 requirements / 153 scenarios, matching the proposal.
- **`.openspec.yaml`** — `schema: spec-gated`, `created: 2026-08-17`.
- **Validation** — `openspec validate --all` passes this change and adds no new failure; the six failures are exactly
  D12's pre-existing set, and `design.md` Decision 17 records them verbatim rather than claiming them.
- **D15, D14, and the no-outbox constraint** — honoured throughout, with dedicated requirements enforcing each.
- **Non-Goals** — every DROP/DEFER this change owns is present, including item 176 recorded as *unachievable* rather
  than dropped.
- **Implementation leakage** — two hits total, both language idioms in the capability whose live text uses the same
  register; the seven new capabilities are free of class, function, and library names.
- **Factual accuracy** — 21 of 22 code claims confirmed against source, including the two known-true facts used
  correctly: the `psycopg`-without-libpq import failure makes the `= Any` fallback the **live** path (the change treats
  it as live, not unreachable), and no Proof depends on an outbox event firing.
- **Change 3 and change 4 boundaries** — no encroachment on `MessagesState`/TypedDict conversion, the tool registry, or
  Cognee; Decision 2 hands the state-type conversion to change 3 explicitly.
- **ADRs** — both are genuinely durable decisions rather than change-local ones, both carry real alternatives with
  reasons for rejection, and `adrs.md:6-8` correctly declines to author the following change's schema contract.

**VERDICT:** `CHANGES-REQUESTED`

VERDICT: CHANGES REQUESTED

---

## Author response

Written by the remediator. Nothing above this line was altered. Five of the eight blocking findings (C, A, F, B, G)
were adjudicated by the orchestrator before remediation and are recorded in `docs/relay/reviews.md` under
"Orchestrator adjudications — change 1's review" (A6–A12); those rulings were followed, not re-litigated. D, E, and I
were verified from source before acting, as were the six non-blocking items.

- **C: fixed** — change 2 wins; a missing extension **fails loudly**. Deleted from
  `specs/hybrid-retrieval-ranking/spec.md`: the lexical extension-precondition requirement (all three scenarios,
  including the degrade-and-continue one), the fuzzy-branch omit-observably requirement, the duplicated fusion
  requirement, and the duplicated single-retrieval-source requirement. That capability now contains no fusion
  requirement, no single-source requirement, and no degrade-and-continue behaviour for a missing database capability.
  Recorded as **Coordination point 1** in `design.md`, with the one deliberate asymmetry stated in the requirement
  body so change 2's reviewer does not read it as a contradiction: an unloadable **re-ranker model** still degrades to
  the fused order, because that is a recoverable runtime condition whose degraded output is still a correct ranking,
  whereas an absent extension means change 0's migration did not run. Decision 6 and Decision 9 were revised;
  Non-Goals and the scope-creep risk updated.
- **A: fixed** — deleted requirement "Consumers of a deliberately unprovisioned checkpointer fail closed" from
  `specs/langgraph-checkpointing/spec.md`. Change 3 owns the read-site 503 per D17; this change keeps checkpointer
  *provisioning* and the guarantee that setup never returns an absent value from a function typed to return a saver.
  **Coordination point 2**; Decision 12 extended. The synchronous *ingestion* surface's fail-closed requirement stays
  in `document-ingestion-pipeline` — different shared object, different router, not the site D17 names.
- **F: fixed** — the requirement no longer says "the application owns the pool". Re-scoped to *the process that
  constructs the checkpointer owns and closes its pool*, which in this change is the queue worker, plus a new scenario
  requiring that the deliberately disabled application construction **stays** disabled, so the requirement can never
  be read as licence to uncomment. The live defect the ruling identified is now specified: teardown must distinguish
  "closed a pool" from "there was nothing to close" from "was handed something that has no pool". Verified while
  fixing, and it sharpens the finding: teardown's pool guard tests `hasattr(checkpointer, "pool")`, but
  `AsyncPostgresSaver.from_conn_string` is decorated `@asynccontextmanager`
  (`.venv/.../langgraph/checkpoint/postgres/aio.py:56`), so the value it guards is an async context manager with no
  `pool` attribute — the pool would go unclosed **silently** even when one existed. Import-provable. Lifespan wiring is
  now an explicit Non-Goal citing D17. **Coordination point 3.**
- **B: fixed** — deleted requirement "Each database URL flavour has exactly one accessor". The checkpointer
  requirement was rewritten as a *consumer* contract: it takes its string from change 0's accessor for its flavour,
  repairs nothing at the call site, never receives the dialect alias, never logs the string. Decision 14 retitled and
  corrected on the fact: **two** flavours, not three, and the memory subsystem is **not a URL consumer at all**
  (`findings-database.md` §9 — its config exposes discrete host/port/user/password fields and no connection-string
  field, so handing it a URL is unimplementable, not merely inelegant). Recorded that the plain client-library flavour
  exists *because of* this checkpointer. **Coordination point 4.**
- **G: fixed, and the proposal was rewritten rather than patched** — `proposal.md`'s "the one genuinely missing third
  of the hybrid contract" is gone. New **Decision 19** states the harvest: the re-ranker exists, is wired as a graph
  edge (`retrieval_kb/graph.py:49` adds the node, `:60-61` edges `hybrid_postgres → reranker → context_grader`), and
  **self-provisions** because `nodes.py:203` resolves `reranker or CrossEncoderReranker()` despite the
  `| None = None` signature; a second ad-hoc path at `documents/service.py:426` constructs one **per call**. The
  requirement was reshaped from "build a re-ranker" to *one implementation, model loaded once per process, every
  ranked path re-ranks*, with a dedicated scenario for the single genuine gap — the direct fused path at
  `search/service.py:161`. Decision 19 also records the second-order narrowing of disposition 176:
  `sentence_transformers` **stays**, only the tokenizer half remains in scope. And it records the standing rule the
  third such correction has now earned: grep for a symbol's *edge wiring*, and follow one layer past `| None = None`.
- **D: fixed** — confirmed from source before acting. `kb_retry.py:41-43` raises `TransientExternalError(msg) from exc`
  inside `except Exception`; `TransientExternalError` (`:15`) derives from `Exception`, and `nodes.py:236` catches
  `LangChainException`, which is not in that hierarchy — so the scenario "that catch SHALL still match" was
  unsatisfiable by the remedy the requirement body prescribed. Took the reviewer's option **(b)**, which is Decision
  1's direction: the boundary raises **one typed transient failure chained to the original**, and callers are
  **converted** to catch it. The old scenario is gone; three replace it — the cause is recoverable, every caller's
  degradation branch matches what the boundary raises, and a degradation branch actually fires on an exhausted retry.
  Decision 1 now states the mistake explicitly (chaining preserves the *cause*, never the *type*) so it cannot be
  reintroduced. All four requirements stay **ADDED** and none collides with change 0's single `MODIFIED` asyncpg
  requirement.
- **E: fixed — but as a closure, not as an Open Question.** The reviewer was right that Decision 6 read F8 as settled
  when D14.2 and D14.4 left it open, and the fix was going to be a fourth Open Question. Mid-remediation the
  orchestrator closed it: the user authorised `CREATE EXTENSION pg_textsearch` against the live instance, scoped to
  that one statement, and the probe ran (`findings-database.md` §10). So F8 is recorded as **closed with its answer**
  instead. The access method **is** `bm25`; operator classes `text_bm25_ops` / `text_array_bm25_ops`; `to_bm25query`
  has two overloads, one of which takes the index name. Two consequences are now in Decision 6 and **Coordination
  point 5**: the repository's lexical SQL is **already correct** and is a harvest, not a rewrite (`repository.py:415`,
  `:417`, `:419`, `:430`, `:432`, `:433` use the two-argument index-scoped overload with the right negation and
  ordering); and BM25 still cannot run because **no `bm25` index exists anywhere**, which — since the overload takes
  the index name as a *literal argument* pinned at `search/constants.py:15` — makes the index name part of the query
  contract and its creation a named dependency on change 0, not work this change owns. The retrieval requirement now
  says so. Recorded in `## Open Questions → Closed since the first draft` rather than deleted, so a reader who
  remembers it as open can see how it closed.
- **I: fixed** — the impossible scenario is gone. `document-ingestion-pipeline`'s schema requirement now splits by
  *which* table is missing, exactly as the reviewer proposed: document table present and chunk table absent → terminal
  failure status recordable, requirement meaningful; document table itself absent → failure surfaces through the task
  result and log with **no document row implied**, and the requirement body states that no reading of it may demand a
  status transition with nowhere to be written. The dependency on change 0's single migration is named in the
  requirement body. Both scenarios are executable after that migration; neither needs `alembic upgrade head --sql`.

### Non-blocking

- **N1: fixed** — the archived `celery-task-registry` harvest is now on the record. Verified it exists at
  `openspec/changes/archive/2026-06-22-quality-fixes-batch-2/specs/celery-task-registry/spec.md` and is absent from
  the live capability directory, so it is the Decision 16 situation and is handled the same way: harvested into
  `celery-worker-deployment`'s task-name requirement, not forked into a second capability under the same name. Added
  to Decision 18, with one deliberate tightening stated — the archived text let an unregistered name fall through
  permissively behind a warning, which is the invisible-failure shape this change exists to remove, so the requirement
  demands it be *reported as a failure*.
- **N3: fixed, and N1 supplied the seam** — the scenario that reached through the relay ("an event dispatches a task
  name that is not registered") is now a unit-level check on the **dispatch helper**, invoked directly, requiring no
  durable outbound event. A second scenario covers a malformed payload for a registered name. This is the archived
  registry's own mechanism, so N1 and N3 closed together.
- **N5: fixed** — both tightened. "continue or restart" became "SHALL resume that document's ingestion at its first
  incomplete stage", so the interesting half is falsifiable; "identical regardless of entry point" became "exactly one
  ingestion entry point exists, and it is the multi-stage pipeline", which is checkable by inspection and cannot be
  satisfied vacuously.
- **N2: acknowledged, kept** — verified each of the four is checkable against a named inspectable artifact. Two are
  moot now: the fusion-constant scenario went with the deleted fusion requirement (and was already satisfied by
  `search/constants.py:8 RRF_K = 60`). The remaining two are proven in `tasks.md` against the artifact rather than
  against behaviour, which is what makes them checks rather than prose.
- **N4: partially fixed** — normalised the one phrasing that described the code rather than the behaviour: `:11`'s "the
  code catches the provider's own exception type, adds a note, and raises" is now "a typed project exception SHALL be
  raised, carrying the provider's own exception as its cause and a note naming …". Left `exc.add_note()` and
  `raise … from exc` where they name Python *language* idioms, matching the live capability's register throughout, as
  the reviewer allowed.
- **N6: acknowledged, and the task list obeys it** — verified there is nothing to test against, so the proof for "a
  changed dimension against stored vectors is refused" is a unit test over a **stubbed** stored width, never a data
  check. Added as a Risk in its own right so a later reader does not read the absent data check as an omission.

