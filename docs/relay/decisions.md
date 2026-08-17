# Refactor decisions — locked

Source of truth for the big refactor driven by `tests/performance/todo.md` items **210** (`:297`) and **155** (`:278`).
These were answered by the user directly. They are **constraints, not suggestions** — no leg of the relay may
re-litigate them, and any plan that contradicts one is wrong by definition.

Recorded 2026-08-17.

---

## D1 — Ingestion pipeline location

**Promote `ingestion_kb/` to live.**

`ingestion_kb/` becomes the real pipeline. It is currently unreachable (its lifespan wiring is commented out and
its router is never mounted) but it is the *better* implementation: a real 7-node graph with `Send` fan-out and
reducer-based chunk accumulation. The currently-live `features/documents/ingestion_graph.py` path — a one-node
graph wrapping 7 opaque Python stages — is what gets replaced or retargeted.

## D2 — Graphiti and Cognee both survive, with two distinct roles

Not a replacement of one by the other. Graphiti is load-bearing today (four agent tools, search service,
verifier, ingestion node). Cognee runs nothing at all — `CogneeStore` is a stub, and only `setup_cognee` is
ever called. So "replace reconciliation with Cognee memory" is **building the Cognee side from near-zero**,
not swapping one working system for another.

The exact role boundary was not specified by the user. Proposing a defensible one is a deliverable of the
memory scout report.

## D3 — Item 210's arrow is a work order

`ingestion -> documents -> tools -> cognee` means **sequence of work**: fix ingestion first, then documents,
then tools, then Cognee. It is *not* a data-flow diagram to be wired up.

## D4 — Aggressive sweep, with explicit carve-outs

Delete confirmed-dead code. A file is only a deletion candidate if its caller set is *proven* empty
(`graphify affected`). Carve-outs that stay despite looking dead:

| Kept | Why |
|---|---|
| `ingest_v2.py` | Legitimately different use case: batch/local-folder ingester (glob over pdf/docx/pptx/xlsx + audio), `clean_databases`, `create_ingestion_pipeline`. Not the per-upload S3 path. |
| `embedder.py` | `ingest_v2.py:18` imports `embed_chunks` from it. It only looked zero-caller because `ingest_v2.py` was itself unreachable. |
| `tasks/pageindex_tasks.py` | Todo (b) defers pageindex. Currently raises `NotImplementedError`. |
| `write_final_report.py`, `memory_pipeline.py` | Deletion **deferred to change 4**, not change 0 — they are the only existing reference for how Cognee writes are meant to work, and change 4 rebuilds exactly that. |

## D5 — ~~`features/search/` is entirely out of scope~~ → **REVERSED, see D5.1 at the end of this file**

> The text immediately below is the **superseded** D5.0, kept for the record. The binding decision is D5.1.

Tables, router, and service all untouched. It **stays unmounted** (so text search remains unreachable — that is
the accepted status quo, not an oversight to fix). `process_ingestion_document` stays. It keeps its own
`build_embedding_client`.

**Accepted consequence:** the unified `langchain_layer` embedder is adopted by the ingestion/documents path
only. The repo ends with **2 embedding paths, not 1** — down from 4, not collapsed entirely. This is the
honest cost of keeping search as-is.

## D6 — ToolRegistry: unify in `langchain_layer`, keep tags

Three classes are named `ToolRegistry`. The `langchain_layer` one survives; tag-based tool selection is
preserved as a capability. All importers get rewritten to the survivor.

## D7 — `open_deep_search` is out of scope

A parallel stack with its own tools, three message channels, and its own retry mechanism. No work scheduled
there. Where it duplicates something being unified elsewhere, that is *recorded as a future hazard* — not fixed
now.

## D8 — Openspec structure: cleanup + 4 sequenced changes

Five change directories total:

- **change 0 — cleanup/foundation.** Alembic head merge, `env.py` model registration, dead-code deletion
  (reconciliation, orphan schema, stubs, zero-byte files), and the confirmed runtime breaks.
- **change 1 — ingestion.** Promote `ingestion_kb`, hierarchical chunking for legal docs, unified embedder,
  unblock the docling event loop, Celery offload, graph in `app.state`, tenacity.
- **change 2 — documents.** Consolidate onto `UnifiedDocument`/`UnifiedChunk`.
- **change 3 — tools.** Registry/idempotency/memory-scope unification, retarget tools off nonexistent tables,
  prompt + TOON adoption, `MessagesState`, agent config `Field(description=...)`.
- **change 4 — cognee.** Build the Cognee memory side, retire the deferred report/memory-pipeline files.

Ordering is load-bearing: the alembic merge gates change 1, the registry unification gates change 3, and
change 4 depends on reconciliation already being gone.

---

## Standing ambiguities resolved along the way

- **Item 210's "tools"** = agent tools (the `langchain_layer/agents/tools/` + `features/*/tools` surface),
  not CLI tooling.
- **Todo (g)** is *not* a Pydantic deprecation fix — that was already resolved by an earlier reorg
  (`configuration.py`/`deep_researcher.py` no longer exist, no `Field(...optional=|metadata=)` misuse remains).
  It is: add `Field(description=...)` to agent config models, starting with `AgentSpec`.
- **Todo (a)'s dataclass→pydantic conversion** is already done in the ingestion area —
  `document_processing/models.py` is all `BaseModel`.
- **Todo (1)'s "Template vs ChatPromptTemplate" question** is already answered by the code: both paths were
  built. `SystemPromptParts` does `string.Template.safe_substitute` *and* `.to_chat_template()`. The real
  problem is **adoption** (~30 bare-string sites, plus a competing `render_prompt_sections` helper).
- **`serialize_to_toon` is already reusable** — one definition, 16 call sites. The defect is import
  inconsistency, not duplication.

## Process rule adopted mid-relay

Every scout/planner report is **written to `docs/relay/` on disk** and only an index is returned to the
orchestrator. Reports that live solely in orchestrator context are destroyed by the first summarization
boundary — which is exactly what happened to the first scout pass on 2026-08-17.

---

# Second decision round — 2026-08-17 (later)

## D5.1 — `features/search/` is IN scope (replaces D5.0)

Three scout findings made the carve-out untenable:

1. **BM25 + RRF already exist, and only in `search/`.** Not tsvector — the `pg_textsearch` extension:
   `features/search/repository.py:415,417,419` run `c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')`;
   RRF fusion is `features/search/fusion.py:28` with `k=60` (`features/search/constants.py:8`). Sub-todo item 195
   asks for exactly this. Under D5.0 we would have **rebuilt from scratch, in a second place, what already
   works** — while the working copy stayed unreachable. Item 195 is therefore **not greenfield**; the only
   genuinely missing piece is **re-ranking**.
2. **`build_retrieval_graph` (`shared/langgraph_layer/retrieval_kb/graph.py:28`) has exactly one caller** —
   `features/search/service.py:259`. Freezing search freezes the retrieval graph, which is squarely change 1.
3. **Item 185 was unactionable under D5.0** — it requires editing `features/search/model.py:73-79`, which D5.0
   barred. Contradiction #2 in `todo-overlap.md` §5 dissolves.

**Consequences replacing the old accepted trade:**

- Embedder collapses to **1 path, not 2**. `features/documents/service.py:24` already imports search's
  `build_embedding_client` (`features/search/embeddings.py:10`) — documents and search are *already* one path.
  The real second path is `ingestion_kb`'s duck-typed `embedding_fn` (`ingestion_kb/nodes.py:738-745`), which
  caches in redis, normalizes and retries but passes **no `task_type`** and embeds one text at a time.
- **Migration shape is `DROP TABLE` + retarget, NOT backfill.** Every write path into `search_documents` /
  `search_chunks` is dead: `ingest_document` (`features/search/service.py:72`) is reachable only via the
  unmounted `POST /search/ingest`; `process_ingestion_document` (`:291`) fires only from an outbox event
  emitted inside that same unreachable call. No seed, fixture, or factory exists.
- **`content_tsv` is pure subtraction** — STORED generated column, live GIN index, **zero readers**.
- Two search capabilities have **no target equivalent** and must be preserved deliberately or dropped on the
  record: `trigram_search` (`features/search/repository.py:236`, index `ix_search_chunks_content_trgm`, 1 of 3
  RRF branches) and upsert `updated_at` (`UnifiedChunk` has no such column).
- **Mounting the search router is NOT part of this reversal.** In scope means refactor and unify. Mounting is
  gated on D5.2 and stays out.

## D5.2 — `UserIdDep` is broken repo-wide; it moves to change 0

`features/search/dependencies.py:45` reads `request.state.user_id`; nothing in `src/` ever assigns it and no
auth middleware exists (`main.py:77-94`). **`features/documents/dependencies.py:61-62` is identical** — and the
documents router **is** mounted (`api/v1.py:15`). Previously logged as unverified Fog from `notes.md`;
**now confirmed**, and it is a live `AttributeError` on already-shipped surface, not a search-only latent bug.

Blocking sub-issue for change 2: `UnifiedChunk.user_id` and `UnifiedDocument.object_uri` are NOT NULL with no
default and **no source value** in the search ingest path (`features/documents/model.py:40,43,87` vs
`features/search/service.py:74`). Dedup semantics also change: global `content_hash` unique
(`features/search/model.py:34`) → `(user_id, content_hash)` (`features/documents/model.py:32`).

## D6.1 — ToolRegistry survivor named

`shared/langchain_layer/agents/tools/base.py:58` is the survivor (D6 said "unify in langchain_layer, keep
tags"; this names the file). Note `shared/rag/graphiti/registry.py` is **not deletable** — `:34-122` is live
code and its `ToolRegistry` at `:56` is consumed by `agent_saul/graph.py:16,91` and
`agents/factory.py:182,205`. Its eager imports at `:41-46` impose an ordering constraint: **rewrite importers
before deleting `shared/agents/**`, or you get `ImportError` at import time.**

## D9 — LangExtract stays; item 195 is its target

Item 136 (`todo.md:225`) marks LangExtract `ABANDONDED`; item 43 (`:67`) shipped it; item 195 (`:282`) makes it
a **prerequisite stage**. Three positions, one feature. **Resolved: 195 wins.** LangExtract is repositioned
**upstream** of the postgres and graphiti writes, as part of change 1. Item 136's abandonment is superseded.

## D10 — Memory decay is DROPPED; the gap is recorded

Item 170 (cron for memory decay → celery) is **not** being built. `src/tasks/memory_decay_reconciliation_tasks.py:51`
`_compute_decay` is the repo's **only** decay formula and it dies with the reconciliation deletion. This is
accepted, on the record, because:

- The task was never registered anyway — no `@celery_app.task` decorator in the module, module absent from
  `connections/celery.py:191-196` `include`, and `beat_schedule` (`:259-276`) holds only 4 billing entries.
  Its four tables were never created by any migration.
- The repo's own change doc agrees the replacement does not cover it:
  `openspec/changes/cognee-saul-memory-migration/proposal.md:20-21` — *"Cognee v1.1 has no built-in
  curation/decay/dedup"* — and marks `saul-cognee-maintenance-worker` / `saul-cognee-reconciliation`
  **deferred**.

**Recorded gap:** after change 4, Cognee memory grows without decay, curation, or dedup. Change 4's `design.md`
must carry this as an explicit Non-Goal, and `adrs.md` must name it as a known consequence. Item 155's word
"entirely" is honoured for *reconciliation removal*, not for *capability parity*.

## D11 — `todo_temp.py` is deleted in change 0 (Fog closed)

`shared/rag/document_processing/todo_temp.py`, 783 lines: `ast.parse` raises `IndentationError` at `:406`
(`__all__` closes at `:404`, then `:405-406` is an orphaned class-body docstring + `__init__` with no `class`).
**A live caller is impossible — the app would not boot.** `graphify affected` → no nodes. It is a duplicated
draft (`create_extraction_tools` at `:360` and `:773`; `process_document_full` at `:221` and `:632`). Both
ruff `invalid-syntax` errors in the 125-error baseline are this file (`:406`, `:773`), so deleting it moves the
lint baseline to 123.

## D12 — Openspec mechanics settled by CLI observation

- CLI is `openspec` v1.8.0 at `/home/harmeet/.bun/bin/openspec`; `validate --all` is the working invocation.
- **Baseline is 16 passed / 6 failed of 22 items.** Pre-existing failures: `spec/cognee-v1-api`,
  `change/mintlify-documentation`, `spec/noqa-documentation`, `spec/pattern-matching-standard`,
  `spec/transactional-outbox`, `spec/typed-exception-handling`. Acceptance criterion is therefore
  **"no new failures beyond these 6"**, never "validate --all passes".
- **`review.md` is NOT enforced by the CLI.** `change/cognee-saul-memory-migration` passes validation while
  declaring `schema: spec-driven` and shipping no `review.md`. The gate is instructional
  (`schema.yaml:394-396`) — we honour it by choice, and a fresh subagent writes each review, never the author.
- **Change IDs are bare slugs.** Archive adds the `YYYY-MM-DD-` prefix. The one bare-slug archived change
  (`integrate-deep-research-into-saul`) was added 2026-06-21, *before* the dated ones — a legacy artifact, not
  a counterexample.
- All five changes are class **L**; `design.md` is mandatory for each.
- **Scenario headers take exactly four hashtags** (`schema.yaml:164-165` — three fails *silently*).

## D13 — Disposition of the previously-uncaptured backlog items

`todo-overlap.md` found backlog items belonging to this refactor that were not in the user's list. Dispositions
are recorded in `docs/relay/dispositions.md`. Anything marked DROP or DEFER there is a **recorded gap**, to be
surfaced in the owning change's `design.md` Non-Goals — not silently omitted.

---

# Third decision round — 2026-08-17 (after the live-database probe)

All four answered by the user directly. Evidence behind them: `docs/relay/findings-database.md`.

## D14 — Migration repair: **merge the two heads, then ONE new migration creating the target schema**

The live DB is stamped at `0004` while the entire document/vector/search branch was never applied
(`findings-database.md` §4). Of three options, the user chose the lowest-risk one:

```
c0c17c6eb1cc → 2bc7726317f6 →┬→ a71f0d7d9c12 ─┐
                             └→ 8a7d9b1c2e3f  │  (all stamped,
                                └→ 9f4a1b7c6d2e│   none ever ran)
                                  └→ 0001..0004┤
                                               ↓
                                        merge_heads
                                               ↓
                              NEW: create the target schema
```

**Explicitly rejected:** editing the unapplied revisions in place (`8a7d9b1c2e3f` / `9f4a1b7c6d2e` /
`a71f0d7d9c12`) — even though they were never applied here, we cannot prove no other environment applied them;
and a full rebaseline/squash — the 15 genuinely-existing billing tables would have to be reconciled by hand
against a rewritten root.

**Accepted cost, on the record:** the revision chain permanently reads as a lie — three revisions stay stamped
while creating nothing. Change 0's `design.md` must state this, and the new migration must be authoritative for
the target schema rather than additive to what those revisions claim to have built.

**Consequence for change 2's revised step 7:** its *preferred* fix (change 0 strips the search DDL from
`8a7d9b1c2e3f` during a rebaseline) is **not available** under D14 — that is the rejected option. Its stated
fallback governs: change 2 ships no DDL, deletes the ORM models, and proves it with
`alembic upgrade head --sql | grep -c 'CREATE TABLE search_'` = 0, which guards fresh databases where the risk
now lives.

## D15 — ADR first: change 1 writes `chunks`, never `clauses`

Change 2's schema ADR is authored and accepted **before** change 1 implements its persistence nodes. Change 1's
`ingestion_kb` persistence nodes (`ingestion_kb/nodes.py:497,551,597,660`) target `chunks` from the start.

Item 210's work-order arrow (D3) bends exactly once, for the schema contract only. Justification: `clauses` does
not exist and never has, so writing to it first would preserve nothing and would roughly double change 2 with a
`clauses`→`chunks` migration for zero rows.

## D16 — `UnifiedChunk` gets `updated_at`

Included in change 0's `CREATE TABLE` (D14's new migration) and written from both the upsert conflict set and
`build_chunk_rows`. ~10 lines. Reason it is not cosmetic: it is the only way a later re-embedding campaign can
distinguish a current-generation embedding from a carried-over one — which bears directly on change 1's
embedding-dimension work (item 198.3, where `document_processing/embedder.py:26-29` still returns 1536 against
`Vector(768)` columns).

## D17 — The unwired Saul graph was **deliberate**, and it stays commented

User: *"just comment it and yes that was deliberate at that time"*. So `lifespan.py:234-247` and `:294-305` are
**not a regression**. This overturns the working assumption every plan carried (which followed D5.2's logic that
the identical defect on the documents router is live).

Consequences:

- **Change 3 step 18 is not a restoration.** The wiring stays commented. Do not enable it, and do not introduce
  a flag that defaults on.
- **`get_saul_graph` failing closed matters MORE, not less** (change 3 step 1). An intentional gap must return a
  clean 503, never `AttributeError`. `features/agent_saul/dependencies.py:45` reading
  `app.state.langgraph_checkpointer` unguarded is the defect to fix, and it is now the *primary* justification
  for that step rather than a side effect of restoring wiring.
- **Proofs for those steps become import-level and type-level only.** Commented code cannot be type-checked,
  linted, or tested, so it will rot; every plan step touching it must prove correctness by construction
  (`ty`, import, unit test against the constructor) and never by running the graph.
- **Change 4's step 12 read seam is speculative** and must be labelled as such in its `design.md`. Change 4
  already concedes it cannot be proven by running the product; D17 makes that permanent rather than temporary.
- The phrase *"at that time"* is noted: this is a decision about the present state, not a commitment that the
  graph stays unwired forever. Nothing in these five changes should make re-enabling it harder.

---

## D14.1 — Orchestrator adjudication: D14 **stands**; the change-0 planner's rejection is overruled, its finding is adopted

On its resumed pass the change-0 planner **adopted** "merge the two heads" but **rejected** "one new migration
creates the target schema", proposing instead that change 0 create *only* the two outbox tables and record the
remaining phantom revisions as knowingly-phantom. Its two stated reasons, and why each fails:

| Planner's reason | Adjudication |
|---|---|
| "It smuggles changes 1–2's unsettled schema decisions into change 0." | **Dissolved by D15.** Change 2's schema ADR is authored and *accepted before* change 1 implements, which is exactly the mechanism that settles the schema ahead of change 0's migration. The decision is not unsettled; it is sequenced. |
| "It creates tables change 2 drops." | **Factually wrong under D14.** Change 2 ships **no DDL** and drops nothing — there is nothing to drop, because nothing exists (`findings-database.md` §4). This reasoning comes from change 2's *superseded* "DROP TABLE + retarget" framing under the original D5.1, not from the D14-refined version that governs. |

D14 and D16 are user decisions, and D16 is explicit that `updated_at` is *"Included in change 0's `CREATE TABLE`
(D14's new migration)"*. The user placed the target schema in change 0. A planner may not relocate it.

**However, the planner's positive finding is adopted in full, and it is stronger than the planner argued** — see
`findings-database.md` §8, independently verified by the orchestrator:

- `outbox_events` / `dead_letter_events` are created only by the stamped-but-unapplied `0001`, and do not exist.
- `POST /auth/forgot-password` and `POST /auth/resend-verification` are **mounted, public** endpoints that
  **500 today**, and they fail *after* persisting a reset/verification token — a partial write on shipped surface.
  This outranks the document/search schema hole in severity.
- The relay itself fails **soft**, not hard: `relay.py:66`'s catch-all `except (PostgresError, Exception)` means
  the app boots and the outbox is silently, permanently dead. The planner's summary implied a boot break; it is
  not one. Boot survives only by accident, through a broad `except` that any tightening pass would remove.

**So the scope of D14's single new migration is: the two outbox tables *and* the target document/chunk schema.**
The outbox half is justified independently (it repairs a 500ing public endpoint) and must be ordered first within
the migration. Nothing else about D14 changes.

**Two ordering constraints this adds, both load-bearing:**

1. Fixing D5.2's `UserIdDep` **without** creating the outbox tables does not repair `POST /documents` — it moves
   the 500 from the dependency layer down to the outbox `INSERT` (`documents/service.py:184`). They land together
   or the repair is illusory.
2. Tightening `relay.py:66` / `:80`'s catch-alls must come **after** the tables exist, or a silent degradation
   becomes a boot failure.

**And it revises change 0's URL-accessor task:** there are three flavours to serve, not two —
SQLAlchemy+asyncpg, a plain libpq DSN (`lifespan.py:124` strips `+asyncpg`, and `relay.py:71` strips it again;
the checkpointer needs the same per `findings-database.md` §5), and Cognee's. One accessor returning one string
cannot serve all three.

## D14.2 — Two open preconditions inherited from the planner, neither guessable

- **F8 — does `pg_textsearch` register an access method literally named `bm25`?** Unresolved and *not*
  resolvable read-only: my probe found no `bm25` AM precisely because the extension is not installed (`§3`). The
  repo assumes the name (`search/repository.py:415` calls `to_bm25query(...)`, and the search migration builds
  `search_chunks_bm25_idx`), but no live install has ever confirmed it. Settling it requires
  `CREATE EXTENSION pg_textsearch` — **DDL on the user's live Timescale Cloud database**, which the orchestrator
  has deliberately not run; every probe this relay made was read-only. Assigned to change 1 as a **step-0
  precondition check against a scratch database**, explicitly not a guess, and flagged to the user as the one
  action that needs their authorization.
- **F11 — who ran `alembic stamp`?** Unknowable from the repo. It does not block any change; it is recorded so
  the next person does not assume the chain was honestly migrated.

## D14.3 — Correction: D14's own recorded proof was unachievable

D14 (and D14.1) recorded change 2's proof as
`alembic upgrade head --sql | grep -c 'CREATE TABLE search_'` = **0**. **That proof cannot pass, and the fault is
in the decision record, not in change 2.** Measured by the change-2 author: the command returns **2** today and
will keep returning 2 forever, because offline `--sql` rendering starts from **base** and therefore traverses
`8a7d9b1c2e3f`, whose body creates those tables — and D14 explicitly forbids editing that revision. Zero is
reachable only via the rejected in-place edit or a drop revision D14 also forbids.

Two further facts make the original proof worse than merely unreachable:

- A from-base offline provisioning **cannot complete at all** regardless, because `9f4a1b7c6d2e` alters the
  phantom `clauses` relation. So the command was never measuring what it claimed to measure.
- Change 2 correctly declined to write a requirement asserting the unreachable outcome, and instead constrained
  two things that *are* checkable: the authoritative create-schema revision creates neither `search_` relation,
  and the source tree outside migration history names neither.

**Re-scoped proof, binding:** the assertion applies to **the authoritative revision's own rendering**, not to a
from-base render of the whole chain. Change 0 must scope its equivalent check the same way. Any task or Proof line
still carrying the from-base form is wrong and must be restated.

**Lesson worth keeping:** a Proof that was never executed is not a proof. This one survived three plan documents
and two decision rounds because it *looked* mechanical. Prefer proofs whose command has actually been run once
against reality before being written down.

## D14.4 — Extensions: trigram search is BUILT, and change 0 creates all four extensions explicitly

The change-2 author resolved B4 by probe: `tsdbadmin` is not superuser but holds `CREATEDB`/`CREATEROLE` and **can**
create both `pg_trgm` 1.6 and `pg_textsearch` 1.3.0. (Executed inside a transaction and rolled back; the database
was left clean. Noted on the record because the orchestrator had deliberately confined itself to read-only probes
and had flagged persistent extension installation as needing the user's authorization — that authorization is
still outstanding for any *persistent* install.)

Consequences:

- **B4 resolves to "build it".** Retrieval ships **three** RRF branches, not two — vector, BM25, and trigram.
  The earlier "no target equivalent, preserve deliberately or drop on the record" framing is closed in favour of
  building, because the extension is installable.
- **Change 0 creates all four extensions explicitly** in the authoritative revision. Do not rely on ambient
  availability: the existing `diskann` index survives today only because `vectorscale` happens to be pre-installed
  on this managed instance, and no revision in the chain ever creates it. That is luck, not design, and it will
  not reproduce on a fresh environment.
- **F8 remains open in its precise form.** "Can the extension be created" is now answered yes. "Does
  `pg_textsearch` register an access method literally named `bm25`" was **not** established — creating the
  extension is not the same as confirming the AM name that `search/repository.py:415`'s `to_bm25query(...)` and the
  `search_chunks_bm25_idx` index depend on. Stays assigned to change 1's step 0 as a scratch-database check.
