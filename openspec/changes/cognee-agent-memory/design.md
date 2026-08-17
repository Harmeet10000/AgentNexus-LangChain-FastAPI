> Change class: **L**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

### The thesis: nothing has ever been ingested into agent memory

`rg -n "cognify|cognee\.add|cognee\.search" src/` returns **zero hits**. `cognify` has no call site anywhere in
`src/`. The only Cognee symbol this repository has ever called is `setup_cognee` (`lifespan.py:206`), which
configures an LLM, a graph store and a relational store — and neither an embedder nor a vector store.
`CogneeStore` is a stub whose five overrides return `None`/`[]`. `store_final_report`, `store_relationships` and
`search_episodic_memory` have zero production call sites; only the package `__init__` re-exports them.

`findings-database.md` §7 closes the last doubt: Cognee's own alembic (`.venv/.../cognee/alembic/`) has **never
run against this database**, and `entities` / `relationships` / `events` / `memory_versions` do not exist in any
form — the complete `public` inventory is 16 billing/audit tables. This closes the plan's **F8**.

Three consequences shape everything below:

1. There is no data to migrate, no dual-run, no backfill and no cutover. The "old" system is a set of
   never-called functions.
2. The work is **configuration correctness plus write topology chosen before any data exists** — which is the
   cheapest moment in the project's life to choose it, and the last moment at which choosing it is free.
3. There is no baseline behaviour to regress against, so "it still works" is not available as evidence. Every
   proof in this change is a configuration read-back, a service test against a faked memory module, or one
   scripted round-trip against real infrastructure.

### Why the boundary needed an ADR rather than a decision here

D2 locked that both memory libraries survive with distinct roles but deliberately left the boundary open. Three
places in the code state a boundary and disagree: `cognee_client.py:12-15` gives the final report to agent memory,
`write_final_report.py:8-13` routes it to **both**, and the superseded change's `design.md:3` says agent memory is
primary. The boundary outlives this change and others will build on it, so it is recorded in `adrs.md` (Status:
**Accepted**), not here. This document consumes it.

### The infrastructure this change depends on and cannot create

`findings-deployment.md` §1–§2, verified by command:

- **There is no worker or beat service in `docker-compose.yml` at all.** Services are exactly `rabbitmq`,
  `timescale`, `caddy`, `ai-service-1`, and `ai-service-1` declares no `command:`, so it runs the API image CMD.
  Nothing consumes the queue.
- **The documented way to start a worker is broken.** `Makefile:52` runs `celery -A celery_config`, and
  `celery_config` does not exist anywhere in the repository.

This change's scheduled consolidation therefore **registers a task and a schedule entry, and states plainly that
no process exists to execute them.** Provisioning a worker and a beat service is an operational dependency of this
change, not a step inside it. Proving registration (`celery inspect registered`, `beat_schedule` length) is
possible today; proving execution is not. Anyone reading the consolidation requirement as "consolidation runs
nightly in production" is reading it wrong until that service exists.

- **The application database is Timescale Cloud**, not the local `timescale` compose service
  (`findings-database.md` §1). `CREATE DATABASE` cannot be assumed, which is why isolation is by **schema**.
- **The graph database has no compose service either**, so it is externally managed and this repository cannot
  install the APOC/GDS plugins that consolidation silently requires (backlog item 140).

### The `cognee-v1-api` validation mechanic — read this before concluding the author failed

**The baseline is 16 passed / 6 failed of 22, and it stays 16/6 through this change.** That is not a shortfall; it
is structural, and the reason is worth stating precisely, because two different things about
`openspec/specs/cognee-v1-api/spec.md` are wrong and only one of them is this change's business.

1. **Why it fails validation:** its sole validator error is `Spec must have a Purpose section` — it opens at
   `### Requirement:` on line 1 with no `## Purpose` header (`findings-openspec-baseline.md` §1, established by
   running `openspec validate cognee-v1-api` per-item; `--all` reports *which* items fail, not *why*).
2. **Why its content is wrong:** it mandates an enrichment call after every write, when the write API already
   performs that enrichment itself (`remember.py:915-944`) — so complying means enriching twice. This is conflict
   **C1**, and it is a *correctness* defect in the requirement text, entirely unrelated to the missing header.

**These two defects are repaired by two different mechanisms, and the delta mechanism reaches only one of them.** A
change's `specs/**` deltas add, modify, remove and rename **requirements**, and archiving applies those requirement
deltas to the deployed spec. **Nothing in the delta mechanism writes a `## Purpose` header.** So:

| Defect | Mechanism | Effect on `openspec validate --all` |
|---|---|---|
| The redundant enrichment mandate (C1) | the `## MODIFIED Requirements` delta in this change | **none** — the counts do not move |
| The missing `## Purpose` header | a **direct one-line edit** to `openspec/specs/cognee-v1-api/spec.md`, housekeeping outside the change flow | would move 16/6 → 17/5, if and when it is done |

**Which of the two this change does, stated explicitly so a later reader does not assume the delta covers both:**
this change authors the `MODIFIED` delta for the redundant-enrichment defect **only**. It does **not** hand-edit
`openspec/specs/cognee-v1-api/`. The Purpose-header repair is carried as its own separately-tracked one-line file
edit in `tasks.md`, never folded into a delta that structurally cannot carry it — and if it is descoped, the delta
is unaffected and still correct.

**Corrects `plan-change4.md`:** the plan's step 2 claimed the C1 delta moves the baseline to **17 passed / 5
failed**, and the plan's ordering constraint 1 claimed the header repair must land *before* the delta because an
unparseable spec has no blocks to match. Both are wrong. The counts do not move on the delta, and the spec's
requirement blocks parse today — the validator's only complaint is the missing Purpose — so the `MODIFIED` header
match works against the file as it stands. Anywhere else in this change's artifacts that implies 17/5, read 16/6.

The acceptance criterion is therefore **"no new failures beyond the existing 6"**, never "validate passes", and it
is **6**, not 5. Four of the six are the same missing-Purpose error (`cognee-v1-api`, `noqa-documentation`,
`pattern-matching-standard`, `typed-exception-handling`); a fifth (`transactional-outbox`) is missing normative
keywords; the sixth (`change/mintlify-documentation`) is a change, not a spec. None is in scope here, and D12 was
right to accept them as a baseline rather than promise to fix them.

**One trap this change must not fall into.** The cause of those four stubs is visible in the specs that *do* have a
Purpose: `transactional-outbox`, `outbox-helper-extraction` and `session-required` all read *"TBD - created by
archiving change &lt;x&gt;. Update Purpose after archive."* The archive flow stubs Purpose and nobody returns to it.
Since this change is superseded-and-archived through that same flow, the new capability it authors carries a
**real, written `## Purpose`** — not a stub — so archiving does not add a seventh failure. `cognee-v1-api`'s delta
deliberately carries **no** `## Purpose`: on a delta for an existing capability it is ignored, and including one
would falsely imply the delta repairs the header.

## Goals / Non-Goals

**Goals:**

- Settle the D2 role boundary as an accepted ADR, so the three disagreeing code sites stop disagreeing.
- Make agent memory correctly configured for the first time: embedder pinned to the repository's model and
  dimension, vector store in the managed database, access-control state explicit, connection authenticated.
- Make the subsystem observable before it is used, on **both** health surfaces.
- Build one real write seam and one real read seam onto the existing memory-persist node.
- Honour Trap3 in the write topology, before the first rebuild call site exists.
- Retire the deferred reference files after harvesting what has no other implementation.

**Non-Goals** — every entry is a **recorded gap**, carried here so it is surfaced rather than silently omitted
(D10, D13). NG1–NG5 are the honest reading of D10: item 155's word "entirely" is honoured for **reconciliation
removal**, never for **capability parity**. The repository's own document already conceded this —
`openspec/changes/cognee-saul-memory-migration/proposal.md:20-21`: *"Cognee v1.1 has no built-in
curation/decay/dedup"*, with both maintenance capabilities marked deferred.

| # | Capability not provided | Sole prior implementation (deleted in change 0) | Replacement anywhere in the repo? |
|---|---|---|---|
| **NG1** | **Scored memory decay** (item 170, **DROPPED** per D10) — exponential decay over age × access count × confidence | `src/tasks/memory_decay_reconciliation_tasks.py:51` `_compute_decay`; driver `:64`; entry `:180` | **None.** The memory library's `forget` is deletion by identifier, not a decay score. The knowledge graph has no decay. This was the repository's only decay formula. |
| **NG2** | **Near-duplicate detection over memory entities** — self-join with LLM adjudication | `reconciliation/nodes.py:62,94-95,135`; prompt `prompts.py:23` | **Partial, wrong axis.** The knowledge graph dedups *its own* document-axis nodes natively. Nothing dedups the agent-run axis. |
| **NG3** | **Edge-preserving merge** — merge duplicates while re-pointing relationship rows | `reconciliation/nodes.py:205`; decision model `reconciliation/state.py:30` | **None on the memory axis.** The knowledge graph's edge resolution applies only to its own edges; agent memory has no edge model to preserve. |
| **NG4** | **Memory-entity version history** — append-only version rows per entity change | `reconciliation/nodes.py:274`; table model `src/database/schemas/memory_schema.py` | **No.** `billing/models/audit.py:48` is live and migrated but scoped to billing. |
| **NG5** | **Per-user / fleet-wide reconciliation orchestration** | `memory_decay_reconciliation_tasks.py:186,198` | Shape is re-derivable from `src/tasks/billing_tasks.py:71,253` — and this change's consolidation job is where it is re-derived. Not otherwise preserved. |

**The honest framing of NG1–NG5, stated and not softened:** none of these was ever observable behaviour. There is
no `@celery_app.task` decorator in `memory_decay_reconciliation_tasks.py`; the module is absent from
`connections/celery.py:191-196` `include` (4 entries: auth_email, example, search, billing); `beat_schedule`
(`:259-276`) holds exactly 4 billing entries; and no migration ever created the four tables the code assumed
(`findings-database.md` §7 — none of them exist, in any form). **What is lost is design work, not a regression a
user could notice.** That is what makes the deletion acceptable — and it is also exactly why the deletion produces
no test signal.

**After this change, agent memory grows without decay, curation, or dedup.** This is D10, mandatory and
non-negotiable. It is not mitigated here. `adrs.md` § Consequences names it as a known consequence of the accepted
boundary. The single safeguard added is a size/count metric on the consolidation job, so growth becomes
*observable* before it becomes a problem — no alarm exists otherwise.

Remaining Non-Goals:

- **NG6 — multi-user access control inside the memory library.** Unavailable on this repository's graph backend:
  with the graph provider set to neo4j and the dataset handler left at its default, the default branch reaches
  `multi_user_support_possible()` and **raises `EnvironmentError`** on a handler/provider mismatch. Rejected as
  *unavailable*, not as undesirable. The setting is therefore written explicitly so startup is deterministic
  rather than raising on first write.
- **NG7 — graph-completion / router threshold tuning** (the deferred half of item 140, `todo.md:253`). The
  precondition half of 140 is **in** this change, twice: as a documented operational precondition and as a health
  probe sub-field. The router knobs exist on the recall API and are left at their defaults, deliberately untuned.
- **NG8 — `redisvl` / `langcache` adoption** (the deferred half of item 179, `todo.md:271`). The narrow half is
  answered, not deferred: **agent memory needs no Redis of its own** — the memory library's store surface is
  relational, vector and graph, with no Redis anywhere in its database configuration layer. The adoption research
  for the *application's* caching is deferred and unowned.
- **NG9 — a LangGraph store backed by agent memory.** `CogneeStore` is **deleted, not implemented**. Implementing
  its `search` override would put document semantic retrieval behind a LangGraph store interface, creating a
  fourth retrieval path against D5.1 — the identical reason `app.state.vector_store` was already dropped.
  LangGraph checkpoint and store duties stay on Postgres.
- **NG10 — making the agent graph reachable.** This change makes memory persistence *correct*, never *reached*.
  See § D17 below; this is permanent, not a temporary state.
- **NG11 — exposing a deeper memory-retrieval tool to individual reasoning nodes.** Harvested out of the
  superseded change and handed to **change 3**, because tool exposure is the registry's concern (D6.1) and this
  change must not add a second tool-registration path. The *node-level* half (deep retrieval only for risk
  analysis and compliance) is specified here as a behaviour of the prefetch step; the *tool-registration* half is
  not.
- **NG12 — the alembic head merge and target-schema migration.** Owned by change 0 (D14). This change consumes
  `env.py` being sane before anyone runs `--autogenerate`; it does not perform the merge.
- **NG13 — the single connection-string accessor.** Owned by change 0. This change consumes it: its requirement is
  only that the memory subsystem *receives a usable connection*, not that it repairs the URL itself.
- **Not owned at all:** item 159 (RAGFlow / OpenRAG evaluation) is deferred per D13 with no owning change, and is
  explicitly not claimed here — D5.1 already commits document retrieval to the existing `pg_textsearch` path.

## Decisions

### 1. The boundary is split by each library's own partition key — an API fact, not taste

Recorded in full in `adrs.md` (Status: **Accepted**); summarised here because everything else depends on it. Typed
agent-memory entries **cannot be written without a conversation identity** — `remember.py:274-276` raises
`session_id is required for typed memory entries`. Every knowledge-graph write in this repository is
`group_id=document_id` (`documents/service.py:544`, `graphiti_verifier.py:56`, `ingestion_kb/nodes.py:384,397`).
So each library's partition key already decides which axis it can serve. **Agent memory owns the agent-run/thread
axis; the knowledge graph owns the document/entity axis and all bitemporal validity. The final report goes to agent
memory only.**

- *Alternatives considered:* dual-write the report to both (the status quo of `write_final_report.py`) — rejected on
  cost and on duplicate-extractor damage; agent memory primary with the knowledge graph retired — rejected,
  contradicts D2 and loses bitemporal invalidation outright; knowledge graph primary with agent memory deleted —
  rejected, contradicts D2. Full pros/cons table in `adrs.md` § Rationale / Alternatives.

### 2. Trap3 is honoured with the library's own deferral, not with hand-rolled batching

Trap3 (`todo.md:485`) says a rebuild is full, so batch the writes and defer the rebuild to a nightly job. **The
intent is right; the mechanism is already in the library.** Verified: `remember()` in permanent mode is
`add()` → `cognify(run_in_background=False)` → `improve()`, all awaited (`remember.py:915-944`, with
self-improvement defaulting to `True` at `:610`). The repository's existing `store_final_report`
(`cognee_client.py:150-151`) then awaits `improve()` a **second** time — one full graph rebuild plus **two**
enrichment passes, synchronously, per approved report, inside a graph node. Conversation-scoped `remember()`
appends to the conversation cache and never touches the rebuild (`remember.py:895-900`); scheduled `improve(dataset,
session_ids=[…])` is the documented bridge into the permanent graph.

**Because no rebuild call site exists yet, this constrains the design before it is written — the cheapest possible
moment to honour it.** Written the other way, it would be a rewrite later.

- *Alternatives considered:* hand-rolled batching over `add()` with a nightly rebuild — duplicates the library's own
  session manager for no gain; permanent-mode writes on the request path — the cost above, inline in a graph node;
  self-improvement left enabled with a conversation identity — fires the bridge as a detached
  `asyncio.create_task` inside the caller's event loop, whose failure is logged "non-fatal" and whose lifetime the
  application does not own. An explicit schedule is observable; a detached task is not.

### 3. Item 152, defect one — the embedding model is pinned, and the dimension is asserted equal

`rg -i cognee src/app/config/settings.py` returns **no hits**: there is no memory configuration surface at all
today, and `setup_cognee` never calls the embedding configuration. The default model is
`openai/text-embedding-3-large`, which resolves to **3072** against the repository's **768**
(`settings.py:212` `EMBEDDING_DIMENSION: int = Field(default=768, gt=0)`) — and it would call a third-party
provider with no key configured. Fix: pin the model to the repository's configured embedder and derive the
dimension from `EMBEDDING_DIMENSION`, with a startup assertion that the two are equal.

- *Correction carried from the plan, so it is not re-inherited:* in the installed version, `embedding_dimensions`
  itself defaults to `None` and is resolved in `model_post_init`; the source comment at
  `vector/embeddings/config.py:73-77` records that the 3072 hard-default was removed *because* it silently broke
  non-OpenAI embedders. The mismatch is still real — because the default **model** still resolves to 3072 — but
  **setting the model is the fix** and setting the dimension is belt-and-braces.
- *Alternatives considered:* set the dimension only — insufficient, the wrong provider would still be called; leave
  it and normalise dimensions downstream — a data-shape lie that would corrupt every stored vector.

### 4. Item 152, defect two — the vector store is configured explicitly, isolated by Postgres **schema**

`set_vector_db_config` is never called, so the vector store defaults to `lancedb` (`vector/config.py:30`): memory
embeddings go to **local files**, invisible to the application's database and lost on every container replacement.
Fix: configure the managed relational database explicitly. Isolation is by **schema**, not by database, because the
application database is Timescale Cloud (`findings-database.md` §1) and `CREATE DATABASE` cannot be assumed there.
Belt-and-braces: add an object/name filter to `src/alembic/env.py`, which today sets
`target_metadata = Base.metadata` with **no** filter (`env.py:23-30`), so the next `--autogenerate` would otherwise
emit `op.drop_table(...)` for every third-party memory table.

- *Alternatives considered:* a Cognee-dedicated **database** — the original ADR choice, withdrawn because the
  managed instance almost certainly forbids it; leave the default local files — silently loses all memory on
  restart, and is the current behaviour; rely on `include_schemas` defaulting to `False` alone — true today but one
  flag flip away from data loss, hence the explicit filter as well.
- *Documented fallback, recorded as a decision rather than left to a scramble:* if the precondition check finds the
  application role cannot create a schema or the required extension, use a local-file vector store **on a mounted
  persistent volume**, for memory recall only — never for document retrieval, which D5.1 keeps on `pg_textsearch`.

### 5. The memory subsystem is handed a usable connection; it does not repair one

`cognee_client.py:111` reads `settings.POSTGRES_URL` **raw**. That value carries **no password**
(`findings-database.md` §2) and bypasses `connections/postgres.py:30-71` `get_database_url()`, which is the one
place that injects the credential, rewrites the scheme and strips the transport parameters the driver rejects.
Change 0 owns the fix and its shape is settled there: **the repair belongs in a single accessor so no caller can
obtain an unusable URL**, not in three call sites. This change's requirement is narrower and consumes that: the
memory subsystem receives a connection that authenticates on first use.

- *Alternatives considered:* repair the URL inside the memory client — a third copy of the same repair, and the
  reason the defect exists; rely on the ambient `PGPASSWORD` / `POSTGRES_PASSWORD` environment side effect — works
  by accident today and breaks in any process that does not inherit it.

### 6. Access control is disabled **explicitly**, because leaving it unset raises

This is not tidiness. With the graph provider set to neo4j and the dataset handler at its default, leaving the
environment variable unset takes the default branch at `context_global_variables.py:88-92` into
`multi_user_support_possible()`, which **raises `EnvironmentError`** on a handler/provider mismatch. Setting it
explicitly makes startup deterministic. Tenant isolation moves entirely to the application layer.

- *Correction to an earlier claim:* the list of graph databases with multi-user support feeds
  `is_multi_user_support_possible()`, **not** the gate on the write path; the gate reads the supported-handler
  dictionary, which *does* contain a neo4j entry (Aura-specific, untested here). The conclusion is unchanged
  (access control off) but the mechanism is worse than "silently disabled" — it raises.
- *Alternatives considered:* enable access control for defence in depth — unavailable on this backend (NG6); leave
  it unset — a raise on first write.

### 7. The observability seam is `check_cognee` only — `check_graphiti` already exists

**Correction to `dispositions.md` 198.2, which said the knowledge graph is unprobed.** `check_graphiti` **already
exists** at `features/health/health_check.py:83-90` and is already registered in the probe list. Disposition 198.2
was narrowed for exactly this reason, and this change **does not claim it**. What is missing is `check_cognee` —
and separately, the *second* health surface (`src/app/features/health/service.py`) probes
postgres/redis/mongo/neo4j/celery/memory/disk and carries **neither** subsystem, so the probe is added in both
places or the two surfaces disagree.

The probe must distinguish three states, because a boolean cannot express this subsystem's failure mode: *degraded*
when the configuration is absent, *fail* when the configuration is present but the stores are unreachable, *ok*
otherwise. It also reports the graph-procedure precondition as a **named sub-field** rather than failing the whole
check — that precondition is the only way item 140's silent failure is ever observed.

- *Alternatives considered:* a boolean probe — cannot distinguish "not configured" from "broken", which is the
  distinction that matters here; failing the whole health check on a missing graph procedure — would take the API
  down for a background-job precondition; no probe at all — `lifespan.py:220-223` already proves this repository's
  habit of degrading in silence, which is precisely why the probe is a *requirement* of this change.

### 8. Configuration lands before code; observability lands before behaviour

The ordering is load-bearing, not stylistic. Configuration first, because reversing it means the first write runs
against a local-file vector store, or raises from the access-control gate. Observability second, because both of
this subsystem's headline failure modes are silent by construction — the rebuild fails without raising when the
graph plugins are absent, and this repository already degrades quietly. Deletions **last**, because the two
reference files are the only existing description of how memory writes were meant to work.

- *Alternatives considered:* build the write seam first and configure afterwards — writes land in the wrong store,
  and the first observation of that is missing data; delete the reference files first (change 0's default posture) —
  explicitly carved out by D4 for this reason.

### 9. Two capabilities, not one — and the deployed spec is corrected by delta, not by hand-edit

`cognee-v1-api` is an **API-surface** contract (which memory calls are made, in which mode). `saul-agent-memory` is
a **behavioural** contract (what is remembered, at what scope, what happens on failure, when consolidation runs).
Collapsing them would put behaviour into an API-surface spec. Checked against all 20 existing capabilities under
`openspec/specs/`: `cognee-v1-api` is the only adjacent one; `session-required` is the outbox relay's session
parameter and `settings-validation` is the production secret registry, so neither fits.

The correction to the deployed spec's redundant-enrichment requirement (conflict **C1**) is expressed as a
`## MODIFIED Requirements` delta **inside this change**. See § Context for what that does and does not repair — the
baseline stays **16 passed / 6 failed** through this change, by structure rather than by omission.

- *Alternatives considered:* one combined capability — puts behaviour in an API spec; comply with the deployed spec
  as written — means enriching twice per write, which is the defect; hand-edit
  `openspec/specs/cognee-v1-api/spec.md`'s requirement text now to make validation green — bypasses the delta
  mechanism the archive step depends on, and would erase the record of *why* the requirement changed.

### 10. The read seam is built, and it is **speculative** (D17)

**Labelled explicitly, as D17 requires.** `build_saul_graph` (`agent_saul/graph.py:86`) has no caller, and D17
settled that the unwired agent graph was **deliberate** and **stays commented**. So the read seam added by this
change cannot be exercised by running the product — not temporarily, but permanently under the current decision.
It is built anyway, because without a read seam this change writes memory nobody reads, and because the logic
largely already exists in the file being deleted (`memory_pipeline.py:213,220` already branches on exactly the two
task names; `:258-260` is already the fail-open pattern), so relocating it now is cheaper than rebuilding it later.

**What follows from labelling it speculative:** its proofs are import-level, type-level and unit-level only. Node
reachability is not proven and is not claimed (NG10). A wiring defect between the node and the service would not be
caught until change 3 wires the graph. Nothing in this change may make re-enabling that wiring harder.

- *Alternatives considered:* ship write-only and add the read seam in change 3 — leaves this change with no
  consumer for its own data and pushes the harvest of `memory_pipeline.py` into a change that has no reason to
  touch it; wire the agent graph here to obtain a proof — forbidden by D17.

### 11. The superseded change is **archived, not deleted**

`openspec/changes/cognee-saul-memory-migration` is superseded: it declares `schema: spec-driven` while
`openspec/config.yaml:1` says `spec-gated`; it has no `review.md`, so under the current schema its `tasks.md` is
illegitimate by construction; it is **0/15 tasks after 23 days**; and its central premise is now wrong in one
specific way — `proposal.md:20-21` *defers* a replacement for reconciliation, whereas item 155 removes
reconciliation outright.

It is nonetheless directionally right, so **its content is harvested, not discarded**: its final-report-write
capability is harvested in full (amended so the write is conversation-scoped and an unapproved run writes
**nothing** — the old design's implicit low-trust write of unapproved reports is dropped), and three of the four
requirements of its prefetch capability are harvested; the fourth (tool exposure) goes to change 3 as NG11.

**The directory must be archived, never deleted.** Its `proposal.md:20-21` is the **primary citation** for D10's
recorded gap — the repository's own admission that the memory library has no curation, decay or dedup. That
sentence is load-bearing evidence and must remain quotable. Archiving is also what applies the `YYYY-MM-DD-`
prefix (D12). **This change does not move it** — that is an implementation task, with a `superseded-by` line added
to the archived `.openspec.yaml` so the link is discoverable from both ends.

- *Alternatives considered:* extend it in place — requires migrating it to `spec-gated` and re-verifying every
  delta, for a change with no in-flight work, and would leave a document whose Why no longer matches its What;
  delete it — destroys the provenance of the D10 citation.
- *Open mechanic, resolved at implementation:* whether `openspec archive` accepts a 0/15-task change. If it
  refuses, move the directory by hand and record that in `review.md` rather than ticking 15 tasks that were never
  done.

### 12. Harvest before delete (D4 carve-out)

D4 kept `write_final_report.py` and `memory_pipeline.py` out of change 0 precisely because they are *"the only
existing reference for how Cognee writes are meant to work"*, and this change rebuilds exactly that. Both are dead
(`graphify affected` on their entry points returns only the package `__init__` re-export), so deletion is safe —
but two helpers inside them have **no other implementation in the repository** and must move first: the
tool-message filter (`memory_pipeline.py:129-157`, which strips tool messages *and* pure-tool-call assistant
messages and substitutes one compact summary) and the structured context-prefix builder (`:160-201`). Both belong
in `shared/langchain_layer/messages.py`. The message-trimming step (`:109-116`) is **not** harvested — it is a
duplicate of `messages.py:40-52`, same counter, same strategy, so deleting it is pure subtraction.

Deleted with them: the knowledge-graph final-report episode writer (`rag/graphiti/client.py:311-350`), whose only
caller was `write_final_report.py:110` — this is the boundary decision expressed in code — and its result model.
`store_relationships` is **retired, not ported**: it pushes a relationship graph as **text** into agent memory,
which is a document-axis concern inside the run-axis owner, and relationships already have a knowledge-graph
writer.

**Paired edits are mandatory:** `rag/graphiti/__init__.py:47,59` and `memory/__init__.py:3-9,23-39` re-export the
deleted symbols. Missing either yields `ImportError` **at boot**, not at test time — the eager-import class of
failure D6.1 warns about, which no unit test can see.

- *Alternatives considered:* delete without harvesting — loses two unique helpers; port `store_relationships` —
  violates the accepted boundary; keep `CogneeStore` and implement it — NG9.

### 13. One failure idiom, and one dataset-naming path

Three failure idioms currently coexist in one layer: re-raise (`cognee_client.py:159`), swallow-to-empty-list
(`:257`), and collect-error-strings (`write_final_report.py:156-161`). The service settles on one per
`RESULT-PATTERN.md` / `EXCEPTION-RULES.md`, keeping `e.add_note()` before re-raise, already the house style here
(`cognee_client.py:251`). Separately, the partition name is built by bare interpolation at three sites
(`cognee_client.py:140,189,238`); with access control unavailable (NG6) that name is the **only** tenant boundary,
so it becomes **one validated helper**.

Also fixed while building: recall returns a discriminated union of models, and the existing
`[dict(r) for r in results]` (`cognee_client.py:259`) is a shallow conversion that leaves nested models as objects
— not serialisable. Use a full model dump and **preserve the origin field**, which is how a caller distinguishes a
conversation-cache hit from a permanent-graph hit.

- *Alternatives considered:* keep the per-function idioms — three behaviours for one subsystem, and the reason the
  read path silently returns `[]` today; keep the shallow conversion — passes a type check and fails at
  serialisation time.

### 14. Item 179's narrow half, answered rather than deferred

**Agent memory needs no Redis of its own.** The memory library's store surface in the installed version is
relational + vector + graph; `rg -il "redis"` across its database infrastructure package returns nothing in its
configuration surface. So no Redis instance, connection pool or configuration key is added by this change. The
`redisvl` / `langcache` question for the *application's* caching is untouched research (NG8).

- *Alternatives considered:* provision a Redis for memory pre-emptively — an unused dependency; leave the question
  open — it is answerable by inspection in minutes and blocks a settings decision.

## Risks / Trade-offs

- **[Memory grows without decay, curation, or dedup]** → **Not mitigated — accepted and recorded**, per D10 and
  NG1–NG5. The repository's own document concedes it
  (`cognee-saul-memory-migration/proposal.md:20-21`). The one cheap safeguard added: a size/count metric on the
  consolidation job, so growth is *observable* before it is a problem, since no alarm exists otherwise. Item 155's
  "entirely" is honoured for **reconciliation removal**, not for **capability parity**.
- **[A third-party library performs schema DDL inside the production managed database at first write]** → Its own
  alembic runs lazily on first use, against Timescale Cloud, where the application's live data sits. Isolate to a
  dedicated **schema**, add an object/name filter to `src/alembic/env.py`, verify `CREATE SCHEMA` privilege in the
  precondition audit **before any code lands**, and run the manual round-trip against a **non-production** instance
  first.
- **[The managed instance may refuse the vector extension or the schema]** → Managed instances restrict extensions.
  Checked in the precondition audit; the fallback is decided in advance (Decision 4): a local-file vector store on a
  **mounted persistent volume**, for memory recall only, never for document retrieval.
- **[The permanent-graph rebuild fails silently without the required graph plugins]** → Item 140, and there is **no
  graph service in `docker-compose.yml`**, so this repository cannot install them. Mitigated three ways: a
  documented precondition, a health-probe sub-field, and the one round-trip that observes the
  conversation-cache-to-permanent-graph transition — the **only** check that detects a silent failure at all.
- **[Scheduled consolidation has nothing to execute it]** → There is no worker or beat service in the deployment,
  and `Makefile:52` names a nonexistent module (`findings-deployment.md` §1–§2). Mitigated only by honesty: this
  change registers the task and the schedule entry and proves **registration**, and provisioning the worker/beat
  services is stated as an operational dependency. Do not read the consolidation requirement as "consolidation runs
  nightly" until that service exists.
- **[The partition name is the sole tenant boundary]** → With access control unavailable (NG6), a bug in partition
  naming is a cross-tenant memory leak with no second line of defence. Mitigated by one validated helper replacing
  three interpolations, plus a test asserting two tenants never collide.
- **[This change cannot be proven end-to-end]** → `build_saul_graph` has no caller and stays that way (D17), so
  memory persistence never executes. Mitigated by service-level proofs plus one manual round-trip, and stated as
  NG10. **The residual risk is real:** a wiring defect between node and service surfaces only when change 3 wires
  the graph.
- **[Deletions produce no test signal]** → The reconciliation and memory modules have **zero** test coverage, so a
  green suite proves nothing about a deletion. The substitute evidence, in descending strength: `graphify affected`
  returns no nodes; `rg` over `src/` and `tests/` returns only definitions and re-exports; **the application still
  imports** (`uv run python -c "import app.main"`) — the load-bearing one, because `ImportError` at boot is the
  failure mode unit tests structurally cannot see; ruff and `ty` counts do not increase; `openspec validate --all`
  failures do not increase.
- **[A green pytest run exits 1]** → `--cov-fail-under=80` against 18.38% coverage. Every proof compares the
  **summary line** (`N passed`), never `$?`. Baseline **55 passed**; ruff **123** post-D11; `ty` **46**. A CI gate
  wired to the exit code would fail every step.
- **[Two graph datasets on one graph instance]** → The permanent memory graph and the document entity graph share an
  instance and cannot reference each other's nodes; joining a run to the clauses it analysed is an application-level
  join. Accepted (`adrs.md` § Consequences). Operational hazard: the memory library's prune operation must **never**
  be called against the shared instance — worth an explicit grep guard.
- **[The user-level partition of the knowledge graph becomes empty]** → Two readers filter on it
  (`graphiti_verifier.py:70`, `documents/service.py:753`) and it was populated only by the retired final-report
  write. Left in place deliberately as the natural home for future user-level facts, and recorded in `adrs.md` so it
  is not later mistaken for a bug — but it *is* now a filter that matches nothing.

## Migration Plan

**There is no data migration.** Nothing has ever been written to agent memory (§ Context), so there is no backfill,
no dual-run and no cutover. What follows is an ordering, and the order is the design.

1. **Precondition audit, no code.** Graph plugins (APOC + GDS), managed-database DDL capability (`CREATE SCHEMA` and
   the vector extension), whether the memory library wants a Redis (answered: no), and the orphan-table question
   (**closed** — `findings-database.md` §7: none of the four tables exist). Three of these answers can invalidate
   later design; each has a pre-decided branch (Decision 4).
2. **Apply the `cognee-v1-api` requirement delta** (the redundant-enrichment correction, C1). It has **no
   ordering dependency** on the spec's missing `## Purpose` header: the requirement blocks parse today, so the
   `MODIFIED` header match works against the file as it stands. Separately and optionally, a **one-line direct
   edit** inserting `## Purpose` into `openspec/specs/cognee-v1-api/spec.md` is the only thing that would move
   validation from 16/6 to 17/5 — it is tracked as its own task and is not part of the delta. See § Context.
3. **Configuration** — the `COGNEE_*` settings surface (which does not exist today), then the embedder and vector
   store on the startup helper, plus the explicit access-control setting written into the process environment
   **before** the first memory configuration call, and a typed configuration result in place of the current
   `dict[str, Any]` so the probe has something to assert on.
4. **Observability** — `check_cognee` on both health surfaces, with the graph-procedure precondition as a named
   sub-field. This is the acceptance test for step 1's first finding.
5. **The memory service** — the repository's first memory call site, with four operations: conversation-scoped
   report write, typed trace/QA/feedback writes, recall, and consolidation. The write shape is where Trap3 is
   honoured; the machine-checkable form is *"the rebuild is never called from a request-path method, and enrichment
   is called only by consolidation"*.
6. **Retarget the memory-persist node** onto the service, gated on human approval, keeping the existing fail-open
   shape — a memory failure must not fail a completed legal analysis.
7. **Scheduled consolidation** — a real task decorator, module registered in the task `include` list, one schedule
   entry. **Depends on change 0** having removed the reconciliation re-exports from `src/tasks/__init__.py:6-9,18-20`;
   until then every worker dies at import and registration cannot be proven. Name it distinctly from the existing
   billing reconciliation entry, which shares only the word.
8. **The read seam** — the prefetch node after clarification, relocating the fail-open and task-branching logic out
   of the file step 9 deletes. **Speculative** (Decision 10).
9. **Harvest, then delete** — the two unique helpers move first; then the two reference files, the knowledge-graph
   final-report writer, the stub LangGraph store and the three legacy module-level functions go, with their
   re-exports edited in the same commit.
10. **Dispose of the superseded change** — archive (never delete), with `superseded-by` recorded on both ends.

**Rollback shape:** steps 3–4 are pure additions and revert cleanly. Step 5 adds a service with no callers until
step 6. Steps 9–10 are the only irreversible ones, and they come last for exactly that reason.

## Open Questions

All of the following are **precondition checks with both branches already decided**, not unresolved design. None
would change the specs. Each names what closes it.

- **Are APOC and GDS present on the target graph database?** Cannot be answered from the repository — there is no
  graph service in the compose stack. *Closes with:* `SHOW PROCEDURES` against the real instance, in step 1. *If
  absent:* consolidation is inoperable, this change ships write-only, and that is recorded as a blocking risk rather
  than worked around.
- **May the application role create a schema and the vector extension on the managed instance?** *Closes with:* the
  step-1 `psql` probe. *If not:* the local-file-on-a-volume fallback in Decision 4. **If that fallback is also
  unacceptable to the operator, this becomes a user decision, not an author's guess.**
- **Does the memory library's vector provider honour a non-default Postgres schema?** The isolation strategy prefers
  it; the `env.py` filter carries the goal alone if it does not. *Closes with:* reading the provider's schema
  handling, then observing table placement after the round-trip.
- **Is consolidation incremental on an already-consolidated dataset?** Bears directly on the scheduled job's cost.
  *Closes with:* two timed consecutive runs on a fixed dataset.
- **Have the memory write, consolidate and recall operations *ever* succeeded against this graph and database?** No
  call sites, no tests, no dataset artifact. *Closes with:* the manual round-trip, whose definition of "working" is
  an observable transition — a recall scoped to the conversation returns a conversation-cache hit; after
  consolidation, a recall *without* the conversation scope returns a permanent-graph hit — rather than parity with
  code that never ran.
- **Does `openspec archive` accept a 0/15-task change?** Decides whether step 10 is a CLI call or a manual move.
  *Closes with:* one dry run.

**Deliberately left open, not fog:** `redisvl` / `langcache` adoption for the application's caching (NG8).
