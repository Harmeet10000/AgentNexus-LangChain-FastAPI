> Change class: **L**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

### The thesis: nothing has ever been ingested into agent memory

`rg -n "cognify|cognee\.add|cognee\.search" src/` returns **zero hits**. `cognify` has no call site anywhere in
`src/`. The only Cognee symbol this repository has ever called is `setup_cognee` (`lifespan.py:206`), which
configures an LLM, a graph store and a relational store — and neither an embedder nor a vector store.
`CogneeStore` is a stub whose five overrides return `None`/`[]`. `store_final_report`, `store_relationships` and
`search_episodic_memory` have **no live call sites**. `search_episodic_memory` is genuinely uncalled from anywhere.
The other two are called — at `write_final_report.py:122,146`, through a structural `CogneeService` interface declared
at `:41-50` — from a module that is itself dead. That is dead code calling dead code, and this change deletes both
sides in one step, so nothing follows from it for scope. It follows for **ordering**: the edge is duck-typed through
an interface declaration, so `graphify affected` on the memory functions does not surface it, and a deletion that
removes the callee while leaving the caller is a break no graph query would have warned about.

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

**The baseline is 21 passed / 6 failed of 27, and the failure count stays at 6 through this change.** That is not a
shortfall; it
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
| The missing `## Purpose` header | a **direct one-line edit** to `openspec/specs/cognee-v1-api/spec.md`, housekeeping outside the change flow | would move the failure count 6 → 5, if and when it is done |

**Which of the two this change does, stated explicitly so a later reader does not assume the delta covers both:**
this change authors the `MODIFIED` delta for the redundant-enrichment defect **only**. It does **not** hand-edit
`openspec/specs/cognee-v1-api/`. The Purpose-header repair is carried as its own separately-tracked one-line file
edit in `tasks.md`, never folded into a delta that structurally cannot carry it — and if it is descoped, the delta
is unaffected and still correct.

**Corrects `plan-change4.md`:** the plan's step 2 claimed the C1 delta moves the baseline to **17 passed / 5
failed**, and the plan's ordering constraint 1 claimed the header repair must land *before* the delta because an
unparseable spec has no blocks to match. Both are wrong. The counts do not move on the delta, and the spec's
requirement blocks parse today — the validator's only complaint is the missing Purpose — so the `MODIFIED` header
match works against the file as it stands.

**Corrects this document's own earlier arithmetic (2026-08-18).** Every earlier draft of this section said
*"16 passed / 6 failed of 22"*. Measured today, `openspec validate --all` reports **21 passed / 6 failed of 27** —
the item total grew by five as the sibling changes of this relay were authored, and each new change passes, so the
**pass count is not an invariant and must never be used as an acceptance number.** The invariant is the **failure
count: 6**, and the four missing-Purpose stubs, the missing-keyword spec and the one failing change that make it up
are identical to those enumerated below. Anywhere in this change's artifacts that a pass count appears, read the
failure count instead.

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

### Coordination points with other changes — open obligations, not completed handoffs

Two things this change needs are owned elsewhere. Neither is a requirement here, and neither is finished; both are
recorded so that a later reader can see the seam rather than discover it.

| # | What is needed | Owner | State after change 4 lands | Consequence if the owner never lands it |
|---|---|---|---|---|
| **C-A** | **A registry binding for deeper memory retrieval.** The behaviour is specified here (`saul-agent-memory`: *Deeper memory retrieval is available only to designated reasoning roles*). What is missing is the tool-name binding and the role assignment: change 3's `agent-tool-registry` must add the memory-retrieval tool to the tool set of exactly the risk-analysis and compliance roles and to no other role — its *"every agent role receives the tools assigned to it"* requirement (`agent-tool-registry/spec.md:74-92`) currently enumerates precedent/statute and knowledge-graph tools only — and change 3's `agent-tool-contract` must carry its refusal path, because the operation must refuse rather than return an empty result set (that contract already has *"Unavailability SHALL never be reported as absence"*, which is exactly the shape needed). | **change 3** (`agent-tools-unification`) — **not yet written into its artifacts**, verified 2026-08-18 | The service operation exists with the role restriction specified. No tool is registered. No reasoning node can invoke it. | The constraint remains specified and unexposed. Nothing regresses; the capability simply is not reachable from a node. This is the *stated* cost of not adding a second tool-registration path here (D6.1). |
| **C-B** | **A worker and a beat service to execute scheduled consolidation.** There is no worker and no beat service in `docker-compose.yml` at all, and `Makefile:52` starts one from a `celery_config` module that does not exist. | **change 1** — dispositioned there (`dispositions.md` 198.4), deliberately **not** duplicated as a requirement here | The consolidation task is registered and its beat entry is present. **The entry is inert: no process exists to execute it.** | Memory accumulates in conversation-scoped caches and is never consolidated into the permanent graph. Recall keeps working against the conversation cache; the permanent half of the boundary stays theoretical. See NG14. |

**C-B also has a hard ordering dependency on change 0**, which is separate from the runtime gap: registration cannot
even be *proven* until change 0 removes the reconciliation re-exports from `src/tasks/__init__.py:6-9,18-20`, because
until then any worker importing the task package dies at import.

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
- **NG11 — *registering* a deeper memory-retrieval tool in the agent tool registry.** The **behaviour** is owned here
  (`saul-agent-memory`: *Deeper memory retrieval is available only to designated reasoning roles*) — which roles may
  invoke it, what it returns, what it must refuse. What this change does **not** do is add a second
  tool-registration path: binding that operation to a tool name and handing it to exactly the risk-analysis and
  compliance roles is the registry's concern (D6.1). That is a **coordination point with change 3, not a completed
  handoff** — see § Context. Until change 3 exposes it, the operation exists as a service method with the role
  restriction specified and no tool binding.
- **NG12 — the alembic head merge and target-schema migration.** Owned by change 0 (D14). This change consumes
  `env.py` being sane before anyone runs `--autogenerate`; it does not perform the merge.
- **NG13 — the single connection-string accessor.** Owned by change 0, and **this change no longer depends on it.**
  Restated after the B1 retraction (Decision 5): the memory library has no connection-string field to receive, so
  there is no URL for change 0's accessor to hand it. What this change needs from change 0 is nothing; what it needs
  from itself is that its **discrete** connection fields resolve to the same instance the application's own engine
  resolves. If an implementer chooses to satisfy that by parsing `get_database_url()` into discrete parts, the
  accessor becomes a convenience, never a precondition.
- **NG14 — the process that executes scheduled consolidation.** This change registers a task and a beat entry; it
  does **not** provision a worker or a beat service, and there is no such service in the deployment to register
  against (§ Context). **The beat entry this change adds is inert on the day it lands**, and will stay inert until
  change 1 provisions those services — the runtime gap is dispositioned **in change 1** (`dispositions.md` 198.4),
  so it is deliberately not duplicated as a requirement here. Stated plainly rather than implied: **after this
  change, consolidation never runs.** The requirement *Consolidation into the permanent memory graph runs on a
  schedule* is satisfied by registration and schedule presence, and nothing in it should be read as evidence that a
  consolidation has ever executed.
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
appends to the conversation cache and **returns at `remember.py:900`**, before `_run()` at `:915` — the
`add` → `cognify` → `improve` chain — is ever reached (the detached session-improve bridge occupies `:885-898`);
scheduled `improve(dataset, session_ids=[…])` is the documented bridge into the permanent graph.

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
`target_metadata = Base.metadata` at **`env.py:39`** (`:42` sets it to `None` on the offline branch; `:23-33` are
model imports) with **no** filter of any kind, so the next `--autogenerate` would otherwise
emit `op.drop_table(...)` for every third-party memory table. **The filter lands before the vector store is
configured**, not after: the filter is the only protection that survives someone setting `include_schemas=True`, and
the first memory write is what creates the tables it protects, so configuring the store first would open a window in
which those tables exist unprotected.

- *Alternatives considered:* a Cognee-dedicated **database** — the original ADR choice, withdrawn because the
  managed instance almost certainly forbids it; leave the default local files — silently loses all memory on
  restart, and is the current behaviour; rely on `include_schemas` defaulting to `False` alone — true today but one
  flag flip away from data loss, hence the explicit filter as well.
- *Documented fallback, recorded as a decision rather than left to a scramble:* if the precondition check finds the
  application role cannot create a schema or the required extension, use a local-file vector store **on a mounted
  persistent volume**, for memory recall only — never for document retrieval, which D5.1 keeps on `pg_textsearch`.
  **The delta permits this branch explicitly and does not have to be amended to take it.** The requirement's
  normative test is *durability*, not storage medium — "no store whose data is lost on process or container
  replacement" — with a third scenario, *A durable file-backed store is permitted only for memory recall*, that
  bounds the fallback to recall, forbids it for document retrieval, and requires the health surface to report that
  the subsystem is running on the fallback rather than reporting it fully configured. An earlier draft of the
  requirement forbade *any* local filesystem path absolutely, which outlawed this documented contingency at exactly
  the moment it would be needed; that draft is corrected.

### 5. The memory subsystem is configured with discrete credentials, and they must point at the application's own database

**This decision replaces an earlier one built on a misread, and the misread is recorded rather than quietly
overwritten** (`findings-database.md` §9, verified independently 2026-08-18).

*What was claimed:* `cognee_client.py:111` reads `settings.POSTGRES_URL` **raw**, hands Cognee a credential-less URL,
and the fix is for Cognee to receive the output of change 0's single connection-string accessor.

*What the installed library and the call site actually do:*

- `RelationalConfig` (`.venv/.../cognee/infrastructure/databases/relational/config.py:12-23`) exposes **discrete
  fields only** — `db_path`, `db_name`, `db_host`, `db_port`, `db_username`, `db_password`, `db_provider` — and
  `to_dict()` (`:73-79`) returns those same seven keys. **There is no DSN, URL or connection-string field on it.** A
  requirement mandating single-connection-string configuration is therefore not merely inelegant, it is
  **unimplementable against the installed version**.
- `cognee_client.py:91-101`, inside the `try`, is the **real** configuration, and it **already** passes those
  discrete fields — including a working password via `settings.POSTGRES_PASSWORD.get_secret_value()` at `:98`.
- `cognee_client.py:107-112`, inside the `else`, builds a **separate local dict also named `config`** carrying
  `"postgres_url": settings.POSTGRES_URL`, and merely `return`s it as `app.state.cognee_config`. That value **never
  reaches Cognee**. Two variables named `config` in one function, one of which configures nothing, is what made the
  misread easy — and renaming it is a task of its own.

*The real defect, which survives the retraction and is what the requirement now covers:* `:96` reads
`settings.POSTGRES_HOST` and `:100` reads `settings.POSTGRES_DB_NAME` **independently of `get_database_url()`**, the
accessor that parses `settings.POSTGRES_URL` and is what the application's own engine connects through. Nothing makes
the two agree. Measured today they *happen* to agree — `.env.development` sets `POSTGRES_URL` to
`…qbid1qrc75.nnro3dh8tf.tsdb.cloud.timescale.com:39662/tsdb` and separately sets `POSTGRES_HOST`,
`POSTGRES_PORT=39662` and `POSTGRES_DB_NAME=tsdb` to the same values (`findings-database.md` §1) — but they agree
**by hand-maintained duplication, not by construction**, and the Pydantic defaults diverge outright:
`settings.py:140` defaults `POSTGRES_URL` to `postgresql://user:pass@host/db` while `:141` defaults `POSTGRES_HOST`
to `localhost` and `:145` defaults `POSTGRES_DB_NAME` to `db`. `.env.example` sets none of them. So **any environment
that configures only `POSTGRES_URL` — the one the application itself uses — silently points memory at
`localhost:5432/db`**, and a memory subsystem that connects to the wrong database succeeds quietly rather than
failing loudly. That is a worse failure mode than the credential-less URL originally alleged.

The requirement therefore mandates: discrete fields drawn from the single settings source, **resolving to the same
instance the application's own engine resolves**, with startup failing rather than proceeding on a divergence, no
field satisfied by a placeholder default, and transport security supplied in the form the driver accepts. On that
last point: `POSTGRES_URL` carries `sslmode=require` (and `get_database_url()` strips `sslmode` and
`channel_binding` at `postgres.py:51-54` because asyncpg rejects them), and Cognee's discrete config has **nowhere to
put them** except `database_connect_args`, which nothing sets — so "how does this connection get TLS" is a
precondition-audit question, and it is now asked.

- *Alternatives considered:* keep the connection-string requirement and have change 0's accessor feed Cognee —
  impossible, there is no field to feed; derive the discrete fields by parsing `get_database_url()` inside the memory
  client — better than today because it removes the second source of truth, and it is the shape this requirement
  admits, but it is an implementation choice the spec deliberately does not fix; leave the two sources independent
  and document it — that is the status quo, and the status quo is a silent wrong-database failure.
- *Consequence for the cross-change dependency:* this decision **no longer depends on change 0's connection-string
  accessor**, because Cognee is not a URL consumer. See NG13, restated.

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
exists** at **`src/app/middleware/health_check.py:83-90`** and is already registered in `ALL_PROBES` (`:93-99`, where
it is the fifth entry at `:98`). Disposition 198.2
was narrowed for exactly this reason, and this change **does not claim it**. *(Path correction: earlier drafts of this
decision, and `dispositions.md` 198.2 itself, cite `features/health/health_check.py:83-90` — no such file exists.
`plan-change4.md:273` had it right.)* What is missing is `check_cognee` —
and separately, the *second* health surface (`src/app/features/health/service.py`) probes
postgres/redis/mongo/neo4j/celery/memory/disk and carries **neither** subsystem, so the probe is added in both
places or the two surfaces disagree.

**A name collision on the second surface, which the implementer must not walk into.** That surface already reports a
field called `memory` — `features/health/service.py:69` calls `_check_memory()`, defined at `:200-213`, and it is
**psutil RAM**, nothing to do with agent memory. The agent-memory probe SHALL use a distinct field name on that
surface; reusing `memory` would overwrite a live, unrelated check and make both unreadable.

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
failure count stays at **6** through this change, by structure rather than by omission.

**The deployed capability has four requirements and the delta now touches all four.** The fourth, *No type ignore
suppressions*, pins its only scenario to `uv run ty check` on a **single file path** —
`…/agents/memory/cognee_client.py` — and Decision 12 retires that module's entire contents (three module-level
functions plus `CogneeStore`, replaced by a service). Left untouched, that requirement would **archive pointing at a
path that does not exist**: an accepted, deployed requirement rendered permanently unverifiable, and a hole through
which `# type: ignore` could re-enter the very code that replaces it. It is therefore carried as a fourth `MODIFIED`
block that keeps the prohibition, keeps the original scenario title *Type checker passes* **verbatim**, and restates
the test **path-neutrally** — over "the module or modules that hold the agent-memory call surface" — with an explicit
sentence that retiring or relocating the call surface is not a way to satisfy it. It is `MODIFIED`, not `REMOVED`:
nothing about the prohibition has stopped being desirable, only its address changed, and `REMOVED` would demand a
Reason and a Migration for a rule this change actively wants to keep.

**The three pre-existing `MODIFIED` blocks replace their scenarios wholesale, and that is intended, not a partial
copy.** Each reproduces the deployed `### Requirement:` header character-for-character, but every one of the eight
deployed scenarios is superseded by a rewritten scenario covering the same concern under a new title. `schema.yaml`
warns that a delta which reproduces only part of the original block silently loses the rest, so the mapping is
recorded here to prove the loss is deliberate and complete:

| Deployed scenario | Superseded by | Why the title changed |
|---|---|---|
| `Store final report` | `Approved final report is stored in conversation scope` | the write is now conversation-scoped and approval-gated |
| `Store relationships` | `Relationship summaries are no longer stored in agent memory` | the behaviour is **inverted** by the accepted boundary; keeping the old title would assert the opposite of the body |
| `Process report after store` | `A write does not trigger consolidation` | C1: the mandated post-write enrichment is exactly the defect |
| `Process relationships after store` | `Consolidation is invoked only on a schedule` | as above, plus there are no relationship writes left to follow |
| `Search episodic memory` | `Recall is scoped to the caller's memory partition` | the tenant boundary is the requirement now, not the call shape |
| `Search returns results as dicts` | `Recall results are fully serialisable and retain their origin` | `[dict(r) for r in …]` is a shallow conversion; "as dicts" understates it |
| `Search handles failures gracefully` | `Recall handles failures gracefully` | rename only, for the `search` → `recall` API change |

Every deployed concern is accounted for and every replacement is a deliberate behaviour change. Two of the seven
titles (`Store relationships`, `Process report after store`) **could not** be kept verbatim without a title that
asserts the opposite of its own body, which is a worse failure than a rename. Where a title *could* be kept — the
fourth requirement above — it is kept.

- *Alternatives considered:* one combined capability — puts behaviour in an API spec; comply with the deployed spec
  as written — means enriching twice per write, which is the defect; hand-edit
  `openspec/specs/cognee-v1-api/spec.md`'s requirement text now to make validation green — bypasses the delta
  mechanism the archive step depends on, and would erase the record of *why* the requirement changed.

### 10. The read seam is built, and it is **speculative** (D17)

**Labelled explicitly, as D17 requires.** `build_saul_graph` (`agent_saul/graph.py:86`) has no caller, and D17
settled that the unwired agent graph was **deliberate** and **stays commented**. So the read seam added by this
change cannot be exercised by running the product — not temporarily, but permanently under the current decision.
It is built anyway, because without a read seam this change writes memory nobody reads, and because the logic
largely already exists in the file being deleted (`:258-260` is already the fail-open pattern), so relocating it now
is cheaper than rebuilding it later.

**Correction to an earlier draft of this decision, which misread the code it relocates.** The earlier text said
*"`memory_pipeline.py:213,220` already branches on exactly the two task names"*. Verified in
`src/app/shared/rag/graphiti/memory_pipeline.py`: `:213` is `if task in {"risk_analysis", "obligation_chain"}:` and
`:220` is `elif task == "compliance":`, inside **`_do_retrieve_graphiti_context`** (`:204-237`). So it branches on
**three** task values, not two, and it is the **knowledge-graph supplement** branch — *not* a deep memory-retrieval
branch. Two things follow, and both are settled here rather than left to the implementer:

1. **`obligation_chain` keeps supplement eligibility.** Dropping it would be a silent behaviour regression smuggled in
   under a relocation, and it is eligible today for a reason: an obligation chain is precisely a
   document-graph traversal. The spec scenario is corrected to name all three tasks —
   *The knowledge-graph supplement is fetched only for the tasks that need it*.
2. **The supplement gate and deeper memory retrieval are two different constraints and are now two different
   requirements.** The relocated three-way branch gates the **knowledge-graph supplement** inside prefetch. The
   two-role restriction (risk analysis and compliance only) gates **deeper memory retrieval**, a separate on-demand
   operation, and lives in its own requirement in `saul-agent-memory`. The earlier single scenario conflated them,
   which would have produced a "memory retrieval" constraint implemented over the knowledge-graph path.

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
**nothing** — the old design's implicit low-trust write of unapproved reports is dropped), and **all four**
requirements of its prefetch capability are harvested into `saul-agent-memory`, including the fourth.

**Correction to an earlier draft, which handed the fourth requirement away and left nobody holding it.** That draft
said the tool-exposure requirement (*Deep memory retrieval is limited to selected reasoning nodes*) went to change 3
as NG11. Verified: `openspec/changes/agent-tools-unification/` is fully authored — proposal, design, adrs and seven
spec deltas — and `rg -ni "cognee|memory|deep(er)? retrieval"` over all of it returns **nothing** on this subject; its
`agent-tool-registry/spec.md:74-92` enumerates precedent/statute and knowledge-graph/obligation-chain tools with **no
memory tool and no negative requirement excluding the orchestrator**. So the handoff was to a change that does not
carry it, and archiving this change's predecessor would have deleted the only statement of a cost-limiting constraint
from the whole five-change set. Because the requirement **originated in the change this one supersedes, the harvest
gap is change 4's**, and it is closed here: *Deeper memory retrieval is available only to designated reasoning roles*
is a first-class requirement of `saul-agent-memory`, which also supplies the definition the old scenario lacked —
deeper retrieval as an on-demand operation **distinct from prefetch** — so the constraint is falsifiable by a reader
of the spec alone. What change 3 still owes is recorded as a coordination point in § Context, not as a requirement in a
capability change 3 owns.

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

### 15. The startup posture is stated per failure class, and `lifespan.py:206` becomes guarded

Three requirements across the two capabilities prescribe three different startup postures — *"startup SHALL fail
rather than proceed if the two differ"* (embedding dimension, `cognee-v1-api`), *"startup SHALL report the subsystem
as degraded rather than silently accepting the default"* (no durable vector store, `cognee-v1-api`), and *"the health
check SHALL report the memory subsystem as degraded … and SHALL NOT fail the request"* (absent configuration,
`saul-agent-memory`). That is deliberate, it was nowhere stated, and it is stated here.

The split is on **whether the failure corrupts data or merely withholds it**:

| Failure class | Posture | Why |
|---|---|---|
| Embedding dimension or model disagrees with the application's | **Hard fail at startup** | Every vector written under a wrong dimension is silently unusable and **cannot be repaired by fixing the config later** — it must be re-embedded, and there is no re-embedding path. It is also a pure configuration error, detectable with no I/O, and never an environment outage. Failing here costs a boot; not failing costs a corrupt store. |
| No durable vector store can be configured | **Degrade, and say so** | Nothing is corrupted; memory is simply not persisted. Taking the API down over the *memory* subsystem would trade a legal-analysis outage for a recall outage. The fallback scenario applies, and the health surface must report the fallback rather than reporting fully configured. |
| Stores configured but unreachable, or configuration absent entirely | **Degrade, probe reports it, requests still served** | Identical to how every other optional subsystem in this repository behaves, and agent memory is optional by construction: *Agent memory failures never fail the run*. |

**`lifespan.py:206` becomes guarded.** Today `cognee_config = await setup_cognee(settings)` is bare — and it is the
**only** unguarded optional-subsystem call in that file. Everything around it degrades: Graphiti `:211-223` (`try` /
`except (ConnectionError, TimeoutError, OSError)` / `add_note` / `warning` / `app.state.graphiti = None`), Crawl4AI
`:258`, object storage `:266`, Celery `:273`, outbox `:284`. Commit `1b3891f` — *"make startup resilient to optional
services"* — rewrote 121 lines of that file and **did not touch line 206**, so the current posture is an oversight
rather than a decision. Left as it is, an implementer who reads *"startup SHALL fail"* literally, and a memory
subsystem that now reaches for a database, an embedder, a vector store **and** a graph at boot, together take the
whole API down on an unreachable memory store. The guard therefore wraps it in the same shape as Graphiti's, sets
`app.state.cognee_config = None` on failure, and **re-raises the dimension/model-mismatch class**, which is the one
failure the table above says must stop the boot.

**The stated loss, not hedged (D10's honesty rule).** Adding that guard means a Cognee misconfiguration **no longer
stops a deploy**. Today, in theory, it would. So the guard removes the loudest possible signal about this subsystem
and leaves the health probe as the *only* signal — which is precisely why the probe is a **requirement** of this
change rather than a nicety, and why it must distinguish *not configured* from *unreachable* from *ok*. If the probe
is descoped, this change ships a subsystem whose misconfiguration is invisible. That is the trade, on the record.

- *Alternatives considered:* leave `:206` unguarded so misconfiguration is loud — makes agent memory the only
  optional subsystem that can kill boot, and inverts D10's own priority that a memory failure must never fail a
  completed legal analysis; guard everything including the dimension mismatch — permits a silently corrupt vector
  store, the one outcome no later fix repairs; guard it and add no probe — the misconfiguration becomes
  unobservable, which is the failure mode `lifespan.py:220-223` already demonstrates.

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
- **[The memory subsystem can be pointed at a different database than the application, silently]** → Its discrete
  connection fields (`POSTGRES_HOST`, `POSTGRES_DB_NAME`, …) are read independently of the `POSTGRES_URL` the
  application's own engine parses. They agree in `.env.development` **by hand-maintained duplication**, and their
  Pydantic defaults do not agree at all (`localhost`/`db` versus `postgresql://user:pass@host/db`). A memory
  subsystem connected to the wrong database **succeeds**, so there is no loud failure to notice. Mitigated by a
  requirement that they resolve to the same instance and that startup fail on divergence, and by a health probe that
  reports the store it actually reached — not eliminated, because nothing prevents an operator from setting the two
  independently. See Decision 5.
- **[Transport security has nowhere to go in the memory library's connection config]** → `POSTGRES_URL` carries
  `sslmode=require`; `get_database_url()` strips it (`postgres.py:51-54`) because asyncpg rejects it; the memory
  library's `RelationalConfig` has no place for it except `database_connect_args`, which nothing sets. So it is
  unverified whether this connection to a **managed cloud instance** negotiates TLS at all. Added to the
  precondition audit; not answerable from the repository.
- **[Scheduled consolidation has nothing to execute it]** → There is no worker or beat service in the deployment,
  and `Makefile:52` names a nonexistent module (`findings-deployment.md` §1–§2). Mitigated only by honesty: this
  change registers the task and the schedule entry and proves **registration**, and provisioning the worker/beat
  services is stated as an operational dependency. Do not read the consolidation requirement as "consolidation runs
  nightly" until that service exists. **The runtime gap is change 1's** (`dispositions.md` 198.4) and is not
  duplicated as a requirement here — see NG14 and coordination point C-B.
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
   edit** inserting `## Purpose` into `openspec/specs/cognee-v1-api/spec.md` is the only thing that would move the
   failure count from 6 to 5 — it is tracked as its own task and is not part of the delta. See § Context.
3. **The migration filter, before any store is configured.** The `include_object` / `include_name` filter on
   `src/alembic/env.py`. It comes **before** step 4 deliberately (Decision 4): it is the only protection that
   survives someone setting `include_schemas=True`, and step 4 is what causes the tables it protects to be created.
4. **Configuration** — the `COGNEE_*` settings surface (which does not exist today), then the embedder and vector
   store on the startup helper, plus the explicit access-control setting written into the process environment
   **before** the first memory configuration call, and a typed configuration result in place of the current
   `dict[str, Any]` so the probe has something to assert on. The startup call is **guarded** in the same pass
   (Decision 15) — it is the only unguarded optional-subsystem call in `lifespan.py` today.
5. **Observability** — `check_cognee` on both health surfaces, with the graph-procedure precondition as a named
   sub-field, and a field name on the second surface that does **not** collide with its existing psutil `memory`
   check (Decision 7). This is the acceptance test for step 1's first finding.
6. **The memory service** — the repository's first memory call site, with four operations: conversation-scoped
   report write, typed trace/QA/feedback writes, recall, and consolidation. The write shape is where Trap3 is
   honoured; the machine-checkable form is *"the rebuild is never called from a request-path method, and enrichment
   is called only by consolidation"*. The prune operation is **never** called from anywhere — a grep guard, not a
   remark (Risks).
7. **Retarget the memory-persist node** onto the service, gated on human approval, keeping the existing fail-open
   shape — a memory failure must not fail a completed legal analysis.
8. **Scheduled consolidation** — a real task decorator, module registered in the task `include` list, one schedule
   entry. **Depends on change 0** having removed the reconciliation re-exports from `src/tasks/__init__.py:6-9,18-20`;
   until then every worker dies at import and registration cannot be proven. **Registration is all that lands: the
   beat entry is inert until change 1 provisions a worker and a beat service** (NG14, coordination point C-B). Name it
   distinctly from the existing billing reconciliation entry, which shares only the word.
9. **The read seam** — the prefetch node after clarification, relocating the fail-open and the **three-way**
   supplement branch out of the file step 10 deletes, plus the deeper-retrieval service operation with its role
   restriction (whose tool binding is change 3's, coordination point C-A). **Speculative** (Decision 10).
10. **Harvest, then delete** — the two unique helpers move first; then the two reference files, the knowledge-graph
    final-report writer, the stub LangGraph store and the three legacy module-level functions go, with their
    re-exports edited in the same commit. The `write_final_report.py:122,146` → memory-function edge runs through a
    structural interface, so caller and callee must go together (§ Context).
11. **Dispose of the superseded change** — archive (never delete), with `superseded-by` recorded on both ends.

**Rollback shape:** steps 3–5 are pure additions and revert cleanly. Step 6 adds a service with no callers until
step 7. Steps 10–11 are the only irreversible ones, and they come last for exactly that reason.

## Open Questions

All of the following are **precondition checks with both branches already decided**, not unresolved design. None
would change the specs. Each names what closes it.

> **Answered 2026-08-23 (band F execution, group 1 probes).** Measured values, host/port/database printed only:
>
> - **APOC/GDS present?** The configured Neo4j instance (`*.databases.neo4j.io:7687`) does **not DNS-resolve** from
>   the execution environment — `ServiceUnavailable: Failed to DNS resolve address`. Unreachable is stronger than
>   absent, so the pre-decided branch applies a fortiori: this change ships **write-only**, group 9's task is still
>   registered (registration-only), and the consolidation *refuses to run when its graph preconditions are absent*
>   scenario is the observable behaviour. The probe must be re-run against a reachable instance before the
>   consolidation requirement can ever be read as satisfied.
> - **May the role create a schema / is vector available / do memory tables exist?** `psql`-equivalent probe
>   (psycopg, read-only): `create_schema=True`, `vector_available=1`, `memory_tables_present=0` — the expected
>   `t|1|0`. Decision 4's primary branch (database-backed store) is available; the file fallback stays unused.
> - **TLS on the discrete-fields connection?** Connected via `POSTGRES_HOST/PORT/USERNAME/DB_NAME` and inspected
>   `pg_stat_ssl` for the backend: `ssl=True, TLSv1.3`. Task 4.5 needs **no** `database_connect_args`.
> - **Do discrete settings agree with the accessor?** `get_database_url()` vs `POSTGRES_HOST/PORT/DB_NAME`:
>   `True True True` — agreeing today only because `.env.development` sets both by hand; task 4.5 still asserts
>   equality at startup so hand-synced env files cannot silently drift.
> - **Does `openspec archive` accept a 0/15-task change?** The CLI has no `--dry-run` flag (`error: unknown option`),
>   so the answer cannot be obtained without performing an archive. Task 10.4 proceeds by hand if the CLI refuses;
>   decided there, not here.

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
- **Does `openspec archive` accept a 0/15-task change?** Decides whether step 11 is a CLI call or a manual move.
  *Closes with:* one dry run.
- **Does the memory subsystem's connection to the managed instance negotiate TLS, and how?** Its config has no
  `sslmode` field; only `database_connect_args` could carry one, and nothing sets it. *Closes with:* one connection
  against the real instance with `pg_stat_ssl` inspected for that backend, in step 1. *If it connects unencrypted:*
  `database_connect_args` is set explicitly in step 4 — a configuration line, not a design change, which is why this
  is a precondition and not an open decision.

**Deliberately left open, not fog:** `redisvl` / `langcache` adoption for the application's caching (NG8).
