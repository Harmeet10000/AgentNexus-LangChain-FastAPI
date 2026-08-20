> Change class: **L** cross-cutting (multi-module, new external dependency wired for the first time, new data store, security boundary).

**Supersedes:** `cognee-saul-memory-migration` (0/15 tasks, declares the retired `spec-driven` schema; its two spec deltas are harvested here).

## Why

Cognee has never executed in this repository: `cognify` has zero call sites in `src/`, the only Cognee symbol ever
called is the startup configuration helper, and that helper configures an LLM, a graph store and a relational store
but neither an embedder nor a vector store. This change is therefore the **first-ever wiring** of the agent-memory
side of the D2 boundary — configuration correctness and write topology, decided before any data exists — not a
migration, a backfill, or a cutover.

## What Changes

- **Settle the D2 role boundary in an ADR** (`adrs.md`, Status: Accepted). Three places in the code currently
  disagree about where an approved final report goes. The boundary is decided by each library's own partition key:
  typed agent-memory entries cannot be written without a conversation/session identity, and every knowledge-graph
  write in this repo is partitioned by document. **Agent memory owns the agent-run/thread axis; the knowledge graph
  keeps the document/entity axis and all bitemporal validity. The final report goes to agent memory only.**
- **Fix two live configuration defects** (backlog item 152): memory embeddings currently default to a 3072-dimension
  third-party model against this repo's 768-dimension standard, and the vector store is never configured at all, so
  memory vectors default to local files invisible to the application's database and lost on container replacement.
- **Make the memory subsystem's database connection provably the application's own.** It is configured with discrete
  credential fields — the only form its library accepts, and the password is already correct — but its host and
  database name are read independently of the URL the application's own engine connects through, so nothing stops it
  from succeeding against a *different* database. Same-instance resolution becomes a requirement, and no connection
  field may fall back to a placeholder default.
- **Disable multi-user access control explicitly.** Left unset, the default path raises on this repo's graph backend
  rather than quietly degrading. Tenant isolation is enforced by the application through dataset and thread naming.
- **Make the subsystem observable before it is used** — a `cognee` health probe on both health surfaces, reporting
  *not configured* / *unreachable* / *ok*, and surfacing the graph-procedure precondition (backlog item 140) as a
  named sub-field. Without it, this subsystem's two headline failure modes are both silent.
- **Build one real write seam and one real read seam** on the already-existing but never-executed memory-persist
  node, which already carries a `COGNEE_WRITE_FAILED` error code waiting for exactly this.
- **Honour Trap3 before the first rebuild call site is ever written**: request-path writes append to a
  conversation-scoped cache and never trigger a full graph rebuild; consolidation into the permanent graph runs on a
  schedule.
- **Retire the deferred reference files** (D4 carve-out) — the memory router and the memory pipeline — after
  harvesting the two helpers inside them that have no other implementation in the repo, plus the now-orphaned
  knowledge-graph final-report writer and the stub LangGraph store, which is **deleted, not implemented**.
- **Correct the deployed `cognee-v1-api` spec**, which mandates a redundant enrichment call after every write. The
  write API already performs that enrichment itself, so complying with the deployed spec means enriching twice.

## Scope / Non-Goals

**In scope:** memory configuration, memory observability, the memory write seam, the memory read seam, scheduled
consolidation, and the retirement of the deferred reference files.

**Non-Goals** (full list with rationale in `design.md` § Goals / Non-Goals — every one of these is a **recorded
gap**, not an omission):

- Memory decay, curation, and deduplication (D10). After this change memory grows without any of the three. This is
  accepted and recorded, not solved.
- Multi-user access control inside the memory library (unavailable on this repo's graph backend).
- Router/threshold tuning for graph completion (the deferred half of item 140).
- `redisvl` / `langcache` adoption (the deferred half of item 179).
- Making the agent graph reachable, and therefore any end-to-end proof of this change (D17).
- A LangGraph store backed by agent memory, and any second document-retrieval path (D5.1).
- **Registering** a deeper memory-retrieval tool in the agent tool registry. The *behaviour* — which roles may invoke
  deeper retrieval, what it returns, what it must refuse — is specified **here**; only the tool binding belongs to
  change 3, because tool exposure is the registry's concern (D6.1). Recorded as an open coordination point in
  `design.md` § Context, **not** as a completed handoff: change 3's artifacts do not mention it today.
- **Provisioning the process that runs scheduled consolidation.** There is no worker and no beat service in the
  deployment at all; that gap is dispositioned in change 1. This change registers the task and the schedule entry, and
  **the entry is inert until change 1 lands those services** — after this change, consolidation never runs. Stated
  plainly rather than implied.

**BREAKING:** the knowledge-graph final-report write path is removed. Its only caller is removed in the same change,
so no live consumer breaks; the observable consequence is that the user-level partition of the knowledge graph
becomes empty (recorded in `adrs.md` § Consequences).

## Capabilities

### New Capabilities

- `saul-agent-memory`: what the agent remembers, when, at what scope, what happens when memory fails, and when
  consolidation runs. Checked against the 20 existing capabilities under `openspec/specs/` — no collision, and none
  of them covers agent memory.

### Modified Capabilities

- `cognee-v1-api`: the memory call surface. The write requirement becomes conversation-scoped; the enrichment
  requirement stops mandating a redundant call and confines enrichment to scheduled consolidation; the query
  requirement keeps the origin of each recalled item; the type-suppression prohibition is restated against the
  surviving call surface instead of a deleted file path. Four configuration requirements are **added** (embedding
  dimensionality parity, durable managed-store persistence, explicit access-control state, same-instance authenticated
  connection) because the deployed spec leaves all four unspecified rather than forbidden.

### Removed from Scope

- **No deployed requirement is deleted, and all four are modified.** The deployed `cognee-v1-api` capability has four
  requirements; earlier drafts of this proposal said "no deployed requirement is removed" while the delta touched only
  three. The fourth — *No type ignore suppressions* — pinned its only scenario to `ty check` on a single file path that
  this change retires, so leaving it alone would have archived an accepted requirement pointing at a deleted file. It
  is now a fourth `MODIFIED` block: the prohibition is kept, its original scenario title is kept verbatim, and its test
  is restated path-neutrally over the agent-memory call surface. Nothing is `REMOVED`.
- The retired knowledge-graph final-report write was never specified under `openspec/specs/` — it existed only in the
  superseded change's deltas, which are handled by archiving (never deletion).

## Impact

- `src/app/config/settings.py` — a Cognee configuration surface, which today does not exist at all.
- `src/app/shared/langchain_layer/agents/memory/**` — the memory service, replacing three module-level functions and
  a stub store.
- `src/app/shared/langgraph_layer/agent_saul/nodes.py` — the memory-persist node and a new prefetch node.
- `src/app/middleware/health_check.py` and `src/app/features/health/**` — the `cognee` probe on both surfaces.
- `src/app/connections/celery.py`, `src/tasks/**` — scheduled consolidation.
- `src/app/shared/rag/graphiti/**` — removal of the final-report episode writer and its re-export.
- `src/alembic/env.py` — an object/name filter, so a third-party library's tables are never proposed for dropping.
- Operational: a Postgres schema for memory stores inside the managed instance, and a graph-database plugin
  precondition the repository cannot install for itself.

## Risks

- **Memory grows unbounded.** No decay, curation, or dedup exists anywhere after this change (D10). Not mitigated —
  accepted and recorded; the one safeguard added is a size metric on consolidation so growth is observable.
- **The memory subsystem can silently connect to the wrong database.** Its host and database name are configured
  independently of the URL the application's own engine uses; they agree today only because both are set by hand in the
  same env file, and their code defaults do not agree at all. A wrong-database connection **succeeds**, so there is no
  loud failure. Mitigated by a same-instance requirement, a startup check and a probe that reports the store actually
  reached.
- **Guarding the startup call trades loudness for availability.** The memory startup call is today the only unguarded
  optional-subsystem call in the lifespan; guarding it means a misconfiguration no longer stops a deploy, which makes
  the health probe the only remaining signal. Accepted deliberately, and the reason the probe is a requirement.
- **A third-party library performs schema DDL inside the production managed database at first write.** Mitigated by
  schema isolation plus a migration filter, and by a precondition check before any code lands.
- **The graph rebuild fails silently without the required graph plugins.** Mitigated by a documented precondition, a
  health probe sub-field, and one manual round-trip that observes the conversation-to-permanent-graph transition.
- **Scheduled consolidation has no infrastructure to run on.** There is no worker or beat service in the deployment
  at all, and the documented command to start one names a module that does not exist. That gap is **change 1's**
  (`dispositions.md` 198.4); this change registers the task and the schedule entry, and the entry is **inert** until
  change 1 lands. Stated as an explicit coordination point in `design.md`, not assumed.
- **This change cannot be proven by running the product** (D17). Its proofs are configuration read-backs, service
  tests against a faked memory module, and one scripted round-trip. The residual wiring risk is real and recorded.
- **Dataset and thread naming are the only tenant boundary.** Mitigated by a single validated naming helper
  replacing three separate string interpolations, and a test that two users never collide.
