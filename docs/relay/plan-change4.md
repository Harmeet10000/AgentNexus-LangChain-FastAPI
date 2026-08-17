# Plan — openspec change 4: Cognee agent memory

Planner leg, 2026-08-17. Read-only on `src/`. Inputs: `decisions.md` (D1–D13), `dispositions.md`
(§ "change 4 — cognee"), `scout-memory.md`, `scout-reconciliation.md`,
`conventions-openspec-skeleton.md`. Verified independently against the installed
`cognee==1.1.0` in `.venv/` — several scout claims are corrected below, each with a citation.

Proposed change ID: **`cognee-agent-memory`** (bare slug, per D12).

---

## Shape

Change 4 is **not** a migration and **not** a rewrite. It is the first-ever wiring of a library that has
never executed in this repo: `cognify` has zero call sites in `src/`, `CogneeStore` is a stub, and the only
Cognee symbol ever called is `setup_cognee` (`lifespan.py:206`), which configures an LLM, a graph store and a
relational store but **not** an embedder and **not** a vector store. Because nothing runs, the work is shaped
by configuration correctness and by choosing the write topology before any data exists — not by data
migration. There is no backfill, no dual-run, no cutover: the "old" system is a set of never-called functions.

The change therefore has four movements, in this order: **(1) settle the boundary** (the ADR below — the
artifact with the longest life, and the reason three code sites currently disagree); **(2) make Cognee
configurable and observable** (`COGNEE_*` settings, embedder + vector store, an explicit ACL decision, and a
`check_cognee` health probe — the only signal that any of this took effect, since `lifespan.py:220-223`
already proves this repo degrades silently); **(3) build one real write seam and one real read seam** onto the
already-existing but never-executed `persist_memory` node (`agent_saul/nodes.py:772-814`, which already
carries the error code `COGNEE_WRITE_FAILED` at `:802` waiting for exactly this); **(4) retire the deferred
reference files** `write_final_report.py` and `memory_pipeline.py` (D4), harvesting the two parts of them
that have no other implementation in the repo.

The cut that makes this reviewable is that **configuration lands before code, and observability lands before
behaviour**. Steps 5–7 are independently committable and each leaves the repo bootable, because Cognee is
touched only by `setup_cognee` until step 9 introduces the first call site.

The single hardest constraint is that **change 4 cannot be proven by running the product.**
`build_saul_graph` (`agent_saul/graph.py:86`) has no caller, so `persist_memory` never executes, so no
end-to-end path exists to observe. Every Proof below is therefore one of exactly three kinds — a config
read-back, a service-level test against a faked `cognee` module, or a scripted round-trip against real
Neo4j + Postgres — and the plan says which, per step, rather than pretending an integration test exists.

---

## ADR: Graphiti/Cognee role boundary

> Liftable verbatim into `openspec/changes/cognee-agent-memory/adrs.md`.
> Sections per `conventions-openspec-skeleton.md` § `adrs.md`: Status · Context · Decision ·
> Rationale / Alternatives · Consequences.

### Status

**Accepted** — supersedes the boundary statements in
`openspec/changes/cognee-saul-memory-migration/design.md:3-7` (which is itself superseded as a change; see
§ Disposition of the superseded change) and corrects the three disagreeing in-code statements listed under
Context.

### Context

D2 fixed that **both Graphiti and Cognee survive with two distinct roles** but left the boundary
unspecified. Three places in the repo state a boundary and they do not agree:

| Site | What it says about the final report |
|---|---|
| `cognee_client.py:12-15` | `Graphiti → structural legal knowledge graph; Cognee → episodic + procedural memory` — report is Cognee's |
| `write_final_report.py:8-13`, `:100-154` | a "Memory Router" that writes the report to **both**: Graphiti as a high-trust episode *and* Cognee as episodic memory |
| `openspec/changes/cognee-saul-memory-migration/design.md:3` | "Cognee the primary memory/recall layer … Graphiti remains the structural knowledge-base layer" |

A third question rides on it: **who owns semantic/vector retrieval** — Graphiti, Cognee, or the app's own
pgvector/`pg_textsearch` path. All three physically have a vector index. D5.1 has since committed document
retrieval to raw asyncpg + `pg_textsearch` (`features/search/repository.py:415-419` BM25 via
`to_bm25query`, RRF at `fusion.py:28`), which removes the app from contention as a *candidate* and makes it
the *incumbent*.

Four facts verified in the installed libraries constrain any answer. They are not preferences:

1. **Cognee's typed agent-memory entries cannot be written without a session.**
   `.venv/.../cognee/api/v1/remember/remember.py:274-276` raises
   `session_id is required for typed memory entries` for `MemoryEntry`/`QAEntry`/`TraceEntry`/`FeedbackEntry`.
   Cognee's agent-memory surface is therefore **intrinsically scoped to a run/thread axis**.
2. **Graphiti's partition key in this repo is the document.** Every Graphiti write uses
   `group_id=document_id` / `state.doc_id` — `documents/service.py:544`, `graphiti_verifier.py:56`,
   `ingestion_kb/nodes.py:384,397`. Graphiti is therefore **scoped to a document/entity axis**.
3. **`cognee.remember()` in permanent mode is `add()` + `cognify()` + `improve()`, synchronously.**
   `.venv/.../remember/remember.py:915-944`: `_run()` awaits `add(...)`, then `cognify(...)` with
   `run_in_background=False`, then — because `self_improvement` defaults to `True` (`:610`) — `improve(...)`.
   `cognify()` is a full graph rebuild over the dataset (Trap3, `todo.md:485`). The repo's
   `store_final_report` (`cognee_client.py:150-151`) then awaits `cognee.improve(dataset=...)` a **second**
   time. So the existing reference implementation performs one full graph rebuild plus **two** enrichment
   passes per approved report, inline in a graph node.
4. **Graphiti is the only one of the two with bitemporal invalidation and dedup.**
   `graphiti_core/edges.py:271,277` (`expired_at`, `invalid_at`), auto-invalidation via
   `resolve_extracted_edges` returning `(resolved, invalidated, new)` (`graphiti_core/graphiti.py:648-678`),
   dedup machinery under `graphiti_core/utils/maintenance/`. Cognee 1.1.0 has no edge-invalidation primitive;
   its `forget` (`api/v1/forget/forget.py:16`) is deletion, not scored supersession.

Facts 1 and 2 are the load-bearing pair: **each library's own partition key already decides which axis it can
serve.** This is not a design taste question, it is an API constraint discovered by reading the installed code.

### Decision

**Three owners, split by axis, with one owner per axis:**

1. **Graphiti owns the document/entity axis** — clauses, parties, obligations, precedent chains,
   `REFERENCES_CLAUSE` edges, and *all* bitemporal validity (`valid_at`/`invalid_at`/`expired_at`).
   Partition key: `group_id=document_id`. It keeps its live writers (`graphiti_verifier.py:39-56`,
   `ingestion_kb/nodes.py:384,397`) and its live readers (`search/service.py:264`,
   `documents/service.py:753`).

2. **Cognee owns the agent-run/thread axis, and nothing else** — approved final reports, QA pairs, agent
   traces, and feedback about a Saul run. Partition key: **`session_id` = the Saul `thread_id`**, inside a
   dataset namespaced `{user_id}.legal_reports`. Cognee's writes are **session-mode only**
   (`remember(..., session_id=thread_id, self_improvement=False)`); the permanent-graph consolidation is a
   **nightly Celery beat** calling `improve(dataset=..., session_ids=[...])`.

3. **The application owns semantic/vector retrieval of documents** — raw asyncpg + `pg_textsearch`, per D5.1.
   Each memory library keeps a private internal index that exists solely to serve its own read API
   (`graphiti.search(...)`, `cognee.recall(...)`). **No application code queries a memory library's vector
   index directly, and no memory library is a document-retrieval path.** Concretely: `CogneeStore(BaseStore)`
   is deleted rather than implemented, because implementing `CogneeStore.search`
   (`cognee_client.py:304`) would create a fourth retrieval path behind a LangGraph interface.

4. **The final report goes to Cognee only.** The Graphiti final-report write
   (`client.py:311-350` `write_final_report_episode`) is retired along with its only caller
   (`write_final_report.py:110`). Postgres remains the report *system of record*
   (`features/billing/models/report.py:42` `Report`) and the audit trail (`billing/models/audit.py:48`
   `AuditLog`); Cognee holds a recall-optimised copy, not the record.

5. **Cognee's own stores are configured explicitly, never defaulted.** `vector_db_provider="pgvector"`
   pointed at a **Cognee-dedicated Postgres database** (not the app's alembic-governed database — see
   Consequences), embedder pinned to the repo's Gemini model at `EMBEDDING_DIMENSION` (768), and
   `ENABLE_BACKEND_ACCESS_CONTROL=false` set explicitly. Tenant isolation is enforced by the app through the
   dataset name and session id, **not** by Cognee ACLs.

### Rationale / Alternatives

**Why the report goes to Cognee only.** Fact 3 is decisive on cost: routing the report to both, in
`persist_memory`, means one Graphiti `add_episode` (itself an LLM-driven entity-extraction pass —
`graphiti.py:648-678`) *plus* a full Cognee `cognify()` rebuild *plus* two `improve()` passes, all awaited
inside a graph node, for a single artifact. Fact 1 then decides ownership on the merits rather than on cost:
a final report is a statement about *a run* — who asked, what was concluded, whether a human approved it —
and Cognee's typed entries are the only structure in either library that models that, at the price of
requiring a session id. Facts 2 and 4 decide the other side: an approved report contains obligations whose
validity changes over time, and only Graphiti can express supersession — but it should learn those
obligations from the **clause extraction path it already owns**, keyed by document, not by re-ingesting a
prose summary keyed by user.

**Why session mode rather than Trap3's "batch `add()`, defer `cognify()`".** Trap3's intent is right and its
mechanism is unnecessary: Cognee 1.1.0 already ships the deferral. `remember(session_id=...)` writes to the
session cache only (`remember.py:895-900`) and never touches `cognify`; `improve(dataset, session_ids=[...])`
(`api/v1/improve/improve.py:36`) is the documented consolidation that bridges session Q&A into the permanent
graph, applies feedback weights, and rebuilds triplet embeddings. Hand-rolled batching would duplicate that.
We additionally pass `self_improvement=False`, because with a session id and `self_improvement=True` Cognee
fires the bridge as an unstructured `asyncio.create_task` (`remember.py:~885-890`) inside the caller's event
loop — fire-and-forget work whose failure is logged as "non-fatal" and whose lifetime is not tied to
anything we control. An explicit nightly beat is observable; a detached task is not.

| Alternative | Pros | Cons | Why rejected |
|---|---|---|---|
| **A. Dual-write the report (status quo of `write_final_report.py`)** | Both graphs see the highest-signal artifact; `graphiti_verifier.py:70` and `documents/service.py:753` already read `group_ids=[user_id, …]`, so the Graphiti copy *would* be found | Two graph rebuilds per report; two failure modes in one node; the report becomes a second, prose-shaped source of entities competing with the clause extractor for the same facts, with no dedup between them | Fact 3's cost, plus fact 4: duplicate party/obligation nodes arriving from two extractors is precisely the damage Trap1 (`todo.md:483`) says is cheap now and unfixable later |
| **B. Cognee primary for everything, Graphiti retired** | One memory system; matches a literal reading of item 155 ("replace … with cognee entirely") | Loses bitemporal invalidation and dedup outright (fact 4) with no Cognee equivalent; Graphiti has four live call sites and is the only writer of `REFERENCES_CLAUSE postgres_chunk_id=` (`graphiti_verifier.py:39-56`) | Contradicts D2, which is locked |
| **C. Graphiti primary, Cognee deleted** | Removes an unused dependency and all its config risk; smallest possible change | Contradicts D2; Graphiti has no typed run-memory model (no actor/reason/feedback fields on episodes) so agent-run memory would have to be hand-built on top of episodes | Contradicts D2, which is locked |
| **D. Cognee owns semantic document retrieval via `CogneeStore`** | One `BaseStore` interface for LangGraph; free hybrid retrieval | Directly contradicts D5.1, which commits document retrieval to asyncpg + `pg_textsearch` where BM25 and RRF already work; adds a fourth retrieval path; `CogneeStore` is a stub with five `# type: ignore`d overrides | Contradicts D5.1, and dispositions already DROPped `app.state.vector_store` for the identical "third retrieval path" reason |
| **E. Enable Cognee ACLs for tenant isolation instead of app-level dataset naming** | Isolation enforced by the library, defence in depth | With `graph_database_provider="neo4j"` and the handler left at its default `"ladybug"` (`.venv/.../databases/graph/config.py:45,59`, whose `fill_derived` at `:77-79` only remaps kuzu and postgres), `multi_user_support_possible()` reaches `context_global_variables.py:~60-77` and **raises `EnvironmentError`** because `supported_dataset_database_handlers["ladybug"]["handler_provider"] == "ladybug" != "neo4j"`. With the env var unset this is the **default** path (`:88-92`) | Rejected as unavailable, not as undesirable. The only ACL-capable neo4j handler is `neo4j_aura_dev` (`supported_dataset_database_handlers.py:18-21`), which is Aura-specific and untested here |
| **F. Give Cognee the app's own Postgres database (status quo of `setup_cognee`)** | One database to operate; already coded (`cognee_client.py:92-102` passes `settings.POSTGRES_DB_NAME`) | Cognee creates and migrates its own tables (`.venv/.../cognee/alembic/`, plus `_ensure_migrations_run()` at `remember.py:41`). `src/alembic/env.py` sets `target_metadata = Base.metadata` with **no `include_object`/`include_name` filter** (`env.py:23-30`), so the next `alembic revision --autogenerate` emits `op.drop_table(...)` for every Cognee table | Rejected. A dedicated database is a one-setting change; the alternative is a latent data-loss migration |

**Correction to an earlier claim, recorded so it is not re-inherited.** The relay brief states Cognee
defaults to 3072 dimensions. In 1.1.0 `embedding_dimensions` defaults to **`None`** and is resolved in
`model_post_init`; the code comment at
`.venv/.../vector/embeddings/config.py:73-77` says it "was hard-defaulted to 3072, which silently broke every
non-OpenAI … embedder". The mismatch is nonetheless real, because the default **model** is still
`openai/text-embedding-3-large` (`:72`) and resolves to 3072. The practical consequence changes: setting
`embedding_model` correctly is *sufficient*, and setting `embedding_dimensions` is belt-and-braces rather
than the fix.

**Correction to the ACL claim.** `GRAPH_DBS_WITH_MULTI_USER_SUPPORT` (`context_global_variables.py:96`) is
consumed by `is_multi_user_support_possible()`, **not** by the gate on the write path.
`backend_access_control_enabled()` (`:83-92`) calls `multi_user_support_possible()` (`:34-81`), which gates on
`supported_dataset_database_handlers` — and that dict **does** contain a neo4j entry (`neo4j_aura_dev`). The
conclusion is unchanged (ACL off, explicitly) but the reason is a handler/provider mismatch that *raises*,
not an absent backend that silently disables.

### Consequences

**Positive**

- One owner per axis, and each owner's partition key enforces it mechanically: a thread-scoped write cannot
  land in the document graph, and a document-scoped write cannot land in run memory.
- The write path becomes cheap enough to sit in a graph node: a session-cache append with no `cognify`.
- Trap3 is honoured before the first `cognify` call site is ever written — the cheapest possible moment.
- `openspec/specs/cognee-v1-api/spec.md` stops being aspirational. It currently mandates a redundant
  `improve()` after every `remember()`; this ADR is the reason to change it rather than comply with it.

**Negative / accepted**

- **After this change the `user_id` partition in Graphiti is empty.** `graphiti_verifier.py:70` and
  `documents/service.py:753` read `group_ids=[user_id, document_id]` / `[user_id, *doc_ids_filter]`; the
  `user_id` half was populated only by the retired final-report write. The union degenerates to
  document scope. Left in place deliberately (it is the natural home for future user-level entity facts) and
  recorded here so it is not later mistaken for a bug — but it *is* now a filter that matches nothing.
- **Two graph databases, both on Neo4j, with no cross-links.** Cognee's permanent graph and Graphiti's
  entity graph share a Neo4j instance and cannot reference each other's nodes. Joining a run to the clauses
  it analysed is an application-level join on `doc_id`, not a graph traversal.
- **A second Postgres database to provision and back up**, whose schema is owned by a third-party library's
  migrations and is invisible to `src/alembic/`.
- **No tenant isolation below the application layer.** With ACLs off, dataset naming and session ids are the
  *only* isolation. A bug in dataset-name construction is a cross-tenant memory leak with no second line of
  defence. `store_final_report` builds that name by string interpolation today
  (`cognee_client.py:140`); it must become a single validated helper.
- **An operational precondition with a silent failure mode.** `cognify()` requires APOC + GDS on the target
  Neo4j (item 140, `todo.md:253`) or it fails silently. There is no Neo4j service in `docker-compose.yml`
  (services are `rabbitmq`, `timescale`, `caddy`, `ai-service-1`), so the instance is externally managed and
  the repo cannot guarantee the plugins. This is why the health probe is a *requirement* of this change and
  not a nicety.
- **Memory grows without decay, curation, or dedup** — see § Recorded capability gaps. This ADR does not
  fix it; it names it.

---

## Recorded capability gaps

These go verbatim into `design.md` § **Non-Goals**, and the first four are the honest reading of D10: item
155's word "entirely" is honoured for *reconciliation removal*, never for *capability parity*. The repo's own
change document already conceded this —
`openspec/changes/cognee-saul-memory-migration/proposal.md:20-21`: *"Cognee v1.1 has no built-in
curation/decay/dedup"*, with `saul-cognee-maintenance-worker` and `saul-cognee-reconciliation` marked
**deferred**. Each row names what dies, where it lived, and that nothing else in the repo provides it.

| # | Capability lost | Sole implementation, deleted in change 0 | Replacement anywhere? |
|---|---|---|---|
| **NG1** | **Scored memory decay** — exponential decay from age × access_count × confidence | `src/tasks/memory_decay_reconciliation_tasks.py:51` `_compute_decay` (~13 lines), driver `:64` `_run_decay_async`, entry `:180` `run_memory_decay` | **None.** Cognee's `forget` (`api/v1/forget/forget.py:16`) is deletion by identifier, not a decay score. Graphiti has no decay. This is the repo's only decay formula and it is gone |
| **NG2** | **Near-duplicate detection over memory entities** — raw SQL self-join `entities e1 JOIN entities e2` with LLM adjudication and a JSON-parse chain | `reconciliation/nodes.py:62` `make_fetch_existing_node` (`:94-95` the self-join), `:135` `make_reconcile_node`, prompt `prompts.py:23`, parsers `nodes.py:374,380,385` | **Partial, different axis.** Graphiti dedups *its own* entity graph natively (`graphiti_core/utils/maintenance/dedup_helpers.py`, `node_operations.py`) — but only for document-axis nodes it extracted. Nothing dedups Cognee's run-memory axis |
| **NG3** | **Edge-preserving merge** — merge duplicate entities while re-pointing `relationships` rows, with an explicit merge-vs-keep-both decision schema | `reconciliation/nodes.py:205` `make_apply_changes_node`; decision model `reconciliation/state.py:30` `MergeDecision` | **None on the memory axis.** Graphiti's `resolve_extracted_edges` → `(resolved, invalidated, new)` (`graphiti.py:648-678`) is the graph-side equivalent and applies only to Graphiti's own edges. Cognee has no edge model to preserve |
| **NG4** | **Memory-entity version history** — append-only `memory_versions` rows per entity change | `reconciliation/nodes.py:274` `make_write_versions_node`; table model `src/database/schemas/memory_schema.py` | **No.** `billing/models/audit.py:48` `AuditLog` is live and migrated but scoped to billing. Cognee's `TraceEntry`/`FeedbackEntry` are the nearest shape and are untested here |
| **NG5** | **Per-user / fleet-wide reconciliation orchestration** | `memory_decay_reconciliation_tasks.py:186` `run_reconciliation_for_user`, `:198` `run_reconciliation_for_active_users` | Shape is re-derivable from `src/tasks/billing_tasks.py:71,253`. Not preserved |

**The honest framing of NG1–NG5, which `design.md` must state and not soften:** none of these was ever
observable behaviour. Proof, from `scout-reconciliation.md` §2–§4 and re-confirmed here: zero
`@celery_app.task` decorators in `memory_decay_reconciliation_tasks.py`; the module is absent from
`connections/celery.py:191-196` `include` (4 entries: auth_email, example, search, billing);
`beat_schedule` (`:259-276`) holds exactly 4 billing entries; and no migration ever created `entities`,
`relationships`, `events`, or `memory_versions`. **What is lost is design work, not a regression a user could
notice.** That is what makes the deletion acceptable — and it is also exactly why deletion produces no test
signal (§ Ordering constraints).

Additional Non-Goals for change 4, carried from D13 dispositions and from this ADR:

- **NG6 — Cognee ACLs / multi-user access control.** Unavailable on this repo's Neo4j backend
  (ADR alternative E). `ENABLE_BACKEND_ACCESS_CONTROL=false` is set *explicitly* so the behaviour is
  deterministic rather than a raised `EnvironmentError` on the first write.
- **NG7 — `GRAPH_COMPLETION_COT` / `FEELING_LUCKY` router threshold > 0.8** (item 140, `todo.md:253`).
  Split per dispositions: the APOC/GDS precondition is IN (step 1, step 8); the router tuning is DEFERRED.
  `recall()` exposes the knobs (`recall.py:329` `triplet_distance_penalty=6.5`, `:331` `feedback_influence`,
  `:322` `auto_route`) — they are left at defaults, deliberately untuned.
- **NG8 — `redisvl` / `langcache` adoption** (item 179, `todo.md:271`). Split per dispositions: the narrow
  question "does Cognee need its own Redis" is answered in `design.md` (it does not — Cognee 1.1.0's stores
  are relational + vector + graph; no Redis in its config surface). The adoption research is DEFERRED.
- **NG9 — `CogneeStore(BaseStore)` as a LangGraph store.** Deleted, not implemented (ADR alternative D).
  LangGraph checkpoint/store duties stay on Postgres (`langgraph_layer/checkpointer.py`).
- **NG10 — wiring `build_saul_graph`.** Change 4 does not make the Saul graph reachable; that depends on the
  registry unification (D6.1, change 3). Change 4 makes `persist_memory` *correct*, not *reached*.
- **NG11 — exposing a deeper `retrieve_from_memory` tool to `risk_analysis`/`compliance`.** This one
  requirement is harvested out of the old change and handed to **change 3**, because tool exposure is the
  registry's concern (D6.1) and change 4 must not add a second tool-registration path.

---

## Ordering constraints

**Cross-change inbound dependencies (D8: "change 4 depends on reconciliation already being gone").**

| Dependency | On | Why change 4 cannot start without it | Verified |
|---|---|---|---|
| Reconciliation package deleted (`reconciliation/` 618 L, `memory_decay_reconciliation_tasks.py` 209 L, `memory_schema.py` 302 L) **with the paired edit to `src/tasks/__init__.py:6-9,18-20`** | **change 0** | Change 4 adds a Celery module to `celery.py` `include`. If `tasks/__init__.py` still re-exports the deleted reconciliation helpers, **every celery worker fails at import**, so the new nightly task cannot be proven registered | `scout-reconciliation.md` §3: `tasks/__init__.py:6` is the one live edge |
| `check_cognee` health probe slot | **change 0** (dispositions 198.2) | Coordinate, do not duplicate. **Correction to dispositions:** `check_graphiti` **already exists** and is already in the probe list — `src/app/middleware/health_check.py:83-90`, registered at `:98`. What is missing is `check_cognee`. The *second* health surface, `src/app/features/health/service.py`, probes postgres/redis/mongo/neo4j/celery/memory/disk and has **neither** graphiti nor cognee | Read directly, this pass |
| Alembic head merge + `env.py` model registration | **change 0** | Change 4 provisions a **separate** Cognee database, so it does not depend on the merge for its own schema — but it *does* depend on `env.py` being sane before anyone runs `--autogenerate`, because there is no `include_object` filter (`env.py:23-30`) | Read directly, this pass |
| Ingestion path actually produces content | **change 1** | Cognee writes the Saul final report plus a relationships summary. The relationships come from `relationship_mapping`, and the report from `finalization` (`agent_saul/nodes.py:765-767`) — both downstream of a working ingestion + retrieval path | D8 |
| ToolRegistry unification / Saul graph reachable | **change 3** | Only needed to *exercise* the write end-to-end, never to *implement* it. Change 4 therefore ships its Proofs at service level. **Stated as a limitation, not worked around** | `agent_saul/graph.py:86` `build_saul_graph` has no caller |

**Internal ordering rules.**

1. **The deployed spec is repaired before any delta is authored against it.**
   `openspec/specs/cognee-v1-api/spec.md` has no `## Purpose` and no `## Requirements` header — it opens
   directly at `### Requirement:` (`:1`). `openspec validate cognee-v1-api --type spec` reports
   *"Spec must have a Purpose section … Expected headers: `## Purpose` and `## Requirements`"*. A
   `## MODIFIED Requirements` delta must copy the entire existing requirement block and match its header, and
   a spec that does not parse has no blocks to match — so the delta would silently lose content at archive.
2. **Configuration precedes code.** Steps 5–7 (settings, `setup_cognee`, health probe) must land before step
   9 introduces the repo's first real Cognee call site. Reversing this means the first `remember()` runs
   against `vector_db_provider="lancedb"` and writes embeddings to local files
   (`.venv/.../vector/config.py:30`), and — with `ENABLE_BACKEND_ACCESS_CONTROL` unset — may instead raise
   `EnvironmentError` from `multi_user_support_possible()`.
3. **Observability precedes behaviour.** The health probe (step 7) lands before the write path (step 9),
   because `lifespan.py:220-223` already proves this repo's habit of degrading silently, and Cognee's failure
   mode for a missing APOC/GDS is silent by construction.
4. **Deletions come last** (steps 11–12). `write_final_report.py` and `memory_pipeline.py` are the *only*
   existing reference for how Cognee writes were meant to work (D4's stated reason for deferring them to this
   change). They stay readable until their replacement exists and is tested.
5. **Harvest before delete.** Two things inside the doomed files have no other implementation and must move
   before the files go — see step 11.

**What stands in for test signal on the deletions.** The reconciliation and cognee modules have **zero test
coverage** (codegraph reports "no covering tests found" for `CogneeStore`, `setup_cognee`,
`store_final_report`, `store_relationships`; `rg "reconcil|memory_schema" tests/` hits only prose in
`todo.md`). So a green suite proves nothing about a deletion. The evidence that stands in, in descending
strength:

- **`graphify affected "<symbol>"` returns no nodes** other than the package's own `__init__` re-export —
  the same instrument that closed D11 and proved reconciliation dead.
- **`rg -n '<symbol>' src/ tests/` returns only the definition, `__init__` re-exports, and docstrings.**
- **The app still boots**, i.e. the import graph is intact — the only check that catches the
  `registry.py:41-46` eager-import class of failure (D6.1), which no unit test would.
- **`uv run ruff check src/` and `uv run ty check src/` counts do not increase** — dangling imports and
  now-unused symbols surface here, not in pytest.
- **`openspec validate --all` failure count does not increase** — catches a spec left referencing a deleted
  requirement.

**Baseline traps that every Proof below is written against.**

- `--cov-fail-under=80` against **18.38%** coverage means a fully green suite **still exits 1**. Every pytest
  Proof compares the **summary line** (`N passed`), never `$?`. Baseline: **55 passed**.
- Lint/type baselines: **ruff 125**, **ty 46**. D11 moves ruff to **123** in change 0 (both
  `invalid-syntax` errors are `todo_temp.py`), so change 4's ruff comparand is **123**, not 125.
- `openspec` baseline is **16 passed / 6 failed of 22** (`openspec validate --all`, v1.8.0 at
  `/home/harmeet/.bun/bin/openspec`), confirmed this pass. Acceptance is **"no new failures"**, never
  "validate passes".
- **`spec/cognee-v1-api` is one of the 6 — and change 4 fixes it.** Decision: **fix, do not leave.** It is a
  pure header-structure defect in a spec this change must modify (rule 1 above), the fix changes no
  requirement text, and it moves the baseline to **17 passed / 5 failed**. The other five
  (`change/mintlify-documentation`, `spec/noqa-documentation`, `spec/pattern-matching-standard`,
  `spec/transactional-outbox`, `spec/typed-exception-handling`) are **out of scope and stay failing** —
  change 4's acceptance is *5* remaining failures, and any sixth is a regression.

---

## ADR amendment — the app's Postgres is Timescale Cloud

Arrived after the ADR was written; **it must be folded into the ADR when lifted**, because it invalidates the
mechanism (not the goal) of ADR § Decision item 5 and adds a Consequence.

`.env.development` `POSTGRES_URL` points at `*.tsdb.cloud.timescale.com:39662/tsdb` — a **managed cloud
instance**, not the `timescale` service in `docker-compose.yml`. Two things follow:

1. **"A Cognee-dedicated Postgres database" is probably not available.** Managed Postgres services commonly
   forbid `CREATE DATABASE` on the provisioned instance; Timescale Cloud provisions `tsdb`. So the ADR's
   chosen mitigation for the alembic-autogenerate hazard (alternative F) cannot be assumed.

   **Amended Decision 5:** isolate Cognee by **Postgres schema**, not by database — `cognee` schema in
   `tsdb` — and additionally add an `include_object` / `include_name` filter to `src/alembic/env.py`.
   Rationale: Alembic's `include_schemas` defaults to **False**, so a non-default schema is already invisible
   to `--autogenerate` reflection; the explicit filter is the belt-and-braces that survives someone later
   flipping `include_schemas=True`. If Cognee's pgvector provider turns out not to honour a custom schema
   (Fog F6), the fallback is the `env.py` filter alone, which is sufficient on its own.

2. **Cognee's `vector_db_provider="pgvector"` now depends on a managed extension allow-list.** Cognee's
   pgvector provider issues `CREATE EXTENSION`-class DDL and creates its own tables via its own alembic
   (`.venv/.../cognee/alembic/`, invoked lazily by `_ensure_migrations_run()` at `remember.py:41`). On a
   managed instance that DDL may be refused, and the repo's own retrieval path already depends on
   **`pg_textsearch`** (`to_bm25query`, `features/search/repository.py:415-419`), which is a different extension
   from plain `pgvector`. Whether both are enabled on this Timescale Cloud instance, and whether the
   application role may create a schema and extensions in it, is a **precondition to verify before step 7**,
   not an assumption (step 1, Fog F6).

**New Consequence for the ADR:** Cognee's stores now sit inside the same managed instance as production
application data, isolated only by schema, with DDL performed by a third-party library at first write. The
blast radius of a Cognee migration bug is the production database. If the precondition check in step 1 fails,
the honest fallback — recorded here so it is a decision and not a scramble — is
`vector_db_provider="lancedb"` on a **mounted persistent volume**, accepting local-file vectors for memory
recall only (never for document retrieval, which D5.1 keeps on `pg_textsearch`), and revisiting when a
self-managed Postgres exists.

---

## Steps

Ordered by dependency. Each is independently committable and leaves the repo bootable. `Proof:` is an exact
command with its expected output; where nothing can run today, the Proof says what "working" means and how it
is observed instead of pretending an integration test exists.

Standing convention for every pytest Proof, because `--cov-fail-under=80` against 18.38% coverage makes a
green suite exit 1: **read the summary line, never `$?`.** Baseline **55 passed**, ruff **123** (post-D11),
ty **46**, openspec **16 passed / 6 failed** → **17 / 5** after step 2.

### Step 1 — Precondition audit against the real infrastructure (no code)

**Inbound:** none. Runs first because three of its answers can invalidate later steps' design.

Four facts to establish, all currently unknown and all load-bearing:

1. **APOC + GDS on the target Neo4j** (item 140, `todo.md:253`). Without them `cognify()` **fails silently** —
   no exception, no data. There is **no Neo4j service in `docker-compose.yml`** (services are `rabbitmq`,
   `timescale`, `caddy`, `ai-service-1`), so the instance is externally managed and the repo cannot install
   plugins for it.
2. **Timescale Cloud DDL capability** (ADR amendment): may the application role `CREATE SCHEMA`, and are
   `vector`/`pg_textsearch` extensions available.
3. **Does Cognee want a Redis?** — item 179's narrow half. Answer by inspection of its config surface, not by
   research.
4. **The orphan-table question** — whether `entities` / `relationships` / `events` / `memory_versions` exist
   in the live database out-of-band. Owned by change 0's plan, but change 4 consumes the answer: if they
   exist, deleting `memory_schema.py` leaves live tables with no model.

**Proof:**
```bash
# 1. APOC + GDS — expect both counts > 0 and a version string
cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" \
  "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' RETURN count(*) AS apoc;"
cypher-shell ... "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'gds' RETURN count(*) AS gds;"
cypher-shell ... "RETURN gds.version() AS gds_version;"
# 2. Timescale Cloud DDL + extensions — expect 'vector' present; note whether `pg_textsearch` is listed
psql "$POSTGRES_URL" -c "SELECT name, installed_version FROM pg_available_extensions
                          WHERE name IN ('vector','vchord','vchord_bm25','pg_textsearch');"
psql "$POSTGRES_URL" -c "CREATE SCHEMA IF NOT EXISTS cognee_probe; DROP SCHEMA cognee_probe;"
# 3. Cognee + Redis — expect ZERO hits, i.e. Cognee 1.1.0 has no Redis in its config surface
rg -il "redis" .venv/lib/python3.12/site-packages/cognee/infrastructure/databases/ | head
# 4. Orphan tables — expect zero rows
psql "$POSTGRES_URL" -c "\dt" | rg -w "entities|relationships|events|memory_versions"
```
**Expected:** `apoc > 0`, `gds > 0`, `gds.version()` returns; `vector` available; `CREATE SCHEMA` succeeds;
zero Redis hits in Cognee's database layer (→ item 179's narrow answer is **"no, Cognee needs no Redis"**,
recorded in `design.md` § Decisions); zero orphan tables.
**If 1 fails:** step 11's nightly `improve()` is inoperable and change 4 ships write-only — record as a
blocking Risk, do not proceed to step 11. **If 2 fails:** take the ADR amendment's `lancedb`-on-a-volume
fallback. Findings are written into `design.md` § Context; no code changes in this step.

### Step 2 — Repair the deployed `cognee-v1-api` spec so a delta can be authored against it

**Inbound:** none. Must precede step 4 (Ordering constraint 1).

`openspec/specs/cognee-v1-api/spec.md` opens at `### Requirement:` on line 1 with no `## Purpose` and no
`## Requirements`. It is one of the 6 baseline `openspec validate --all` failures. **Decision: fix it, do not
leave it** — change 4 must emit a `## MODIFIED Requirements` delta against it (step 4), and `MODIFIED` must
copy and header-match the entire existing requirement block, which is impossible while the spec does not
parse: the delta would silently lose content at archive. The fix inserts a `## Purpose` (≥50 chars) and a
`## Requirements` header. **No requirement text changes** — this is structure only, so it is not a behaviour
change smuggled into a formatting commit.

**Proof:**
```bash
timeout 120 openspec validate cognee-v1-api --type spec   # expect: no issues
timeout 180 openspec validate --all 2>&1 | tail -2        # expect: Totals: 17 passed, 5 failed (22 items)
```
The 5 remaining are `change/mintlify-documentation`, `spec/noqa-documentation`,
`spec/pattern-matching-standard`, `spec/transactional-outbox`, `spec/typed-exception-handling` — out of scope,
they stay failing. **Any sixth failure is a regression.**

### Step 3 — Create the change directory and `proposal.md`

**Inbound:** step 1 (its findings shape Scope/Non-Goals).

`openspec/changes/cognee-agent-memory/` with `.openspec.yaml` = `schema: spec-gated` (matching
`openspec/config.yaml:1` — the superseded change's `spec-driven` is what made it unmigratable),
`created: 2026-08-17`. **No `skip_specs`** — this change has real spec deltas. `proposal.md` declares class
**L** on line 1 (multi-module + a new external dependency wired for the first time + a new data store), and
its Capabilities section names the mapping settled in § Openspec mapping below.

**Proof:** `timeout 120 openspec validate cognee-agent-memory --type change` → no issues;
`timeout 180 openspec validate --all 2>&1 | tail -2` → `18 passed, 5 failed (23 items)` (the new change adds
one item). `rg -c "^> Change class" openspec/changes/cognee-agent-memory/proposal.md` → `1`.

### Step 4 — `specs/**` deltas, `design.md`, `adrs.md`

**Inbound:** step 2 (the `MODIFIED` target must parse), step 3 (`specs` and `design` both require `proposal`).

- `adrs.md` = the ADR above **plus the Timescale amendment folded in**.
- `design.md` carries § Non-Goals = **NG1–NG11** verbatim from § Recorded capability gaps (this is D10's
  requirement and D13's), § Decisions each with alternatives considered, § Risks in the literal
  `[Risk] → Mitigation` form, and the item-179 and item-140 answers from step 1.
- Spec deltas: one `## MODIFIED Requirements` against `cognee-v1-api` and one new capability
  `saul-agent-memory` — see § Openspec mapping.

**Trap to avoid, it fails silently:** scenario headers take **exactly four hashtags**
(`schema.yaml:164-165`; three hashtags drop the scenario with no error).

**Proof:**
```bash
timeout 120 openspec validate cognee-agent-memory --type change   # expect: no issues
# every requirement has >=1 four-hash scenario: counts must match
rg -c "^### Requirement:" openspec/changes/cognee-agent-memory/specs/**/spec.md
rg -c "^#### Scenario:" openspec/changes/cognee-agent-memory/specs/**/spec.md
rg -n "^### Scenario:" openspec/changes/cognee-agent-memory/specs/   # expect: ZERO hits (3-hash trap)
```

### Step 5 — `review.md` by a fresh subagent, then `tasks.md`

**Inbound:** step 4 (`review` requires `design`; `tasks` requires `specs` + `design` + **`review`**).

Per `schema.yaml:321,394-396` the review is written **as a reviewer, not the author** — so a fresh subagent
writes it, never the author of the proposal. D12 established `review.md` is **not CLI-enforced** (the
superseded change passes validation without one); we honour it by choice. `tasks.md` is written only after the
`VERDICT:` line is not `CHANGES-REQUESTED`, and each task states its own verification — the relay's "Proof"
rule and openspec's "each task must be verifiable" rule are the same requirement.

**Proof:** `rg -n "^\*\*VERDICT:\*\*" openspec/changes/cognee-agent-memory/review.md` → one line, value
`APPROVED` or `INFO`. `rg -c "^- \[ \] [0-9]+\.[0-9]+ " .../tasks.md` → equals the task count (the apply
phase *parses* this checkbox shape, `schema.yaml:406-409`; a task not in `- [ ] N.M` form is untracked).

### Step 6 — `COGNEE_*` settings (item 152, half one)

**Inbound:** step 1 fact 2 (whether pgvector-on-Timescale is viable decides the default), step 4 (design
settled).

`rg -i cognee src/app/config/settings.py` returns **no hits** — there is no Cognee configuration surface at
all today. Add one, env-driven, with the dimension tied to the existing single source of truth
(`settings.py:212` `EMBEDDING_DIMENSION: int = Field(default=768, gt=0)`, guarded by the validator at `:44`):
embedding provider/model/api-key/dimensions, `COGNEE_VECTOR_DB_PROVIDER`, `COGNEE_DB_SCHEMA`,
`COGNEE_DATASET_PREFIX`, `ENABLE_BACKEND_ACCESS_CONTROL=false` (explicit — see below), and the nightly-improve
toggle + cron for step 11.

**`ENABLE_BACKEND_ACCESS_CONTROL=false` is not tidiness, it prevents a raise.** With
`graph_database_provider="neo4j"` and `graph_dataset_database_handler` left at its default `"ladybug"`
(`.venv/.../databases/graph/config.py:45,59`; `fill_derived` at `:77-79` only remaps kuzu and postgres),
leaving the env var **unset** takes the default branch at `context_global_variables.py:88-92` into
`multi_user_support_possible()`, which raises `EnvironmentError` because
`supported_dataset_database_handlers["ladybug"]["handler_provider"] == "ladybug" != "neo4j"`.

**Proof:**
```bash
uv run python -c "
from app.config import get_settings
s = get_settings()
print(s.COGNEE_VECTOR_DB_PROVIDER, s.COGNEE_EMBEDDING_MODEL, s.COGNEE_EMBEDDING_DIMENSIONS, s.EMBEDDING_DIMENSION)
assert s.COGNEE_EMBEDDING_DIMENSIONS == s.EMBEDDING_DIMENSION, 'dimension drift'
assert s.COGNEE_VECTOR_DB_PROVIDER != 'lancedb' or s.COGNEE_LANCEDB_PATH, 'lancedb needs a persistent path'
"
uv run ruff check src/ 2>&1 | tail -2   # expect: still 123
uv run ty check src/ 2>&1 | tail -3     # expect: still 46
```
**Expected:** prints the Gemini embedding model and `768 768`; the assert is the guard against item 152's
dimension conflict reappearing.

### Step 7 — Make `setup_cognee` configure the embedder and the vector store (item 152, half two)

**Inbound:** step 6 (settings must exist), step 1 fact 2.

`setup_cognee` (`cognee_client.py:58-114`) today calls `set_llm_config`, `set_graph_db_config`,
`set_relational_db_config` — and **not** `set_embedding_config`, **not** `set_vector_db_config`. The embedder
block is commented out at `:46-55` with a `TODO: add this VECTOR_DB_PROVIDER=pgvector` at `:46`. Consequences
of the omission, both verified in the installed library:

- **Embedder:** default `embedding_provider="openai"`, `embedding_model="openai/text-embedding-3-large"`
  (`.venv/.../vector/embeddings/config.py:71-72`) resolving to **3072** dims against the repo's **768**. Note
  the corrected mechanism: `embedding_dimensions` itself now defaults to `None` and is resolved in
  `model_post_init` (`:73-77`), so **setting the model is the fix**; setting dimensions is belt-and-braces.
  Left as-is, Cognee also calls **OpenAI** with no key configured.
- **Vector store:** default `vector_db_provider="lancedb"` (`.venv/.../vector/config.py:30`) → embeddings go
  to **local files**, invisible to the app's Postgres and lost on every container replacement.

Also in this step: `db_path=""` is passed today (`:100`) — set the schema from `COGNEE_DB_SCHEMA`; set
`ENABLE_BACKEND_ACCESS_CONTROL` into the process env **before** the first cognee config call; and replace the
`dict[str, Any]` return (`:107-114`) with a frozen Pydantic model so the health probe in step 8 has something
typed to assert on.

**Proof** — a config read-back, because there is no behaviour to observe yet:
```bash
uv run python -c "
import asyncio, os, cognee
from app.config import get_settings
from app.shared.langchain_layer.agents.memory import setup_cognee
cfg = asyncio.run(setup_cognee(get_settings()))
ec = cognee.config.get_embedding_config() if hasattr(cognee.config,'get_embedding_config') else None
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
v, e = get_vectordb_config(), get_embedding_config()
print('vector_provider =', v.vector_db_provider)
print('embed_model     =', e.embedding_model, '| dims =', e.embedding_dimensions)
print('acl_env         =', os.environ.get('ENABLE_BACKEND_ACCESS_CONTROL'))
assert v.vector_db_provider != 'lancedb', 'still defaulting to local LanceDB files'
assert 'openai' not in (e.embedding_model or ''), 'still defaulting to the OpenAI embedder'
assert e.embedding_dimensions == get_settings().EMBEDDING_DIMENSION
"
```
**Expected:** `vector_provider = pgvector`, `embed_model` is the Gemini model, `dims = 768`,
`acl_env = false`, all three asserts pass. **This is the first moment in the repo's history that Cognee is
correctly configured**, and the three asserts are precisely item 152's two config bugs plus the ACL raise.

### Step 8 — `check_cognee` health probe (item 140's observable half)

**Inbound:** step 7 (there must be a typed config to probe), change 0 (the probe registry pattern).

**Correction to `dispositions.md` 198.2, which said graphiti is unprobed:** `check_graphiti` **already
exists** at `src/app/middleware/health_check.py:83-90` and is already registered in the probe list at `:98`.
What is missing is `check_cognee`. Separately, the **second** health surface,
`src/app/features/health/service.py`, probes postgres/redis/mongo/neo4j/celery/memory/disk and has
**neither** graphiti nor cognee — so a probe must be added in *both* places or the two surfaces disagree.

The probe must distinguish three states, because a boolean cannot express Cognee's failure mode:
`degraded("cognee","not configured")` when the typed config is absent; `fail` when the config is present but
its stores are unreachable; `ok` otherwise. It should also surface the **APOC/GDS** result, since that is the
silent-`cognify()` precondition and a probe is the only way it is ever observed — a cheap
`SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' RETURN count(*)` against the existing
`app.state.neo4j_driver`, reported as a named sub-field rather than failing the whole probe.

This step is what makes every later step observable, and it is the acceptance test for step 1's finding 1. It
lands **before** the write path deliberately (Ordering constraint 3): `lifespan.py:220-223` already sets
`app.state.graphiti = None` and continues, so this repo's established behaviour is to degrade in silence.

**Proof:**
```bash
uv run uvicorn app.main:app --port 8099 &   # then:
curl -s localhost:8099/health | jq '.data.checks | keys'          # expect: includes "cognee"
curl -s localhost:8099/health | jq '.data.checks.cognee'          # expect: status ok|degraded + apoc/gds sub-field
```
**Expected:** a `cognee` key exists on both health surfaces; with step 7 applied it reports `ok`; with
`COGNEE_*` unset it reports `degraded`, **not** `ok` and **not** a 500. Assert the degraded case too — a probe
that cannot go degraded is not a probe.

### Step 9 — Build the real memory service (item 174 — the core; Trap3 — the binding constraint)

**Inbound:** steps 6–8. This is the first Cognee **call site** in the repo's history.

**The ground truth this step exists to change:** `rg -n "cognify|cognee\.add|cognee\.search" src/` returns
**zero hits**. Nothing has ever been ingested into Cognee. `CogneeStore` is a stub whose five overrides return
`None`/`[]`, and only `setup_cognee` is ever called (`lifespan.py:206`).

Add `CogneeMemoryService` in `src/app/shared/langchain_layer/agents/memory/` — a class held on `app.state` per
the project convention that *"shared clients live in lifespan and are read from `connection.app.state`"*
(`openspec/config.yaml:6-35`). It owns exactly four operations, and the **write shape is where Trap3 is
honoured**:

| Operation | Call | Why this shape |
|---|---|---|
| `remember_report(...)` | `cognee.remember(entry, dataset_name=..., session_id=thread_id, self_improvement=False)` | **Session mode never calls `cognify()`** — `remember.py:895-900` appends to the session cache and returns. `self_improvement=False` suppresses the detached `asyncio.create_task` bridge (`remember.py:~885-890`) whose failure Cognee logs as "non-fatal" |
| `remember_trace(...)` / feedback | typed `QAEntry` / `TraceEntry` / `FeedbackEntry` | These **require** a `session_id` — `remember.py:274-276` raises `session_id is required for typed memory entries`. This is the API fact the ADR's boundary rests on |
| `recall(...)` | `cognee.recall(query_text=..., datasets=[...], session_id=..., top_k=...)` | With `session_id` and no `datasets`/`query_type` it searches the session cache **only** and falls through to the graph on no match (`recall.py:338-348`) — the cheap prefetch path |
| `consolidate(...)` | `cognee.improve(dataset=..., session_ids=[...])` | The nightly bridge; called only by step 11, never from a request |

**Trap3 (`todo.md:485`) restated correctly, and this is the single highest-value design decision in the
change.** Trap3 says `cognify()` is a full rebuild, so batch `add()` and defer `cognify()` to a nightly beat.
Verified: `remember()` in **permanent** mode (no `session_id`) is `add()` → `cognify(run_in_background=False)`
→ `improve()`, all awaited, at `.venv/.../remember/remember.py:915-944`, with `self_improvement=True` by
default (`:610`). So the repo's existing `store_final_report` (`cognee_client.py:150-151`) would perform **one
full graph rebuild plus two enrichment passes, synchronously, per approved report** — because it awaits
`remember()` and *then* awaits `improve()` again. Trap3's intent is honoured, but with the library's **own**
mechanism rather than hand-rolled batching: session-mode writes + an explicit nightly `improve()`. Because
there is no `cognify` call site yet, this costs nothing now and would be a rewrite later.

Three further defects to fix while building, all in code the service replaces:
- **Result conversion.** `recall()` returns `list[RecallResponse]`, a **discriminated union of Pydantic
  models** (`cognee/modules/recall/types/RecallResponse.py:26-29`, discriminator `source`). The existing
  `[dict(r) for r in results]` (`cognee_client.py:259`) is a shallow `dict()` that leaves nested models as
  objects — not JSON-safe. Use `model_dump()` and **preserve the `source` field**, which is how a caller tells
  a session hit from a graph hit.
- **Dataset naming.** Built by bare interpolation today (`cognee_client.py:140,189,238`). With ACLs off
  (NG6) the dataset name is the **only** tenant boundary, so it becomes one validated helper, not three
  f-strings.
- **Three failure idioms in one layer** — `store_final_report` re-raises (`:159`), `search_episodic_memory`
  swallows to `[]` (`:257`), `write_final_report_to_memory` collects error strings (`write_final_report.py:156-161`).
  Settle on one per `RESULT-PATTERN.md` / `EXCEPTION-RULES.md`; keep `e.add_note()` before re-raise, which is
  already the house style here (`cognee_client.py:251`).

**Proof** — two tiers, because no end-to-end path exists (`build_saul_graph` has no caller):
```bash
# Tier 1, in CI: service-level tests against a faked cognee module (monkeypatched)
uv run pytest tests/ -k cognee_memory -q 2>&1 | tail -5
```
Assertions that make "working" concrete: (a) `remember` is called with a non-`None` `session_id` and
`self_improvement=False` — **the machine-checkable form of Trap3**; (b) `cognify` is **never** called from any
request-path method; (c) `improve` is called **only** by `consolidate`; (d) a `recall` result keeps its
`source` discriminator; (e) two different `user_id`s produce different dataset names.
```bash
# Tier 2, run once by hand against real Neo4j + Postgres — the only proof Cognee actually works
uv run python scripts/cognee_roundtrip.py --user-id probe --thread-id t1
```
**"Working" is defined as:** `remember()` returns a `RememberResult` with no `error`; `recall()` with the same
`session_id` returns ≥1 result whose `source == "session"`; then `consolidate()` completes and a subsequent
`recall()` **without** `session_id` returns ≥1 result with `source` in `{graph, graph_context}`. That last
transition — session → permanent graph — is the **only** observable evidence that `cognify()`/`improve()` did
anything, and therefore the only test that catches a missing APOC/GDS (item 140's silent failure). Record the
tier-2 output in the change's `review.md`; it is not automatable until real infrastructure is in CI.

### Step 10 — Retarget the `persist_memory` node onto the service

**Inbound:** step 9. Independently committable; the node is unreachable either way (NG10).

`make_persist_memory_node` (`agent_saul/nodes.py:772-814`) takes `_cognee_client` — **underscore-prefixed,
i.e. unused** — and writes nothing: it appends two synthetic ref-key strings to `long_term_refs` and returns
`WorkflowStatus.COMPLETED`. It already carries the error code `COGNEE_WRITE_FAILED` (`:802`), so the seam was
designed for exactly this and never filled. Wire it to `CogneeMemoryService.remember_report`, gate on
`human_approved`, and **keep the existing fail-open shape**: memory write failure must not fail the workflow,
because the legal analysis is already complete. That matches the archived house rule that graph nodes return
fallback state rather than crash.

**Proof:** `uv run pytest tests/ -k persist_memory -q 2>&1 | tail -5` — asserting a write occurs when
`human_approved=True`; **no** write when it is `False` (the old `write_final_report.py:17` gave unapproved
reports `trust_score=0.3` and wrote them anyway — we do not); and that an exception from the service yields
`status=COMPLETED` with one `COGNEE_WRITE_FAILED` entry in `errors`, not a raised exception.
Then `uv run ruff check src/ 2>&1 | tail -2` → 123, `uv run ty check src/ 2>&1 | tail -3` → ≤46.

### Step 11 — Nightly Celery beat consolidation (Trap3's deferral half)

**Inbound:** step 9; **change 0** for the `tasks/__init__.py:6-9,18-20` edit — without it, adding a module to
`celery.py` `include` cannot be proven because the worker dies at import re-exporting deleted reconciliation
helpers. Also step 1 finding 1: **if APOC/GDS is absent this step is inoperable** and change 4 ships write-only.

Add `src/tasks/memory_consolidation_tasks.py` with a real `@celery_app.task` decorator, register the module in
`connections/celery.py:191-196` `include` (today 4 entries: `auth_email`, `example`, `search`, `billing`), and
add a `beat_schedule` entry (today 4 billing entries, `:259-276`). **Naming caution:** the existing
`"billing-reconciliation-daily"` → `billing.reconciliation` (`:272-275`) is billing *payment* reconciliation —
a live, unrelated subsystem sharing only the word. Do not collide with it; name this `memory.consolidation`.

This is where the deleted subsystem's *shape* is legitimately reused: `src/tasks/billing_tasks.py:71`
`_renewal_job` / `:253` `_reconciliation_job` is the working per-tenant batch-orchestration pattern in this
repo (NG5 notes the shape is re-derivable from there — this is the step that re-derives it).

**Proof:**
```bash
uv run celery -A app.connections.celery:celery_app inspect registered 2>&1 | rg "memory.consolidation"
uv run python -c "
from app.connections.celery import celery_app as c
assert 'memory.consolidation' in c.tasks, sorted(k for k in c.tasks if not k.startswith('celery.'))
assert len(c.conf.beat_schedule) == 5, c.conf.beat_schedule.keys()
print('registered + scheduled')
"
```
**Expected:** the task appears in `inspect registered` — the check the deleted decay task would have failed,
since it had **no** `@celery_app.task` decorator at all — and `beat_schedule` has **5** entries, not 4.

### Step 12 — One read seam: the post-`qna` memory prefetch

**Inbound:** step 9. Harvests requirements 1–3 of the superseded change's
`saul-memory-prefetch-and-retrieval`; requirement 4 (tool exposure) goes to change 3 (NG11).

Without a read seam, change 4 writes memory nobody reads. Add the prefetch node after `qna`, Cognee-first with
a small Graphiti supplement, deeper retrieval branching only on `risk_analysis`/`compliance`, and fail-open.
Two useful facts: `memory_pipeline.py:213,220` **already branches on exactly those task names**, and
`:258-260` is already the fail-open pattern — so this step is largely a relocation of logic that step 13
deletes, which is why 12 precedes 13.

**Proof:** `uv run pytest tests/ -k memory_prefetch -q 2>&1 | tail -5` — asserting Cognee is queried before
Graphiti; that a Cognee failure returns partial context rather than raising; and that a non-`risk_analysis`,
non-`compliance` task performs **no** deep retrieval. Node reachability is **not** proven here (NG10).

### Step 13 — Harvest, then retire `write_final_report.py` and `memory_pipeline.py` (D4)

**Inbound:** steps 9, 10, 12. **Last** by Ordering constraint 4: D4 deferred these two files' deletion to
change 4 precisely because they are *"the only existing reference for how Cognee writes are meant to work"*,
so they stay readable until their replacement exists and is tested.

Both are dead — `graphify affected` on `write_final_report_to_memory` (`write_final_report.py:65`) and
`build_agent_context` (`memory_pipeline.py:77`) returns only the package `__init__` re-export.

**Harvest first — two things have no other implementation in the repo:**

1. **`_filter_tool_messages` (`memory_pipeline.py:129-157`)** — strips `ToolMessage`s *and* pure-tool-call
   `AIMessage`s and substitutes one compact summary. Verified unique: `rg "trim_messages|filter_messages"
   src/` shows only `open_deep_search/{graph,utils}.py` (a parallel stack, out of scope per D7) and
   `langchain_layer/messages.py`. Move to `shared/langchain_layer/messages.py`, which is its correct home and
   is not a Graphiti concern.
2. **`_build_context_prefix` (`memory_pipeline.py:160-201`)** — the structured goal/return-format/warnings
   context block. Also unique. Move alongside; it belongs with `SystemPromptParts` adoption (change 3), so
   move it now and let change 3 adopt it.

**Do not harvest** the `trim_messages` step (`memory_pipeline.py:109-116`): it is a **duplicate** of
`langchain_layer/messages.py:40-52` `trim_by_token_count`, same `token_counter=len`, same `strategy="last"`.
Deleting it is pure subtraction.

Then delete both files, plus the now-orphaned Graphiti final-report writer
(`rag/graphiti/client.py:311-350` `write_final_report_episode`) whose only caller was
`write_final_report.py:110` — this is ADR Decision 4 in code. `MemoryPersistResult`
(`write_final_report.py:53-62`) goes with it. Note that `GraphitiService` / `CogneeService` in that file
(`:33-50`) are declared **inside `if TYPE_CHECKING:`** — type stubs with no runtime existence, which is
further evidence the router never ran.

**Paired edits, mandatory:** `rag/graphiti/__init__.py:47,59` re-exports `write_final_report_episode`;
`memory/__init__.py:3-9,23-39` re-exports the memory symbols. Missing either yields `ImportError` **at boot**,
not at test time — the `registry.py:41-46` eager-import class of failure that D6.1 warns about.

**Proof** — these modules have **zero test coverage**, so a green suite proves nothing; this is the substitute
evidence stack from § Ordering constraints:
```bash
graphify affected "write_final_report_to_memory"   # expect: no nodes
graphify affected "build_agent_context"            # expect: no nodes
graphify affected "write_final_report_episode"     # expect: no nodes
rg -n "write_final_report|memory_pipeline|build_agent_context|MemoryPersistResult" src/ tests/   # expect: ZERO
uv run python -c "import app.main; print('import graph intact')"   # the only check that catches boot ImportError
uv run ruff check src/ 2>&1 | tail -2      # expect: <=123, never higher
uv run ty check src/ 2>&1 | tail -3        # expect: <=46, never higher
uv run pytest -q 2>&1 | tail -5            # expect: >=55 passed (read the line, NOT $?)
timeout 180 openspec validate --all 2>&1 | tail -2   # expect: 5 failed, never 6
```

### Step 14 — Delete `CogneeStore` and collapse the legacy module-level functions

**Inbound:** step 13.

`CogneeStore` (`cognee_client.py:273-341`) is deleted, not implemented — ADR Decision 3 / NG9. Implementing
`CogneeStore.search` (`:304`) would put document semantic retrieval behind a LangGraph `BaseStore`, creating a
fourth retrieval path against D5.1; and `dispositions.md` already DROPped `app.state.vector_store` for this
exact "third retrieval path" reason. Its five `# type: ignore`d overrides (`:286,295,304,316,337`) go with it —
they are on `BaseStore` overrides, not on cognee calls, so this is not what
`openspec/specs/cognee-v1-api`'s no-`type: ignore` requirement targets, but removing them is a free win.

Then fold `store_final_report` (`:122`), `store_relationships` (`:174`) and `search_episodic_memory` (`:220`)
into `CogneeMemoryService`. All three have **zero production call sites** (only the `memory/__init__.py`
re-export, which codegraph misreports as a caller). `store_relationships` in particular is an ADR boundary
violation to retire, not port: it pushes the relationship graph as **text** into
`{user_id}.legal_relationships` — an entity-graph concern in the agent-memory owner — while relationships
already have a Graphiti writer (`rag/graphiti/client.py:257` `write_relationship_edge`).

**Proof:**
```bash
rg -n "CogneeStore|store_final_report|store_relationships|search_episodic_memory" src/ tests/  # expect: ZERO
graphify affected "CogneeStore"   # expect: no nodes
uv run python -c "import app.main; print('ok')"
uv run pytest -q 2>&1 | tail -5   # expect: >=55 passed
uv run ty check src/ 2>&1 | tail -3   # expect: strictly < 46 (five type:ignore overrides removed)
```
`ty` going **down** is the observable signal here, and it is the only one — deletion of untested code yields no
test signal (§ Ordering constraints).

### Step 15 — Dispose of the superseded change

**Inbound:** steps 3–5 (the replacement must exist and validate before the old one is retired). See
§ Disposition of the superseded change for the full mapping.

**Proof:** `ls openspec/changes/` shows `cognee-agent-memory` and no live `cognee-saul-memory-migration`;
`timeout 180 openspec validate --all 2>&1 | tail -2` shows the item count unchanged from step 3's and still
**5 failed**.

---

## Disposition of the superseded change

`openspec/changes/cognee-saul-memory-migration` is **superseded, not extended.** The reasons are structural,
not stylistic:

- It declares `schema: spec-driven` in `.openspec.yaml` while `openspec/config.yaml:1` says **`spec-gated`**.
  Extending it means migrating it to the current schema — adding the `review.md` the gate now requires and
  re-verifying every delta against today's formatting rules.
- It has **no `review.md`**, so under `spec-gated` its `tasks.md` is illegitimate by construction
  (`tasks` requires `specs` + `design` + **`review`**, `schema.yaml:394-396`).
- It is **0/15 tasks after 23 days** (`tasks.md`, every line `[ ]`; directory dated Jul 24 vs today
  2026-08-17). There is no in-flight work to preserve.
- Its central premise is now wrong in one specific way: `proposal.md:20-21` *defers* a Cognee replacement for
  reconciliation, whereas item 155 **removes reconciliation outright**. Editing that in place would leave a
  document whose Why no longer matches its What.

It is nonetheless **directionally right** — `design.md:3` already assumes the two-role split — so its content
is harvested rather than discarded.

### Its two spec deltas — harvested / dropped, requirement by requirement

**`specs/saul-cognee-final-report-write/spec.md` → HARVESTED in full**, into the new capability
`saul-agent-memory`. Its single requirement ("Saul `persist_memory` writes approved final reports to Cognee",
one scenario) is exactly ADR Decision 4 and step 10. Amended on harvest: the write becomes **session-scoped**
(`session_id` = `thread_id`) and `human_approved=False` writes **nothing** — the old design's implicit
`trust_score=0.3` low-trust write (`write_final_report.py:15-17`) is dropped.

**`specs/saul-memory-prefetch-and-retrieval/spec.md` → HARVESTED 3 of 4 requirements:**

| Requirement | Disposition |
|---|---|
| "Saul SHALL prefetch memory after qna" | **Harvested** → step 12 |
| "Prefetch SHALL be Cognee-first" (2 scenarios) | **Harvested** → step 12. Now grounded rather than asserted: `recall(session_id=…)` with no `datasets` searches the session cache first and falls through to the graph (`recall.py:338-348`) |
| "Memory retrieval failures fail open" | **Harvested** → step 12. Pattern already exists at `memory_pipeline.py:258-260` |
| "Deep memory retrieval is limited to selected reasoning nodes" (4 scenarios, incl. orchestrator-denied) | **DROPPED from change 4, reassigned to change 3** (NG11). Tool exposure is the registry's concern (D6.1, `langchain_layer/agents/tools/base.py:58`), and change 4 must not add a second tool-registration path |

### Its `tasks.md` — carried forward where still valid

Per `scout-memory.md` §5: tasks 1.1–1.4, 2.2–2.4, 3.1–3.3, 4.1–4.2 remain valid and reappear as steps 9–13.
**Task 2.1 ("post-`qna` prefetch node") needed rewriting** because it presumed a wired Saul graph — step 12
ships the node and NG10 records that it stays unreached. **Not carried forward:** the proposal's deferral of
`saul-cognee-maintenance-worker` / `saul-cognee-reconciliation`, which becomes NG1–NG5 (recorded gaps, per
D10) instead of a deferral promising future work.

**What is entirely absent from the old change and new here:** the config-correctness work (item 152 — embedder
and vector store, steps 6–7), the explicit ACL decision (NG6), the APOC/GDS precondition and probe (item 140,
steps 1 and 8), Trap3's write topology (step 9), and the `cognee-v1-api` spec repair (step 2).

### The directory: archive, do not delete

**Archive it** — `openspec/changes/archive/2026-08-17-cognee-saul-memory-migration/`. Rationale: D12 records
that archiving is what adds the `YYYY-MM-DD-` prefix, and the archive already holds the historical record this
refactor keeps citing (`scout-reconciliation.md` §7 cites
`archive/2026-07-22-noqa-exception-handling-migration/`). Deleting it would destroy the provenance of
`proposal.md:20-21`, which is the **primary citation** for D10's recorded gap — the repo's own admission that
Cognee has no curation/decay/dedup. That sentence is load-bearing evidence and must remain quotable.

Add a `superseded-by: cognee-agent-memory` line to the archived change's `.openspec.yaml`, and a
`Supersedes: cognee-saul-memory-migration` line in the new `proposal.md`, so the link is discoverable from
both ends. **Caveat to verify:** `openspec archive` may require the change to be complete (0/15 tasks may
block it); if it refuses, move the directory by hand and note it in `review.md` rather than ticking 15 tasks
that were never done.

---

## Openspec mapping

`openspec/specs/` holds 20 capabilities. `cognee-v1-api` is the **only** adjacent one — verified by listing
the directory: the other 19 are `datetime-utc-cleanup`, `llm-injection`, the eight `mcp-*`,
`mcp-{telemetry,testing}`, `noqa-documentation`, `outbox-helper-extraction`, `pattern-matching-standard`,
`session-required`, `settings-validation`, `test-mock-isolation`, `transactional-outbox`,
`typed-exception-handling`. None touches memory.

**Decision: do both — MODIFY the existing capability and ADD one new one.** They are different kinds of
contract and collapsing them would put behaviour into an API-surface spec:

### 1. `cognee-v1-api` — `## MODIFIED Requirements` (the call surface)

This spec is an **API-surface contract**: which Cognee functions we call. Change 4 changes two of them, so
`MODIFIED` is exactly right. Requires step 2's repair first, and each `MODIFIED` block must copy the **entire**
original requirement including all scenarios, header text matching whitespace-insensitively.

| Existing requirement (`spec.md`) | Action | Why |
|---|---|---|
| "Store content via remember" (`:1-10`) | **MODIFIED** | Still `remember`, but session-scoped with `self_improvement=False`. Its two scenarios name `store_final_report()`/`store_relationships()`, both deleted in step 14, so the scenarios must be restated against the service |
| "Process content via improve" (`:12-21`) | **MODIFIED** | **Its current requirement is actively wrong.** It mandates `improve()` after every `remember()` — but `remember()` already runs `improve()` itself in permanent mode (`remember.py:940-944`), so complying means running it twice. Restated: `improve()` is called **only** by the nightly consolidation |
| "Query memory via recall" (`:23-36`) | **MODIFIED** | Keeps `recall` + `auto_route=True` + empty-list-on-failure. Sharpened: results are converted with `model_dump()` and **retain the `source` discriminator** (`RecallResponse.py:26-29`); the current "converted to a dict" scenario (`:30-32`) is satisfied by a shallow `dict()` that leaves nested models unserialised |
| "No type ignore suppressions" (`:38-43`) | **unchanged** | Still honoured; step 14 removes five `# type: ignore`s (on `BaseStore` overrides, which this requirement never covered) |
| — | **ADDED** ×3 | Behaviourally-phrased config requirements — memory embeddings use the **same dimensionality as document embeddings**; memory embeddings are stored in the **application's managed Postgres, not local files**; multi-user access control is **explicitly disabled** so startup is deterministic. `scout-memory.md` §5 confirms embeddings/vector-store/ACL are *unspecified, not forbidden* by this spec |

Phrase all three behaviourally: `vector_db_provider="pgvector"` is a library choice and
`schema.yaml:109-111`'s test ("if the implementation can change without changing externally visible behaviour,
it does not belong in the spec") excludes it. "Embeddings are not written to the local filesystem" is
observable; "pgvector" is not.

### 2. `saul-agent-memory` — new capability, `## ADDED Requirements` + `## Purpose`

The **behavioural** contract, which does not belong in an API-surface spec: what gets remembered, when, at what
scope, what happens on failure, and when consolidation runs. Kebab-case, checked against the existing 20 (no
collision). Needs a `## Purpose` of ≥50 characters (NEW capabilities only).

Requirements: (a) only **human-approved** final reports are persisted; (b) memory is scoped to
`(user_id, thread_id)` and cross-tenant reads are impossible; (c) memory writes on the request path **never**
trigger a full graph rebuild — the machine-checkable form of Trap3; (d) consolidation into the permanent graph
runs on a **schedule**, not inline; (e) memory read failures **fail open** and the run continues; (f) prefetch
is Cognee-first with a bounded Graphiti supplement. Every requirement gets ≥1 scenario at **exactly four
hashtags**.

### 3. `## REMOVED Requirements` — none

Change 4 removes no *deployed* requirement. The retired final-report-to-Graphiti write was never specified in
`openspec/specs/`; it existed only in the superseded change's deltas, which are handled by archiving. Note
`REMOVED` would require both **Reason** and **Migration** — not applicable here.

### Acceptance criterion

**"No new failures beyond the existing 6 of 22"** — and change 4 improves on it: step 2 fixes
`spec/cognee-v1-api`, so the target is **5 failed**, with the count of items rising by one for the new change
directory. `timeout 180 openspec validate --all 2>&1 | tail -2` is the check. Never "validate --all passes".

---

## Conflicts surfaced

Contradictions found while planning. Each is resolved here or escalated; none is left implicit.

| # | Conflict | Resolution |
|---|---|---|
| **C1** | **The deployed spec mandates a redundant call.** `openspec/specs/cognee-v1-api/spec.md:12-21` requires `improve()` after every `remember()`. But `remember()` in permanent mode already runs `add()` → `cognify()` → `improve()` (`remember.py:915-944`, `self_improvement=True` by default at `:610`). Complying with the deployed spec means enriching twice per write | **The spec is wrong; change it** (step 2 repair, step 4 `MODIFIED`). This is the clearest case in the change of a spec that was written against an assumed API rather than the installed one |
| **C2** | **Three code sites disagree on where the final report goes** — `cognee_client.py:12-15` (Cognee), `write_final_report.py:8-13` (both), `cognee-saul-memory-migration/design.md:3` (Cognee primary) | Resolved by the ADR: **Cognee only**. Decided on each library's partition key (`session_id` required for typed entries, `remember.py:274-276`; Graphiti writes keyed `group_id=document_id`), not on preference |
| **C3** | **Trap3's prescribed mechanism is not the library's.** Trap3 (`todo.md:485`) says batch `add()` and defer `cognify()` to a nightly beat. Cognee 1.1.0 already provides the deferral as session mode + `improve(session_ids=…)` | Trap3's **intent** is honoured, its **mechanism** is replaced (step 9). Hand-rolled batching over `add()` would duplicate `SessionManager` |
| **C4** | **`dispositions.md` 198.2 says graphiti is unprobed.** It is probed — `middleware/health_check.py:83-90`, registered `:98` | Corrected in step 8. The real gap is `check_cognee` on that surface, and **both** graphiti and cognee on the *other* surface (`features/health/service.py`) |
| **C5** | **The brief's dimension claim.** Brief says Cognee hard-defaults to 3072. In 1.1.0 `embedding_dimensions` defaults to **`None`**, resolved in `model_post_init`; the source comment (`vector/embeddings/config.py:73-77`) says the 3072 hard-default was removed *because* it silently broke non-OpenAI embedders | The mismatch is real (default **model** is `openai/text-embedding-3-large`, resolving to 3072) but the **fix changes**: set the model, and dimensions follow. Recorded in the ADR and step 7 |
| **C6** | **The brief's ACL claim.** Brief cites neo4j's absence from `GRAPH_DBS_WITH_MULTI_USER_SUPPORT` (`context_global_variables.py:96`). That list feeds `is_multi_user_support_possible()`, **not** the gate on the write path; `backend_access_control_enabled()` (`:83-92`) gates on `supported_dataset_database_handlers`, which **does** contain a neo4j entry (`neo4j_aura_dev`) | Conclusion unchanged (ACLs off, explicitly) but the mechanism is worse than described: with the env var **unset**, the default branch **raises `EnvironmentError`** on a handler/provider mismatch rather than quietly disabling. Hence step 6 sets it explicitly |
| **C7** | **ADR Decision 5 vs. managed Postgres.** The ADR chose a Cognee-dedicated *database* to keep its self-managed tables away from `alembic --autogenerate` (`src/alembic/env.py:23-30` has **no** `include_object` filter). The app's Postgres is **Timescale Cloud**, where `CREATE DATABASE` is likely unavailable | Amended in-plan (§ ADR amendment): isolate by **schema** + add an `env.py` filter. Must be folded into the ADR when lifted, or the lifted ADR is wrong |
| **C8** | **`documents/service.py:596-601` opens a second Graphiti connection per ingestion call** while `app.state.graphiti` already exists (`lifespan.py:216`) — a boundary/lifecycle violation in the Graphiti half | **Not change 4's.** It is a documented convention violation ("shared clients live in lifespan", `openspec/config.yaml:6-35`) on the ingestion path → **change 1**. Recorded here so it is not lost |
| **C9** | **D8 says change 4 depends on change 1 for content**, but the artifact Cognee writes (the final report) comes from the Saul graph's `finalization` node, not the ingestion pipeline — and the Saul graph is unwired (change 3) | Both dependencies are real and neither is a blocker for *implementing* change 4. Change 4's Proofs are service-level by design (steps 9–12); end-to-end proof is explicitly deferred (NG10) rather than faked |

---

## Risks

Format follows `design.md`'s literal `[Risk] → Mitigation`.

- **[Cognee performs DDL at first write inside the production managed database]** → Cognee's own alembic runs
  lazily via `_ensure_migrations_run()` (`remember.py:41`) against Timescale Cloud, where the app's live data
  sits. Isolate to a dedicated **schema**, add an `include_object` filter to `src/alembic/env.py`, and run
  step 9's tier-2 round-trip against a **non-production** instance first. Precondition-check `CREATE SCHEMA`
  privilege in step 1 before writing any code.
- **[`vector_db_provider="pgvector"` may be unavailable on Timescale Cloud]** → Managed instances restrict
  extensions, and the repo's retrieval path already depends on **`pg_textsearch`** (`to_bm25query`,
  `search/repository.py:415-419`), a different extension from plain `pgvector`. Verified in step 1; documented
  fallback is `lancedb` on a **mounted persistent volume** for memory only, never for document retrieval.
- **[`cognify()` fails silently without APOC + GDS]** → Item 140. There is **no Neo4j in
  `docker-compose.yml`**, so the plugins cannot be guaranteed by the repo. Mitigated by a documented
  precondition (step 1), a health probe reporting APOC/GDS as a named sub-field (step 8), and step 9's tier-2
  session→graph transition test, which is the **only** check that actually detects the silent failure.
- **[Dataset naming is the sole tenant boundary]** → With ACLs unavailable (NG6), a bug in dataset-name
  construction is a cross-tenant memory leak. Mitigated by one validated helper (step 9) replacing three
  f-strings (`cognee_client.py:140,189,238`), plus a test asserting two `user_id`s never collide.
- **[Unbounded memory growth]** → NG1–NG5. Cognee accumulates with no decay, curation or dedup; the repo's own
  proposal concedes it (`cognee-saul-memory-migration/proposal.md:20-21`). **Not mitigated — accepted and
  recorded** per D10. The one cheap safeguard worth adding: a size/count metric on the consolidation task
  (step 11) so growth is *observable* before it is a problem, since no alarm exists otherwise.
- **[Change 4 cannot be proven end-to-end]** → `build_saul_graph` (`agent_saul/graph.py:86`) has no caller, so
  `persist_memory` never runs. Mitigated by the two-tier proof (step 9) and stated as NG10. **The residual
  risk is real:** a wiring defect between the node and the service would not be caught until change 3 wires
  the graph.
- **[Deletions produce no test signal]** → The reconciliation and cognee modules have **zero** coverage. The
  substitute evidence stack is specified in § Ordering constraints; `uv run python -c "import app.main"` is
  the load-bearing one, because `ImportError` at boot is the failure mode unit tests structurally cannot see.
- **[A green pytest run exits 1]** → `--cov-fail-under=80` against 18.38%. Every Proof compares the **summary
  line**, never `$?`. A CI gate wired to the exit code would fail every step in this plan.
- **[Two graph databases on one Neo4j instance]** → Cognee's permanent graph and Graphiti's entity graph
  cannot reference each other's nodes; joining a run to the clauses it analysed is an application-level join
  on `doc_id`. Accepted (ADR Consequences). The operational hazard is `cognee.prune()`, which must **never** be
  called against the shared instance — worth an explicit lint/grep guard.

---

## Fog

Open questions, each with what would close it. Nothing below is guessed at in the plan above.

- **F1 — APOC + GDS on the target Neo4j.** Unknown, and it decides whether step 11 is operable at all. No
  Neo4j service exists in `docker-compose.yml`, so this cannot be answered from the repo. **Closes with:**
  step 1's `SHOW PROCEDURES` queries against the real instance.
- **F2 — Whether `remember`/`improve`/`recall` have *ever* succeeded against this Neo4j + Postgres
  config.** No call sites, no tests, no dataset artifact. Step 9 may be rebuilding something that never ran,
  which is why the plan defines "working" by an observable transition rather than by parity with the old code.
  **Closes with:** step 9's tier-2 round-trip.
- **F3 — `cognee.improve()` idempotency on an already-improved dataset.** Whether it re-embeds everything
  matters directly for the nightly beat's cost, and `improve()`'s docstring describes a five-stage pipeline
  without stating incrementality. **Closes with:** reading `api/v1/improve/improve.py` in depth, or two timed
  consecutive runs on a fixed dataset.
- **F4 — Whether the unwired Saul graph is intentional or a regression.** `lifespan.py:234-247` and `:294-305`
  are commented-out blocks with no dated note. Treated as a **regression** (per D5.2's logic — the identical
  defect on the documents router is live), which is overturnable. If intentional, change 4's read seam
  (step 12) is speculative work. **Closes only by asking the user.**
- **F5 — Which Cognee tables land where, and their size.** Needed to size the schema and back-ups, and to
  judge the `alembic --autogenerate` blast radius concretely rather than in principle. **Closes with:**
  `\dn` + `\dt cognee.*` after step 9's tier-2 run.
- **F6 — Whether Cognee's pgvector provider honours a non-default Postgres schema.** The whole amended
  isolation strategy rests on it; if not, the `env.py` `include_object` filter carries it alone.
  **Closes with:** reading `.venv/.../databases/vector/pgvector/` for schema handling, then observing
  `search_path`/table placement after tier-2.
- **F7 — Whether `openspec archive` accepts a 0/15-task change.** Decides whether step 15 is a CLI call or a
  manual `git mv`. **Closes with:** `openspec archive cognee-saul-memory-migration --help` and one dry run.
- **F8 — Live-vs-orphan status of `entities`/`relationships`/`events`/`memory_versions`.** Assigned to change
  0 but consumed by change 4 (step 1 finding 4). Absent from *this repo's* migrations is proven; absent from a
  running Postgres is not — Cognee's own alembic (`.venv/.../cognee/alembic/`) is a candidate creator.
  **Closes with:** `\dt` against the dev database.
- **F9 — Item 179's deferred half.** Cognee needs no Redis (step 1 finding 3 answers the narrow question), but
  whether `redisvl`/`langcache` should back the *app's* caching is untouched research. **Deliberately left
  open** as NG8, not fog to be closed inside change 4.

---

**Orchestrator correction (2026-08-17):** 9 occurrence(s) of the vendor name "VectorChord" were
replaced with `pg_textsearch` throughout this file. The BM25 index access method and `to_bm25query()` come from
**`pg_textsearch`** (Timescale/TigerData), verified present at version 1.3.0 on the live server. `vchord` 0.5.3
is available on that server but unused, and `vchord_bm25` is **not available at all** — so the earlier
attribution was not merely a naming slip, it named an extension the deployment cannot install. See
`docs/relay/findings-database.md` §3.
