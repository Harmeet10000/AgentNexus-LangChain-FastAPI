# ADR — The Graphiti / Cognee role boundary

## Status

**Accepted** (2026-08-17).

Supersedes the boundary statements in `openspec/changes/cognee-saul-memory-migration/design.md:3-7` — a change that
is itself superseded by `cognee-agent-memory` and archived, not deleted — and corrects the three disagreeing in-code
statements listed under Context.

Scope note: this ADR records a boundary that **outlives the change**. Others will build on it, and it is the reason
three code sites currently disagree, so it is recorded here rather than inside `design.md`.

## Context

D2 locked that **both Graphiti and Cognee survive with two distinct roles**, and deliberately left the boundary
unspecified. Three places in the repository state a boundary and they do not agree:

| Site | What it says about the final report |
|---|---|
| `cognee_client.py:12-15` | *"Graphiti → structural legal knowledge graph; Cognee → episodic + procedural memory"* — the report is Cognee's |
| `write_final_report.py:8-13`, `:100-154` | a "Memory Router" that writes the report to **both** — Graphiti as a high-trust episode *and* Cognee as episodic memory |
| `openspec/changes/cognee-saul-memory-migration/design.md:3` | *"Cognee the primary memory/recall layer … Graphiti remains the structural knowledge-base layer"* |

A third question rides on it: **who owns semantic/vector retrieval** — Graphiti, Cognee, or the application's own
`pg_textsearch` path. All three physically hold a vector index. D5.1 has since committed document retrieval to raw
asyncpg + `pg_textsearch` (`features/search/repository.py:415-419` BM25 via `to_bm25query`, RRF at `fusion.py:28`),
which removes the application from contention as a *candidate* and makes it the *incumbent*.

The starting condition matters: **Cognee has never executed here.** `cognify` has zero call sites in `src/`;
`CogneeStore` is a stub; the only Cognee symbol ever called is `setup_cognee` (`lifespan.py:206`), which configures
an LLM, a graph store and a relational store but neither an embedder nor a vector store. `findings-database.md` §7
confirms Cognee's own alembic has never run against this database, and none of `entities`, `relationships`,
`events`, `memory_versions` exists in any form. So this boundary is being chosen **before any data exists** — the
cheapest moment, and the last free one.

### Four facts verified in the installed libraries. These are constraints, not preferences.

1. **Cognee's typed agent-memory entries cannot be written without a session.**
   `.venv/.../cognee/api/v1/remember/remember.py:274-276` raises `session_id is required for typed memory entries`
   for `MemoryEntry` / `QAEntry` / `TraceEntry` / `FeedbackEntry`. Cognee's agent-memory surface is therefore
   **intrinsically scoped to a run/thread axis**.
2. **Graphiti's partition key in this repository is the document.** Every Graphiti write uses
   `group_id=document_id` / `state.doc_id` — `documents/service.py:544`, `graphiti_verifier.py:56`,
   `ingestion_kb/nodes.py:384,397`. Graphiti is therefore **scoped to a document/entity axis**.
3. **`cognee.remember()` in permanent mode is `add()` + `cognify()` + `improve()`, synchronously.**
   `remember.py:915-944`: `_run()` awaits `add(...)`, then `cognify(...)` with `run_in_background=False`, then —
   because `self_improvement` defaults to `True` (`:610`) — `improve(...)`. `cognify()` is a full graph rebuild over
   the dataset (Trap3, `todo.md:485`). The repository's `store_final_report` (`cognee_client.py:150-151`) then
   awaits `cognee.improve(dataset=...)` a **second** time. The existing reference implementation therefore performs
   one full graph rebuild plus **two** enrichment passes per approved report, inline in a graph node.
4. **Graphiti is the only one of the two with bitemporal invalidation and dedup.**
   `graphiti_core/edges.py:271,277` (`expired_at`, `invalid_at`); auto-invalidation via `resolve_extracted_edges`
   returning `(resolved, invalidated, new)` (`graphiti_core/graphiti.py:648-678`); dedup machinery under
   `graphiti_core/utils/maintenance/`. Cognee 1.1.0 has no edge-invalidation primitive; its `forget`
   (`api/v1/forget/forget.py:16`) is deletion by identifier, not scored supersession.

**Facts 1 and 2 are the load-bearing pair: each library's own partition key already decides which axis it can
serve.** That is what makes this decision an API fact rather than a matter of taste.

## Decision

**Split by each library's own partition key. Three owners, one owner per axis.**

1. **Graphiti owns the document/entity axis** — clauses, parties, obligations, precedent chains,
   `REFERENCES_CLAUSE` edges, and **all** bitemporal validity (`valid_at` / `invalid_at` / `expired_at`).
   Partition key: `group_id=document_id`. It keeps its live writers (`graphiti_verifier.py:39-56`,
   `ingestion_kb/nodes.py:384,397`) and its live readers (`search/service.py:264`, `documents/service.py:753`).

2. **Cognee owns the agent-run/thread axis, and nothing else** — approved final reports, QA pairs, agent traces, and
   feedback about a Saul run. Partition key: **`session_id` = the Saul `thread_id`**, inside a dataset namespaced
   `{user_id}.legal_reports`. Cognee's request-path writes are **session-mode only**
   (`remember(..., session_id=thread_id, self_improvement=False)`); permanent-graph consolidation is a **scheduled
   Celery beat** calling `improve(dataset=..., session_ids=[...])`.

3. **The application owns semantic/vector retrieval of documents** — raw asyncpg + `pg_textsearch`, per D5.1. Each
   memory library keeps a private internal index that exists solely to serve its own read API
   (`graphiti.search(...)`, `cognee.recall(...)`). **No application code queries a memory library's vector index
   directly, and no memory library is a document-retrieval path.** Concretely, `CogneeStore(BaseStore)` is
   **deleted rather than implemented**, because implementing `CogneeStore.search` (`cognee_client.py:304`) would
   create a fourth retrieval path behind a LangGraph interface.

4. **The final report goes to Cognee only.** The Graphiti final-report write (`rag/graphiti/client.py:311-350`
   `write_final_report_episode`) is retired along with its only caller (`write_final_report.py:110`). Postgres
   remains the report *system of record* (`features/billing/models/report.py:42` `Report`) and the audit trail
   (`billing/models/audit.py:48` `AuditLog`); Cognee holds a recall-optimised copy, **not** the record.

5. **Cognee's own stores are configured explicitly, never defaulted.** The embedder is pinned to the repository's
   Gemini model at `EMBEDDING_DIMENSION` (768, `settings.py:212`); the vector store is `pgvector` against the
   application's managed Postgres, isolated by **Postgres schema** (see the amendment below);
   `ENABLE_BACKEND_ACCESS_CONTROL=false` is set **explicitly**, before the first Cognee config call. Tenant
   isolation is enforced by the application through the dataset name and session id, **not** by Cognee ACLs, and the
   dataset name is produced by one validated helper rather than three f-strings
   (`cognee_client.py:140,189,238`).

6. **Cognee receives a usable database connection.** `cognee_client.py:111` today reads `settings.POSTGRES_URL`
   **raw**, and that value carries **no password** (`findings-database.md` §2), bypassing
   `connections/postgres.py:30-71` `get_database_url()`, which is the one place that injects the credential,
   rewrites the scheme and strips the transport parameters asyncpg rejects. Change 0 owns the single-accessor fix;
   this ADR's requirement is only that Cognee is handed a connection that authenticates on first use.

### Amendment folded in — the application's Postgres is Timescale **Cloud**

`.env.development` `POSTGRES_URL` points at `*.tsdb.cloud.timescale.com:39662/tsdb` — a **managed** instance, not
the `timescale` service in `docker-compose.yml` (`findings-database.md` §1). This invalidates the *mechanism*
originally chosen for Decision 5, not its goal:

- **A Cognee-dedicated Postgres *database* is not available.** Managed services commonly forbid `CREATE DATABASE`;
  Timescale Cloud provisions `tsdb`.
- **Amended Decision 5:** isolate Cognee by **Postgres schema** (`cognee` inside `tsdb`), **and additionally** add an
  `include_object` / `include_name` filter to `src/alembic/env.py`. Rationale: Alembic's `include_schemas` defaults
  to `False`, so a non-default schema is already invisible to `--autogenerate` reflection; the explicit filter is the
  belt-and-braces that survives someone later flipping `include_schemas=True`. `src/alembic/env.py:23-30` sets
  `target_metadata = Base.metadata` with **no** filter today, so without this the next `--autogenerate` emits
  `op.drop_table(...)` for every Cognee table.
- **`vector_db_provider="pgvector"` now depends on a managed extension allow-list.** Cognee's pgvector provider
  issues `CREATE EXTENSION`-class DDL and creates its own tables via its own alembic
  (`.venv/.../cognee/alembic/`, invoked lazily by `_ensure_migrations_run()` at `remember.py:41`). Whether the
  application role may create a schema and the extension is a **precondition to verify before any code lands**, not
  an assumption. `findings-database.md` §3 is favourable on availability (`vector` is installed; `pg_textsearch`
  1.3.0 is available), but privilege is unverified.
- **Recorded fallback, so it is a decision rather than a scramble:** if the precondition fails, use
  `vector_db_provider="lancedb"` on a **mounted persistent volume**, accepting local-file vectors for memory recall
  only — never for document retrieval, which D5.1 keeps on `pg_textsearch` — and revisit when a self-managed
  Postgres exists.

## Rationale / Alternatives

**Why the report goes to Cognee only.** Fact 3 is decisive on cost: routing the report to both, inside
`persist_memory`, means one Graphiti `add_episode` (itself an LLM-driven entity-extraction pass,
`graphiti.py:648-678`) *plus* a full Cognee `cognify()` rebuild *plus* two `improve()` passes, all awaited inside a
graph node, for a single artifact. Fact 1 then decides ownership **on the merits rather than on cost**: a final
report is a statement about *a run* — who asked, what was concluded, whether a human approved it — and Cognee's
typed entries are the only structure in either library that models that, at the price of requiring a session id.
Facts 2 and 4 decide the other side: an approved report contains obligations whose validity changes over time, and
only Graphiti can express supersession — but it should learn those obligations from the **clause-extraction path it
already owns**, keyed by document, not by re-ingesting a prose summary keyed by user.

**Why session mode rather than Trap3's "batch `add()`, defer `cognify()`".** Trap3's intent is right and its
mechanism is unnecessary, because Cognee 1.1.0 already ships the deferral: `remember(session_id=...)` writes to the
session cache only (`remember.py:895-900`) and never touches `cognify`, and `improve(dataset, session_ids=[...])`
(`api/v1/improve/improve.py:36`) is the documented consolidation that bridges session Q&A into the permanent graph,
applies feedback weights and rebuilds triplet embeddings. Hand-rolled batching would duplicate that. We additionally
pass `self_improvement=False`, because with a session id **and** `self_improvement=True` Cognee fires the bridge as
an unstructured `asyncio.create_task` (`remember.py:~885-890`) inside the caller's event loop — fire-and-forget work
whose failure is logged as "non-fatal" and whose lifetime is not tied to anything we control. **An explicit
scheduled job is observable; a detached task is not.** Because no `cognify` call site exists yet, honouring Trap3
costs nothing now and would be a rewrite later.

| Alternative | Pros | Cons | Why rejected |
|---|---|---|---|
| **A. Dual-write the report** (status quo of `write_final_report.py`) | Both graphs see the highest-signal artifact; `graphiti_verifier.py:70` and `documents/service.py:753` already read `group_ids=[user_id, …]`, so the Graphiti copy *would* be found | Two graph rebuilds per report; two failure modes in one node; the report becomes a second, prose-shaped source of entities competing with the clause extractor for the same facts, with no dedup between them | Fact 3's cost, plus fact 4: duplicate party/obligation nodes arriving from two extractors is exactly the damage Trap1 (`todo.md:483`) says is cheap now and unfixable later |
| **B. Cognee primary for everything, Graphiti retired** | One memory system; matches a literal reading of item 155 ("replace … with cognee entirely") | Loses bitemporal invalidation and dedup outright (fact 4) with no Cognee equivalent; Graphiti has four live call sites and is the only writer of `REFERENCES_CLAUSE postgres_chunk_id=` (`graphiti_verifier.py:39-56`) | Contradicts D2, which is locked |
| **C. Graphiti primary, Cognee deleted** | Removes an unused dependency and all its config risk; smallest possible change | Contradicts D2; Graphiti has no typed run-memory model (no actor/reason/feedback fields on episodes), so agent-run memory would be hand-built on top of episodes | Contradicts D2, which is locked |
| **D. Cognee owns semantic document retrieval via `CogneeStore`** | One `BaseStore` interface for LangGraph; free hybrid retrieval | Directly contradicts D5.1, which commits document retrieval to asyncpg + `pg_textsearch` where BM25 and RRF already work; adds a fourth retrieval path; `CogneeStore` is a stub with five `# type: ignore`d overrides | Contradicts D5.1, and `app.state.vector_store` was already dropped for the identical "third retrieval path" reason |
| **E. Enable Cognee ACLs for tenant isolation instead of app-level dataset naming** | Isolation enforced by the library; defence in depth | With `graph_database_provider="neo4j"` and the handler left at its default `"ladybug"` (`databases/graph/config.py:45,59`, whose `fill_derived` at `:77-79` only remaps kuzu and postgres), `multi_user_support_possible()` reaches `context_global_variables.py:~60-77` and **raises `EnvironmentError`**, because `supported_dataset_database_handlers["ladybug"]["handler_provider"] == "ladybug" != "neo4j"`. With the env var unset this is the **default** path (`:88-92`) | Rejected as **unavailable**, not as undesirable. The only ACL-capable neo4j handler is `neo4j_aura_dev` (`supported_dataset_database_handlers.py:18-21`), which is Aura-specific and untested here |
| **F. Give Cognee the application's own Postgres database** (status quo of `setup_cognee`) | One database to operate; already coded (`cognee_client.py:92-102` passes `settings.POSTGRES_DB_NAME`) | Cognee creates and migrates its own tables, and `src/alembic/env.py:23-30` has **no** `include_object` filter, so the next `--autogenerate` emits `op.drop_table(...)` for every Cognee table | Rejected. A dedicated *database* is unavailable on managed Postgres anyway (see the amendment); schema isolation plus an `env.py` filter achieves the goal |

### Two corrections recorded so they are not re-inherited

- **The 3072-dimension claim.** In Cognee 1.1.0 `embedding_dimensions` defaults to **`None`** and is resolved in
  `model_post_init`; the comment at `vector/embeddings/config.py:73-77` records that the 3072 hard-default was
  removed *because* it "silently broke every non-OpenAI … embedder". The mismatch is nonetheless real, because the
  default **model** is still `openai/text-embedding-3-large` (`:72`) and resolves to 3072. **The practical
  consequence changes:** setting `embedding_model` correctly is *sufficient*; setting `embedding_dimensions` is
  belt-and-braces rather than the fix.
- **The ACL claim.** `GRAPH_DBS_WITH_MULTI_USER_SUPPORT` (`context_global_variables.py:96`) is consumed by
  `is_multi_user_support_possible()`, **not** by the gate on the write path. `backend_access_control_enabled()`
  (`:83-92`) calls `multi_user_support_possible()` (`:34-81`), which gates on
  `supported_dataset_database_handlers` — and that dict **does** contain a neo4j entry. The conclusion is unchanged
  (ACLs off, explicitly) but the reason is a handler/provider mismatch that **raises**, not an absent backend that
  silently disables.

## Consequences

### Positive

- One owner per axis, and each owner's partition key enforces it **mechanically**: a thread-scoped write cannot land
  in the document graph, and a document-scoped write cannot land in run memory. The boundary is not a convention
  someone has to remember.
- The write path becomes cheap enough to sit in a graph node: a session-cache append with no `cognify()`.
- Trap3 is honoured **before the first `cognify` call site is ever written** — the cheapest possible moment, and the
  only one at which it is free.
- `openspec/specs/cognee-v1-api/spec.md` stops being aspirational. It currently mandates an `improve()` call after
  every `remember()`; this ADR is the reason to change that requirement rather than comply with it.

### Negative / accepted

- **Cognee memory grows without decay, curation, or dedup.** This is a **known consequence** of the accepted
  boundary, and it is mandated by **D10**, which drops item 170 outright. Cognee 1.1.0 has no decay primitive (its
  `forget` is deletion by identifier), no dedup on the run axis, and no version history; Graphiti's dedup and
  bitemporal invalidation apply only to its own document-axis nodes and cannot reach agent memory. The repository's
  own document already conceded it — `openspec/changes/cognee-saul-memory-migration/proposal.md:20-21`: *"Cognee v1.1
  has no built-in curation/decay/dedup"*. The five capabilities that die with the reconciliation deletion (scored
  decay, near-duplicate detection, edge-preserving merge, memory version history, per-tenant reconciliation
  orchestration) have **no replacement anywhere in the repository**; they are enumerated as NG1–NG5 in `design.md` §
  Goals / Non-Goals. **This ADR does not fix it; it names it.** Item 155's word "entirely" is honoured for
  **reconciliation removal**, never for **capability parity** — none of the five was ever observable behaviour (no
  task decorator, module absent from the Celery `include`, four tables that never existed), so what is lost is
  design work, not a regression a user could notice. The only safeguard added is a size/count metric on the
  consolidation job, so growth is observable before it is a problem.
- **After this change the `user_id` partition in Graphiti is empty.** `graphiti_verifier.py:70` and
  `documents/service.py:753` read `group_ids=[user_id, document_id]` / `[user_id, *doc_ids_filter]`; the `user_id`
  half was populated only by the retired final-report write, so the union degenerates to document scope. Left in
  place deliberately — it is the natural home for future user-level entity facts — and recorded here so it is not
  later mistaken for a bug. But it **is** now a filter that matches nothing.
- **Two graph databases, both on Neo4j, with no cross-links.** Cognee's permanent graph and Graphiti's entity graph
  share a Neo4j instance and cannot reference each other's nodes. Joining a run to the clauses it analysed is an
  application-level join on `doc_id`, not a graph traversal. Operational corollary: `cognee.prune()` must **never**
  be called against the shared instance.
- **Cognee's stores sit inside the same managed instance as production application data**, isolated only by schema,
  with DDL performed by a third-party library at first write. The blast radius of a Cognee migration bug is the
  production database. Mitigated by the `env.py` filter, the schema boundary, and running the first round-trip
  against a non-production instance — not eliminated.
- **A schema whose contents are owned by a third-party library's migrations and invisible to `src/alembic/`.** Its
  table set and size are unknown until the first round-trip runs.
- **No tenant isolation below the application layer.** With ACLs unavailable (alternative E), dataset naming and
  session ids are the **only** isolation. A bug in dataset-name construction is a cross-tenant memory leak with no
  second line of defence — which is why the name becomes one validated helper rather than the three string
  interpolations at `cognee_client.py:140,189,238`.
- **An operational precondition with a silent failure mode.** `cognify()` requires APOC + GDS on the target Neo4j
  (item 140, `todo.md:253`) or it **fails silently** — no exception, no data. There is no Neo4j service in
  `docker-compose.yml` (services are `rabbitmq`, `timescale`, `caddy`, `ai-service-1`), so the instance is
  externally managed and the repository cannot guarantee the plugins. This is why the health probe is a
  **requirement** of this change and not a nicety, and why the probe reports APOC/GDS as a named sub-field.
- **Scheduled consolidation has no infrastructure to run on yet.** `docker-compose.yml` contains **no worker and no
  beat service at all**, and `Makefile:52` starts a worker from a `celery_config` module that does not exist
  (`findings-deployment.md` §1–§2). This ADR's Decision 2 depends on a scheduled job; provisioning the process that
  executes it is an operational dependency of the decision, and until it exists the permanent-graph half of the
  boundary is registered but not running.
- **The boundary cannot be proven by running the product.** `build_saul_graph` (`agent_saul/graph.py:86`) has no
  caller, and D17 settled that the unwired graph was **deliberate and stays commented**. So `persist_memory` never
  executes, the read seam is speculative, and the evidence for this ADR is API-level and service-level only. A
  wiring defect between node and service surfaces only when change 3 wires the graph.
