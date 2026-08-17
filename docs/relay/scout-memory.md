# Scout — Memory / Knowledge-Graph Layer (Cognee + Graphiti)

Versions pinned from the lockfile and the installed venv:
- `cognee==1.1.0` — `uv.lock:934-935`, `pyproject.toml:111`, `.venv/lib/python3.12/site-packages/cognee-1.1.0.dist-info`
- `graphiti-core==0.29.1` (extra `google-genai`) — `uv.lock:2529-2530`, `pyproject.toml:62`

---

## 1. Cognee inventory

All in `src/app/shared/langchain_layer/agents/memory/cognee_client.py` (342 lines).

| Symbol | Line | State | Production call sites |
|---|---|---|---|
| module docstring ("Cognee does NOT replace Graphiti") | :1-23 | doc | — |
| `setup_cognee(settings)` | :58 | implemented (partial) | `src/app/lifecycle/lifespan.py:206` only |
| `store_final_report(...)` | :122 | implemented | **none** |
| `store_relationships(...)` | :174 | implemented | **none** |
| `search_episodic_memory(...)` | :220 | implemented | **none** |
| `CogneeStore(BaseStore)` | :273 | **stub** | **none** |
| `CogneeStore.put` | :286 | empty body — returns `None` implicitly | — |
| `CogneeStore.get` | :295 | `return None` | — |
| `CogneeStore.search` | :304 | `return []` | — |
| `CogneeStore.delete` | :316 | empty body | — |
| `CogneeStore.list_keys` | :324 | `return []` | — |
| `CogneeStore._make_key` | :332 | implemented (unused) | — |
| `CogneeStore._matches_filter` | :337 | implemented (unused) | — |

All five are re-exported from `src/app/shared/langchain_layer/agents/memory/__init__.py:3-9,23-39` — that re-export is the *only* thing codegraph reports as a "caller" for the four unused symbols.

`setup_cognee` returns a plain `dict[str, Any]` (`cognee_client.py:107-114`), stored at `app.state.cognee_config` (`lifespan.py:207`). There is no Cognee client object, no shutdown hook.

### cognee 1.1.0 API surface actually available
From `.venv/lib/python3.12/site-packages/cognee/__init__.py`:
- classic pipeline: `add`, `cognify`, `search`/`SearchType`, `delete`, `update`, `prune`, `datasets`, `run_custom_pipeline` (:21-30, :39)
- **memory API (what this repo uses)**: `remember`, `RememberResult`, `recall`, `improve`, `forget`, `serve`, `disconnect` (:48)
- typed memory entries: `MemoryEntry`, `QAEntry`, `TraceEntry`, `FeedbackEntry` (:49)
- `memify` (:24), `agent_memory` (:61), `session` + `SessionRecord`/`SessionModelUsage` (:36, :64)
- config setters: `set_llm_config`, `set_embedding_provider/_model/_dimensions/_endpoint/_api_key/_config`, `set_relational_db_config`, `set_graph_db_config`, `set_vector_db_config` — `.venv/.../cognee/api/v1/config/config.py:358,371,383,395,423,435,447,553,575,586`

`remember` is at `cognee/api/v1/remember/remember.py:593`; `recall` at `cognee/api/v1/recall/recall.py:314`; `improve` at `cognee/api/v1/improve/improve.py:36`. The three calls the repo makes exist in 1.1.0 — the repo is not calling a phantom API.

---

## 2. Graphiti inventory — live vs dead

`src/app/shared/rag/graphiti/` = **2049 lines** (confirmed): `client.py` 605, `subgraph.py` 325, `write_clause_episodes.py` 288, `schemas.py` 227, `memory_pipeline.py` 260, `write_final_report.py` 161, `registry.py` 122, `__init__.py` 61.

### 2a. LIVE at runtime

| Site | Reads / writes | Role |
|---|---|---|
| `lifespan.py:33,212-217` `setup_graphiti` + `setup_graphiti_indices` → `app.state.graphiti` | opens Neo4j-backed `Graphiti`, builds indices | infra |
| `lifespan.py:225-232` | logs Neo4j/Graphiti state inconsistency | infra |
| `lifespan.py:335` `close_graphiti` | shutdown | infra |
| `src/app/middleware/health_check.py:83-90,98` `check_graphiti` | reads `app.state.graphiti`, degraded if `None` | infra |
| `src/app/features/search/dependencies.py:40` | injects `getattr(app.state,"graphiti",None)` into search service | entity-graph read |
| `src/app/features/search/service.py:46,65,70,264` | passes `self.graphiti` into `build_retrieval_graph` | entity-graph read |
| `src/app/features/documents/service.py:38,596-601,622` | its **own second** `setup_graphiti`/`close_graphiti` per ingestion call | entity-graph write |
| `src/app/features/documents/service.py:65,673` → `graphiti_verifier.py:28` `write_and_verify_chunk` | `graphiti.add_episode(..., group_id=document_id)` writing `REFERENCES_CLAUSE postgres_chunk_id=` (`graphiti_verifier.py:39-56`) | entity-graph write |
| `src/app/features/documents/ingestion_graph.py:15,43,55,69,82` | threads `graphiti` into the document ingestion graph | entity-graph write |

### 2b. DEAD — zero production callers (verified)

`rg -n 'build_saul_graph|build_tool_registry|build_ingestion_graph|build_agent_context' src/ tests/` returns only definitions, `__init__` re-exports, docstrings, and the **commented-out** block `lifespan.py:234-247`.

- `build_tool_registry` — `rag/graphiti/registry.py:98`. Never called → the four Graphiti-consuming agent tools are unreachable:
  - `agents/tools/search_legal_precedents.py:38,45,94` → `search_for_precedent_chains`
  - `agents/tools/query_knowledge_graph.py:31,35,79` → `search_for_risk_context`
  - `agents/tools/precedent_tools.py:31,36,100` → `search_for_precedent_chains` + `Neo4jSubgraphConfig`
  - `agents/tools/get_obligation_chain.py:35,40,82` → `get_obligation_chain`
- `build_saul_graph` — `langgraph_layer/agent_saul/graph.py:86`. Never called → `app.state.saul_graph` is **never assigned**, yet `features/agent_saul/dependencies.py:40-41` returns `request.app.state.saul_graph` (AttributeError at request time).
- `build_ingestion_graph` — `langgraph_layer/ingestion_kb/graph.py:37`; its `graphiti_upsert` node (`graph.py:73` → `ingestion_kb/nodes.py:354` `make_graphiti_upsert_node`, and `nodes.py:757` `_graphiti_add_episode`) is dead. `lifespan.py:241-245` is the commented-out wiring.
- `build_agent_context` — `rag/graphiti/memory_pipeline.py:77`. Never called.
- `write_final_report_to_memory` — `rag/graphiti/write_final_report.py:65`. Never called.
- `rag/graphiti/subgraph.py` `expand_from_seeds`:133 / `get_obligation_chain`:178 / `detect_conflicts`:214 — reached only from `precedent_tools.py` (dead).
- Reconciliation: `build_reconciliation_graph` (`langgraph_layer/reconciliation/graph.py:24`) — `graphify affected` returns exactly one edge, `reconciliation/__init__.py:1`. **Zero external callers.** Area = 618 lines (`nodes.py` 442, `state.py` 78, `graph.py` 42, `prompts.py` 23, `__init__.py` 33).

`rag/graphiti/registry.py` lines 1-32 **are** a module docstring containing example lifespan wiring (including `app.state.saul_graph = build_saul_graph(... cognee_client=app.state.cognee_config ...)` at :25-31) — but the file continues to 122 lines with a real `ToolRegistry` BaseModel (:56) and `build_tool_registry` (:98).

---

## 3. Proposed role boundary

Grounded in what each installed version provides.

**Graphiti 0.29.1 provides** (verified in venv): bitemporal edges — `EntityEdge.expired_at` (`graphiti_core/edges.py:271`, "datetime of when the node was invalidated") and `invalid_at` (:277), both serialised at :352-354; LLM-driven edge invalidation via `resolve_extracted_edges` returning `(resolved, invalidated, new)` (`graphiti_core/graphiti.py:648-678, 917-924, 1740-1757`), telemetry `edge.invalidated_count` (:1204); dedup + community machinery under `graphiti_core/utils/maintenance/` (`dedup_helpers.py`, `node_operations.py`, `edge_operations.py`, `community_operations.py`, `combined_extraction.py`); `group_id` scoping; hybrid semantic+BM25+graph search.

**Cognee 1.1.0 provides**: dataset-scoped `remember`/`improve`/`recall`/`forget`, auto-routed recall, `memify`, `agent_memory`, sessions, and typed `MemoryEntry`/`QAEntry`/`TraceEntry`/`FeedbackEntry`. No bitemporal edge invalidation primitive.

| Concern | Owner | Justification |
|---|---|---|
| Episodic — what the agent did/decided, approved reports, QA pairs, traces, feedback | **Cognee** | `MemoryEntry`/`QAEntry`/`TraceEntry`/`FeedbackEntry` (`cognee/__init__.py:49`) + `agent_memory` (:61) are literally agent-run memory types; `recall` auto-routes without a hand-built retriever |
| Semantic — "have I seen a clause like this / what pattern resolved it" | **Cognee** | `improve` enriches the dataset into queryable knowledge; `recall` is the auto-routed reader |
| Entity graph — clauses, parties, obligations, precedent chains, `REFERENCES_CLAUSE` | **Graphiti** | `group_id` scoping + hybrid graph search; already the live writer (`graphiti_verifier.py:39-56`) and reader (`search/service.py:264`) |
| Temporal / bitemporal facts — "this obligation was superseded on date X" | **Graphiti** | only library here with `valid_at`/`invalid_at`/`expired_at` and automatic invalidation (`edges.py:271-277`, `graphiti.py:669`) |
| Report store / audit trail / idempotency | **Postgres** | `features/billing/models/report.py:42` `Report`; `AuditLog` in the same models package; Graphiti episodes are not an audit log |
| LangGraph checkpoint + store | **Postgres** | `langgraph_layer/checkpointer.py` (`teardown_langgraph_checkpointer`, `lifespan.py:31,316-317`) |

### Boundary violations in current code

1. `rag/graphiti/write_final_report.py:100-114` writes the final report to Graphiti as a high-trust episode — report-store concern in the entity-graph owner. (Dead code, but it is the reference implementation.)
2. `cognee_client.py:174-212` `store_relationships` pushes the relationship graph as text into Cognee `{user_id}.legal_relationships` — entity-graph concern in the agent-memory owner. Relationships already have a Graphiti writer (`client.py:257` `write_relationship_edge`).
3. `rag/graphiti/memory_pipeline.py:6` claims "Memory Retrieval (Graphiti + Cognee)" but `_do_retrieve_graphiti_context`:204-236 and `_retrieve_graphiti_context`:239-260 are Graphiti-only. The Cognee half of the router was never written.
4. `CogneeStore(BaseStore)` `cognee_client.py:273` puts LangGraph store duties on Cognee — that is Postgres's row in the table, and it is a stub anyway.
5. Layering: `registry.py`, `memory_pipeline.py`, `write_final_report.py` live under `rag/graphiti/` but are memory-router / tool-registry concerns; `registry.py:110` builds `retrieve_statute_section` (pure Postgres) from inside the Graphiti package.
6. `documents/service.py:596` opens a **second** `Graphiti` connection per ingestion call while `app.state.graphiti` already exists (`lifespan.py:216`).

---

## 4. Config correctness register

| Concern | What the repo sets | What it must set | Citation |
|---|---|---|---|
| LLM provider | `llm_provider="google_genai"`, `llm_model=GEMINI_FLASH_MODEL`, Gemini key | correct as-is | `cognee_client.py:77-83` |
| **Embedding provider** | **nothing** — commented-out TODO block | `cognee.config.set_embedding_config({...})` or `set_embedding_provider`/`_model`/`_api_key`/`_dimensions` | repo: `cognee_client.py:46-55`; cognee default `embedding_provider="openai"`, `embedding_model="openai/text-embedding-3-large"` → `.venv/.../cognee/infrastructure/databases/vector/embeddings/config.py:71-72`. Repo is Gemini @ 768 dims (`config/settings.py:194,212`). **Confirmed mismatch**, plus a 3072-vs-768 dimension mismatch. |
| **Vector store** | **nothing** — `set_vector_db_config` never called | pgvector (the TODO at `cognee_client.py:47` says `VECTOR_DB_PROVIDER=pgvector`) | cognee default `vector_db_provider = "lancedb"` → `.venv/.../databases/vector/config.py:30`. Cognee will silently write embeddings to local LanceDB files, not the app's Postgres. |
| Graph store | `graph_database_provider="neo4j"` + app URI/user/password | as-is | `cognee_client.py:84-91` |
| Relational store | `db_provider="postgres"` + app Postgres, `db_path=""` | as-is | `cognee_client.py:92-102` |
| ACL / Cognee user | nothing | see below | — |
| Dataset namespacing | `f"{user_id}.legal_reports"` (`cognee_client.py:140,238`), `f"{user_id}.legal_relationships"` (:189) | plus an owning Cognee user if ACL is on | — |
| `COGNEE_*` settings | **none exist** in `src/app/config/settings.py` | env-driven config for the above | `rg -i cognee src/app/config/settings.py` → no hits |

### ACL — previous claim CORRECTED
`ENABLE_BACKEND_ACCESS_CONTROL` is **not** unconditionally on in 1.1.0. `.venv/.../cognee/context_global_variables.py:83-92`: if the env var is unset it returns `multi_user_support_possible()`; if `"true"` it *also* requires `multi_user_support_possible()`. Support lists at :95-96: `VECTOR_DBS_WITH_MULTI_USER_SUPPORT = ["lancedb","pgvector","falkor"]`, `GRAPH_DBS_WITH_MULTI_USER_SUPPORT = ["ladybug","kuzu","falkor","postgres"]` — **neo4j is absent**. `multi_user_support_possible()` (:34-45) inspects `graph_dataset_database_handler` / `vector_dataset_database_handler` and **raises `EnvironmentError`** if the handler is unsupported (:44). So with this repo's neo4j graph config the realistic outcome is ACL off (or a startup `EnvironmentError`), not a silent ACL layer. Set `ENABLE_BACKEND_ACCESS_CONTROL=false` explicitly to make it deterministic.

---

## 5. Reuse assessment — `openspec/changes/cognee-saul-memory-migration/`

Artifacts: `proposal.md`, `design.md`, `tasks.md`, `.openspec.yaml`, `specs/saul-cognee-final-report-write/spec.md`, `specs/saul-memory-prefetch-and-retrieval/spec.md`. All four artifact types present. `tasks.md` has 13 tasks in 4 sections, **`[ ]` on every one — 0 done**. Dated `Jul 24` (dir mtime) vs today 2026-08-17.

`proposal.md:20-21` defers `saul-cognee-maintenance-worker` and `saul-cognee-reconciliation`. `design.md` states "Cognee v1.1 has no built-in dedup/decay/reconciliation, but memory drift is acceptable at v1 scale."

| Task | Verdict |
|---|---|
| 1.1 real Cognee memory service facade | **valid** — this is the core of the rebuild |
| 1.2 wire `persist_memory` → Cognee | **valid** |
| 1.3 `persist_memory` stops calling Graphiti final-report write | **valid** — matches the boundary (report-store ≠ entity-graph) |
| 1.4 tests: only approved reports persisted | **valid** |
| 2.1 post-`qna` prefetch node | **needs rewrite** — presumes a wired Saul graph; `build_saul_graph` has no caller |
| 2.2 prefetch Cognee-first + limited Graphiti supplement | **valid and now central** — this *is* the two-role split; `memory_pipeline.py:204-236` is the Graphiti half already written |
| 2.3 deeper retrieval only for `risk_analysis`/`compliance` | **valid** — `memory_pipeline.py:213,220` already branches on exactly those task names |
| 2.4 fail-open memory retrieval | **valid** — pattern exists at `memory_pipeline.py:258-260` and `cognee_client.py:250-257` |
| 3.1 remove final-report persistence from Graphiti memory route | **valid** |
| 3.2 keep Graphiti writes on KB extraction + relationships | **valid — and now the headline requirement**, needs promotion from cleanup to a first-class spec |
| 3.3 tests: Saul reports don't write Graphiti | **valid** |
| 4.1 targeted unit tests | valid |
| 4.2 `ruff check` + `ty check` | valid |
| — proposal's *deferral* of reconciliation | **invalidated by item 155**, which removes reconciliation outright rather than deferring a Cognee replacement for it |

The change is **not invalidated** by the two-distinct-roles decision — it already assumes it (`design.md`: "Cognee the primary memory/recall layer … Graphiti remains the structural knowledge-base layer"). What it lacks: the config-correctness work (§4), the reconciliation *removal*, and the fact that the whole Saul graph is unwired. Extending it requires adding those; superseding it would discard artifacts that are already directionally right.

### Also existing (prior art)
`openspec/specs/cognee-v1-api/spec.md` (43 lines) is a **deployed** spec that already locks the API surface: `remember` not `add`, `improve` not `cognify`, `recall` not `search` with `auto_route=True`, results→dicts, empty-list-on-failure, and "no `# type: ignore` on cognee calls". Any Cognee rewrite is bound by it. It says nothing about embeddings, vector store, or ACL — those are unspecified, not forbidden.

Searched for prior art under domain concepts, not the request's wording: "episodic/procedural/semantic memory", "memory store", "recall", "agent memory", "knowledge graph", "dedup", "decay", "audit trail" across `src/`, `openspec/specs/`, `openspec/changes/`. Only hits are the files named above plus `langchain_layer/agents/memory/memory_scope.py` (MemoryScope / MemorySource / MemoryEntityType / MemoryTimeFilter + five pre-built scopes, `memory/__init__.py:10-21`) — a retrieval-policy layer that is **already role-agnostic** and consumed by `rag/graphiti/subgraph.py:30,136`. It is reusable for both owners as-is.

---

## 6. Capability gap

| Capability | Graphiti 0.29.1 | Cognee 1.1.0 | Verdict |
|---|---|---|---|
| Bitemporal validity / supersession | **native** — `invalid_at`/`expired_at` (`edges.py:271,277`), auto-invalidation (`graphiti.py:669,1740-1757`) | no equivalent primitive | Graphiti owns |
| Near-duplicate detection | **native** — `graphiti_core/utils/maintenance/dedup_helpers.py`, `node_operations.py` | none found; `design.md` concurs | Graphiti owns; Cognee side must be built or dropped |
| Edge-preserving merge | **native** via `resolve_extracted_edges` → `(resolved, invalidated, new)` (`graphiti.py:648-678`) | no edge model to preserve | Graphiti owns |
| Decay | none | `forget` (`cognee/api/v1/forget/forget.py:16`) is deletion, not scored decay | **must be built** if wanted |
| Audit trail of memory mutations | episodes are append-only but carry no actor/reason field | `TraceEntry`/`FeedbackEntry` (`cognee/__init__.py:49`) are the closest, untested here | **must be built** — Postgres `AuditLog` is the natural home |
| Community/cluster summaries | `community_operations.py` | `memify` | either |

The reconciliation code being deleted (`reconciliation/nodes.py:135` reconcile, :205 apply-changes, :274 write-versions, :62 fetch-existing) implements dedup + versioning **in Postgres via asyncpg**. Graphiti already does the graph-side equivalent natively; Cognee does not do it at all. So "Cognee absorbs reconciliation" is not supportable from the installed API — confirmed.

---

## 7. Blast radius of the intended change

- Deleting reconciliation: 618 lines, **zero external callers** (`graphify affected build_reconciliation_graph`). Only `reconciliation/__init__.py:1` imports it. Archived openspec references exist (`openspec/changes/archive/2026-07-22-noqa-exception-handling-migration/`) — historical, non-binding. A `src/tasks/memory_decay_reconciliation_tasks.py` is referenced in that archive; confirm whether the file still exists.
- Removing Graphiti init from ingestion: touches `lifespan.py:33,212-217,225-232,335`, `documents/service.py:38,596-601,622`, `health_check.py:83-90,98`, `documents/ingestion_graph.py:43,69`, `graphiti_verifier.py:31`, `search/dependencies.py:40`, `search/service.py:65,264`. Removing `app.state.graphiti` breaks the health check and the search read path — those are the entity-graph role and by the boundary above they **stay**.
- No tests cover any of this: codegraph reports "⚠️ no covering tests found" for `CogneeStore`, `setup_cognee`, `store_final_report`, `store_relationships`.

---

## 8. Constraints in force

- `RESULT-PATTERN.md` / `EXCEPTION-RULES.md`: current memory code does neither cleanly — `store_final_report` re-raises (`cognee_client.py:159`), `search_episodic_memory` swallows into `[]` (:257), `write_final_report_to_memory` collects error strings into `MemoryPersistResult.errors` (`write_final_report.py:156-161`). Three different failure idioms in one layer.
- `e.add_note()` before re-raise is the house style — `cognee_client.py:251`, `graphiti_verifier.py:60`, `lifespan.py:221`.
- Graph nodes must return fallback state, never crash (archived spec + `ingestion_kb/nodes.py:358`, `memory_pipeline.py:258-260`).
- `openspec/specs/cognee-v1-api/spec.md` binds the Cognee call surface and forbids `# type: ignore` on cognee calls. Note `CogneeStore` currently carries `# type: ignore` on `put`/`get`/`search`/`delete`/`_matches_filter` (`cognee_client.py:286,295,304,316,337`) — those are on `BaseStore` overrides, not cognee calls, so not a violation of that spec as written.
- Async-first: every symbol above is `async def`. `setup_cognee` is async but does only sync config calls (`cognee_client.py:58-102`).

---

## 9. Fog

- **The exact role boundary is not decidable from the code alone.** §3 is my proposal, derived from installed-library capability, not from a repo statement. The code contains a *stated* boundary in three places — `cognee_client.py:12-15` ("Graphiti → structural legal knowledge graph; Cognee → episodic + procedural memory"), `write_final_report.py:8-13` (a memory-router comment routing the final report to **both**), and `design.md` ("Cognee primary memory/recall, Graphiti structural KB") — and they **disagree on the final report**: the docstring gives it to Cognee, the router writes it to both. I cannot tell which was the intent.
- **Where the semantic/vector-retrieval role sits is genuinely ambiguous.** Both libraries do hybrid retrieval, and the repo *also* has its own pgvector retrieval (`ingestion_kb/nodes.py:716,738` `_cached_embedding`/`_call_embedding_fn`, `retrieval_kb/nodes.py`). Three candidate owners for one concern; nothing in the repo picks.
- **Whether the Saul graph being unwired is intentional or a regression.** `lifespan.py:234-247` and `:294-305` are commented-out blocks with no dated note. If it is temporary, "four agent tools depend on Graphiti" becomes true again after re-wiring and the blast radius in §7 grows. Resolvable only by asking the user.
- **Cognee ACL end state under neo4j.** I established the gate logic (`context_global_variables.py:83-96`) and that neo4j is not in the supported-graph list, but `multi_user_support_possible()` keys off `graph_dataset_database_handler` (a separate setting) and can raise `EnvironmentError`. I did not run cognee to see which branch this repo actually hits. Resolvable by executing `setup_cognee` against the real config and observing whether startup raises.
- **Whether `store_final_report`/`store_relationships` were ever exercised.** No tests, no call sites, no migration or dataset artifact. I cannot confirm `remember`+`improve` ever succeeded against this Neo4j+Postgres config, so "rebuild exactly that" may be rebuilding something that never ran.
- **`src/tasks/memory_decay_reconciliation_tasks.py`** — referenced by an archived openspec change; my attempt to list `src/tasks/` was blocked by the permission system, so I could not confirm the file still exists or whether a Celery beat schedule registers it. Resolvable with one directory listing.
- **`cognee.improve` semantics on an already-improved dataset** — whether it is idempotent or re-embeds everything. Not determinable without reading `improve.py` in depth or running it; matters for cost.
