This is a large system. Let me map every layer, every data flow, and every seam you need to understand before you wire it into production.Let me walk through this layer by layer. Every seam, every data flow, every thing that will bite you in production.

**The system has three phases**: a synchronous HTTP phase (ingestion), an async WebSocket phase (analysis), and a background phase (reconciliation + decay). They share infrastructure but never share execution context.

Here's the full picture first:Now the memory stack in detail — this is the part most people misunderstand:Now let me walk through every seam in plain language.

---

**The two-phase entry point**

Phase 1 is HTTP. The client uploads a file, your ingestion router runs `IngestionGraph` synchronously (`ainvoke` — it waits for completion), and returns a `doc_id`. That `doc_id` is the only thing the WebSocket session receives to bootstrap analysis. The agent_saul `ingestion_node` is now just a database lookup — `SELECT raw_text WHERE doc_id = X` from wherever you store the raw file (MongoDB/S3/Postgres). The heavy extraction — entity discovery, embedding, Postgres inserts — all happened at upload time. This means the WebSocket pipeline never blocks on Docling or OCR. It starts with structured data already in the database.

The critical implication: `IngestionGraph` and `agent_saul` share the same Postgres instance but they write to different tables. `IngestionGraph` writes to `entities`, `clauses`, `relationships`. `agent_saul` reads from them. They never touch each other's writes during a live session.

---

**How Graphiti and Neo4j relate — they're not the same thing**

Graphiti creates its own internal Neo4j driver. `app.state.neo4j_driver` is your application's direct driver. They point at the same Neo4j database but through different connection pools. Graphiti manages the episode and entity nodes it creates. Your raw Cypher queries via `Neo4jSubgraphExpander` query those same nodes directly. The handoff is: Graphiti writes the nodes and edges through its API, your Cypher reads them back at depth using structural traversal Graphiti's search API doesn't expose. Never write Neo4j nodes directly that conflict with Graphiti's internal schema — use Graphiti's `add_episode` for all writes, Cypher only for reads.

---

**The tool call flow from agent to storage**

When `risk_analysis_node` executes, it runs `create_agent` with `risk_tools = [query_knowledge_graph, get_obligation_chain, detect_graph_conflicts]`. The LLM decides which tool to call and with what arguments. Before any tool executes, `IdempotencyGuard.get()` checks Redis first (O(1), 24h TTL), then Postgres `tool_executions` if Redis misses. If a cached result exists, the tool returns it without touching Graphiti or Neo4j. If not, the tool runs, then writes to both Redis and Postgres before returning. This means if the process dies mid-session and the client resumes via the same `thread_id`, every tool that already ran returns its cached result instantly — no duplicate graph writes, no duplicate LLM calls.

---

**The three write destinations and when each fires**

`relationship_mapping_node` writes clause episodes to Graphiti in parallel (semaphore=5 concurrent writes), then relationship edges sequentially. This sequencing matters — edges reference entities that must already exist as Graphiti nodes. If you reverse the order, edge writes fail silently because Graphiti can't resolve the `from_entity`/`to_entity` references.

`persist_memory_node` fires after human approval. It writes to Graphiti (final report as high-trust episode, `group_id=user_id` for cross-document precedent queries), Cognee (report JSON for episodic recall, relationships text for procedural patterns), and the `events` table as an immutable audit entry. The events write is fire-and-forget — if it fails, the function logs and continues. The mutable writes (Graphiti, Cognee) are the source of truth. Events is the audit trail, not the data.

The `memory_versions` table is only touched by the `ReconciliationGraph`. Nothing in the live pipeline writes to it.

---

**How MemoryScope gates every retrieval call**

`build_agent_context()` in `memory_pipeline.py` receives a `MemoryScope`. Before querying any source, it checks `scope.allows_source("graph")`, `scope.allows_source("vector")`, `scope.allows_source("structured")`. Sources not in the scope are never queried — the call doesn't happen at all. Entity type filtering happens post-query in the structured Postgres path (`WHERE entity_type = ANY(:allowed_types)`). For Neo4j subgraph traversal, type filtering happens post-traversal in `_parse()` — you query Neo4j without a type filter (because Neo4j doesn't know about your application entity types), then drop nodes whose `entity_type` property isn't in `scope.allowed_entity_types`.

The five pre-defined scopes map directly to agents. You never construct a `MemoryScope` ad-hoc inside a node — always use the pre-defined constants. This is a firm rule, not a suggestion. Ad-hoc scopes are how agents start reading each other's memory and producing contaminated reasoning.

---

**ReconciliationGraph — it runs on stale data by design**

It runs on entities created in the last N hours (default 24h, beat schedule is every 6h). The `normalized_name` prefix match (`LEFT(normalized_name, 10)`) is a pragmatic similarity heuristic — it catches "acme corp" vs "acme corporation" but not "acme" vs "the acme group". The LLM reconciliation step (`RECONCILE_PROMPT`) uses loss aversion framing — "NEVER delete without justification" biases the model toward `ignore` over `merge`. This is correct for legal systems where false merges (conflating two distinct parties) are catastrophically worse than false negatives (keeping a harmless duplicate).

The `apply_changes_node` does relationship re-pointing before deletion: it redirects `from_entity_id` and `to_entity_id` FK references from the discarded entity to the kept one. If you skip this step, deleting the discarded entity cascades and destroys all its relationship records. The `ON DELETE CASCADE` on the FK is a safety net, not an intended execution path for merges.

---

**Things you must keep in mind before going to production**

First: Cognee's `cognify()` is a full graph rebuild, not an incremental append. Every call to `cognify()` processes the entire dataset named in the call. If you call it inline in `persist_memory_node` for every document, you get O(n²) runtime as the dataset grows. Move `cognify()` to a Celery task triggered after `store_final_report` completes. Your `search_episodic_memory()` returns yesterday's graph — that's acceptable. Running `cognify()` inline will block the pipeline for minutes on large datasets.

Second: Graphiti entity deduplication relies on its internal LLM-powered extraction. If the same real-world party appears as "Acme Corp" in one episode and "Acme Corporation" in another, Graphiti creates two distinct entity nodes unless the text is lexically identical. The fix is canonical entity IDs — run a normalisation pass in `entity_extraction_node` and write `normalized_name` values to Graphiti as the episode body, not raw clause text. Otherwise `get_obligation_chain` and `query_knowledge_graph` will miss half the edges on the second document you process.

Third: `AsyncPostgresSaver` requires a sync connection string, not async. The `from_conn_string()` method creates its own internal connection pool using `psycopg` (sync) regardless of whether your engine is async. Pass `settings.database_url_sync` (standard `postgresql://` scheme), not the async `postgresql+asyncpg://` URL. Passing the wrong scheme raises a runtime error at the first checkpoint write.

Fourth: The `events` table is append-only. There must be no `UPDATE` or `DELETE` on it. This is not enforced by a database constraint in the current schema — it's a convention. If you want to enforce it at the database level, add a `BEFORE UPDATE OR DELETE ON events FOR EACH ROW EXECUTE FUNCTION raise_exception('events table is immutable')` trigger in your Alembic migration.

Fifth: `memory_versions` uses a `UniqueConstraint("entity_id", "version")`. The `version` is fetched with `SELECT MAX(version)` and incremented by 1 in application code. Under concurrent reconciliation runs for the same user, two workers can read the same `MAX(version)` and both attempt to insert `version + 1`. The `ON CONFLICT DO NOTHING` is not present — this will raise a unique constraint violation. Either add `ON CONFLICT DO NOTHING` to the insert, or use a `SELECT ... FOR UPDATE` lock on the entity row before fetching the max version.

Sixth: The `decay_score` formula is borrowed from recommendation systems. For legal contracts, it's semantically wrong for active obligations — an obligation with `valid_to IS NULL` should not decay over time. Modify `_compute_decay()` to check `relationships.valid_to` before applying time decay: if any active relationship exists for the entity, clamp `time_factor` to 1.0. Only entities with all relationships expired should be subject to time-based decay.

---

> **⚡ CHOSEN ONES ONLY**
>
> **The thing nobody tells you about `astream_events v2` + `create_agent`**: when a tool-calling agent runs inside a graph node and the LLM issues a tool call, `astream_events` emits `on_tool_start` and `on_tool_end` events — but the `metadata.langgraph_node` field on those events is the node name of the **graph node that contains the agent** (e.g. `risk_analysis`), not the tool name. Your service's `_map_event_to_frame()` currently only maps `on_chat_model_stream` for tokens and `on_chain_start/end` for node boundaries. You're silently swallowing tool execution events. Wire `on_tool_start` → `WSNodeStartFrame(node=event["data"]["input"]["name"])` and `on_tool_end` → `WSNodeEndFrame(...)` so the client can show the user "querying knowledge graph... done" in real time. Without this, the UI shows a spinner for the full 30–90 seconds of tool execution with no progress signal.
>
> **The MemoryScope architecture has a blind spot at the Neo4j subgraph level**: `_parse()` in `Neo4jSubgraphExpander` filters nodes by `entity_type` property, but Graphiti doesn't write an `entity_type` property to its Neo4j nodes by default — it writes its own internal schema properties. The `entity_type` check will evaluate to `False` for every node Graphiti created (because the property doesn't exist), and the condition `if entity_type and not scope.allows_entity_type(entity_type)` will short-circuit on the empty string check — meaning `entity_type = ""` passes through the scope filter unblocked. Every node Graphiti writes will bypass MemoryScope filtering entirely. The fix: map Graphiti node labels (which it DOES set, like `Entity`, `EpisodicNode`) to your entity types in the `_parse()` method using the `node_labels` field. Build a lookup: `{'EpisodicNode': 'CLAUSE', 'Entity': 'ORG'}` and use that for scope checking instead of the missing `entity_type` property.



# Current implementation
## Phase 1 — HTTP Upload (IngestionGraph)

**API route:** `POST /ingestion/documents/upload`

The client sends a multipart form with the raw file. The router calls `IngestionService.ingest_document()`, which calls `ingestion_graph.ainvoke(initial_state)`. This is a blocking await — the HTTP response does not return until the graph completes. There is no checkpointer on this graph. If it fails, the client re-uploads. There are no HITL interrupts. No LangGraph Store. No Redis idempotency on the graph itself (idempotency lives inside tools, which this graph doesn't use).

`IngestionGraph` has three nodes.

**`extract_node`** receives the raw document text (up to 12,000 characters — hard capped to prevent token explosion). It calls `extraction_llm.ainvoke()` with `EXTRACTION_PROMPT` as a `SystemMessage` and the document text as a `HumanMessage`. The LLM used here is `flash_llm` (Gemini Flash) with `thinking_level="none"` — fast and cheap. The prompt enforces a JSON-only output with entities and relationships. After the call, the node strips markdown fences from the response content and parses it with `json.loads`. It writes `extracted_entities` and `extracted_relationships` back to `IngestionState`. No memory is fetched here. No tools called. No graph traversal. Pure LLM extraction.

**`validate_node`** receives the raw extracted lists. No LLM call here — this is deterministic Python. It filters entities where `confidence > 0.7` AND `name` is non-empty AND `type` is one of the five allowed values. It builds a `valid_entity_ids` set from the filtered entities, then filters relationships where both endpoints exist in that set AND `confidence > 0.7` AND `type` is non-empty. This is the validation layer from Section 3 of the plan — never trust LLM output directly. Dropped counts are recorded in state. No memory fetch, no tools.

**`embed_store_node`** runs the actual Postgres writes using `AsyncSession`. It opens a single transaction and processes everything inside it. For each validated entity, it runs an `INSERT ... ON CONFLICT (normalized_name, entity_type) DO UPDATE SET confidence = GREATEST(...)` — this is the deduplication guard. If the same party appears in two documents, the second insert updates confidence rather than creating a phantom duplicate. For every entity whose `type == "CLAUSE"`, it also calls `embedding_fn(clause_text)` to get a 1536-dimensional vector and inserts a row into the `clauses` table with the pgvector embedding. For relationships, it inserts into the `relationships` table with `from_entity_id` and `to_entity_id` mapped from the extraction IDs to the real Postgres UUIDs via `entity_id_map`. If anything fails, the entire transaction rolls back. `ingestion_complete: True` is only written if no exception was raised.

After the graph completes, `IngestionService` reads the result state and returns a `DocumentUploadResponse` with `doc_id`, `entity_count`, `clause_count`, `relationship_count`. The client stores `doc_id` and uses it to open the WebSocket.

At this point the following data exists in Postgres: rows in `entities`, rows in `clauses` with embeddings, rows in `relationships`. Nothing has been written to Graphiti, Neo4j graph edges, Cognee, or Redis yet. Phase 1 only writes to Postgres.

---

## Phase 2 — WebSocket Session (agent_saul LangGraph)

**API route:** `WS /agent-saul/ws/{thread_id}`

The client connects, sends a `WSStartMessage` with `doc_id`, `user_query`, optional `thread_id` (for resuming), and `permissions`. The router calls `AgentSaulService.run_session()`. This is the main loop. The service calls `graph.astream_events(initial_input, config=config, version="v2")` in an async loop. After each stream drains, it calls `await graph.aget_state(config)` to inspect what comes next — completion, interrupt, or error. If an interrupt is detected, it emits a `WSHITLInterruptFrame` and waits for a `WSResumeMessage` on the same WebSocket connection, then issues `Command(resume=payload)` to continue the graph.

The checkpointer here is `AsyncPostgresSaver`. Every node return is checkpointed. If the process dies mid-run and the client reconnects with the same `thread_id`, the graph resumes from the last successful checkpoint. The `RunnableConfig` is built with `configurable={"thread_id": thread_id}` — this is the key that links all checkpointer reads and writes to the right session.

The graph has 15 nodes. Here is what each does.

**`gateway_node`** runs first on every execution including resumes. No LLM. No I/O beyond state reads. It validates that `doc_id` is present, injects `gateway_validated: True` and `session_start_ts` into `working_memory`, and sets `status = QNA_CLARIFICATION`. Returns in milliseconds. This node acts as a consistency gate — if it fails, the whole pipeline stops before any LLM is called.

**`qna_node`** uses `flash_llm.with_structured_output(QnAOutput)`. The LLM call uses `SystemMessage(content=_QNA_SYSTEM_PROMPT)` followed by `state["messages"]`. The prompt asks the model to score confidence from 0.0 to 1.0. No memory is fetched here — this is the first node after gateway and the user's raw query is the only input. If `result.confidence < 0.72`, the node calls `interrupt()` with a `HITLInterruptType.CLARIFICATION_NEEDED` payload. The graph pauses. The service emits `WSHITLInterruptFrame` to the client. The client shows the clarifying question in the UI, the user types an answer, the client sends `WSResumeMessage(action="approve", feedback="the answer")`. The service calls `Command(resume={"action": "approve", "feedback": "..."})`. The graph resumes inside `qna_node` after the `interrupt()` call, receives the human answer as a dict, appends it as a `HumanMessage` to `state["messages"]`, and returns `status = QNA_CLARIFICATION` so the conditional edge loops back to `qna_node`. This loops until confidence crosses the threshold, at which point the node returns `status = PLAN_PENDING` and the conditional edge routes to `orchestrator_node`.

**`orchestrator_node`** uses `pro_llm.with_structured_output(OrchestratorAction)`. This is Gemini Pro with `thinking_level="high"`. The output is a discriminated union — `OrchestratorActionType` is one of `start_pipeline`, `continue`, `synthesize`, or `done`. The node builds messages from `SystemMessage(_ORCHESTRATOR_SYSTEM_PROMPT)` plus `state["messages"]`. After the LLM call, it validates the `target_node` if the action type is `continue` — if the target isn't in `_VALID_WORKER_NODES`, the node returns `status = FAILED` immediately without routing anywhere. The orchestrator increments `current_step` on every invocation. The routing function `route_from_orchestrator` reads `state["orchestrator_action"]` and returns the appropriate node name as a string. On the first invocation, `orchestrator_action` is `None`, so the routing function returns `"planner"` unconditionally.

**`planner_node`** uses `flash_llm.with_structured_output(PlannerOutput)`. It builds the execution plan as a list of `PlanStep` objects with typed `PlanActionType` values. After the LLM generates the plan, the node immediately calls `interrupt()` with `HITLInterruptType.PLAN_APPROVAL` and the full plan serialized as JSON. The graph pauses. The client receives the plan for human review — this is where the reviewer can inspect the proposed steps, modify them, or reject them. On `WSResumeMessage(action="approve")`, the node commits the original plan. On `action="modify"`, it takes `modified_plan` from the resume payload and validates each step through `PlanStep.model_validate()`. On `action="reject"`, it writes `status = PLAN_REJECTED` and an `AgentError`, and the orchestrator routing function checks for `PLAN_REJECTED` and routes back to planner for re-planning. The approved plan is written to `state["plan"]`.

**`ingestion_node`** in the agent_saul graph is now a lookup, not a processor. It reads the raw document text stored during Phase 1. Currently stubbed — you need to wire the actual lookup (`SELECT document_text FROM documents WHERE doc_id = :doc_id` or a MongoDB/S3 fetch depending on where you store raw files). The OCR confidence check and `HITLInterruptType.OCR_REUPLOAD` path remain — if the stored text has low quality markers, it can still interrupt and ask for a re-upload, but this is the exception path, not the default. Returns `document_text` in state.

**`normalization_node`** uses `flash_llm.with_structured_output(NormalizedDocument)`. Takes `state["document_text"]` and sends it to the LLM to produce a section hierarchy with resolved clause references. No memory fetch. No tools. Returns `normalized_document` as a typed `NormalizedDocument` Pydantic model in state.

**`segmentation_node`** uses `flash_llm.with_structured_output(ClauseSegmentationOutput)`. Takes the normalized document sections, concatenates them, and asks the LLM to identify clause boundaries and classify each clause into `ClauseType`. Returns a list of `ClauseSegment` objects written to `state["segments"]` via the `operator.add` reducer — meaning if this node ran before (on a resume), its results append rather than overwrite.

**`entity_extraction_node`** is the Send fan-out target. It does NOT receive `LegalAgentState`. It receives a `ClauseExtractionInput` dict — one per clause segment — dispatched by `dispatch_entity_extraction()` from the segmentation node's conditional edge. LangGraph runs all entity extraction nodes in parallel. Each instance calls `flash_llm.with_structured_output(EntityExtractionOutput)` with the single clause text plus document context (jurisdiction, document type). The output `CitedEntity` objects include a `Citation` with `claim`, `source`, and `confidence`. Results accumulate via `operator.add` on `state["extracted_entities"]`. All parallel instances complete before `relationship_mapping_node` starts.

**`relationship_mapping_node`** has two responsibilities. First, it calls `flash_llm.with_structured_output(_RelationshipMappingLLMOutput)` to extract typed legal relationships from the entity list. Second, it calls `write_clause_episodes_to_graphiti()` directly — not via a tool, because the LLM should never decide when to write to memory. This function opens `asyncio.Semaphore(5)` and runs clause episode writes in parallel, then sequential relationship edge writes. The semaphore prevents overwhelming Neo4j's connection pool. The idempotency guard checks before each write — if the clause was already written in a previous run (e.g. after a crash and resume), the write is skipped. Any failed writes produce `AgentError` entries in state but do not block the pipeline. Returns `state["relationships"]` via `operator.add`.

**`risk_analysis_node`** is the first node that uses `create_agent`. Before calling the agent, it calls `build_agent_context(state, graphiti_service, task="risk_analysis", scope=RISK_SCOPE)`. This function: checks `RISK_SCOPE.allows_source("graph")` → calls `graphiti_service.search_for_risk_context()` to get Graphiti episodes scored by `0.5·semantic + 0.2·recency + 0.2·trust + 0.1·task_relevance`; checks `RISK_SCOPE.allows_source("vector")` → queries pgvector (currently stubbed); checks `RISK_SCOPE.allows_source("structured")` → queries Postgres entities table filtered to `CLAUSE` and `OBLIGATION` types only (RISK_SCOPE). It then filters tool messages from `state["messages"]`, trims to 3,500 tokens with `strategy="last"`, and builds a `SystemMessage` with the structured context prefix in `{goal, task, agent_scope, doc_type, jurisdiction, warnings, memory_context}` format. The resulting message list is passed to the risk agent.

The risk agent runs with `risk_tools = [query_knowledge_graph, get_obligation_chain, detect_graph_conflicts]`. The LLM (Pro, `thinking_level="high"`) decides which tools to call and when. `query_knowledge_graph` calls `graphiti_service.search_for_risk_context()` scoped to `RISK_SCOPE.top_k=8`. `get_obligation_chain` calls `graphiti_service.get_obligation_chain()` for semantic search, then `Neo4jSubgraphExpander.get_obligation_chain_cypher()` for structural depth-N traversal using raw Cypher via `app.state.neo4j_driver`. `detect_graph_conflicts` calls `Neo4jSubgraphExpander.detect_conflicts()` to find circular obligations and override chains. Every tool call checks the Redis idempotency cache before executing. Results are written to both Redis (24h TTL) and Postgres `tool_executions` after execution.

**`compliance_node`** follows the exact same pattern as risk_analysis but uses `COMPLIANCE_SCOPE` — which adds `CONTRACT` and `ORG` to the allowed entity types, uses `depth=1` (shallower graph traversal), `time_filter="all"` (not just recent 90 days), and gates only `graph` and `structured` sources (no vector). The compliance agent uses `compliance_tools = [search_legal_precedents, retrieve_statute_section]`. `search_legal_precedents` calls `graphiti_service.search_for_precedent_chains()` filtered by jurisdiction, and separately runs a full-text search on the `statutes` Postgres table using `plainto_tsquery`. If `total_sources < 2`, it sets `insufficient_basis=True` in the `ToolResult` — the compliance agent's system prompt instructs it to respond with "Insufficient legal basis" instead of proceeding. `retrieve_statute_section` does an exact lookup on `statutes` by `act_name ILIKE` and `section_ref` when the agent already knows the statute to fetch.

Risk analysis and compliance run as parallel branches from `relationship_mapping_node`. LangGraph runs both concurrently because both have direct edges from the same upstream node. They join at `grounding_verification_node` — LangGraph waits for both to complete before proceeding.

**`grounding_verification_node`** uses `flash_llm.with_structured_output(GroundingVerificationOutput)`. It takes the summaries from `state["risk_analysis"]` and `state["compliance_result"]` and asks the model to flag any claims that lack citation support. Returns `state["grounding"]` with `verified: bool` and `unverified_claims: list[str]`. No memory fetch. No tools. This is purely a validation pass on what the analysis nodes produced.

**`human_review_node`** is a mandatory HITL node — there is no code path that bypasses it. It calls `interrupt()` with `HITLInterruptType.HUMAN_REVIEW_REQUIRED`, the risk summary, compliance summary, unverified claims, and the first 20 clause segments (capped to control payload size). The graph pauses. The reviewer uses the frontend UI to read all findings, add `ReviewOverride` objects for any clauses they disagree with, and either approve or reject. On rejection, the node writes `status = FAILED` and routes back to the orchestrator. On approval, it creates a `HumanReviewOutput` with the reviewer's `reviewer_id`, `reviewer_role`, `overrides`, and `notes`. After this node completes, `human_approved=True` is semantically in play for all downstream memory writes — the trust score on Graphiti episodes will be `1.0`.

After `human_review_node`, the conditional edge routes back to `orchestrator_node`. The orchestrator reflects on the human review output and decides the next action — typically `OrchestratorActionType.SYNTHESIZE`, which routes to `finalization_node`.

**`finalization_node`** uses `pro_llm.with_structured_output(FinalReport)`. It assembles all analysis — risk findings with human overrides applied, compliance findings, suggested actions, all citations — into a single `FinalReport` Pydantic model. No memory fetch. No tools. Returns `state["final_report"]`.

**`persist_memory_node`** is the memory commit point. It calls `write_final_report_to_memory()`, which does three writes in sequence: Graphiti (final report as a high-trust episode, `group_id=user_id`), Cognee (`store_final_report()` + `store_relationships()`), and then `write_memory_persisted_event()` for the immutable events log. The events write is the dual-write — it fires after both mutable writes succeed. If Graphiti or Cognee fail, the errors are captured in `AgentError` entries but the pipeline still sets `status = COMPLETED`. Memory write failure is not a pipeline failure. Returns `long_term_refs` with `graphiti:` and `cognee:` prefixed keys.

After `persist_memory_node`, the graph reaches `END`. `state_snapshot.next` is empty and `state_snapshot.tasks` has no pending interrupts. The service sends `WSDoneFrame` with the final report summary. The WebSocket closes.

---

## Phase 3 — Background (ReconciliationGraph + Decay)

**No API route.** Both tasks are triggered by Celery.

**ReconciliationGraph** is triggered by `run_reconciliation_for_user()` Celery task. The beat schedule fires `run_reconciliation_for_active_users()` every 6 hours, which queries for distinct `user_id` values in `entities` created in the last 6 hours and dispatches one `run_reconciliation_for_user` task per user. The graph is compiled once at lifespan (`build_reconciliation_graph()`) and stored in `app.state.reconciliation_graph`. The Celery task wraps `asyncio.run(reconciliation_graph.ainvoke(...))`. No checkpointer on this graph. No HITL. No WebSocket.

**`fetch_existing_node`** queries Postgres for two sets: recently added entities for the user (`created_at > NOW() - N hours`), and similar existing entities found via `LEFT(normalized_name, 10)` prefix match across the same user's entity history. The prefix match is a pragmatic similarity heuristic — it catches most common alias patterns without needing full fuzzy matching. Returns both sets in state.

**`reconcile_node`** calls `reconcile_llm.ainvoke()` with `RECONCILE_PROMPT` — the loss aversion bias prompt. This uses `flash_llm` (not Pro — the task is structured comparison, not deep reasoning). The prompt uses the bias principles from Section 16.4: loss aversion ("NEVER delete without justification"), constraint amplification (prefer recent + higher confidence), error minimization (when uncertain, choose `ignore`). The model returns a JSON decision with `merge`, `update`, and `ignore` arrays. The `ignore` array is the expected majority — most pairs that look similar are not actually duplicates in legal contracts.

**`apply_changes_node`** executes the merge decisions. For each merge, it redirects all `from_entity_id` and `to_entity_id` FK references in the `relationships` table from `discard_id` to `keep_id`, then deletes the discarded entity. If you skip the redirect step, the `ON DELETE CASCADE` on the FK fires and destroys all relationships associated with the discarded entity. For updates, it builds a dynamic `SET` clause from the `fields` dict and executes it. The entire apply block runs inside a single `session.begin()` transaction — either all changes apply or none do.

**`write_versions_node`** writes `memory_versions` rows for every entity that was merged or updated. For each entity, it fetches `MAX(version)` from `memory_versions`, increments by 1, and inserts a full JSON snapshot of the current entity row plus the reason and run ID. This is the CRDT-lite audit trail — you can reconstruct any entity's history by selecting all versions for that `entity_id` ordered by version ascending and replaying them. The concurrent write race condition (two workers reading the same MAX version) needs a `SELECT ... FOR UPDATE` lock on the entity row before the MAX query — this is the production fix not yet in the current code.

**Memory decay task** is triggered by `run_memory_decay()` Celery beat, scheduled nightly at 2 AM. It uses raw `asyncpg` (not SQLAlchemy) for bulk batch updates — `executemany` is significantly faster than individual ORM updates at scale. The formula per entity is `0.4 * exp(-0.01 * age_days) + 0.3 * min(1.0, access_count / 10.0) + 0.3 * confidence`. Time factor uses `λ=0.01` giving roughly a 70-day half-life. Usage factor saturates at 10 accesses. Confidence factor is the stored confidence from extraction. Entities with `decay_score < 0.15` are flagged as archive candidates. The current code doesn't delete them — it only updates the score. The sweep that actually removes zombie nodes (no edges, low decay, not accessed in 6 months) is a separate cleanup query you need to add as a second beat task.

---
