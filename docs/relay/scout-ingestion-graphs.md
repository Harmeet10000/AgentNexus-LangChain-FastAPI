# Scout — Ingestion graphs, runtime wiring, Celery, app.state

## 1. The two pipelines side by side

| | `documents/` (LIVE today) | `ingestion_kb/` (to be promoted) |
|---|---|---|
| Graph factory | `src/app/features/documents/ingestion_graph.py` | `build_ingestion_graph` — `src/app/shared/langgraph_layer/ingestion_kb/graph.py:37` |
| Entry point | HTTP `documents/router.py` → `upload_document` (`service.py:118`) → S3 + outbox row → Celery `ingest_document` → `run_document_ingestion_task` (`service.py:580`) | none mounted; `build_ingestion_graph` has **only** commented-out call in `lifespan.py` |
| State class | `DocumentIngestionState` (`ingestion_graph.py`) | `IngestionState` — `state.py:166`, `BaseModel`, `extra="forbid"`, `arbitrary_types_allowed=True` |
| Node count | see per-node notes below | **7** |
| Reducers | — | exactly **one**: `contextualized_chunks: Annotated[list[ContextualizedChunk], operator.add]` (`state.py:181`) |
| Checkpointer | none (`compile()` with no arg) | none — `graph.compile()` `graph.py:84`, no `checkpointer=` |
| Resumable | No | No (no checkpointer, `raw_bytes: bytes` channel is not serializable-friendly) |

### `ingestion_kb/` node list, in graph order

| # | Node (graph.py line) | Factory | Reads from state | Writes channels |
|---|---|---|---|---|
| 1 | `parse_document` `:46` (entry `:75`) | `make_parse_document_node` `nodes.py:90` | `raw_bytes`, `filename`, `source` | `parsed_document` \| `failure`+`ingestion_complete=False` |
| 2 | `extract_schema` `:47` | `make_extract_schema_node` `nodes.py:110` | `parsed_document`, `document_type`, `jurisdiction` | `contract_metadata` |
| 3 | `segment_document` `:51` | `make_segment_document_node` `nodes.py:150` | `parsed_document`, `contract_metadata` | `segments` |
| 4 | `contextualize_chunks` `:57` | `make_contextualize_chunk_node` `nodes.py:212` — **takes `dict`, not `IngestionState`** | `Send` payload dict only | `contextualized_chunks: [chunk]` (reduced by `operator.add`) |
| 5 | `classify_extract_entities` `:63` | `make_classify_extract_node` `nodes.py:259` | `contract_metadata`, `contextualized_chunks` | `extracted_entities`, `extracted_relationships` |
| 6 | `embed_store` `:69` | `make_embed_store_node` `nodes.py:303` (`db_engine`, `embedding_fn`, `redis`) | `parsed_document`, `contract_metadata`, entities, chunks | `parent_doc_id`, `stored_clause_ids`, `stored_chunks`, `stored_entity_ids`, `stored_relationship_ids` |
| 7 | `graphiti_upsert` `:73` | `make_graphiti_upsert_node` `nodes.py:354` | `contextualized_chunks`, `stored_chunks`, `contract_metadata` | `graphiti_episode_ids`, `ingestion_complete=True` |

**CORRECTION to brief:** the `Send` fan-out is *declared* at `graph.py:78` (`add_conditional_edges("segment_document", dispatch_contextualize_chunks)`) but the `Send` objects are constructed at **`nodes.py:200`** inside `dispatch_contextualize_chunks` (`nodes.py:194`). Confirmed 7 nodes, 1 reducer.

Error handling in `ingestion_kb`: two disjoint styles. Guard clauses return `_state_failure` (`nodes.py:70`) → sets `failure` channel but **the graph has no conditional edge on `failure`**, so downstream nodes still run and hit their own guard. LLM nodes catch `LangChainException` and degrade (`nodes.py:182`, `:236`, `:289`) — `_fallback_segments` (`nodes.py:442`), deterministic preamble, empty `EntityExtractionResult`. `retry_immediate` (`shared/langgraph_layer/kb_retry.py:19`) wraps every I/O call.

**Bug found:** `nodes.py:238` — `contextualize_chunk_node` receives a plain `dict` (`state: dict[str, Any]`, `nodes.py:215`) but the except branch calls `state.doc_id` → `AttributeError` masking the original `LangChainException`.

### `documents/` node list — CONFIRMED: **one node**

`build_document_ingestion_graph` (`features/documents/ingestion_graph.py:39`) adds exactly one node `ingest_document` (`:50`), `set_entry_point` `:60`, `add_edge(..., END)` `:61`, `compile()` `:62`. The node (`_make_ingest_document_node` `:65`) is a pure pass-through that forwards 5 state fields + 4 injected deps to `ingest_document_fn`.

`DocumentIngestionState` (`ingestion_graph.py:25`) — 9 scalar channels, **zero `Annotated` reducers**: `document_id`, `user_id`, `filename`, `content_type`, `object_uri`, `status`, `chunk_count`, `verified_chunk_count`, `document_kind`. Note `status`/`chunk_count`/`verified_chunk_count`/`document_kind` are declared but the node returns them from `process_document_ingestion`'s dict (`service.py:571-577`) — 4 of 9 channels are write-only outputs.

**CONFIRMED — the graph is decorative.** All work runs inside `process_document_ingestion` (`service.py:465`) as straight-line Python. Stage list, in order:

1. `object_store.get_object(key_from_s3_uri(object_uri))` `:477`
2. `parse_document` `:478`
3. `classify_document` `:481`
4. `extract_legal_metadata` `:485` (conditional on `classified.graphiti_required`)
5. `repo.update_document_status(status="parsed")` `:490`
6. `segment_chunks` `:507` + `enrich_legal_chunks` `:509`
7. `_embed_chunks` `:514` (`service.py:626`, batched Gemini `aembed_documents` `:636`)
8. `repo.upsert_chunks` `:520`, `analyze_chunks` `:527`, status `stored_postgres` `:528`
9. Graphiti contract-event episodes `:531-552` + `_verify_legal_chunks` `:553` (`service.py:663`, per-chunk `write_and_verify_chunk` `:673`, then a **second** `upsert_chunks` `:686`)
10. final `update_document_status` `:570`

Net effect confirmed: no checkpointing, no per-stage retry, no per-stage state. A crash at stage 9 replays stages 1-8 from scratch. Only recovery signal is the `documents.status` column.

`run_document_ingestion_task` (`service.py:580`) builds **per-invocation**: `init_db()` `:588`, `StorageService.from_settings` `:590`, `_build_chat_model` `:591`, `setup_graphiti` `:596` + `setup_graphiti_indices` `:601`, then `build_document_ingestion_graph` `:605` **inside** the session block, `ainvoke` `:612`, `finally` `close_graphiti` + `engine.dispose()` `:621-623`. **CONFIRMED** `:596` and `:601` are live Graphiti call sites; `:673` → `write_and_verify_chunk` is the third.

Error handling in `documents/`: `Failure` → `app_error_to_exception` raise (`:525`, `:691`); Graphiti episode failures swallowed with `continue` (`:546-552`). No retry wrapper anywhere — contrast `retry_immediate` in `ingestion_kb`.

## 2. Celery wiring

`create_celery_app` — `src/app/connections/celery.py:185`. `include=[...]` at **`:191-196`**:

```
"tasks.auth_email_tasks", "tasks.example", "tasks.search_tasks", "tasks.billing_tasks"
```

**MAJOR FINDING — `tasks.document_tasks` is NOT in `include`.** The live ingestion task `ingest_document` (`src/tasks/document_tasks.py:32`, `name="tasks.documents_ingest"` `:28`) is therefore never imported by the worker process. Unless something else imports it at worker boot, the outbox publish lands a message the worker answers with `NotRegistered`. Same for `tasks.pageindex_tasks` (`:5` `ingest_pageindex_document`), `tasks.document_extraction_tasks` (`:13` `document_extraction.legal_batch`), and `tasks.auth_email_tasks_typed`.

Every `@celery_app.task` in the repo:

| Module | Task name | In `include`? |
|---|---|---|
| `src/tasks/document_tasks.py:27` | `tasks.documents_ingest` | **NO** |
| `src/tasks/search_tasks.py:28` | `tasks.search_ingest` | yes |
| `src/tasks/pageindex_tasks.py:4` | (auto: `tasks.pageindex_tasks.ingest_pageindex_document`) | **NO** |
| `src/tasks/document_extraction_tasks.py:13` | `document_extraction.legal_batch` | **NO** |
| `src/tasks/billing_tasks.py:321,326,331,336,341,346` | `billing.*` (6) | yes |
| `src/tasks/auth_email_tasks.py:35,74` | 2 auth email | yes |
| `src/tasks/auth_email_tasks_typed.py:68,112` | typed auth email | **NO** |
| `src/tasks/example.py:8,16` | `tasks.add`, + 1 | yes |
| `src/app/connections/celery_registry.py:13` | `auth.send_verification_email` | via registry |

`task_routes` `:253-258` sends only `tasks.*` to the default queue; `task_create_missing_queues=False` `:229`. So `billing.*` and `document_extraction.legal_batch` do **not** match `tasks.*` and fall to `task_default_queue` `:225` by default anyway, but there is **no dedicated ingestion queue** — one queue + one DLQ only (`task_queues` `:233-252`).

`beat_schedule` **`:259-276`** — **CONFIRMED 4 entries, all billing**: `billing-invoice-daily`, `billing-dunning-daily`, `billing-receipt-daily`, `billing-reconciliation-daily`. No ingestion, no outbox-relay, no reconciliation of stuck documents.

### Offloaded today vs inline

| Work | Where it runs |
|---|---|
| S3 upload + document row + outbox row | **inline** in request (`documents/service.py:118` `upload_document`) |
| Parse/classify/segment/embed/store/Graphiti (all 10 stages) | Celery `ingest_document` → `run_document_ingestion_task` (`service.py:580`) — offloaded, but as **one indivisible task** |
| Per-stage retry | nowhere — task-level `ResilientTask` retry only, which replays from stage 1 |
| Per-chunk contextualization (`ingestion_kb` `Send` fan-out) | **not offloaded** — in-process LangGraph fan-out, one Gemini call per clause, serial-ish |
| Per-chunk Graphiti verification (`_verify_legal_chunks` `service.py:672`) | **not offloaded** — sequential `for` loop inside the single task |
| Embedding batches (`_embed_chunks` `service.py:635`) | **not offloaded** — sequential batch loop |
| `ingestion_kb` graph as a whole | **not offloaded** — no Celery task references it at all |

Todo (e) gap, precisely: no queue exists for ingestion; the three fan-out-shaped loops above (embedding batches, per-clause contextualize, per-chunk Graphiti verify) are the unoffloaded work; and `document_tasks` is not even registered.

## 3. `app.state` contract — mismatch table

`config.yaml:16-17` documents "shared clients live in lifespan and are read from app.state", so each row below is a **convention violation**, not merely a bug.

### SET by `lifecycle/lifespan.py`
`object_store` `:108,112,270` · `outbox_relay_task` `:131` · `outbox_relay` `:132,289` · `db_engine` `:171` · `db_session_local` `:171` · `mongo_client` `:180,182,185` · `db` `:180,183,186` · `redis` `:190,192` · `neo4j_driver` `:196,198` · `websocket_security` `:200` · `cognee_config` `:207` · `graphiti` `:218,223` · `httpx_client` `:251` · `tavily_http_client` `:254` · `crawl4ai_crawler` `:259,263` · `celery` `:275,278,281`

Also set outside lifespan: `graphiti` (`shared/rag/graphiti/__init__.py:33`), `langgraph_checkpointer` (`shared/langgraph_layer/checkpointer.py:11`), `cognee_config` (`shared/langchain_layer/agents/memory/cognee_client.py:20`).

### READ but NEVER set in lifespan

| Attribute read | Read site | Set in lifespan? |
|---|---|---|
| `saul_graph` | `features/agent_saul/dependencies.py:41` | **NO** — only in the docstring at `registry.py:25` |
| `ingestion_graph` | `features/ingestion/dependencies.py:8` | **NO** — commented `lifespan.py:241` |
| `storage` | `features/profile/router.py:29` | **NO** — lifespan sets `object_store` (`:108`) |
| `mongodb` | `features/profile/router.py:30` | **NO** — lifespan sets `db` (`:180`) |
| `langgraph_checkpointer` | `features/agent_saul/dependencies.py:45`, `shared/langgraph_layer/checkpointer.py:16`, **and read on shutdown at `lifespan.py:317`** | **NO** — commented `:299,305` |
| `tool_registry`, `idempotency_guard`, `saul_checkpointer` | only in `registry.py` docstring `:21,22,26` | **NO** |
| `pageindex_client` | `shared/rag/pageindex/client.py:67` names it in a docstring; no `request.app.state.pageindex_client` read found | commented `:249` |

**NEW mismatches beyond the three known:** `ingestion_graph` (read at `features/ingestion/dependencies.py:8`) and `langgraph_checkpointer` — the latter is read by lifespan's own shutdown path at `:317` guarded by `hasattr` at `:316`, but `features/agent_saul/dependencies.py:45` is unguarded.

`features/ingestion/router.py` is **not** in `src/app/api/v1.py` (7 imports, 6 mounts, `documents_router` and `agent_saul_router` present, no `ingestion_router`) — confirming the brief.

Reads that DO match: `redis` (`connections/redis.py:55`, `crawler/dependencies.py`, `api/generation_with_cb.py:16`, `middleware/health_check.py:43`, `server_middleware.py:164`, `utils/rate_limit/dependencies.py:21`, `agent_saul/dependencies.py:53`), `db` (`connections/mongodb.py:46`), `neo4j_driver` (`connections/neo4j.py:48`, `health_check.py:71`), `db_session_local` (`connections/postgres.py:145`), `db_engine` (`health_check.py:28`), `mongo_client` (`health_check.py:57`), `httpx_client` (`connections/httpx_client.py:67`, `open_deep_search/config.py:72`), `tavily_http_client` (`connections/tavily.py:39`), `websocket_security` (`agent_saul/dependencies.py:67`, `agent_saul/router.py:97,146,155,170,183`).

## 4. Every commented-out block in `lifespan.py`

| Lines | What it would wire | Downstream dependency on it |
|---|---|---|
| `235-240` | `ingestion_llm = ChatGoogleGenerativeAI(...)` (Gemini Flash, `temperature=0.1`, `retries=0`) | sole input to `:241` |
| `241-247` | `app.state.ingestion_graph = build_ingestion_graph(extraction_llm, db_engine, build_embedding_client(), graphiti, redis)` | `features/ingestion/dependencies.py:8` → `IngestionGraphDep` → `features/ingestion/router.py`. Router unmounted, so nothing 500s today. |
| `248` | `logger.info("Contract KB ingestion graph initialized")` | none |
| `249` | `app.state.pageindex_client = PageIndexClient()` | no `app.state.pageindex_client` read found in `src/` |
| `295-305` | `setup_langgraph_checkpointer(conn_string=settings.POSTGRES_URL)` → `app.state.langgraph_checkpointer` | **YES** — `lifespan.py:317` (guarded), `agent_saul/dependencies.py:45` (unguarded), `shared/langgraph_layer/checkpointer.py:16`. Also the only thing that could make either ingestion graph resumable. |
| `shared/rag/strategies.py:305-306` | a strategy taking `app.state.db_engine`/`db_session_local` | outside lifespan; noted for completeness |

Note: the brief's `lifespan.py:235-249` block does **not** contain `IdempotencyGuard`/`tool_registry`/`saul_graph` — that wiring exists only as a docstring example in `shared/rag/graphiti/registry.py:13-29`. The commented lifespan block is purely the ingestion graph + PageIndex client.

**CONFIRMED** `documents/service.py:188` — `event_type="tasks.documents_ingest"` passed to `with_outbox` (`:181-195`), payload `document_id/user_id/filename/content_type/object_uri`. Dispatch is by string; `OutboxRelay._publish` (`shared/outbox/relay.py:118`) resolves it against Celery's registry, which is where the missing `include` entry (§2) bites.

## 5. Todo (f) target state

The attachment point already exists in three places, all consistent:

- **Where:** `lifespan.py:241` — `app.state.ingestion_graph = build_ingestion_graph(...)`, immediately after the Graphiti block (`:210-223`) and after Postgres/Redis are assigned (`:171`, `:190`).
- **Reader:** `features/ingestion/dependencies.py:8` → `IngestionGraphDep` (`:15`) → `features/ingestion/router.py:41` `upload_document(file, graph, user_id, ...)` → `IngestionService(ingestion_graph=graph)` (`features/ingestion/service.py:31`) → `self._graph.ainvoke(initial_state)` (`:67`).
- **Mount:** `src/app/api/v1.py` — `ingestion_router` import + `include_router` are absent.

Constructed **once** at startup (all are closure-captured by the node factories, `graph.py:46-73`): the Gemini `extraction_llm` (`lifespan.py:236-240`), `db_engine` (`:171`), `embedding_fn` from `build_embedding_client()` (`:244`), `graphiti_service` (`:218`), `redis` (`:190`). `build_ingestion_graph` docstring says "once during application startup" (`graph.py:44`).

Constructed **per-request**: only the `IngestionState` dict (`features/ingestion/service.py`, feeds `ainvoke` `:67`) — `raw_bytes`, `doc_id`, `user_id`, `thread_id`, `source`, `filename`, `document_type`, `jurisdiction`.

Blockers, cited:
1. `graph.py:84` — `graph.compile()` takes no `checkpointer=`, so even with `lifespan.py:295-305` uncommented the ingestion graph gets no persistence. `checkpointer` is not a parameter of `build_ingestion_graph` (`graph.py:37-43`).
2. `lifespan.py:295-305` commented — no `AsyncPostgresSaver` exists to pass.
3. `IngestionState.raw_bytes: bytes` (`state.py:172`) and `AppError` in `failure` (`state.py:194`, `arbitrary_types_allowed=True` `:167`) are checkpoint-serialization hazards.
4. `graph.py:78` has no `failure` short-circuit edge, so a resumed run would still walk dead nodes.
5. `ingestion_router` not mounted (`api/v1.py`).
6. `embed_store` opens its own `AsyncSession(db_engine)` (`nodes.py:320`) — the graph owns its transaction, unlike `documents/` where the session is injected by `run_document_ingestion_task` (`service.py:603`).

## 6. `documents/` ↔ `ingestion_kb/` overlap (item 190)

| Concern | `documents/` | `ingestion_kb/` | Verdict |
|---|---|---|---|
| Fetch bytes from S3 | `service.py:477` `object_store.get_object` | none — `raw_bytes` arrives in state (`state.py:172`) from the HTTP upload | **unique to `documents/`** |
| Docling parse | `parse_document` (`service.py:478`) | `_parse_document_with_docling` (`nodes.py:407`), `_sync_parse` `:412`, `asyncer.asyncify` `:439` | **overlap** |
| Classify / metadata extract | `classify_document` `:481` + `extract_legal_metadata` `:485` (2 steps) | `make_extract_schema_node` (`nodes.py:110`) → `ContractMetadata` (1 LLM step) | **overlap, different shape** |
| Segmentation | `segment_chunks` `:507` | `make_segment_document_node` `nodes.py:150` + `_fallback_segments` `:442` | **overlap** — only `ingestion_kb` has a fallback |
| Enrichment / contextualization | `enrich_legal_chunks` `:509` (deterministic) | `make_contextualize_chunk_node` `nodes.py:212` (LLM, `Send` fan-out) + `_build_preamble` `:476` | **overlap, `ingestion_kb` richer** |
| Entity/relationship extraction | **absent** | `make_classify_extract_node` `nodes.py:259` → `_store_entities`, `_store_relationships` | **unique to `ingestion_kb`** |
| Embed | `_embed_chunks` `service.py:626` (batched Gemini) | `_cached_embedding` `nodes.py:716` + `_call_embedding_fn` `nodes.py:738` (Redis-cached) | **overlap** — caching only in `ingestion_kb` |
| Persist chunks | `repo.upsert_chunks` `:520` + `build_chunk_rows` (`repository.py:601`) + `analyze_chunks` `:527` | `_store_chunks` `nodes.py:629` + `_upsert_parent_document` `nodes.py:488` + `_force_merge_bm25` (`nodes.py`, called `:339`) | **overlap, different tables** |
| Status transitions | `repo.update_document_status` `:490,528,570` (`parsed`→`stored_postgres`→`completed*`) | none — only `ingestion_complete: bool` (`state.py:193`) | **unique to `documents/`** |
| Graphiti episodes | contract events `:531-552` + `write_and_verify_chunk` per chunk `:673` | `make_graphiti_upsert_node` `nodes.py:354`, `_graphiti_add_episode`, `_contract_events` `:389` | **overlap** — verification only in `documents/` |
| Retry | none | `retry_immediate` (`shared/langgraph_layer/kb_retry.py:19`) on every I/O | **unique to `ingestion_kb`** |
| Query/search side | `DocumentQueryService` `:229` (`search` `:244`, `rag` `:319`, `ask` `:357`, `_build_query_plan` `:695`, `_graphiti_filter_chunk_ids` `:741`, `_grade_context` `:768`, `_generate_answer` `:791`) | none — retrieval lives in `retrieval_kb/nodes.py` | **seam with `features/search/` — sibling scout's area** |

Union: 11 shared concerns, 3 unique to `documents/` (S3 fetch, status column, Graphiti verify), 2 unique to `ingestion_kb/` (entity/relationship extraction, retry), plus `DocumentQueryService` which is retrieval and not ingestion at all.

## Prior art / specs

`openspec/specs/` has **no ingestion spec**. Nearest in force: `transactional-outbox`, `outbox-helper-extraction`, `llm-injection` (mentions `process_document_ingestion`), `session-required`, `typed-exception-handling`. Archived and relevant: `openspec/changes/archive/2026-06-14-result-adoption-phases-2-5/specs/langgraph-node-result-pattern/spec.md` (binds how graph nodes signal failure — explains the `_state_failure`/`Failure` split at `nodes.py:70-87`), `2026-06-22-celery-outbox-idempotency/specs/transactional-outbox/spec.md`, `2026-06-22-quality-fixes-batch-2/specs/celery-task-registry/spec.md`. In-flight `openspec/changes/`: `cognee-saul-memory-migration`, `mintlify-documentation` — neither touches ingestion.

**No tests found** referencing `ingestion_graph`, `ingestion_kb`, `documents_ingest`, or `process_document_ingestion`. codegraph reports "no covering tests found" for all 7 `ingestion_kb` node factories, `build_document_ingestion_graph`, `process_document_ingestion`, and `run_document_ingestion_task`.

## Fog

- **Whether the Celery worker actually fails on `tasks.documents_ingest`.** I confirmed the module is absent from `include` (`celery.py:191-196`) but did not find the worker's entrypoint/`celeryconfig` or a `conftest`/`__init__` side-effect import of `tasks.document_tasks`. To settle: locate the `celery -A ... worker` command (Dockerfile/compose/Procfile) and check for `imports=` in settings, or run `celery -A app.connections.celery:celery_app inspect registered`.
- **`config.yaml:16-17`** — I did not open it; the convention claim is carried from the brief, not independently cited.
- **`features/ingestion/service.py` error path** (`:69` `log.exception("ingestion_graph_failed")`) — I saw the line via grep but not the full except clause, so I cannot say whether it re-raises or swallows.
- **Whether `IngestionState.raw_bytes` survives `ainvoke`'s Pydantic coercion** across the `Send` boundary at `nodes.py:200` — the fan-out payload is a plain `dict` and `contextualize_chunk_node` types its state as `dict[str, Any]` (`nodes.py:215`) while the graph is compiled with `StateGraph(IngestionState)` (`graph.py:45`, `extra="forbid"` `state.py:167`). Whether LangGraph validates the `Send` dict against `IngestionState` (which would reject `segment`/`contract_metadata` keys) is unresolved; it would take running the graph or reading the installed `langgraph` `Send` handling.
- **Number of `documents/` stages**: I counted 10 discrete operations, not the 7 the brief asserted. If "7 stages" refers to a named list elsewhere, I did not find that list.
