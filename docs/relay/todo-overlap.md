# Todo backlog overlap sweep — refactor items 210 / 155

Scope: full sweep of `tests/performance/todo.md` (726 lines), `notes.md` (141 lines),
`src/app/shared/rag/document_processing/todo_temp.py`. Cross-checked against
`docs/relay/decisions.md` D1–D8 so "NOT captured" excludes anything already locked there.

Counts: **ALREADY DONE 11 · IN SCOPE captured 14 · IN SCOPE NOT captured 27 · OUT OF SCOPE 12 · UNCLEAR 5**

---

## 1. Item 138, verbatim

`tests/performance/todo.md:149`

> `138. add neo4j driver, DB session from request.app.state in Graphiti, Cognee, AsyncPostgresCheckpointer, vector_store and other places where required in tools and do the same for DB, redis            DONE`

**It is marked DONE. It is not done.** Sub-todo 1 cites it, and the residue is precisely the
unfinished half:

| Resource | On `app.state`? | Evidence |
|---|---|---|
| `neo4j_driver` | yes | `src/app/lifecycle/lifespan.py:196` |
| `graphiti` | yes | `src/app/lifecycle/lifespan.py:218` |
| `cognee_config` | yes (config object, not a client) | `src/app/lifecycle/lifespan.py:207` |
| `redis` | yes | `src/app/lifecycle/lifespan.py:190` |
| `db_engine` / `db_session_local` | yes | `src/app/lifecycle/lifespan.py:171` |
| **`AsyncPostgresCheckpointer`** | **no — commented out** | `src/app/lifecycle/lifespan.py:294-305` |
| **`vector_store`** | **no — zero `app.state.vector_store` sites** | grep across `src/` returns nothing |
| **`ingestion_graph`** | **no — commented out** | `src/app/lifecycle/lifespan.py:241-248` |
| `pageindex_client` | no — commented out | `src/app/lifecycle/lifespan.py:249` |
| `tool_registry` / `idempotency_guard` / `saul_graph` | **docstring only** | `src/app/shared/rag/graphiti/registry.py:1-33`; `build_saul_graph` has **zero callers** |

**Implication:** item 138 = sub-todo (f) *plus two uncaptured pieces* — the `AsyncPostgresSaver`
checkpointer and a `vector_store` singleton. Teardown at `lifespan.py:316-317` already calls
`teardown_langgraph_checkpointer` on an attribute that is never set.

## 2. Item 177, verbatim

`tests/performance/todo.md:152-162`

> `177.`
> ` PydanticDeprecatedSince20: json_encoders is deprecated. …`
> `…/open_deep_search/deep_researcher.py:701: LangGraphDeprecatedSinceV10: config_schema is deprecated … use context_schema`
> `…:701: LangGraphDeprecatedSinceV05: input is deprecated … use input_schema`
> `…:589: LangGraphDeprecatedSinceV05: output is deprecated … use output_schema`
> `…/open_deep_search/configuration.py:214: PydanticDeprecatedSince20: Using extra keyword arguments on Field … (Extra keys: 'optional', 'metadata') … mcp_config: MCPConfig | None = Field(   DONE`

**Genuinely resolved.** Confirmed on disk:
- `json_encoders` — zero hits anywhere outside `todo.md:153` itself.
- `config_schema` — zero hits in `src/`; call sites now `context_schema`
  (`src/app/shared/langgraph_layer/open_deep_search/graph.py:267,463,540`).
- `input=`/`output=` → `input_schema`/`output_schema`
  (`open_deep_search/graph.py:462,539`; `agent_saul/graph.py:104`).
- Both cited files are gone: `deep_researcher.py` and `configuration.py` no longer exist
  (dir now holds `config.py graph.py prompts.py state.py tools.py utils.py __init__.py`); no
  `Field(...optional=|metadata=)` misuse remains.

Matches `decisions.md:90-92`. Todo (g) is therefore the `Field(description=...)` work, not this.

---

## 3. Overlap table

`✚` = NOT captured (payload). Line numbers are `tests/performance/todo.md`.

| # | Line | Verbatim (trimmed) | Area | Class |
|---|---|---|---|---|
| 138 | 149 | add neo4j driver, DB session from request.app.state in Graphiti, Cognee, AsyncPostgresCheckpointer, vector_store … | LangGraph/DI | ✚ change 1 (checkpointer + vector_store) / captured (graph→app.state = todo f) |
| 177 | 152 | PydanticDeprecatedSince20 `json_encoders` … `config_schema` … Field extra keys | prompts/config | ALREADY DONE |
| 43 | 67 | add langextract to agent tools   DONE | tools | ALREADY DONE (`src/app/shared/rag/langextract/`, `src/tasks/document_extraction_tasks.py`) |
| 33 | 68 | add pageindex properly and include it in agent tools   DONE | tools | OUT (deferred, todo b; `src/tasks/pageindex_tasks.py` raises NotImplementedError per `decisions.md:D4`) |
| 79 | 138 | perf optimisation in pageindex/langextract, pydantic or dataclass, replace asyncio with asyncer | ingestion | ALREADY DONE (`document_processing/models.py` all BaseModel — `decisions.md:93-94`) |
| 136 | 225 | use LangExtract outputs to build rich graph knowledge from your legal documents.  **ABANDONDED** | graphiti | **CONTRADICTION** — see §5 |
| 57 | 280 | No agent-to-agent message passing format standard … standardized AIMessage/ToolMessage | LangGraph state | captured (todo i, `MessagesState`) |
| 67 | 249 | go and learn … multi-agent communication system … structured message bus, ACP, persistent shared state | LangGraph state | ✚ change 3 |
| 140 | 253 | cognee GRAPH_COMPLETION_COT if FEELING_LUCKY router > 0.8 … Neo4j must have APOC + GDS or cognify() fails silently | Cognee | ✚ change 4 |
| 148 | 279 | figure out the types of memory an agent can have … cognee, honcho, episodic | Cognee | ✚ change 4 |
| 151 | 255 | add langchain-cisco-aidefense, compact-middleware, langchain-collapse | middleware | UNCLEAR |
| 152 | 256 | see cogneeRetriver how does vertex ai differ from google_genai | Cognee/embeddings | ✚ change 4 |
| 153 | 257-259 | add a hydration node after checkpointer … StateHydrationNode … schema_version | LangGraph state | ✚ change 3 — `schema_version: int` **exists** at `src/app/shared/langgraph_layer/agent_saul/state.py` ("guards hydration node") but **no hydration node is implemented** |
| 162 | 261 | what kind of text splitters do i need. diff in PGvector and pgvectorstore in langchain | ingestion | ✚ change 1 |
| 163 | 262 | refactor vectorStore code   TSVECTOR, | documents | ✚ change 1/2 |
| 164 | 263 | refactor RAG code | ingestion | ✚ change 1 |
| 165 | 281 | implement RAG inspired from uber enhanced-agentic-rag | ingestion | ✚ change 1 (dup of Upgrades#3) |
| 170 | 265 | write cron job for memory decay and then send to celery for off loading | Celery/Cognee | ✚ change 4 — task file exists (`src/tasks/memory_decay_reconciliation_tasks.py`) but is **absent from `beat_schedule`** (`src/app/connections/celery.py:259-276` holds only 4 billing entries) |
| 171 | 266 | use CacheBackedEmbeddings for reusing embeddings | embeddings | ✚ change 1 — zero `CacheBackedEmbeddings` hits in `src/` |
| 172 | 267 | use prebuilt and custom middlewares in langchain | agent config | ✚ change 3 |
| 173 | 268 | rewrite the tools for the new grpahiti, cognee etc | tools | ✚ change 3 |
| 174 | 269 | add proper cognee functions, graphiti from docs | Cognee | ✚ change 4 — **`cognify` has zero call sites in `src/`** |
| 176 | 270 | check sentence_transformers, AutoTokenizer … or replaced by a langchain package | embeddings | ✚ change 1 |
| 179 | 271 | proper plan for caching … redisvl, langcache, does cognee take a redis instance too? | Cognee/cache | ✚ change 4 |
| 184 | 272-275 | documents/chunks should be the sole retrieval truth … Agent Saul / precedent / reconciliation code still reads clauses directly. Option A: leave old clause code stale/disabled | documents | ✚ change 2 — **decision A/B never recorded**; 20 files still reference clauses |
| 185 | 276 | remove ts_vector … from search/document and write correct SQL for documents/ | documents | ✚ change 2 (documents half only) — **collides with D5**, see §5 |
| 186 | 172 | documents/ uses docling from shared and doesnt uses its own one.  DONE | ingestion | ALREADY DONE (`src/app/features/documents/parser.py:10` imports `create_document_converter` from `app.shared.rag.document_processing`) |
| 190 | 298 | see if documents/ can be moved in ingestion pipeline with langextract, pageindex, graphiti, postgres | ingestion | ✚ change 1/2 — sits immediately after 155's sub-todos and is unlisted |
| 195 | 282 | ingestion pipeline postgres + extensions for vector + BM25 + RRF, graphiti …, **langextract before these**, pageindex parallel … hybrid-search-and-re-ranking | ingestion | ✚ change 1 — the *ordering constraint* (langextract upstream) is new |
| 196 | 283 | asyncio.gather in researcher_subgraph / crawl_executor | ODS | OUT (D7) |
| 198.1 | 195-196 | HYBRID SEARCH CACHING RACE … use redis.setnx as computing lock | documents | ALREADY DONE (`src/app/features/documents/service.py:254-255`) |
| 198.2 | 197-198 | GRAPHITI INITIALIZATION ORDER … add health check endpoint verifying all clients | lifespan | **PARTIAL** — inconsistency warning added (`lifespan.py:225-233`), health service checks neo4j (`src/app/features/health/service.py:160`) but **not graphiti/cognee** → ✚ change 0 |
| 198.3 | 199-200 | EMBEDDING DIMENSION HARDCODING … make it configurable | ingestion | **PARTIAL** — `EMBEDDING_DIMENSION: int = Field(default=768, gt=0)` + validator in `src/app/config/settings.py`, `get_embedding_dimension()` in `document_processing/embedder.py`, but ORM columns still hardcode `Vector(768)` at `src/app/features/documents/model.py:94` and `src/app/features/search/model.py:73` → ✚ change 1/2 |
| 198.4 | 201-202 | CELERY TASK DEFINITIONS SCATTERED … invoked via string names, no type safety | Celery | ✚ change 1 — still string dispatch: `event_type="tasks.documents_ingest"` at `src/app/features/documents/service.py:188` |
| 198.5 | 203-204 | MIDDLEWARE ORDER SUBTLE BUG | middleware | OUT (unrelated area) |
| 199 | 190 | `DocumentQueryService.__init__` uses `object \| None` for redis/graphiti — should be `Redis \| None` / `Graphiti \| None` | documents | ✚ change 2 (marked DONE; typing claim unverified → Fog) |
| 204 | 206-211 | Cognee 1.0 changes: remember/recall/forget/improve; session memory; multi-user access control | Cognee | captured by in-flight `openspec/changes/cognee-saul-memory-migration` |
| 178 | 181 | Cognee multi-user access control … migrate to version 1.1 | Cognee | ALREADY DONE (spec `openspec/specs/cognee-v1-api/`) |
| 211 | 224 | check agent router usage | agent config | captured (change 3) |
| 220 | 302 | check the alembic warning having 2 heads | migrations | captured (change 0, `decisions.md:72`) — **confirmed real**: `8a7d9b1c2e3f` and `a71f0d7d9c12` both `down_revision="2bc7726317f6"` (`src/alembic/versions/8a7d9b1c2e3f_add_search_documents_and_chunks.py:19`, `.../a71f0d7d9c12_add_unified_documents_and_chunks.py:17`) → heads `0004` + `a71f0d7d9c12` |
| Up#2 | 359 | add celery for offloading ingestion to a queue.  DONE | Celery | **PARTIAL** — outbox event emitted (`documents/service.py:188`) but `ingestion_kb` graph has no Celery path; captured as todo (e) |
| Up#3 | 360 | make ingestion pipeline inspired from uber | ingestion | ✚ change 1 |
| Up#4 | 361 | add pageindex for vectorless RAG, **markitdown** | ingestion | OUT for pageindex (D4); `markitdown` ✚ change 1 |
| Up#5 | 362-373 | Pointer State Pattern — do not store document content in `state["messages"]`; store UUIDs, fetch in StateModifier | LangGraph state | ✚ change 1/3 — gates the checkpointer from #138 |
| Up#6 | 375-385 | "Lost in the Middle" — enforce prompt order: context top, history middle, system prompt + schema bottom | prompts | ✚ change 3 |
| Up#7 | 387-395 | State Schema Migrations / StateHydrationNode (dup of 153) | LangGraph state | ✚ change 3 |
| Up#7b | 398-412 | Add Idempotency Layer — `idempotency_key = hash(step_id + input + user_id)` | tools | captured (D8 change 3, idempotency unification) |
| Up#9 | 423-431 | Introduce Result Validation Layer via pydantic (Post-LLM) → Accept/Retry/Escalate | agent config | ✚ change 3 |
| Up#10 | 432-440 | Introduce Tool Output Normalization Layer — all tools output `ToolResult(success, data, error, metadata)` | tools | ✚ change 3 — **three competing `ToolResult` classes**: `src/app/shared/langchain_layer/agents/tools/idempotency.py:34`, `src/app/shared/agents/tools/idempotency.py:11`, `src/app/shared/rag/document_processing/models.py:318` |
| Up#11 | 441-447 | Citation Enforcement Layer — claim/source/confidence | agent config | ✚ change 3 |
| Up#13 | 449-481 | Memory Architecture A persistent / B graph / C episodic | Cognee/Graphiti | ✚ change 4 |
| Trap1 | 483 | Graphiti entity deduplication trap … run entity canonicalisation, write `party_id` not raw text | Graphiti | ✚ change 1/4 — zero `canonical*` hits in `src/` |
| Trap2 | 484 | Idempotency key collision … always hash structural IDs (clause_id, doc_id), never content | tools | ✚ change 3 |
| Trap3 | 485 | `cognify()` is a full graph rebuild, not an append … batch `add()`, defer `cognify()` to a nightly Celery beat | Cognee/Celery | ✚ change 4 — no `cognify` call exists yet, so this constrains the design before it is written |
| 8 | 413-421 | Execution Budgeting System (max_tokens/tool_calls/cost/latency) | agent config | OUT (marked "maybe in future") |
| 12 | 448 | JIT permission, IAM model | auth | OUT (future) |
| 1 | 358 | DSPy prompt compilation  ABANDONED | prompts | OUT (abandoned) |
| 159 | 241 | discover RAGFlow, OpenRAG if or if not to use it | ingestion | UNCLEAR (research) |
| 194 | 260 | add headroom-ai for **comrpression** | prompts | UNCLEAR |
| 161 | 227-233 | FP patterns, ROP, flow()/bind()/map()  DELAYED | cross-cutting | OUT (delayed) |
| 155(old) | 235 | ripgrep/tree-sitter/zoekt as a search tool replacing a vector DB  DELAYED | ingestion | OUT (delayed; note the **duplicate item number 155**) |
| 64 · 116 · 61 · 53 · 99 · 44 · 115 · 153(perf) · 156 · 157 · 158 · 160 · 218 · 219 | 245-252, 236-243, 300-301 | eval framework, rate-limit/CB redesign, PDF gen, voice, promptfoo, crawler fix, inter-layer logs, perf tests, langsmith, terraform, tests, ports&adapters, glossary, subagent prompts | unrelated | OUT |

### Other backlog files

- **`notes.md`** (141 lines, repo root) — a code-review transcript, not a numbered backlog. Overlapping
  claims, all verified: `build_saul_graph` has zero callers and `app.state.saul_graph` exists only in the
  `src/app/shared/rag/graphiti/registry.py:1-33` docstring (**confirmed**); checkpointer block commented
  (`lifespan.py:294-305`, **confirmed**); alembic forked (**confirmed**); `documents/dependencies.py:62`
  reads `request.state.user_id` that nothing sets (**unverified — Fog**); `nodes.py:347` hardcodes
  `text = ""` and `_extract_risk_output` returns empty findings (**unverified — Fog**, file path ambiguous
  between `ingestion_kb/nodes.py` and `agent_saul/nodes.py`).
- **`src/app/shared/rag/document_processing/todo_temp.py`** — 783 lines of *live-looking* code
  (`extract_tables`, imports `docling_enhanced`, `entity_extractor`, `models`), not a todo list. Name says
  temporary; caller set unverified. **Change 0 deletion candidate — Fog.**
- No `TODO`/`BACKLOG` file exists anywhere outside `.venv`.

---

## 4. NOT-captured items, grouped by target change

**change 0 — cleanup:** 198.2 (health endpoint must cover graphiti + cognee, not just neo4j);
`todo_temp.py` triage.

**change 1 — ingestion:** 138-residue (`AsyncPostgresCheckpointer` + `vector_store` on `app.state`);
195 (langextract *upstream* of postgres/graphiti; BM25+RRF; re-ranking); 190 (fold `documents/` into the
pipeline); 164; 163; 162 (splitter choice; PGVector vs PGVectorStore); 171 (`CacheBackedEmbeddings`);
176 (sentence_transformers/AutoTokenizer); 165 + Up#3 (Uber agentic-RAG shape); Up#4-markitdown;
198.3 (un-hardcode `Vector(768)`); 198.4 (typed Celery task signatures); Up#5 (Pointer State — gates the
checkpointer); Trap1 (entity canonicalisation before Graphiti write).

**change 2 — documents:** 184 (A-vs-B decision on clause-reading code — 20 files affected);
185 (documents-side `ts_vector` removal only); 199 (`object | None` → `Redis | None`/`Graphiti | None`).

**change 3 — tools:** 173; Up#10 (three competing `ToolResult` classes); Up#9; Up#11; Up#6 (prompt
ordering); 153 + Up#7 (hydration node — `schema_version` exists, node does not); Trap2 (hash structural
IDs, never clause text); 67 (structured message bus / ACP); 172 (langchain middlewares).

**change 4 — cognee:** 174 (**no `cognify` call exists in `src/`**); Trap3 (batch `add`, nightly beat
`cognify`); 170 (memory-decay task exists but is not in `beat_schedule`); 140 (COT/router + APOC/GDS
prerequisite); 152 (cogneeRetriever, vertex vs google_genai); 179 (redisvl/langcache; cognee's redis);
148 (memory taxonomy); Up#13 (A/B/C memory architecture).

---

## 5. Contradictions

1. **LangExtract: abandoned vs load-bearing.** Item 136 (`:225`) `use LangExtract outputs to build rich
   graph knowledge from your legal documents.  ABANDONDED` — but sub-todo (b) schedules langextract, and
   item 195 (`:282`) makes it a *prerequisite* stage (`need to have langextract before these as well`).
   Item 43 (`:67`) already shipped it. Three positions, one feature.
2. **Item 185 vs D5.** `185. remove ts_vector … from search/document` requires editing
   `src/app/features/search/model.py:73` — `decisions.md:44` says search is *entirely* out of scope,
   tables included. The documents half is actionable; the search half is barred.
3. **"Remove reconciliation" vs "break up reconciliation".** Item 155 (`:285`) says *remove* it entirely;
   sub-todo 1 says *break up* the reconciliation code in `langgraph_layer/` and `features/`. Also
   `src/tasks/memory_decay_reconciliation_tasks.py` is the vehicle item 170 asks for — deleting
   reconciliation deletes the memory-decay cron with it.
4. **Item 138 marked DONE vs sub-todo (f).** Sub-todo (f) re-asks for exactly what 138 claims to have
   done. The `DONE` marker is false (§1) — any plan that trusts it skips the checkpointer.
5. **Item 33 `pageindex … DONE` vs sub-todo (b) "leave this for now".** `src/tasks/pageindex_tasks.py`
   raises `NotImplementedError` (`decisions.md:D4`), so `DONE` is false, and `decisions.md` defers it.
6. **Duplicate item numbers.** `155`, `151`, `152`, `153`, `172`, `197`, `76`, `77`, `32`, `65`, `66`
   each appear twice with unrelated content. Any plan citing "item 15x" by number alone is ambiguous.

---

## 6. Already-done ledger

| # | Proof |
|---|---|
| 177 | `json_encoders`/`config_schema` zero hits in `src/`; `deep_researcher.py` + `configuration.py` deleted; `open_deep_search/graph.py:267,462-463,539-540` use `context_schema`/`input_schema`/`output_schema` |
| 186 | `src/app/features/documents/parser.py:10` → `from app.shared.rag.document_processing import create_document_converter` |
| 198.1 | `src/app/features/documents/service.py:254-255` `setnx` computing lock |
| 43 | `src/app/shared/rag/langextract/langextract_batch_processor.py`, `langextract_to_graph.py`, `src/tasks/document_extraction_tasks.py` |
| 79 | `src/app/shared/rag/document_processing/models.py` — all `BaseModel` (per `decisions.md:93-94`) |
| 178 / 204 | spec `openspec/specs/cognee-v1-api/`; in-flight `openspec/changes/cognee-saul-memory-migration/` |
| Up#7b (partial) | `ToolResult` + guard at `src/app/shared/langchain_layer/agents/tools/idempotency.py:34` |
| tenacity (todo j, partial) | `src/app/shared/langgraph_layer/kb_retry.py`, `src/app/connections/redis.py`, `src/app/features/billing/clients/razorpay_client.py` |
| toons (todo 1, partial) | `src/app/utils/toon_parser.py` + 7 call-site files (`shared/langchain_layer/prompts.py`, `retrieval_kb/nodes.py`, `ingestion_kb/nodes.py`, `features/documents/service.py`, …) |
| SystemPromptParts (todo 1, partial) | `src/app/shared/langchain_layer/prompts.py`, `agents/registry.py`, `agents/factory.py` |
| 220 | 2 heads confirmed real (see overlap table) — todo was right, work is outstanding |

**`init_embedding` (todo 1) is NOT done** — zero hits in `src/`.

---

## 7. Fog

- **`todo_temp.py` caller set.** 783 lines with real imports. I did not run `graphify affected` on its
  symbols, so I cannot say whether it is dead. D4 requires proven-empty callers before deletion.
- **`notes.md` runtime-break claims** (`documents/dependencies.py:62` unset `request.state.user_id`;
  `nodes.py:347` `text = ""`; `_extract_risk_output` empty findings). Cited by line but from an
  unknown-vintage transcript; the `nodes.py` path is ambiguous between two files. Establishing: read
  those three exact lines.
- **Item 199's `object | None` typing.** Marked DONE; I did not open
  `DocumentQueryService.__init__` to check the annotations.
- **Whether `ingestion_kb` has any Celery path at all.** `documents/service.py:188` emits an outbox
  event; `ingestion_kb/nodes.py` was not inspected for task dispatch. Establishing:
  `graphify explain "ingestion_kb/nodes.py"`.
- **Items 151(`:255`), 159, 194** are research prompts with no acceptance criterion. Cannot classify
  without the user stating intent.
- **Sub-todo (h) "research for RAG pipeline with Gemini"** has no todo-file counterpart I could find,
  so I cannot tell whether items 162/195 are its intended content or separate work.
