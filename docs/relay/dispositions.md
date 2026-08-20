# Disposition ledger — previously-uncaptured backlog items

Every item `docs/relay/todo-overlap.md` marked `✚` (in scope for this refactor, absent from the user's list),
with a recommended disposition. Line refs are `tests/performance/todo.md` unless stated.

**Count note, on the record:** the sweep's headline said **27**. The overlap table actually carries **37 `✚`
marks**; 2 collapse as duplicates (165≡Up#3, Up#7≡153), leaving **35**. The gap is a counting-method artifact —
the headline appears to have counted only *numbered* todo items (25 of them) and not the `Up#`/`Trap` entries
from the unnumbered "Upgrades"/"Traps" sections (10 more). All 35 are listed here; none are dropped silently.

Disposition vocabulary:

| Code | Meaning |
|---|---|
| **IN n** | Do it, in change *n*. Becomes a task with a Proof. |
| **DEFER** | Real but out of this refactor. Recorded as a Non-Goal in the owning change's `design.md`. |
| **DROP** | Decided against on the merits. Recorded with the reason. |
| **MERGE** | Duplicate of another row; folded into it. |

---

## change 0 — cleanup / foundation

| # | Line | Item | Disposition | Reasoning |
|---|---|---|---|---|
| 198.2 | 199-200 | health check endpoint verifying all clients | **IN 0, narrowed** | **Half of this row is wrong** — `check_graphiti` **already exists** at `features/health/health_check.py:83-90` (found by the change-4 planner, C4). The real gap is **`check_cognee`**, which does not exist at all. Lifespan already degrades silently (`lifespan.py:220-223` sets `app.state.graphiti = None` and continues), so a probe remains the only observable signal that degradation happened — but the work is one probe, not two. Change 4 owns `check_cognee` (its step 8); change 0 keeps only whatever `features/health/service.py:160` still misses. |
| — | — | `todo_temp.py` triage | **IN 0 — delete** | Fog closed: does not parse (`IndentationError` at `:406`), `graphify affected` empty, duplicated draft. See D11. Also moves the ruff baseline 125 → 123. |
| 199 | 190 | `DocumentQueryService.__init__` uses `object \| None` for redis/graphiti | **IN 0** | Was listed under change 2, but it is a pure annotation fix with no schema dependency, so it costs nothing here and unblocks `ty` noise early. |

Also newly promoted into change 0 by the second scout round (not from the `✚` list, recorded here so the
count is honest): **D5.2 `UserIdDep`** (`documents/dependencies.py:61-62` — live `AttributeError` on a mounted
router), **`tasks/__init__.py:6-9`** (imports the reconciliation helpers and re-exports at `:18-20`; deleting
the module without editing this breaks every celery worker at import), **`features/__init__.py:3,8,9`**
(imports `knowledge_base`, `web_scraping` — both all-zero-byte deletion targets), and the
**`profile/router.py:29,30`** `app.state.storage`/`mongodb` name mismatches (lifespan sets `object_store` at
`lifespan.py:108` and `db` at `:180`).

## change 1 — ingestion

| # | Line | Item | Disposition | Reasoning |
|---|---|---|---|---|
| 138 (residue a) | 149 | `AsyncPostgresCheckpointer` on `app.state` | **IN 1** | `agent_saul/dependencies.py:45` reads `app.state.langgraph_checkpointer` **unguarded** and lifespan's own shutdown calls `teardown_langgraph_checkpointer` on it (`:317`), while the setup block is commented out (`:294-305`). Broken today. Docs mandate an async saver in production (`brief:ref:1465`). |
| 138 (residue b) | 149 | `vector_store` singleton on `app.state` | **DROP** | Zero `app.state.vector_store` sites exist. With D5.1, retrieval is raw asyncpg + `pg_textsearch`/pgvector (`search/repository.py:415`), not a LangChain `VectorStore` object. Adding one creates a **third** retrieval path. Gap recorded. |
| 195 | 282 | postgres vector + BM25 + RRF, langextract **before** these, hybrid + re-ranking | **IN 1, reshaped** | Not greenfield (D5.1): BM25 (`search/repository.py:415-419`) and RRF (`fusion.py:28`, k=60) exist. **CORRECTED 2026-08-18 — re-ranking is NOT missing** (see `reviews.md` A6): `CrossEncoderReranker` (`retrieval_kb/reranker.py:19`) is wired as a graph edge, `hybrid_postgres → reranker → context_grader` (`retrieval_kb/graph.py:60-61`), and `nodes.py:203` does `reranker or CrossEncoderReranker()` so it **self-provisions** when nothing injects one. A second ad-hoc path exists at `documents/service.py:426`. Work = harvest BM25/RRF/re-rank into the unified path + **unify the two re-rank call paths** + fill the one real gap (`search/service.py:161 hybrid_search` fuses but never re-ranks) + move langextract upstream (D9). |
| 190 | 298 | can `documents/` move into the ingestion pipeline | **IN 1/2 — the spine** | This *is* the structural question behind D1. It sits immediately after 155's sub-todos and was simply unlisted. `documents/ingestion_graph.py:39` is a 1-node pass-through (`:73-84`) with all 10 stages as straight-line Python in `service.py:465`; `ingestion_kb` is a real 7-node graph. Folding is the change. |
| 162 | 261 | which text splitters; PGVector vs PGVectorStore | **IN 1 as a design decision** | Answer both in `design.md`, not as code tasks: splitter = docling `HybridChunker` (legal hierarchy, todo (a)); PGVector-vs-PGVectorStore = **neither**, the repo uses raw asyncpg + `pg_textsearch`/pgvector. Prevents a third retrieval path being introduced by accident. |
| 163 | 262 | refactor vectorStore code, TSVECTOR | **MERGE → 185** | The TSVECTOR half is item 185 verbatim; the "vectorStore" half is the 0-byte `shared/vectorstore/` trio already in change 0's deletion manifest. Nothing left of its own. |
| 164 | 263 | refactor RAG code | **DROP as umbrella** | No acceptance criterion; it restates "do change 1". Keeping it as a task would produce an unverifiable checkbox, which `schema.yaml` forbids. |
| 171 | 266 | `CacheBackedEmbeddings` for reusing embeddings | **IN 1** | Zero hits in `src/`. Docs name the exact defect: *"`aembed_batch` calls the API every time… LRU cache keyed on SHA256(text)"* (`brief:ref:2049`) and prescribe `CacheBackedEmbeddings` (`brief:13-…:30-54`). Caveat to design: it lives in `langchain_classic`, which collides with the dedicated-import rule — resolve in `design.md`. |
| 176 | 270 | check `sentence_transformers` / `AutoTokenizer`, or a langchain replacement | **IN 1, narrowed further** | **CORRECTED 2026-08-18:** the `sentence_transformers` half is **settled — it stays.** `retrieval_kb/reranker.py:8` imports `CrossEncoder` from it for real cross-encoder re-ranking that is live in the graph (item 195), so it is not "only token counting" and cannot be dropped. Remaining scope is `AutoTokenizer`/`transformers` **only**: if that is used solely for token counting, drop the direct `transformers` dependency. Verification = that import no longer present, with `sentence_transformers` explicitly retained. |
| 165 + Up#3 | 281, 360 | RAG inspired by Uber's enhanced-agentic-RAG | **DEFER** | Aspirational with no acceptance criterion; adopting a whole external architecture would balloon change 1 past reviewable size. The concrete pieces we *do* want from it (hybrid retrieval, re-ranking, agentic query rewriting) are already captured by 195 and todo (d). Gap recorded. |
| Up#4 | 361 | `markitdown` | **DROP** | docling already owns parsing and `documents/parser.py:10` was consolidated onto it (item 186, done). Adding markitdown is a second parser for the same job. pageindex half stays deferred (D4). |
| 198.3 | 201-202 | embedding dimension hardcoding | **IN 1 — urgent** | Partially done (`settings.EMBEDDING_DIMENSION` + validator, `get_embedding_dimension()`), but ORM columns still hardcode `Vector(768)` (`documents/model.py:94`, `search/model.py:73`, `memory_schema.py:218`) **and there is a live conflict**: `document_processing/embedder.py:26-29` returns `{"dimensions": 1536}` for `gemini-embedding-001`. A 1536-vec insert into `Vector(768)` raises `DataError`. This is a correctness bug, not tidying. |
| 198.4 | 203-204 | Celery task definitions scattered, string dispatch, no type safety | **IN 1, re-ranked** | **Superseded framing — see `docs/relay/findings-deployment.md` §1-§3.** The `include` omission (`connections/celery.py:191-196`) is a *latent fragility*, not a live break: `tasks/__init__.py:4` imports `document_tasks`, so the task IS registered transitively and breaks only when that file is tidied. The live breaks are bigger and rank ahead of it: (1) **no worker or beat service exists in `docker-compose.yml` at all** — nothing consumes the queue, so every dispatched task enqueues forever; (2) `Makefile:52` runs `celery -A celery_config`, and **`celery_config` does not exist** in the repo. Also still missing from `include`: `pageindex_tasks`, `document_extraction_tasks`, `auth_email_tasks_typed`. |
| Up#5 | 362-373 | Pointer State — store UUIDs in state, fetch in StateModifier | **IN 1** | Gates the checkpointer from 138: legal documents in `state["messages"]` means every checkpoint write serializes full document text. Docs corroborate via `JsonPlusSerializer` behaviour (`brief:ref:1604-1609`). |
| Trap1 | 483 | Graphiti entity dedup — canonicalise, write `party_id` not raw text | **IN 1** | Zero `canonical*` hits in `src/`. Must land **before** any Graphiti write or the graph accumulates duplicate party nodes that no later pass can separate. Cheap now, unfixable later. |

## change 2 — documents

| # | Line | Item | Disposition | Reasoning |
|---|---|---|---|---|
| 184 | 272-275 | documents/chunks as sole retrieval truth; Option A = leave clause code stale | **IN 2, as an ADR** | The A/B decision was never recorded. Recommendation: **A+** — do not leave it stale, *retarget* it. The clause readers are `precedent_tools.py:237` (a stub returning `[]`) and `search/repository.py:308-405`; under D5.1 both are in scope. Leaving 20 files reading a table that no migration creates is how the invisible-failure register got this long. |
| 185 | 276 | remove `ts_vector` from search/document, write correct SQL | **IN 2, both halves** | Unblocked by D5.1. `content_tsv` (`search/model.py:75-79`) is a STORED generated column with a live GIN index and **zero readers** → pure subtraction. |

Change 2 also owns the four non-mechanical cells of the schema collapse (from `scout-search.md` §2), which
are prerequisites rather than backlog items: `chunks.user_id` / `documents.object_uri` NOT NULL with no source
value; `UnifiedChunk` has no `updated_at`; the hardcoded constraint name in
`search/repository.py:157` (`on_conflict_do_update(constraint="uq_search_chunks_document_chunk_index")`); and
`ix_search_chunks_content_trgm` having no target equivalent.

## change 3 — tools

| # | Line | Item | Disposition | Reasoning |
|---|---|---|---|---|
| 173 | 268 | rewrite the tools for the new graphiti/cognee | **IN 3 — the core** | This is change 3's whole thesis. |
| Up#10 | 432-440 | tool output normalization — one `ToolResult` | **IN 3** | **CORRECTED 2026-08-18 — there are FOUR competing definitions, not three** (found by change 3's reviewer, verified by the orchestrator). The fourth is **`ToolOutput`** (`shared/langchain_layer/agents/tools/base.py:30`) — `success`/`data`/`error`/`metadata` with `ok()`/`fail()` classmethods and a `to_agent_string()` that returns `f"ERROR: {self.error}"`, i.e. the string-as-error anti-pattern itself — with **13 uses in `tools/shell.py`**. Critically, it is the class the **deployed** `typed-exception-handling` spec already governs, naming `ToolOutput.fail()` in five scenarios (`openspec/specs/typed-exception-handling/spec.md:219,223,227,235,239`). The other three: `langchain_layer/agents/tools/idempotency.py:34` (survivor), `shared/agents/tools/idempotency.py:11`, `rag/document_processing/models.py:318`. Direct extension of D6.1. **Consequence:** any gate of the form `rg -c "^class ToolResult"` 3→1 passes while `ToolOutput` survives, so the gate must count all four envelopes. |
| Up#11 | 441-447 | citation enforcement — claim / source / confidence | **IN 3** | Cheap (an output schema plus a validator) and disproportionately valuable for a legal product where an uncited assertion is the failure mode. |
| Up#6 | 375-385 | "Lost in the Middle" prompt ordering | **IN 3** | A prompt-assembly ordering rule, near-free to implement, and it lands naturally with the `SystemPromptParts` adoption work already in change 3. |
| 153 + Up#7 | 257-259, 387-395 | hydration node after checkpointer, `schema_version` | **IN 3** | Half-built already: `agent_saul/state.py` carries `schema_version: int` documented as guarding a hydration node, and **no node exists**. Finishing a half-built thing is cheaper than the ambiguity it leaves. |
| Trap2 | 484 | hash structural IDs (clause_id, doc_id), never content | **IN 3** | A one-line rule inside the surviving `IdempotencyGuard`, and docs independently require idempotency keys because nodes replay from the beginning after `interrupt` (`brief:ref:1614,1628`). |
| 172 | 267 | prebuilt + custom langchain middlewares | **IN 3, narrowly** | Scope to `@wrap_model_call` — the docs' retry seam (`brief:05-…:93-105`) — plus `ToolNode(handle_tool_errors=…)` (`brief:ref:38`). **This carries a real conflict with sub-todo (j)**: the reference doc has **zero** mentions of `tenacity`, `RetryPolicy`, or `.with_retry()`, and `brief:ref:1633` forbids wrapping `interrupt` in a bare `try/except` — which condemns tenacity's catch-all inside graph nodes. Recommendation: tenacity stays at I/O-client boundaries (where it already is: `kb_retry.py`, `connections/redis.py`, `razorpay_client.py`), and **middleware, not tenacity, owns model/tool retries**. |
| Up#9 | 423-431 | Result Validation Layer → Accept / Retry / Escalate | **SPLIT: IN 3 (cheap half) + DEFER (state machine)** | The cheap half is already supported: `create_agent(response_format=…)` (`brief:04-…:53-72`) or `with_structured_output(include_raw=True)` (`:79`), and `ProviderStrategy.strict` needs `>=1.2` — installed langchain is 1.2.12. The Accept/Retry/Escalate escalation machine is a separate subsystem; defer. |
| 67 | 249 | structured message bus / ACP / persistent shared state | **DROP** | LangGraph's native answer is already prescribed: handoff as an `AIMessage` carrying a `transfer_to_*` tool call with a router edge on it (`brief:ref:1473-1479`), which is what sub-todo (i) asks for. A bespoke message bus on top is duplicate machinery. Gap recorded. |

**Correction that change 3 must respect** (from `brief-langgraph-practices.md`, and it contradicts sub-todo (i)
as literally written): the docs never use `MessagesState`. They prescribe a `TypedDict` with
`Annotated[list, add_messages]` plus sibling channels, and are explicit that *"custom state schemas must be
TypedDict… Pydantic models and dataclasses are no longer supported"* (`brief:ref:1341-1345`). Sub-todo (i)'s
intent (standardise A→B message passing) is honoured; its named vehicle is not the documented one.

## change 4 — cognee

| # | Line | Item | Disposition | Reasoning |
|---|---|---|---|---|
| 174 | 269 | proper cognee functions from docs | **IN 4 — the core** | **`cognify` has zero call sites in `src/`.** Nothing has ever been ingested into Cognee. |
| Trap3 | 485 | `cognify()` is a full rebuild — batch `add()`, defer `cognify()` to nightly beat | **IN 4 as a binding design constraint** | Because no `cognify` call exists yet, this constrains the design *before* it is written rather than forcing a rewrite after. Cheapest possible moment to honour it. |
| 152 | 256 | cogneeRetriever — vertex ai vs google_genai | **IN 4 — load-bearing** | Cognee defaults to `openai/text-embedding-3-large` @ **3072** dims against the repo's **768** (`settings.py:212`), and `set_vector_db_config` is never called so Cognee defaults to `vector_db_provider="lancedb"` — local files, not the app's Postgres. Two config bugs, not a research question. |
| 140 | 253 | GRAPH_COMPLETION_COT / FEELING_LUCKY router > 0.8; Neo4j needs APOC + GDS or `cognify()` fails silently | **SPLIT: IN 4 (prerequisite) + DEFER (router)** | The APOC/GDS prerequisite is an operational blocker with a silent failure mode — it must be a documented precondition and a health probe. The COT/router threshold is tuning; defer. |
| 148 + Up#13 | 279, 449-481 | memory taxonomy; architecture A persistent / B graph / C episodic | **MERGE → IN 4 as the ADR** | These two are one question, and it is the **D2 role boundary still open**. It must be settled because three places in the code disagree: `cognee_client.py:12-15` gives final reports to Cognee, `write_final_report.py:8-13` routes them to **both**, and the existing `design.md` says Cognee primary. An ADR is required (it outlives the change). |
| 179 | 271 | caching plan — redisvl, langcache, does cognee take a redis instance | **SPLIT: IN 4 (the narrow question) + DEFER (redisvl/langcache)** | "Does Cognee need its own redis" is a config fact change 4 must know. redisvl/langcache adoption is unrelated research. |
| 170 | 265 | cron for memory decay → celery | **DROP — see D10** | Never registered (no task decorator, module absent from `include`, `beat_schedule` has 4 billing entries only, its 4 tables never created). Its `_compute_decay` (`memory_decay_reconciliation_tasks.py:51`) is the repo's only decay formula and dies with reconciliation. **The gap is real and must appear in change 4's Non-Goals**: Cognee memory will grow without decay, curation, or dedup — which the repo's own proposal already concedes (`cognee-saul-memory-migration/proposal.md:20-21`). |

---

## Unclassifiable without user intent (from `todo-overlap.md` §7)

| # | Line | Item | Recommendation |
|---|---|---|---|
| 151 | 255 | add `langchain-cisco-aidefense`, `compact-middleware`, `langchain-collapse` | **DEFER** — three new dependencies, no stated criterion. Partially overlaps 172's middleware work; note it there. |
| 159 | 241 | discover RAGFlow / OpenRAG, use or not | **DEFER** — research, and D5.1 already commits to the existing `pg_textsearch` path. |
| 194 | 260 | `headroom-ai` for compression | **DEFER** — new dependency, no criterion. |
| (h) | — | "research for RAG pipeline with Gemini" | **Treat as satisfied by items 162 + 195**, both IN 1. No separate todo-file counterpart exists. Stated as an assumption, overturnable. |

## Fog closed since the sweep

- `todo_temp.py` caller set → **closed**, dead (D11).
- `notes.md` claim `documents/dependencies.py:62` unset `request.state.user_id` → **closed, CONFIRMED** (D5.2).
- Item 199's `object | None` typing → still unopened, but the fix is unconditional, so it is a task either way.
- Whether `ingestion_kb` has a Celery path → **closed**: it has none; the outbox event at
  `documents/service.py:188` is the only dispatch, and its target task is unregistered (198.4).

## Fog still open and assigned

- Live-vs-orphan status of `parent_documents`, `events`, `memory_versions` — needs `\dt` against a real DB.
  Assigned to change 0's plan as a precondition check, not a guess.
- Whether the unwired Saul graph is intentional or a regression. Treated as **regression** (D5.2 logic: an
  identical defect on the documents router is live), overturnable.
- `shared/agents/**` shadow duplicates: `precedent_tools.py:21` imports the **30-byte shadow**
  `memory/memory_scope.py` while `graphiti/subgraph.py:30` imports the real 7189-byte path. Both live. Change 3
  must rewrite importers **before** change 0 deletes the shadow, or `registry.py:41-46`'s eager imports raise
  `ImportError` at boot (D6.1).
