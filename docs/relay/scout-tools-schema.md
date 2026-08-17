# Scout — Tools leg: which tables agent tools query, and their UnifiedDocument mapping

Scope: `src/app/shared/langchain_layer/agents/tools/*` + tool modules reachable from an agent factory.
Excluded by user: `features/search/`, `open_deep_search`.

## Claims verdict

| Claim | Verdict |
|---|---|
| `statutes` in NO model, NO migration; only raw SQL at `search_legal_precedents.py:193` + `retrieve_statute_section.py:138`; both swallow `SQLAlchemyError` | **CONFIRMED** |
| Both tools have been failing invisibly forever | **CONFIRMED but MOOT** — neither tool is ever constructed at runtime (see Dead wiring) |
| `vectorstore/{vector_store,insert_vectors,similarity_search}.py` are 0 bytes | **CONFIRMED** (`ls`); nothing imports them, `vectorstore/__init__.py:4` is `__all__ = []` |
| Five parallel document schemas | **CORRECTED — the tools touch ZERO of them.** Tools touch only `statutes` (nonexistent) and `clauses` (via a stubbed function returning `[]`) |

Only `statutes` refs in whole repo: `search_legal_precedents.py:9,64,67,102,130,158,164,167,177,180,193,223` and `retrieve_statute_section.py:103,138`. No `__tablename__ = "statutes"`, no `create_table("statutes")` in any of the 10 files in `src/alembic/versions/`.

## Tool inventory

| Tool | path:line | Does | Registered how |
|---|---|---|---|
| `search_legal_precedents` | `search_legal_precedents.py:55` (factory `:44`) | Graphiti precedent chains + Postgres `statutes` FTS; sets `insufficient_basis` if <2 sources | `graphiti/registry.py:105` → `ToolRegistry.search_legal_precedents` (`:78`), `compliance_tools` (`:87`), `deep_research_tool` (`:95`) |
| `retrieve_statute_section` | `retrieve_statute_section.py:42` (factory `:36`) | Point lookup of one statute section by act+section+jurisdiction | `graphiti/registry.py:110` → `compliance_tools` (`:87`) |
| `query_knowledge_graph` | `query_knowledge_graph.py:40` (factory `:34`) | Graphiti multi-hop semantic search, doc+user scoped. **No SQL** | `graphiti/registry.py:114` → `risk_tools` (`:91`) |
| `get_obligation_chain` | `get_obligation_chain.py:46` (factory `:39`) | Forward-chains obligations from an entity via Graphiti. **No SQL** | `graphiti/registry.py:118` → `risk_tools` (`:91`) |
| `hybrid_retrieve_precedents` | `precedent_tools.py:50` (factory `:35`) | pgvector `clauses` + Graphiti + Neo4j subgraph merge | **NOT REGISTERED** — factory never called, not in `tools/__init__.py:17-32` |
| `detect_graph_conflicts` | `precedent_tools.py:166` (factory `:154`) | Neo4j Cypher circular-obligation / override-chain detection | **NOT REGISTERED** (same) |
| `web_search` | `web_search.py:18,80` | Tavily web search | `tools/registry.py:19` → `get_all_tools()` (`:53`) |
| `crawl_url` | `crawl.py:28,114` | Crawl4AI page fetch | `tools/registry.py:20` |
| `shell` | `shell.py:1` | Shell exec tool | `tools/base.py:99` module-level `registry` |
| pydantic-ai RAG agent tools (5) | `rag_agent_advanced.py:52,97,180,252,296,353` | `match_chunks()` vector RPC + `documents` table | `Agent(...)` at `:488`; module imported by nothing |

## Schema migration table (tool × table)

| Tool | Table | Cols read/written | Model | Migration | Query kind | UnifiedDocument/Chunk target |
|---|---|---|---|---|---|---|
| `search_legal_precedents` | `statutes` | R: `id, title, section_ref, body, jurisdiction, act_name, year, fts_vector` | **none** | **none** | raw `text()` `:182-200` | `id`→`UnifiedChunk.id` (`model.py:81`); `title`→`UnifiedDocument.title` (`:41`); `body`→`UnifiedChunk.content` (`:90`); `jurisdiction`→`UnifiedDocument.jurisdiction` (`:47`); `fts_vector`→`UnifiedChunk.search_text` (`:100`) + `chunks_bm25_idx` (`a71f0d7d9c12:102`); **`section_ref`, `act_name`, `year` — no equivalent**; nearest carriers `UnifiedChunk.clause_type` (`:92`) / `metadata_` (`:95`) / `UnifiedDocument.metadata_` (`:50`) |
| `retrieve_statute_section` | `statutes` | R: `id, act_name, section_ref, title, body, jurisdiction, year` | **none** | **none** | raw `text()` `:128-146` | same as above; needs an `(act_name, section_ref)` point-lookup key that `documents`/`chunks` has no index for |
| `hybrid_retrieve_precedents` | `clauses` (intended) | none — stub | `src/database/schemas/memory_schema.py:190` | `9f4a1b7c6d2e` (adds `chunk_id`, `chunk_text`, `clauses_embedding_idx`, `clauses_bm25_idx`) | **stub, `return []`** `:237` | `clauses.chunk_text`→`UnifiedChunk.content` (`:90`); embedding→`UnifiedChunk.embedding Vector(768)` (`:94`) |
| RAG agent (5 tools) | `match_chunks()` fn + `documents` | R: `documents.title`, chunk rows | `documents`→`UnifiedDocument` `model.py:27/30` **(name collision)** | `match_chunks` defined in **no migration** | raw asyncpg `$1::vector` `:134,210,276,386,450` | `documents` already IS the target table; `match_chunks` must become a query over `chunks.embedding` |
| `query_knowledge_graph`, `get_obligation_chain`, `detect_graph_conflicts` | none (Neo4j/Graphiti only) | — | — | — | — | no Postgres migration needed |
| `web_search`, `crawl_url`, `shell` | none | — | — | — | — | out of scope |

Migration branch hazard: `2bc7726317f6` has **two children** — `8a7d9b1c2e3f` (`8a7d9b1c2e3f:19`) and `a71f0d7d9c12` (`a71f0d7d9c12:17`). The unified `documents`/`chunks` migration is on an **unmerged head**; `9f4a1b7c6d2e`→`0001`→`0004` is the other. `alembic upgrade head` is ambiguous.

## Invisible-failure register (ranked by load-bearing)

1. **`search_legal_precedents.py:227-229`** — `except SQLAlchemyError` → `logger.warning("statute_postgres_search_failed")` → `return []`. Table does not exist, so this fires on **every** call. Caller sees `statutes: []` and `total_sources` = graphiti-only. If Graphiti returns ≥2, `insufficient_basis=False` (`:110`) — the compliance agent proceeds believing it has statutory basis it never retrieved. Docstring at `:179-180` calls this intentional ("lets you deploy before the statutes table is populated").
2. **`retrieve_statute_section.py:170-172`** — `except SQLAlchemyError` → `return None`, which `:87-92` converts to `ToolResult.fail("Section X of Y not found in Z")`. **Missing table is reported to the LLM as "section does not exist."** Worst signal in the file: the agent may conclude a statute has no such section.
3. **`precedent_tools.py:221-237`** — `_vector_search_clauses` unconditionally `return []` with a `TODO` at `:234`. No log line at all. `hybrid_retrieve_precedents` reports `total_sources` (`:130`) as graphiti-only while advertising pgvector in its docstring (`:62`).
4. **`rag_agent_advanced.py:169-172, 241-244, 290-293, 342-345, 478-481`** — `except (OpenAIError, GoogleAPIError)` → `return f"Search error: {e!s}"` string to the LLM. `:420-422` and `:460-462` `logger.warning` then continue with ungraded results. Narrow except tuple means `asyncpg`/`UndefinedFunctionError` from the missing `match_chunks` propagates — this one is loud, not silent.
5. **`lifespan.py:220-223`** — Graphiti startup failure sets `app.state.graphiti = None` and continues. Combined with (1)+(2), the compliance path can run with *both* backends dead.

## Dead wiring (largest finding)

`build_tool_registry` (`src/app/shared/rag/graphiti/registry.py:98`) is **never called** — the only other occurrences are inside its own docstring (`:4,9,16`). `lifespan.py` never constructs `IdempotencyGuard`, `tool_registry`, or `saul_graph`; the block that would (`lifespan.py:235-249`) is commented out. `build_saul_graph` (`agent_saul/graph.py:86`) has no caller. `precedent_tools.py` is imported by nothing.

Consequence: the four legal tools plus the two precedent tools are **not reachable at runtime today**. Their invisible failures are latent, not active.

## Vector search surface

- Only `precedent_tools.py:221` claims pgvector — a stub returning `[]` (`:237`). No `<=>` operator anywhere in `src/app` outside `features/search/` and the commented-out `strategies.py:643`.
- `src/app/shared/vectorstore/{vector_store,insert_vectors,similarity_search}.py` = **0 bytes** each. `__init__.py:4` `__all__ = []`. **Zero importers** repo-wide.
- Target dimension: `UnifiedChunk.embedding = Vector(768)` (`model.py:94`); index `chunks_embedding_idx` (`a71f0d7d9c12:99`). Matches `settings.py:51-52` (`gemini-embedding-2-preview`, `text-embedding-004` = 768). `settings.py:44` validates dim; `src/app/utils/embedding.py:8` `normalize_embedding` pads/truncates.
- No tool constructs an embedding client. `lifespan.py:244` `embedding_fn=build_embedding_client()` is commented out.

## Postgres RAG path (todo d)

- Entry: `src/app/shared/rag/rag_agent_advanced.py:488` `agent = Agent(...)` (pydantic-ai). Tools at `:52, 97, 180, 252, 296, 353`. CLI-only driver `run_cli()` `:517`, `main()` `:568`.
- Query shape: asyncpg `SELECT * FROM match_chunks($1::vector, $2)` at `:134, 210, 276, 386, 450`; plus `FROM documents` at `:317, 329`.
- `match_chunks` is defined in **no migration and no source file**.
- Iterative already exists: `search_with_self_reflection` (`:353`) grades results (`:420`) and refines the query (`:460`) in a loop. `search_with_multi_query` (`:97`) fans out via `expand_query_variations` (`:52`). `search_knowledge_base` (`:252`) is single-shot.
- Embedder: `from ingestion.embedder import create_embedder` at `:119, 198, 267, 373` — **`ingestion` package does not exist** in this repo (`src/ingestion` absent, no top-level `ingestion/`). Every tool ImportErrors on first call.
- `src/app/shared/rag/strategies.py` is **100% commented out** (all 800+ lines), including its `document_vectors` INSERT (`:407-409`) and pgvector search (`:643-644`).
- `src/app/features/documents/repository.py` is the live ORM path over `UnifiedDocument`/`UnifiedChunk` (`get_by_content_hash`, `get_by_id`, bulk `insert`); it has **no similarity-search method**.

## Prior art searched

- `openspec/changes/`: `cognee-saul-memory-migration`, `mintlify-documentation`. **No change covers tools↔document schema.**
- `openspec/specs/`: 21 specs, all MCP/typing/outbox/settings. **No document-schema or retrieval spec.**
- Searched by concept, not wording: `statutes`, `match_chunks`, `<=>`/`cosine_distance`, `pgvector`, `embedding`, `vector_store`, `similarity`, `__tablename__`, `create_table`, `build_tool_registry`.

## Fog

- **Runtime state of `statutes` unknown.** Could not query the live DB. It may exist as a hand-created table outside alembic; migration+model absence is the only evidence.
- **Whether `match_chunks` exists in the deployed DB** — same reason. Not in any repo file.
- **Which of the 5 schemas the ingestion side writes** — sibling scout's leg. `ingest_v2.py:220-276` handles embeddings; not traced here.
- **Whether the dead wiring is intentional staging or regression.** `lifespan.py:235-249` commented block and `graphiti/registry.py:4-31` docstring both describe the wiring as if it were live. `1b3891f fix: make startup resilient to optional services` is the likely commit; not confirmed by blame.
- **`clauses` vs `chunks` overlap.** `9f4a1b7c6d2e` added `chunk_id`/`chunk_text` to `clauses`, suggesting a prior half-migration toward chunks. Whether `clauses` is meant to survive is undetermined.
- No tests cover any tool: codegraph reports "no covering tests found" for `get_obligation_chain`, `get_all_tools`, and all three `ToolRegistry` classes.
