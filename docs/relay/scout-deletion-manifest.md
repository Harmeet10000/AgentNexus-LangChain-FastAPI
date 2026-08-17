# Scout — Deletion Manifest & Runtime-Break Register

Leg 1 (Scout). Tool routing: `graphify affected` on every deletion candidate.
All claims carry `path:line`. Written incrementally.

## 1. `todo_temp.py` — VERDICT: DEAD, and worse than dead

`src/app/shared/rag/document_processing/todo_temp.py` — **783 lines**.

**It does not parse.** `ast.parse` fails:

```
File "<unknown>", line 406
    """Collection of document extraction tools for agents."""
IndentationError: unexpected indent
```

`__all__` closes at `todo_temp.py:404`; `todo_temp.py:405-406` is an orphaned
class-body docstring + indented `__init__` with no enclosing `class` statement.
Any `import` of this module raises `IndentationError` at import time, so a live
caller is *impossible* — the app would not boot.

**Caller set — `graphify affected "todo_temp.py"`: `No affected nodes found.`**

Per-symbol `graphify affected` (all edges are intra-file):

| symbol | line | affected |
|---|---|---|
| `process_document_full` | 221 | `create_extraction_tools()` @ todo_temp.py:388 (self) |
| `create_extraction_tools` | 360 | **empty** |
| `search_rag_data` | 170 | `create_extraction_tools()` @ todo_temp.py:385 (self) |
| `_save_to_rag_data` | 265 | `process_document_full()` @ :233, `create_extraction_tools()` @ :388 (self) |
| `extract_tables` / `extract_images` / `extract_code_blocks` | 11/82/49 | **empty** |
| `extract_entities` | 117 | `create_extraction_tools()` @ todo_temp.py:382 (self) |

Repo-wide `rg "todo_temp"` outside the file itself: **zero hits** (excluding
graphify-out/ and docs/).

The file is a **duplicated draft of `docs`-style tooling**: `create_extraction_tools`
is defined twice (`:360` and `:773`), `process_document_full` twice (`:221` module-level,
`:632` method), `_save_to_rag_data` twice (`:265`, `:675`). The second half is a
class-based rewrite of the first half, pasted in without the `class` line.

Its only outbound deps are `docling_enhanced.py` (`:6`), `entity_extractor.py` (`:7`),
`document_processing/models.py` (`:8`), `utils/logger.py` (`:4`) — all of which live
independently. Deleting it breaks nothing.

**Single largest deletion in the refactor: 783 lines, change 0 (cleanup).**

## 3. Runtime-break register (ranked)

### RANK 1 — breaks on first request

| # | read site | expects | lifespan actually sets | reachability proof |
|---|---|---|---|---|
| B1 | `features/profile/router.py:29` `request.app.state.storage` | `app.state.storage` | `app.state.object_store` @ `lifespan.py:108,112,270` — **never `storage`** | `profile_router` mounted `api/v1.py:15` → `main.py:159`. Every endpoint using `_get_profile_service` |
| B2 | `features/profile/router.py:30` `request.app.state.mongodb` | `app.state.mongodb` | `app.state.db` @ `lifespan.py:180,183,186` — **never `mongodb`** | same as B1 |
| B3 | `features/agent_saul/dependencies.py:41` `request.app.state.saul_graph` | `app.state.saul_graph` | **nothing.** Only assignment text is inside the `registry.py:25` module docstring | `agent_saul_router` mounted `api/v1.py:17`. `get_saul_graph` → `get_agent_saul_deps` (`dependencies.py:90`) → `AgentSaulDepsAnnotated` (`:126`) |
| B4 | `features/agent_saul/dependencies.py:45` `app.state.langgraph_checkpointer` | set at startup | **commented out** `lifespan.py:295-305` → `AttributeError`, not the intended `ServiceUnavailableException` at `:48` | same as B3 |

Minimal fixes: B1/B2 rename the read (`object_store`, `db`) or the write.
B3 requires `build_tool_registry` + `build_saul_graph` wiring (see below).
B4 uncomment `lifespan.py:295-305` **or** `setattr(app.state,'langgraph_checkpointer',None)` so `:46` guard fires.

### RANK 2 — rare path

| # | site | fact | reachability |
|---|---|---|---|
| B5 | `rag/document_processing/entity_extractor.py:78` `from graphiti_graph import Graphiti` | package **not declared** — `pyproject.toml:62` declares `graphiti-core`, not `graphiti_graph`. Always `ImportError` | `extract_with_graphiti` reached via `extract()` `:346`, `extract_entities_batch()` `:258`, and re-exported `document_processing/__init__.py:27`; consumed by `features/documents/classification.py:10` and `parser.py:10` (documents_router mounted `api/v1.py:16`). Silently falls back via `extract_with_fallback` `:123` — never raises to caller |
| B6 | `features/health/service.py:160` | checks neo4j; **no** graphiti/cognee probe though `lifespan.py:218` sets `app.state.graphiti` and `:207` `app.state.cognee_config` | health_router mounted `api/v1.py:13` **and** `api/v2.py:9`. Not a crash — a silent blind spot |

### RANK 3 — latent, unreachable (cleanup, not bug)

| # | site | fact |
|---|---|---|
| B7 | `utils/toon_parser.py:13` | `parse()` returns `toons.dumps(text)` — serialises instead of parsing. `toons>=0.5.4` IS declared (`pyproject.toml:133`), so the import is fine; only the logic is inverted. `graphify affected "toon_parser"` → **empty**. Module-level side effect: builds a `ChatPromptTemplate` at import (`:17`) |
| B8 | `shared/vectorstore/{vector_store,insert_vectors,similarity_search}.py` | **0 bytes** each. `vectorstore/__init__.py` deliberately does not import them (`__all__ = []`). `graphify affected` → empty for all three |
| B9 | `shared/rag/graphiti/registry.py` `build_tool_registry` `:98` | `graphify affected` → **empty**. Real code, zero callers |
| B10 | `langgraph_layer/agent_saul/graph.py:86` `build_saul_graph` | `graphify affected` → one edge only: `agent_saul/__init__.py:3` re-export. No real caller. **Confirmed** |

### CORRECTION to the brief

`src/app/shared/rag/graphiti/registry.py` is **NOT** "entirely a module docstring."
Lines 1–32 are the docstring; **`:34-122` is live code** — `class ToolRegistry(BaseModel)` at `:56`
and `build_tool_registry()` at `:98`, both real. `ToolRegistry` IS consumed:
`agent_saul/graph.py:16,91` and `factory.py:182,205` (`tool_registry.deep_research_tool`).
Only the *wiring* named in the docstring (`app.state.tool_registry`, `app.state.saul_graph`)
was never transcribed into `lifespan.py`. Registry.py is **not** a deletion candidate.

Second correction: the docstring at `registry.py:9` names `app.shared.agents.tools.registry`
— a module that does not exist. Real path is `app.shared.rag.graphiti.registry`.

## 2. Deletion manifest

`affected` column = verbatim `graphify affected "<name>"` result. `ch` = openspec change.

| path | lines | affected | why safe | ch |
|---|---|---|---|---|
| `src/app/shared/rag/document_processing/todo_temp.py` | 783 | empty | **does not parse** (IndentationError :406); zero importers | 0 |
| `src/app/utils/toon_parser.py` | 36 | empty | zero callers; `parse()` inverted (`:13`); import-time side effect `:17` | 0 |
| `src/app/shared/vectorstore/vector_store.py` | 0 | empty | 0 bytes; `__init__.py` does not import it | 0 |
| `src/app/shared/vectorstore/insert_vectors.py` | 0 | empty | 0 bytes | 0 |
| `src/app/shared/vectorstore/similarity_search.py` | 0 | empty | 0 bytes | 0 |
| `src/app/shared/langchain_layer/agents/orchestration_type/handoff.py` | 0 | empty | 0 bytes; `__init__.py` also 0 bytes | 0 |
| `.../orchestration_type/llm_router.py` | 0 | empty | 0 bytes | 0 |
| `.../orchestration_type/router.py` | 0 | empty | 0 bytes | 0 |
| `.../orchestration_type/subagents.py` | 0 | empty | 0 bytes | 0 |
| `.../orchestration_type/__init__.py` | 0 | empty | 0 bytes; nothing imports the package | 0 |
| `src/app/features/knowledge_base/` (7 files) | 0 each + 8 in `__init__` | (graphify returns nothing for the pkg) | all 7 modules 0 bytes; `__init__.py` imports none of them (`__all__ = []`) | 0 |
| `src/app/features/web_scraping/` (8 files) | 0 each + 8 in `__init__` | same | all 8 modules 0 bytes; `__init__.py` `__all__ = []` | 0 |

**Coupled edit required for the two feature packages:** `src/app/features/__init__.py:3`
does `from . import documents, health, knowledge_base, web_scraping` and lists both at
`:8` and `:9`. Deleting the directories without editing that file is an ImportError at
app boot. This is the only non-obvious coupling in the manifest.

### EXCLUDED (user decisions — do not delete)

| path | reason |
|---|---|
| `rag/document_processing/ingest_v2.py` | batch/local-folder ingester; distinct use case; STAYS |
| `rag/document_processing/embedder.py` | `ingest_v2.py:18` imports `embed_chunks`; STAYS |
| `tasks/pageindex_tasks.py` | pageindex deferred; STAYS (raises `NotImplementedError`) |
| `write_final_report.py`, `memory_pipeline.py` | deletion DEFERRED; sole reference for intended Cognee writes. Characterised only |
| all of `features/search/`, `process_ingestion_document` | out of scope; stays unmounted |
| `langgraph_layer/open_deep_search/` | out of scope |
| `langgraph_layer/ingestion_kb/` | **NOT dead** — being PROMOTED to the live pipeline |
| reconciliation | sibling scout owns the inventory; belongs to the same deletion change |
| `shared/rag/graphiti/registry.py` | see correction in §3 — live code, `ToolRegistry` consumed by `graph.py:16,91` + `factory.py:182,205` |

## 4. Zero-byte / stub-file sweep

**0 bytes (29 files).** All in §2 above, plus these which are 0-byte `__init__.py` and
are **required** by Python package resolution (deleting them breaks the package):
`src/alembic/__init__.py`, `src/app/examples/__init__.py`, `src/app/shared/agents/__init__.py`,
`src/app/shared/circuit_breaker/__init__.py`, `src/database/seeders/__init__.py`,
`src/mcp_core/{client,common,server}/__init__.py`.

**Empty file that something imports — one hit, and it is NOT a break:**
`src/app/shared/agents/memory/memory_scope.py` is 30 bytes / **1 line**
(`PRECEDENT_SCOPE = "precedent"`) and IS imported at
`src/app/shared/agents/memory/__init__.py:1` and, live,
`src/app/shared/langchain_layer/agents/tools/precedent_tools.py:21`
(`from app.shared.agents.memory.memory_scope import PRECEDENT_SCOPE`).
**Verified importable:** `uv run python -c "import ...precedent_tools"` → `OK`.

### DELETION CANDIDATE WITH A LIVE CALLER — `src/app/shared/agents/`

`src/app/shared/agents/` is a **shadow duplicate** of `src/app/shared/langchain_layer/agents/`:

| shadow | real | sizes |
|---|---|---|
| `shared/agents/memory/memory_scope.py` | `langchain_layer/agents/memory/memory_scope.py` | **30 B** vs 7189 B |
| `shared/agents/tools/idempotency.py` | `langchain_layer/agents/tools/idempotency.py` | **695 B** vs 6913 B |

`precedent_tools.py:21` imports the **30-byte shadow**, not the 7 KB real one, while
`shared/rag/graphiti/subgraph.py:30` imports `MemoryScope` from the **real** path.
Two parallel memory-scope modules are live simultaneously. `shared/agents/` looks
deletable by size and by `graphify affected` (empty for `idempotency`), but it is not —
it has a live importer. **Goes to Fog, not the manifest.**

## 5. Commented-out-wiring sweep

### `src/app/lifecycle/lifespan.py`

| lines | would have wired | restore-or-delete signal |
|---|---|---|
| `235-240` | `ingestion_llm = ChatGoogleGenerativeAI(...)` (`GEMINI_FLASH_MODEL`, temp 0.1, retries 0) | feeds the block below |
| `241-248` | `app.state.ingestion_graph = build_ingestion_graph(extraction_llm, db_engine, embedding_fn=build_embedding_client(), graphiti_service=graphiti, redis)` + `logger.info("Contract KB ingestion graph initialized")` | this **is** the `ingestion_kb` promotion target — restore (change 1) |
| `249` | `app.state.pageindex_client = PageIndexClient()` | pageindex deferred → stays commented (matches the `pageindex_tasks.py` carve-out) |
| `291` | header-only: "FastAPI-Guard setup (depends on Redis, but non-blocking)" — no code beneath | `guard` IS imported live at `main.py:8,31` and `middleware/server_middleware.py:9,10`, so the comment is stale, not pending |
| `294-305` | `setup_langgraph_checkpointer(conn_string=settings.POSTGRES_URL)` → `app.state.langgraph_checkpointer`, with `except (ConnectionError, TimeoutError, OSError)` → `None` | **directly causes B4.** Restoring this is a prerequisite for agent_saul working at all |

Note `lifespan.py:241-248` and `294-305` are the only two blocks whose absence produces a
*reachable* break. `:235-240`/`:249` are supporting/deferred.

### `src/app/connections/` — no dead wiring, only tuning knobs

| file:line | commented |
|---|---|
| `postgres.py:98-99` | `poolclass=NullPool`, `connect_args={...}` |
| `neo4j.py:33-34` | `max_retry_time=30`, `database=settings.NEO4J_DATABASE` |
| `redis.py:7,28-29` | `from tenacity import retry, ...`; `host=get_settings().REDIS_HOST`, `max_connections=50` |

None of these wire a service that anything reads. They are parameter alternatives.

## 6. Dependency reality check

Every top-level import in `src/` cross-referenced against `pyproject.toml`
`[project.dependencies]` + `[dependency-groups]`, then probed with `importlib.util.find_spec`.

| module | import site | declared? | installed? | verdict |
|---|---|---|---|---|
| `graphiti_graph` | `entity_extractor.py:78` | no (`graphiti-core` @ `pyproject.toml:62`) | **MISSING** | **real break — B5** |
| `ingestion` | `rag_agent_advanced.py:119,198,267,373` (`from ingestion.embedder import create_embedder`) | no | **MISSING** | **real break — NEW, see below** |
| `ty_extensions` | `postgres.py:25`, `documents/service.py:83`, `ingestion_kb/nodes.py:59` | no | MISSING at runtime | **not a break** — all three are inside `if TYPE_CHECKING:` (`postgres.py:20`); `ty` supplies it at check time |
| `docling_core` | `documents/parser.py:16`, `chunker.py:15`, `docling_enhanced.py:25`, `ingestion_kb/nodes.py:54` | no | INSTALLED (transitive of `docling` `:74`) | works today; undeclared transitive |
| `fpdf` | `billing/services/pdf.py:7` | no (dist is `fpdf2`) | INSTALLED | works; undeclared |
| `guard` | `main.py:8,31`, `server_middleware.py:9,10` | no (dist `fastapi-guard`) | INSTALLED | works; undeclared |
| `joserfc` | `auth/security.py:12-15` | no | INSTALLED | works; undeclared — **auth depends on it** |
| `pyrate_limiter` | `auth/websocket_security.py:15,16` | no | INSTALLED | works; undeclared |
| `kombu` | `connections/celery.py:17` | no | INSTALLED (transitive of celery) | works; undeclared |
| `mcp` | `mcp_core/client/auth.py:17` | no | INSTALLED | works; undeclared |
| `playwright` | `shared/crawler/crawler.py:21`, `open_deep_search/utils.py:24` | no | INSTALLED (transitive of crawl4ai) | works; undeclared |

**NEW break, not in the brief — `rag_agent_advanced.py`:**
`src/app/shared/rag/rag_agent_advanced.py` (586 lines) does deferred
`from ingestion.embedder import create_embedder` at `:119`, `:198`, `:267`, `:373`.
No top-level `ingestion` package exists (`find_spec` → MISSING); the real path is
`app.shared.rag.document_processing.embedder`. `rg "rag_agent_advanced" src/` → **zero
importers**, and `graphify affected "rag_agent_advanced.py"` returns only intra-file
edges. So: **latent, unreachable (RANK 3)** — a deletion candidate, but it was not
sanctioned by the carve-outs and it touches `embedder.py`, so it goes to Fog.

## Fog — candidates I could NOT prove dead

1. **`src/app/shared/agents/` (shadow package)** — `memory/memory_scope.py` has a LIVE
   importer at `precedent_tools.py:21`, and `tools/idempotency.py` returns empty from
   `graphify affected` but is a 695 B stub of a 6913 B real module. Whether
   `precedent_tools.py` should import the real path instead is a **plan decision**, not a
   scout finding. To establish: `graphify affected "precedent_tools"` is empty, so trace
   who reaches `precedent_tools` at all — I could not.
2. **`src/app/shared/rag/rag_agent_advanced.py` (586 lines)** — zero importers by both `rg`
   and graphify, and structurally broken (`ingestion.embedder`). Not on the carve-out list
   either way. Needs a user decision like `ingest_v2.py` got.
3. **`entity_extractor.py:23-121` `extract_with_graphiti`** — dead *in effect* (always
   ImportError) but its symbol is re-exported at `document_processing/__init__.py:27` and
   consumed by `documents/classification.py:10` + `parser.py:10`. Cannot be deleted
   without touching live document classification. Fix vs delete is a plan call.
4. **`features/knowledge_base` / `web_scraping`** — in the manifest, but only because I
   found the `features/__init__.py:3` coupling. I did NOT verify no Alembic migration,
   test, or `openspec/specs/` entry references them by name.
5. **`ingest_v2.py` / `embedder.py` reachability** — carved out, so I did not trace
   whether anything *else* imports them. If the answer matters to change 1, it is unproven.
6. **Test coverage of anything here** — codegraph reported "no covering tests found" for
   `build_saul_graph`, `get_agent_saul_deps`, `ToolRegistry`. I did not enumerate `tests/`,
   so I cannot state that deleting the manifest breaks zero tests.
7. **openspec** — `openspec/changes/` holds only `cognee-saul-memory-migration` and
   `mintlify-documentation`; `openspec/specs/` has 21 specs, none named for deletion,
   ingestion, or vectorstore. I matched on directory names only, not spec contents, so an
   in-flight change touching this area may exist under an unrelated name.
