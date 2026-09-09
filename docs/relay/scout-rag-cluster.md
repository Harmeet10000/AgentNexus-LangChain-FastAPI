# Scout — RAG cluster (todos 235, 240, 163, 164, 185, 162, 165, 195, 176)

Leg 1 of relay. Written 2026-09-09. Branch `main`, tip `7cca750`. Terrain only — no plan.

Every claim carries a `path:line`, measured against the working tree on this date. Where a prior
`docs/relay/*` report disagrees with disk, the disagreement is called out.

---

## 0. Headline: the working tree does not import

`import app.main` **fails** today:

```
  File "src/app/features/documents/classification.py", line 13, in <module>
    from app.shared.rag.document_processing import IngestionConfig
ModuleNotFoundError: No module named 'app.shared.rag.document_processing'
```

Three independent probes fail the same way:

| Probe | Failing import site |
|---|---|
| `import app.shared.rag.docling` | `src/app/shared/rag/docling/__init__.py:3` |
| `import app.features.documents.parser` | `src/app/features/documents/parser.py:11` |
| `import app.main` | `src/app/features/documents/classification.py:13` |

Cause: `src/app/shared/rag/document_processing/` was **deleted** (7 files, staged ` D` in
`git status`) and its contents copied to an **untracked** `src/app/shared/rag/docling/`. Nothing was
repointed — including `docling/__init__.py`, which still imports the package it replaced.

**Production importers still naming the deleted path (5 files, 8 sites):**

- `src/app/shared/rag/docling/__init__.py:3,9,21,27,37` — the new package imports the old one
- `src/app/features/documents/classification.py:13,14`
- `src/app/features/documents/parser.py:11,12`
- `src/app/shared/langgraph_layer/ingestion_kb/nodes.py:31` (`table_markdown`)
- `src/app/examples/policy_examples.py:36,40`

**Test importers (4 files):** `tests/unit/test_auth_documents_feature_errors.py:19`,
`tests/unit/shared/rag/test_chunker_tokenizer_cache.py:33,34,40`,
`tests/unit/shared/rag/test_embedder_no_substitution.py:26,27,34`,
`tests/unit/shared/rag/test_rag_agent_embedder_import.py:38` (a string target, not an import).

**Doc/comment references (not load-bearing):** `src/app/utils/embedding.py:23`,
`src/alembic/versions/0013_...py:101`, `src/app/shared/rag/langextract/langextract_to_graph.py:1`.

**`pyproject.toml:539-553` per-file-ignores still name the dead path**, granting `PLC0415` to
`document_processing/{embedder,docling_enhanced,chunker,entity_extractor}.py`. The files moved, so
those 7 late imports are now unignored.

### Gate state, measured now vs the handover's claimed baseline

| Gate | Handover claim (`1ce52ec`, Aug 23) | Measured `main` @ `7cca750`, Sep 9 |
|---|---|---|
| `uv run ruff check --no-cache src/` | All checks passed | **24 errors** |
| `import app.main` | (implied importable) | **ModuleNotFoundError** |
| alembic head | `b3e7c41d92af` | migration files top out at **`0017`** |
| `openspec list` | 6 active-ish bands | **"No active changes found."** |

The 24 ruff errors: `INP001` x12 (every `src/app/features/*/__init__.py` — there is no
`src/app/features/__init__.py`; last touched `8e25352`), `PLC0415` x7 (all in the untracked
`docling/` tree, orphaned by the stale ignores above), `PLC2701` x3 (`documents/dependencies.py:13`,
`documents/service.py:19,26` — `_build_chat_model`, `_extract_postgres_chunk_ids`), `I001` x2
(`documents/classification.py:3`, `documents/parser.py:3`).

The three `PLC2701` hits on `_build_chat_model` are the literal subject of the second todo numbered
240 (`tests/performance/todo.md:362`).

---

## 1. Todo source — exact current text

`tests/performance/todo.md`. **No item in this block is ticked** — bare `N. text` lines, no `- [ ]`
checkbox, so there is no tick state to read. Last commit touching the file: `87f746e`; working copy
is ` M`.

| Line | Verbatim |
|---|---|
| 350 | `235. need to have all graph such as ingestion graph in the lifespan rather than in service` |
| 351 | `240. document processing and crawler needs db instance injection and docling processong needs improvement as well. need to think of a suitable chunking strategy for legal docs, and everything that is considered best practice for docling` |
| 352 | `163. refactor vectorStore code        TSVECTOR,` |
| 353 | `164. refactor RAG code` |
| 354 | `185. remove ts_vector(think if it is required here or other extension can do the job here) from search/document and write correct SQL query for documents/ taking skills for pgvector/pgvectorscale ` |
| 355 | `162. what kind of text splitters do i need. diff in PGvector and pgvectorstore in langchain` |
| 356 | `165. implement RAG by getting inspired from this https://www.uber.com/en-IN/blog/enhanced-agentic-rag/?uclick_id=...` |
| 357 | `195. in ingestion pipeline postgres + extensions for vector + BM25 + RRF and more, graphiti for what we already did, need to have langextract before these as well, and a pageindex parallel to postgres graphiti and learn from https://towardsdatascience.com/hybrid-search-and-re-ranking-in-production-rag/` |
| 358 | `176. check sentence_transformers, AutoTokenizer from transformer package do i need it or can it be replaced by a langchain package` |

**The brief's paraphrase of 163/164 is richer than the file.** On disk 163 is
`refactor vectorStore code        TSVECTOR,` and 164 is bare `refactor RAG code`. The
"i think i have already deleted vectorstore/" and "see which of the code is suitable for
connections/, features/document/, shared/rag/ ... reuse embedder that lives in langchain_layer"
clauses exist **only in the user's chat message**.

**Adjacent items in the same block, not listed in the brief, that overlap this cluster:**

- `:347` — `190. see if documents/ can be moved in ingestion pipeline with langextract, pageindex, graphiti, postgres,`
- `:348` — `136. use LangExtract outputs to build rich graph knowledge from your legal documents. in document processing`
- `:362` — a **second item numbered 240**: `remove build chat model from documents/ and review chunking strategy used here and in crawler and find out from where to add them`
- `:344` — `f. insert the langgraph in app.state in lifespan` (a duplicate of 235 in an unrelated sub-list)

Two distinct todos share the number **240**.

---

## 2. Graph lifecycle (235)

### The three `StateGraph` builders and where each compiles

| Builder | File:line | Compile | Checkpointer |
|---|---|---|---|
| `build_ingestion_graph` | `src/app/shared/langgraph_layer/ingestion_kb/graph.py:58` | `:118` `graph.compile(checkpointer=checkpointer)` | optional param; `None` logs `ingestion_graph_built_without_checkpointer` (`:75-78`) |
| `build_document_ingestion_graph` | `src/app/features/documents/ingestion_graph.py:47` | `:74` `graph.compile()` | **none — no parameter at all** |
| `build_retrieval_graph` | `src/app/shared/langgraph_layer/retrieval_kb/graph.py` | `:73` `graph.compile()` | none |

Plus three **module-global** compiles at import time in
`src/app/shared/langgraph_layer/open_deep_search/graph.py:278` (`supervisor_subgraph`), `:478`
(`researcher_subgraph`), `:555` (`deep_researcher`) — module globals, not `app.state`.
And `src/app/shared/langgraph_layer/agent_saul/graph.py:115` compiles with a checkpointer.

### Who holds the compiled graph today

- **`app.state.ingestion_graph` is commented out** — `src/app/lifecycle/lifespan.py:522-537`. The
  `ingestion_llm` construction and the whole `build_ingestion_graph(...)` call are commented, with an
  in-place note at `:534-536` forbidding restoration of an `embedding_fn=` argument.
- **`app.state.pageindex_client` is commented out** — `lifespan.py:538`.
- **The LangGraph checkpointer block is commented out** — `lifespan.py:549-565`, with a note that a
  re-enable must use `get_database_url(flavour="plain")` because the saver is psycopg-backed and
  cannot parse the async dialect scheme.
- **`app.state.saul_graph` is never assigned anywhere.**
  `src/app/features/agent_saul/dependencies.py:37-45` reads it via `getattr(..., None)` and raises
  `ServiceUnavailableException` when absent — so that dependency 503s unconditionally today. Same
  file's `get_saul_checkpointer` (`:48-53`) reads `app.state.langgraph_checkpointer`, also never set.
- **`IngestionService` takes the graph by constructor injection** —
  `src/app/features/ingestion/service.py:33-35`
  `def __init__(self, ingestion_graph: CompiledStateGraph[Any])`, invoked at `:71`
  `await self._graph.ainvoke(initial_state)`. It does **not** build the graph. The todo's premise
  ("graph in the service") is not literally true for `IngestionService`.
- **`build_document_ingestion_graph` IS built per-invocation inside a Celery task** — codegraph edge
  `run_document_ingestion_task -> build_document_ingestion_graph`. That is the real
  "compiled in the wrong place" site: a fresh compile per job.

### What `lifespan.py` actually builds into `app.state`

`object_store` (`:134`), `outbox_relay` + `outbox_relay_task` (`:166-167`), `cognee_config` (`:193`),
`graphiti` (`:205`), `crawl4ai_crawler` (`:213`), `crawler_processor` (`:214`), `celery` (`:226`),
`db_engine` + `db_session_local` (`:464`), `mongo_client` + `db` (`:473-479`), `redis` (`:483`),
`neo4j_driver` (`:489`), `websocket_security` + `websocket_revocation_task` (`:496-502`),
`httpx_client` (`:540`), `tavily_http_client` (`:543`).

Optional resources boot through `STARTUP_POLICIES` (`lifespan.py:509-510`, `_run_startup_policy` at
`:345`). Teardown is `_shutdown_resources` (`:362`).

### Does compilation need a live DB pool?

`build_ingestion_graph(extraction_llm, db_engine: AsyncEngine, graphiti_service, redis, checkpointer)`
— `graph.py:58-64`; `db_engine` threads into `make_embed_store_node(db_engine, redis)` at `:105`.
`app.state.db_engine` is assigned at `lifespan.py:464`, **before** the commented block at `:528`, so
ordering already permits it. The engine is a handle, not a live connection — compile does no I/O.

`build_document_ingestion_graph` needs a **`DocumentRepository`** (`ingestion_graph.py:51`), which
holds an `AsyncSession` (`repository.py:62-63`) — a request/job-scoped object, not lifespan-scoped.
That is the structural obstacle to hoisting *this* graph into `app.state` as written.

---

## 3. Document processing + crawler (240)

### `src/app/shared/rag/` on disk now

```
rag/__init__.py          re-exports pageindex ONLY (9 symbols)
rag/errors.py            RagProviderError, RagResult
rag/strategies.py        lines 331-343 are COMMENTED-OUT sentence_transformers/CrossEncoder code
rag/docling/             UNTRACKED: chunker, docling_enhanced, embedder, entity_extractor,
                         ingest_v2, models, __init__   (mtimes Sep 8-9)
rag/graphiti/            client, registry, schemas, subgraph, write_clause_episodes
rag/pageindex/           client, functions
rag/langextract/         langextract_to_graph, docling_preprocessor, langextract_batch_processor
rag/document_processing/ GONE (deleted, staged)
```

`src/app/shared/rag/__init__.py` exports **only** pageindex symbols — docling, graphiti and
langextract are not re-exported at package level.

### Chunking — three implementations, all live

1. `src/app/shared/rag/docling/chunker.py` — Docling `HybridChunker`. `_hybrid_chunk_documents:240`,
   `_chunk_document_impl`, `_simple_fallback_chunk` on HybridChunker failure (`:234`). Tokenizer via
   `AutoTokenizer.from_pretrained(model_id)` at `:87`, imported at `:55`.
2. `src/app/features/documents/chunking.py:18` `chunk_text(text, *, chunk_size, chunk_overlap)` —
   **whitespace token windows** (`normalized.split(" ")` at `:33`). Not a tokenizer, not
   structure-aware. 5 callers. `INGEST_CHUNK_SIZE = 512`, `INGEST_CHUNK_OVERLAP = 64`
   (`src/app/features/documents/constants.py:31-32`).
3. `src/app/shared/crawler/chunker.py` — `Chunk:9`, `smart_chunk_markdown:128`, `truncate_content:196`.

Nothing here is legal-document-aware beyond `features/documents/classification.py:116,124`
(`_HEADING_RE`, `_CLAUSE_START_RE`) and `legal_metadata.py`.

### Crawler DB injection — there is none

`rg 'AsyncSession|session_factory|get_db|async_sessionmaker|db_engine|create_async_engine'` across
`src/app/shared/crawler/*.py` and `src/app/shared/rag/docling/*.py` returns **zero hits**. Neither
subsystem touches the database at all today — "needs db instance injection" is adding a seam, not
fixing a self-constructed one.

`src/app/shared/crawler/processor.py:137` `GeminiProcessor`; `:392` `async def get_processor()` — a
module-level factory called from `lifespan.py:214`. It imports `_build_chat_model` from
`app.shared.langchain_layer.models` at `processor.py:14` — the same private-name import ruff flags in
`documents/`.

Crawler modules: `__init__, chunker, config, crawler, errors, processor, validator` — no repository,
no model, no session.

---

## 4. Vector store (163/164)

**No `vectorstore/` directory survives anywhere in `src/`.** A `find` for `*vector*` directories
returns only `.github/skills/pgvector-skill`, `.venv/.../pgvector`, and unrelated site-packages. The
user's "i think i have already deleted vectorstore/" is **confirmed**.

Vector-store behaviour now lives as raw SQL in `src/app/features/documents/repository.py:452-504`
(`vector_search`). There is **no LangChain `VectorStore`/`PGVector` object anywhere in `src/`** — the
repo does not use the LangChain vector-store abstraction at all. Todo 162's "diff in PGvector and
pgvectorstore in langchain" is therefore a research question with no incumbent code.

### The embedder situation — two implementations, both live

| Module | Symbols | Model |
|---|---|---|
| `src/app/shared/langchain_layer/embeddings.py` | `EmbeddingTaskType:59`, `get_embedding_client:87`, `_cache_key:137`, `embed_text:168`, `embed_texts:216`, `_CACHE_NAMESPACE = "embedding:v1"` (`:83`) | from settings; `EMBEDDING_DIMENSION` default 768 (`settings.py:263`) |
| `src/app/shared/rag/docling/embedder.py` | `get_embedding_dimension:58`, `_provider_failure:74`, `_validated_width:113`, `generate_embedding:135`, `generate_embeddings_batch`, `create_embedder`, `embed_chunks` | **hardcoded** `_PROVIDER_EMBEDDING_MODEL = "gemini-embedding-001"` (`:48`) |

`docling/embedder.py:44-47` documents the divergence in its own comment: `"gemini-embedding-001"`
here against `"gemini-embedding-2-preview"` in configuration, and says collapsing the four embedding
paths "belongs to B1".

**Reuse vs re-implement:** `src/app/features/documents/service.py:420` calls
`embed_text(..., task_type=EmbeddingTaskType.QUERY, redis=...)` — reuses `langchain_layer`.
`docling/ingest_v2.py:18` imports `embed_chunks` from the local `embedder` — re-implements. A third
site exists at `src/app/utils/embedding.py` (`:23` cross-references the rag embedder's
`_validated_width`).

`docs/relay/decisions.md:55` records: *"the unified `langchain_layer` embedder is adopted by the
ingestion/documents path"* — already a locked decision.

---

## 5. TSVECTOR (163/185) — mostly already gone

**Zero occurrences of `tsvector` in application code.** `rg 'TSVECTOR|tsvector|ts_vector|tsquery|search_vector'`
across `src/**/*.py` hits **only two migration files**:

- `src/alembic/versions/0004_add_search_documents_and_chunks.py:54-55` — `postgresql.TSVECTOR()` with
  `sa.Computed("to_tsvector('english', content)", persisted=True)`; GIN index at `:94`
  (`ix_search_chunks_content_tsv_gin`). On `search_chunks`.
- `src/alembic/versions/0014_create_the_five_phantom_relations.py:79-81, 208, 218-219` — recreates
  `content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED`. Its docstring
  at `:79-81` explains it uses the **two-argument** `to_tsvector(regconfig, text)` deliberately
  because only that overload is `IMMUTABLE`; the one-argument form is merely `STABLE`.

**`0014` is titled "create the five phantom relations"** — it exists to materialise tables that
earlier `stamp`-marked revisions never created. This corroborates the memory that the
document/vector/search schema was never created on the live DB.

**Correction to memory:** memory says "live DB is stamped at alembic `0004`". Disk carries
migrations `0001`-`0017` (linear except two merges: `0011` down = `("0009","0003")`, `0012` down =
`("0010","0011")`). **File head is `0017`** (`0017_scope_statute_identity_index.py:28`, down `0016`;
nothing points at `0017`). The handover's claimed head `b3e7c41d92af` matches **no revision id on
disk**. I did **not** connect to the live database, so the *stamped* revision remains unverified.

**No ORM column uses TSVECTOR.** `src/app/features/documents/model.py` uses `search_text` with a
**bm25** index (`:114 postgresql_using="bm25"`), a **diskann** index (`:122`), and GIN on `metadata_`
(`:60`, `:101`). The comment at `model.py:109`: *"bm25 comes from pg_textsearch, diskann from
vectorscale, gin_trgm_ops"*.

For 185, "remove ts_vector from search/document" appears **already done** in the ORM and the query
layer; the remaining tsvector is confined to historical/phantom-relation migrations.

---

## 6. Search SQL + extensions (185)

### Where the SQL lives

`src/app/features/documents/repository.py`, three branch methods sharing `_FILTER_SQL` (`:47-56` —
filters on `document_ids`, `chunk_kind`, `metadata_->>'jurisdiction'`, `metadata_->>'contract_type'`,
`clause_type`, `graphiti_verified`, jsonb `@>` on `metadata_` and `metadata_->'parties'`):

| Method | Line | Mechanism |
|---|---|---|
| `bm25_search` | `:405` | `c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx')`, negated for score, `WHERE ... < 0`, `ORDER BY ... ASC` |
| `vector_search` | `:452` | `1 - (c.embedding <=> CAST(:embedding AS vector))`; sets `SET LOCAL diskann.query_search_list_size` (`:479`) and `diskann.query_rescore` (`:483`) |
| `trigram_search` | `:507` | pg_trgm |

Fusion: `src/app/features/documents/fusion.py:28` `reciprocal_rank_fusion(*result_sets, k, limit)` —
standard `1/(k+rank)`. `RRF_K = 60`, `HYBRID_CANDIDATE_LIMIT = 50` (`constants.py:23-24`).
Assembly: `src/app/features/documents/rag.py:37` `assemble_rag_context(...)` — groups by document,
restores chunk order, merges adjacent chunks, budgets by `len(content.split())` (a **word** count,
not a token count — `rag.py:79`).
Orchestration: `src/app/features/documents/service.py:393` `search(...)` with a Redis cache + `setnx`
lock at `:396-418` and a 30x0.05s poll loop at `:408-410`; branches fused via `_fuse_search_branches`
at `:428`.

### Extensions actually declared

`CREATE EXTENSION` appears only in migrations — **not** in `docker-compose.yml`,
`docker-compose.prod.yml`, or `infra/`:

| Migration | Extensions |
|---|---|
| `0003:23-26` | `uuid-ossp`, `vector`, `pg_trgm`, `pg_textsearch` |
| `0004:25-29` | `vector`, `vectorscale`, `pg_textsearch`, `pg_trgm`, `unaccent` |
| `0005:24-26` | `uuid-ossp`, `vector`, `pg_textsearch` |
| `0013:180-183` | `vector`, `vectorscale`, `pg_trgm`, `pg_textsearch` |

`0013:62-67` carries a load-bearing note: `CREATE EXTENSION IF NOT EXISTS` does **not** soften a
missing extension, so the `bm25` and `diskann` index branches **fail hard** rather than degrading.
`vectorscale` (pgvectorscale) supplies `diskann`; `pg_textsearch` supplies `bm25`. `pg_search` /
ParadeDB is **not** used anywhere.

---

## 7. Layering inventory (164)

`src/app/connections/` — one module per external client, all returning shared/singleton handles:
`celery.py`, `celery_task_names.py`, `crawl4ai.py`, `httpx_client.py` (`get_shared_httpx_client:60`),
`mongodb.py`, `neo4j.py`, `postgres.py` (`init_db:174`, `get_database_url`), `redis.py`, `tavily.py`
(`get_shared_tavily_http_client:16`, `create_tavily_http_client:42`).

`src/app/features/documents/` (**there is no `features/document/` singular**):
`chunking.py` (word-window splitter) | `classification.py` (heading/clause regex, doc-kind) |
`constants.py` (RRF_K, limits, chunk sizes) | `dependencies.py` (DI) | `dto.py`
(`DocumentSearchResultItem:84`) | `errors.py` (`DocumentDatabaseError:58`) | `fusion.py` (RRF) |
`graphiti_verifier.py` | `ingestion_graph.py` (per-job LangGraph) | `legal_metadata.py` | `model.py`
(ORM: bm25/diskann/gin indexes) | `parser.py` (docling converter wrapper) | `rag.py` (context
assembly) | `repository.py` (all search SQL) | `router.py` | `service.py`.

`src/app/features/search/` — **`__init__.py` only.** Retired by the archived
`2026-09-07-documents-unified-schema` change; the directory is an empty shell.

`src/app/shared/rag/` — see section 3. `src/app/shared/langchain_layer/` — `agents/`, `callback.py`,
`chains.py`, `dto.py`, `embeddings.py`, `messages.py`, `models.py`, `prompts.py`.
`src/app/shared/langgraph_layer/` — `ingestion_kb/`, `retrieval_kb/`, `agent_saul/`,
`open_deep_search/`, `checkpointer.py`, `kb_retry.py`.

The **ingestion_kb / docling / documents triangle** is the layering knot: `ingestion_kb/nodes.py:31`
(shared/langgraph) imports shared/rag; `documents/parser.py:11` and `classification.py:13` (features)
import shared/rag; `shared/rag/graphiti/write_clause_episodes.py:42` imports `shared/rag/langextract`.

---

## 8. Tokenizer dependencies (176)

**Runtime imports — only 3 in `src/`:**

- `src/app/shared/rag/docling/chunker.py:55` — `from transformers import AutoTokenizer, PreTrainedTokenizerBase`;
  used at `:87` `AutoTokenizer.from_pretrained(model_id)` to feed Docling's `HybridChunker`.
- `src/app/shared/langgraph_layer/retrieval_kb/reranker.py:9` — `from sentence_transformers import CrossEncoder`.
  The **only** live `sentence_transformers` use.
- `src/app/shared/rag/strategies.py:331-343` — `sentence_transformers` / `CrossEncoder` /
  `SentenceTransformer`, entirely **commented out**.

**Test imports:** `tests/unit/shared/rag/test_chunker_tokenizer_cache.py:31,104` asserts the tokenizer
is cached — an existing regression guard on `AutoTokenizer` construction.

**`tiktoken`: zero imports in `src/` or `tests/`, and absent from `pyproject.toml`.**

**`pyproject.toml`:** `sentence-transformers>=5.1.2` (`:51`), `transformers>=5.12.0` (`:188` — a
different group from line 51), `langchain-docling>=2.0.0` (`:43`), `docling>=2.72.0` (`:77`),
`langextract>=1.2.0` (`:78`), `graphiti-core[google-genai]>=0.29.1` (`:62`), `pgvector>=0.4.2`
(`:56`). `docling` is in `exempt-modules` at `:409`.

`langchain-docling` is a declared dependency with **no import anywhere** — the code uses `docling`
directly. `src/app/examples/rag_agent_advanced.py:31-44,176-236` re-implements CrossEncoder reranking
in an example.

---

## 9. 195 — what already exists

| Piece | State | Evidence |
|---|---|---|
| Postgres + `vector` + `vectorscale` | migrations create them | `0013:180-183` |
| BM25 | **built** | `repository.py:405`; index `chunks_bm25_idx`; `model.py:114` |
| RRF | **built** | `fusion.py:28`; `RRF_K=60` `constants.py:24` |
| Trigram (3rd branch) | **built** | `repository.py:507` |
| Graphiti | **built + wired into lifespan** | `shared/rag/graphiti/{client,registry,schemas,subgraph,write_clause_episodes}.py`; `lifespan.py:205` `app.state.graphiti`; `setup_graphiti:90`, `setup_graphiti_indices:159`, `close_graphiti:179` |
| LangExtract | **code exists, unwired** | `shared/rag/langextract/` (3 modules). Only non-package references: `settings.py:24,252` (`LANGEXTRACT_API_KEY`, default `"empty-langextract-api-key"`) and a `TYPE_CHECKING` import at `graphiti/write_clause_episodes.py:42`. No runtime call site. |
| PageIndex | **code exists, unwired** | `shared/rag/pageindex/{client,functions}.py`, re-exported by `shared/rag/__init__.py:3`. Only construction site is **commented out**: `lifespan.py:538`. |
| Reranker | **exists, wiring unverified** | `retrieval_kb/reranker.py` (CrossEncoder). No `rerank` call from `documents/service.py`. |

The Postgres/BM25/RRF half of 195 is shipped; the langextract-before and pageindex-parallel half is
code-on-disk-but-not-called.

---

## 10. The two named skills

Both are a routing-layer `SKILL.md` plus a large `references/*.md`.

**`.github/skills/pgvector-skill/`** (SKILL.md 127 lines, `references/pgvector.md` 540 lines).
Prescribes: confirm whether the DB needs `vector` only or `vector` + `vectorscale`; enable extensions
**first** (`CREATE EXTENSION IF NOT EXISTS vector; ... vectorscale;`); keep raw text and embedding
**in the same row**; pick the distance metric **before** designing indexes or queries; create the ANN
index whose **operator class matches the query operator**; tune query-time settings only after the
query shape is correct. Fixed-dimension `VECTOR(n)` when the model is stable, plain `VECTOR` only
when variable dimensions are intentional. Canonical shape
`SELECT * FROM t ORDER BY embedding <=> $1 LIMIT 10`. Cosine = `<=>`. Index sections cover
StreamingDiskANN, HNSW, ivfflat.

**`.github/skills/pg-textsearch-skill/`** (SKILL.md 131 lines, `references/pg_textsearch.md` 844
lines). Prescribes: confirm Tiger Cloud vs self-hosted; `CREATE EXTENSION pg_textsearch` then verify
via `pg_extension`; **self-hosted requires `shared_preload_libraries = 'pg_textsearch'` before
restart — Tiger Cloud does not**; create a **single-column** BM25 index
(`USING bm25(col) WITH (text_config = 'english')`); **BM25 indexes are single-column only**; load data
first then build the index; use implicit `<@>` in `ORDER BY` for simple ranked queries; switch to
**`to_bm25query()` when filtering in `WHERE`, naming the index explicitly, or inside PL/pgSQL**; tune
`k1`/`b` only when custom ranking is actually needed. The reference has a dedicated reciprocal-rank-
fusion hybrid-search section plus crash-recovery and self-hosted caveats.

**Current code already follows the `to_bm25query()` rule** — `repository.py:405` uses the explicit
form with the index named, in both `WHERE` and `ORDER BY`.

---

## 11. OpenSpec state

- **`openspec/changes/` contains only `archive/`.** `openspec list` prints **"No active changes
  found."** 35 archived changes; the newest six dated `2026-09-07`: `agent-tools-unification`,
  `cleanup-foundation`, `cognee-agent-memory`, `documents-unified-schema`,
  `error-handling-foundation`, `ingestion-pipeline-unification`. **All six bands A-F from the
  handover are archived** — the handover's "D 14-16 remain, E and F not started" is stale.
- **`openspec/specs/` holds 28 capability specs, none retrieval/RAG-related.** Full list:
  cognee-v1-api, datetime-utc-cleanup, feature-error-contract, http-result-rendering, llm-injection,
  mcp-context-di, mcp-directory-restructure, mcp-server-codemode, mcp-server-composition,
  mcp-server-pagination, mcp-server-prompts, mcp-server-resources, mcp-telemetry, mcp-testing,
  migration-completion, noqa-documentation, outbox-helper-extraction, pattern-matching-standard,
  repository-transaction-safety, result-layer-boundaries, saul-cognee-final-report-write,
  saul-memory-prefetch-and-retrieval, session-required, settings-validation,
  shared-infrastructure-errors, test-mock-isolation, transactional-outbox, typed-exception-handling.
- **`openspec/specs/document-retrieval-schema/` does not exist**, even though
  `openspec/changes/archive/2026-09-07-documents-unified-schema/specs/document-retrieval-schema/spec.md`
  exists with `## ADDED Requirements`. Either the archive did not sync the delta into `specs/`, or it
  was rolled back. **No capability spec covers documents, retrieval, search, chunking, ingestion or
  RAG anywhere in `openspec/specs/`.** This territory is spec-greenfield.
- `openspec/config.yaml` — `schema: spec-gated`; `openspec/schemas/spec-gated/`.

### Change-proposal shape (template: `2026-09-07-documents-unified-schema`)

```
<change-dir>/
  .openspec.yaml        # schema: spec-gated  +  created: YYYY-MM-DD
  proposal.md           # "> Change class: **L** — cross-cutting." then ## Why, ## What Changes
  design.md
  adrs.md
  tasks.md              # "- [ ] N.M Description" under "## N" headings, each with a **Proof** command
  review.md
  specs/<capability>/spec.md   # ## Purpose, then ## ADDED Requirements
```

Spec-delta grammar (from `openspec/config.yaml`): `### Requirement: <name>` with SHALL/MUST (avoid
should/may); `#### Scenario:` with **exactly four hashes** followed by WHEN/THEN; **MODIFIED
requirements copy the ENTIRE existing requirement block**. Proposal must name kebab-case
capabilities, checking `openspec/specs/` for existing names first; 1-2 pages; no implementation
detail. Tasks small enough for one session, ordered by dependency, each verifiable.

`tasks.md` in the template carries a rule worth copying: *"Compare against a captured baseline, never
an absolute number"* — and self-documents that its own Proofs violated it three times.

Locked prior decisions constraining this area (`docs/relay/decisions.md`):
`:55` the unified `langchain_layer` embedder is adopted by the ingestion/documents path;
`:76-77` change 1 promotes `ingestion_kb`, hierarchical chunking for legal docs, unified embedder,
unblocks the docling event loop, Celery offload, **graph in `app.state`**, tenacity;
`:117-118` BM25 + RRF use `pg_textsearch`, not tsvector;
`:257-264` "change 1 writes `chunks`, never `clauses`";
`:388` retrieval ships **three** RRF branches — vector, BM25, trigram.

Repo-wide rules that bind (from `openspec/config.yaml` context and `CLAUDE.md`): thin routers ->
service -> repository, no HTTP concerns in repos; classes only for stateful components; shared clients
live in lifespan and are read from `app.state`; async-first, no blocking calls in async functions;
per-feature typed `FeatureError` union in `features/<name>/errors.py`, never `match/case` on
Success/Failure (ADR-002); `SecretStr.get_secret_value()`; `APIResponse` envelope; ruff + ty gate the
merge; always `uv run`.

---

## Fog

1. **The working tree is broken and I do not know whose breakage it is.** `import app.main` fails.
   The `document_processing/` -> `docling/` move is uncommitted, half-done, and matches **no commit
   message in `git log`** — `git log --oneline -8 -- src/app/shared/rag/` shows only unrelated
   policy/logging/observability commits. I cannot tell whether this is (a) the user's in-progress
   manual rename, (b) an abandoned prior agent run, or (c) something meant to be reverted. **A
   planner must not assume the tree is a valid baseline.** Resolving this needs the user to say
   whether `src/app/shared/rag/docling/` is intended to survive. Everything in todos 240/164 depends
   on that answer.

2. **`pytest` state is unknown.** Given `import app.main` fails, collection almost certainly aborts,
   but I did not run the suite — a single `uv run python -c "import ..."` probe already exceeded 120s
   in this sandbox and I chose not to burn the budget. `uv run pytest -q --collect-only` settles it.

3. **Live database state is unverified.** Memory says "stamped at `0004`, document/vector/search
   schema never created." Disk has migrations through `0017`, and `0014` is literally named "create
   the five phantom relations" — which *implies* the phantom problem was addressed in code, but says
   nothing about whether `alembic upgrade head` ever ran against the live Timescale Cloud instance. I
   did not connect. The handover's claimed head `b3e7c41d92af` matches **no revision id on disk**, so
   at least one of {handover, memory, disk} is wrong. Settling it requires `alembic current` against
   the live DB — and memory warns `alembic/env.py` breaks migration commands.

4. **Whether `pg_textsearch` and `vectorscale` are actually installed in any environment I cannot
   see.** `CREATE EXTENSION` appears only in migrations — **never** in `docker-compose.yml`,
   `docker-compose.prod.yml`, or `infra/`. `docs/relay/decisions.md:345` already flagged this as
   unsettled ("no live install has ever confirmed it"). `0013:62-67` says the migration **fails hard**
   if the extension is absent, so the entire BM25 branch is either working or has never run. The
   pg-textsearch skill's `shared_preload_libraries` requirement applies only to self-hosted, and I
   **could not determine whether the deployment is Tiger Cloud or self-hosted** from the repo.

5. **Two different todos are both numbered 240** (`todo.md:351` and `:362`). Any spec, task ID, or
   commit message keyed to "240" is ambiguous. The user's chat message quoted only the first.

6. **The chat text for 163/164 is strictly richer than `todo.md`.** `todo.md:352-353` say only
   `refactor vectorStore code TSVECTOR,` and `refactor RAG code`. The clauses about `connections/
   features/document/ shared/rag/` and "reuse embedder that lives in langchain_layer" exist only in
   the request. Note also the user wrote `features/document/` (singular) — the directory is
   `features/documents/` (plural), and `features/search/` is now an empty `__init__.py`.

7. **Todo 185's premise is largely already satisfied and I cannot tell if the user knows.** No
   application code contains `tsvector` — the ORM already uses `bm25` + `diskann` + `gin_trgm`, and
   the SQL already uses `to_bm25query()` exactly as the skill prescribes. The only surviving
   `TSVECTOR` is in migrations `0004` and `0014`, on `search_chunks` — a table the archived
   `documents-unified-schema` change declared "not a retrieval table". Whether "remove ts_vector"
   means "write a down-migration dropping the column" or "it's done, close it" is a user question.

8. **Todo 195's "langextract before these" and "pageindex parallel" may already be code.** Both
   packages exist under `src/app/shared/rag/` with real modules. Neither has a runtime call site —
   pageindex's only construction is commented out at `lifespan.py:538`. I could not determine whether
   they are unfinished, deliberately parked, or dead. `graphify`'s index is **stale** here (it still
   reports `document_processing/` nodes), so its edge data on these packages is untrustworthy.

9. **I did not read the Uber "enhanced agentic RAG" blog (165) or the Towards Data Science hybrid
   search article (195).** Both are external URLs; nothing in the repo cites them. Todo 165 has no
   corresponding code to compare against.

10. **`build_document_ingestion_graph` cannot be hoisted to `app.state` as written, and I did not
    establish what the alternative shape is.** It requires a `DocumentRepository`
    (`ingestion_graph.py:51`) wrapping a job-scoped `AsyncSession` (`repository.py:62-63`). That is a
    genuine design question, not a lookup — but the planner should know the constraint is real and
    not incidental.

11. **The 12 `INP001` ruff errors may be pre-existing, not new.** They fire because
    `src/app/features/__init__.py` does not exist; that file was last touched in `8e25352`. The
    handover claims `ruff check src/` was clean at `1ce52ec` (a different branch). I could not
    determine whether the rule was enabled since, whether a per-file-ignore was removed, or whether
    the handover's measurement was simply wrong. Only the 7 `PLC0415` errors are unambiguously caused
    by the uncommitted rename (stale per-file-ignores at `pyproject.toml:539-553`).

12. **Two divergent chunk-size regimes coexist and I could not tell which is authoritative.**
    `features/documents/constants.py:31-32` says 512/64 in whitespace-split "tokens"
    (`chunking.py:33` — not real tokens), while `rag/docling/chunker.py` uses a real
    `AutoTokenizer`-backed Docling `HybridChunker`. Nothing reconciles them. Todo 240's "suitable
    chunking strategy for legal docs" lands squarely on this gap, and there is **no legal-aware
    splitter anywhere** — only regexes at `features/documents/classification.py:116,124`.

13. **`langchain-docling>=2.0.0` is a declared dependency (`pyproject.toml:43`) with zero imports.**
    Relevant to 162 and 176 — the repo may already own the LangChain-native path it is asking about,
    unused. I did not investigate what that package provides.

14. **`docs/relay/handover-210-D14-to-F.md` has an mtime of Sep 9 but its content says "Written
    2026-08-23" and describes branch `refactor/todo-210-sequence` @ `1ce52ec`.** We are on `main` @
    `7cca750`. The mtime is misleading; the content is three weeks stale and is contradicted by the
    OpenSpec archive (section 11). Do not treat its gate table as a baseline.
