# Scout Report — `features/search/` (Leg 1, newly in scope)

Repo: `/home/harmeet/Desktop/Projects/langchain-fastapi-production`
Date: 2026-08-17

## Corrections to the briefing (read first)

1. **`build_embedding_client` is NOT in `search/service.py`.** It is at
   `src/app/features/search/embeddings.py:10` (whole file is 17 lines).
   `service.py` only *imports* it (`service.py:38`) and calls it at
   `service.py:170`, `:262`, `:299`.
2. **`model.py:73` is not the `ts_vector` column.** `:73` is
   `embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)`.
   The full-text column is named `content_tsv` and lives at
   `src/app/features/search/model.py:75-79`.
3. **`ts_vector`/`content_tsv` is already dead weight.** BM25 does not use it —
   `repository.py:415,417,419` query `c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')`
   (VectorChord-BM25 operator, applied to the raw `content` column). No `ts_rank`,
   no `@@`, no `plainto_tsquery` anywhere in the search feature.
4. **BM25 + RRF already exist** (`repository.py:189` bm25_search, `fusion.py:28`
   `reciprocal_rank_fusion`, `RRF_K=60` at `constants.py:8`). Item 195 is not
   greenfield. Re-ranking is the only missing third of it.

## 1. Full inventory of `features/search/`

| File | Lines | Public symbols | Status |
|---|---|---|---|
| `__init__.py` | 33 | re-exports `chunk_text`, `build_embedding_client`, `RankedChunk`, `RankedResultRow`, `reciprocal_rank_fusion`, `SearchChunk`, `SearchDocument`, `SearchChunkRecord`, `assemble_rag_context`, `router`, 6 constants (`:16-33`) | live — the import surface `documents/` uses |
| `model.py` | 92 | `SearchDocument` (`:22`), `SearchChunk` (`:51`) | live tables, unreachable via HTTP |
| `repository.py` | 545 | `SearchRepository` (`:57`), `build_chunk_rows` (`:526`) + 5 module helpers | live via `retrieval_kb` graph |
| `service.py` | 439 | `SearchService` (`:57`), `process_ingestion_document` (`:291`), `run_ingestion_task` (`:349`) | mixed (see §4/§7) |
| `dto.py` | 152 | request/response models | live only through unmounted router + documents reuse |
| `router.py` | 81 | `router` — 5 endpoints | **NOT MOUNTED** (§7) |
| `dependencies.py` | 49 | `get_search_repository` (`:17`), `get_search_service` (`:31`), `get_current_user_id` (`:44`), `SearchServiceDep`, `UserIdDep` | dead while router unmounted |
| `embeddings.py` | 17 | `build_embedding_client` (`:10`) | live |
| `chunking.py` | 51 | `TextChunk` (`:8`), `chunk_text` (`:18`) | live |
| `fusion.py` | 46 | `RankedResultRow` (`:8`), `RankedChunk` (`:18`), `reciprocal_rank_fusion` (`:28`) | live |
| `rag.py` | 103 | `SearchChunkRecord`, `assemble_rag_context` | live |
| `constants.py` | 16 | 13 constants incl. `SEARCH_CHUNKS_BM25_INDEX_NAME` (`:15`), DiskANN tuning (`:13-14`) | live |

Total 1,624 lines of Python (plus stale `__pycache__/` holding `dependency.cpython-312.pyc`
and `handler.cpython-312.pyc` — files that no longer exist in source).

## 2. Schema-collapse table

Source of truth: `src/app/features/search/model.py` + migration
`src/alembic/versions/8a7d9b1c2e3f_add_search_documents_and_chunks.py`
Target: `src/app/features/documents/model.py` (`UnifiedDocument:27` → table `documents`,
`UnifiedChunk:69` → table `chunks`).

### `search_documents` → `documents` (`UnifiedDocument`)

| `search_documents` col | type / null | → `UnifiedDocument` | note |
|---|---|---|---|
| `id` (`model.py:27`) | PGUUID PK, default uuid4 | `id` (`documents/model.py:39`) | identical |
| `source_uri` (`:32`) | Text, NULL | `source_uri` (`:42`) | identical |
| `title` (`:33`) | String(500), NOT NULL | `title` (`:41`) | identical |
| `content_hash` (`:34`) | String(64), **globally UNIQUE**, NOT NULL | `content_hash` (`:44`) | **semantic change** — target uniqueness is `(user_id, content_hash)` (`:32`). Global dedup becomes per-user dedup. |
| `doc_metadata` (`:35`) | JSONB NOT NULL default `{}` | `metadata_` (`:50`) | **rename** `doc_metadata`→`metadata_` |
| `ingested_at` (`:36`) | tz DateTime NOT NULL | `created_at` (`:51`) | **rename**; no `ingested_at` on target |
| `updated_at` (`:41`) | tz DateTime NOT NULL, onupdate | `updated_at` (`:56`) | identical |
| — | | `user_id` (`:40`) String(255) **NOT NULL, no default** | **no source value** — search has no user scoping at all. Backfill needs a sentinel/system user or the column made nullable. |
| — | | `object_uri` (`:43`) Text **NOT NULL, no default** | **no source value** — search ingests raw text via API body (`service.py:74`), never an S3 object. Hard blocker for a straight `INSERT..SELECT`. |
| — | | `document_kind` (`:45`) default `"generic"` | server default covers backfill |
| — | | `status` (`:46`) default `"received"` | search rows are already chunked → backfill as a terminal status, not `received` |
| — | | `jurisdiction` (`:47`), `contract_type` (`:48`), `parties` (`:49`) | nullable / default; no source value |
| indexes | `uq_search_documents_content_hash` (mig `:41`) | `uq_documents_user_content_hash` (`:32`) + 4 indexes (`:33-36`) | drop the global unique |
| relationship | `chunks` (`:48`), **no cascade** | `chunks` (`:63`) `cascade="all, delete-orphan"` | ORM-level cascade gained |

### `search_chunks` → `chunks` (`UnifiedChunk`)

| `search_chunks` col | type / null | → `UnifiedChunk` | note |
|---|---|---|---|
| `id` (`model.py:61`) | PGUUID PK | `id` (`:81`) | identical |
| `document_id` (`:66`) | FK `search_documents.id` ON DELETE CASCADE | `document_id` (`:82`) FK `documents.id` CASCADE | FK retarget only |
| `chunk_index` (`:71`) | Integer NOT NULL | `chunk_index` (`:88`) | identical |
| `content` (`:72`) | Text NOT NULL | `content` (`:90`) | identical |
| `embedding` (`:73`) | `Vector(768)` NULL | `embedding` (`:94`) `Vector(768)` NULL | identical — **both hardcode 768** while `settings.py:212` has `EMBEDDING_DIMENSION` |
| `chunk_metadata` (`:74`) | JSONB NOT NULL default `{}` | `metadata_` (`:95`) | **rename**; target also has separate `custom_metadata` (`:96`) — decide which receives it |
| `content_tsv` (`:75-79`) | TSVECTOR **generated** `to_tsvector('english', content)` persisted, NOT NULL | **DROP** | no reader anywhere (see §8). Target's analogue is `search_text` (`:100`), a *plain Text* generated column concatenating `clause_type‖preamble‖content`, fed to BM25. |
| `created_at` (`:80`) | tz DateTime NOT NULL | `created_at` (`:110`) | identical |
| `updated_at` (`:85`) | tz DateTime NOT NULL, onupdate | **no equivalent — needs a new column** | `UnifiedChunk` has no `updated_at`. `search/repository.py:162` upserts it in `on_conflict_do_update`; `build_chunk_rows` writes it (`repository.py:542`). Either add `updated_at` to `UnifiedChunk` or drop upsert-timestamp behaviour. |
| — | | `user_id` (`:87`) String(255) NOT NULL | **no source value** (same blocker as parent) |
| — | | `chunk_kind` (`:89`) default `"generic"` | default covers it |
| — | | `preamble` (`:91`) Text NOT NULL default `""`, `clause_type` (`:92`) NULL, `page_no` (`:93`) default 0 | no source value; `preamble` feeds `search_text` |
| — | | `quality_warnings` (`:97`), `graphiti_episode_id` (`:98`), `graphiti_verified` (`:99`) | defaults cover backfill |
| unique | `uq_search_chunks_document_chunk_index` (`:56`) | `uq_chunks_document_chunk_index` (`:74`) | **constraint name changes** — `search/repository.py:157` names it literally in `on_conflict_do_update(constraint=...)`; must be edited |
| idx: `ix_search_chunks_document_id` (mig `:74`) | btree | covered by `ix_chunks_user_document` (`:75`) | prefix is `user_id`, not `document_id` — a document-only lookup loses its index |
| idx: `ix_search_chunks_document_chunk_index` (mig `:76`) | btree | subsumed by unique constraint | drop |
| idx: `search_chunks_embedding_idx` (mig `:82`) | **diskann** `vector_cosine_ops` | must exist on `chunks.embedding` | **verify target has a diskann index** — not declared in `documents/model.py:73-79` |
| idx: `search_chunks_bm25_idx` (mig `:86`) | `USING bm25(content) WITH (text_config='english')` | target uses `clauses_bm25_idx` on `search_text` (`search/repository.py:356`) | index name is hardcoded in SQL at `search/repository.py:415,417,419` and `constants.py:15` |
| idx: `ix_search_chunks_chunk_metadata_gin` (mig `:90`) | GIN jsonb | `ix_chunks_metadata_gin` (`:77`) | equivalent |
| idx: `ix_search_chunks_content_tsv_gin` (mig `:94`) | **GIN on tsvector** | **DROP** with the column — zero readers | |
| idx: `ix_search_chunks_content_trgm` (mig `:97`) | `gin (content gin_trgm_ops)` | **no equivalent — needs a new index** | `trigram_search` (`search/repository.py:236`) is one of three RRF branches; without this index it seq-scans |

## 3. Data-migration risk: **this is an empty-table collapse**

Evidence for "no rows anywhere":

- The only write path into `search_documents` is `SearchService.ingest_document`
  (`search/service.py:72`), reachable **only** via `POST /search/ingest`
  (`search/router.py:25`) — and that router is not mounted (§7). No other caller.
- The only write path into `search_chunks` is `process_ingestion_document`
  (`search/service.py:291`) → `repo.upsert_chunks` (`repository.py:147`), invoked by
  `run_ingestion_task` (`service.py:349`), invoked by the Celery task
  `src/tasks/search_tasks.py:10`. That task is triggered by the outbox event
  `tasks.search_ingest` emitted at `search/service.py:110` — inside the unreachable
  ingest endpoint. Dead chain end-to-end.
- **No seed, fixture, or factory populates these tables.** Files referencing the table
  names at all: the migration, `search/{constants,repository,model,__init__}.py`, and docs.
  `scripts/` contains no seeder (`alembic.sh`, `docker.sh`, `replay_outbox.py`, graph tooling).
- Tests never touch a database: `tests/integration/test_search.py:34` patches
  `SearchRepository` and `:38` patches `build_embedding_client`; the session is an
  `AsyncMock` (`:45`). `tests/conftest.py` has **no** engine/`create_all`/testcontainer
  fixture. `tests/unit/search/{test_chunking,test_fusion,test_rag}.py` are pure functions.

Consequence: the collapse can be a `DROP TABLE` + code retarget, not a backfill.
**The `user_id`/`object_uri` NOT-NULL problem in §2 is therefore theoretical for data but
real for code** — every write path must start supplying both.

## 4. The two (actually three) ingestion entry points

| | `search.process_ingestion_document` (`search/service.py:291`) | `documents` upload path (`documents/service.py:118` `upload_document` → `:465` `process_document_ingestion`) | `ingestion_kb` graph (`shared/langgraph_layer/ingestion_kb/`) |
|---|---|---|---|
| Input | raw `content: str` from API body | uploaded file → S3 (`build_s3_key`, `documents/service.py:40`) | parsed doc through a LangGraph state machine |
| Chunking | `chunk_text` word-window (`search/chunking.py:18`), 512/64 (`constants.py:9-10`) | `segment_chunks` from `.classification` (`documents/service.py:52`) | `dispatch_contextualize_chunks` fan-out (`nodes.py:194`) + LLM contextualization (`:254`) |
| Embedding | `build_embedding_client()` direct, batched 200 (`service.py:299,314`) | same `build_embedding_client` (`documents/service.py:633`), batched 200 (`:635`) | injected `embedding_fn`, **one text at a time**, redis-cached 24 h (`nodes.py:716-736`) |
| Normalization | **none** | `normalize_embedding` on query only (`documents/service.py:829`) | `normalize_embedding` on every stored vector (`nodes.py:733`) |
| Tables written | `search_chunks` | `documents` / `chunks` | `parent_documents`, `clauses`, `entities`, `relationships` (`nodes.py:497,660,551,597`) |
| Graphiti | no | `write_and_verify_chunk` (`documents/service.py:65`) | `make_graphiti_upsert_node` (`nodes.py:354`) |
| Metadata extraction | none | `extract_legal_metadata`, `enrich_legal_chunks` (`documents/service.py:67-71`) | LLM prompts in `ingestion_kb/prompts.py` |
| BM25 maintenance | `ANALYZE search_chunks` above 10k chunks (`repository.py:186`, `constants.py:12`) | — | `bm25_force_merge('clauses_bm25_idx')` (`nodes.py:749`) |

`process_ingestion_document` does **nothing the other two don't**, except: word-window
chunking with overlap, embedding-batching of 200, and the `ANALYZE` threshold. It has no
classification, no legal metadata, no Graphiti, no contextualization, no quality warnings.
Under item 190 (fold `documents/` into the ingestion pipeline) it is the **weakest** of the
three and is a candidate for deletion rather than absorption — the only behaviour worth
preserving is the batch-of-200 `aembed_documents` call, which `ingestion_kb`'s
one-at-a-time `_cached_embedding` lacks.

## 5. The coupling — one-directional

`search/` imports **nothing** from `documents/` (verified: no `features.documents` reference
under `src/app/features/search/`). Every edge points documents → search.

| Consumer | `path:line` | Symbol (defined at) | What it does | After collapse |
|---|---|---|---|---|
| `documents/service.py` | `:16` | `ANALYZE_THRESHOLD_CHUNKS` (`search/constants.py:12`) | 10 000-chunk `ANALYZE` trigger | **internal** — move constant to documents/ingestion |
| `documents/service.py` | `:17` | `DEFAULT_SEARCH_CACHE_TTL_SECONDS` (`constants.py:5`) | 900 s hybrid-search cache TTL | **internal** |
| `documents/service.py` | `:18` | `INGEST_EMBEDDING_BATCH_SIZE` (`constants.py:11`) | 200-doc embed batch | **internal** |
| `documents/service.py` | `:19` | `RRF_K` (`constants.py:8`) | RRF k=60 | **internal** |
| `documents/service.py` | `:20` | `RankedChunk` (`fusion.py:18`) | fused rank DTO | **internal** |
| `documents/service.py` | `:21` | `RankedResultRow` (`fusion.py:8`) | per-branch ranked row DTO | **internal** |
| `documents/service.py` | `:22` | `SearchChunkRecord` (`rag.py`) | hydrated chunk for context assembly | **internal** |
| `documents/service.py` | `:23` | `assemble_rag_context` (`rag.py`) | token-budgeted context sectioning | **internal** |
| `documents/service.py` | `:24` | `build_embedding_client` (`embeddings.py:10`) | the shared Gemini client | **internal** → moves to `langchain_layer` (§6) |
| `documents/service.py` | `:25` | `reciprocal_rank_fusion` (`fusion.py:28`) | RRF | **internal** |
| `documents/service.py` | `:86` (TYPE_CHECKING) | `ContextSection` (`search/rag.py`) | return type of `assemble_rag_context` | **internal** |
| `documents/dto.py` | `:7` | `DEFAULT_PAGE_SIZE`, `DEFAULT_RAG_TOKEN_BUDGET`, `MAX_PAGE_SIZE` (`constants.py:3,6,4`) | pagination + token budget defaults | **internal** |
| `documents/repository.py` | `:15-19` | `DISKANN_QUERY_RESCORE`, `DISKANN_QUERY_SEARCH_LIST_SIZE`, `TRIGRAM_SIMILARITY_THRESHOLD` (`constants.py:14,13,16`) | DiskANN `SET LOCAL` tuning + trigram floor | **internal** |
| `src/tasks/search_tasks.py` | `:10` | `run_ingestion_task` (`service.py:349`) | Celery entry for search ingest | **disappears** if `process_ingestion_document` is deleted (§4) |
| `shared/langgraph_layer/retrieval_kb/nodes.py` | `:26`, `:172`, `:181` | `SearchRepository` type + `legal_rrf_search` (`search/repository.py:308`) | the graph calls search's repo | **inverted dependency** — a `shared/` module typed on a `features/` class; §8 note |

So: 15 of 16 couplings are constants/DTO/helper reuse that become intra-module once search
folds into documents/ingestion. The two structural ones are `search_tasks.py:10` and the
`shared → features` inversion at `retrieval_kb/nodes.py:26`.

**Cross-schema oddity:** `SearchRepository.legal_rrf_search` (`search/repository.py:308-405`)
does not query `search_chunks` at all — it queries the **`clauses`** table
(`:337`, `:383`) with `clauses_bm25_idx` (`:356`), i.e. `ingestion_kb`'s schema. A
search-feature method already reaching into a third schema.

## 6. Embedding-path unification

`build_embedding_client` — `src/app/features/search/embeddings.py:10-17`:

| | search/documents | ingestion_kb |
|---|---|---|
| Provider | `GoogleGenerativeAIEmbeddings` (`embeddings.py:5,13`) | injected `embedding_fn`, duck-typed (`ingestion_kb/nodes.py:738-745`: tries `aembed_query`, then `ainvoke`, then `__call__`) |
| Model | `settings.GEMINI_EMBEDDING_MODEL` (`embeddings.py:14`; default `gemini-embedding-2-preview`, `settings.py:194`) | caller's choice |
| Dimension | **`output_dimensionality=768` hardcoded** (`embeddings.py:16`) — ignores `settings.EMBEDDING_DIMENSION` (`settings.py:212`), which even has a model/dim consistency validator (`settings.py:48-60`) | inherited |
| Batch | `aembed_documents` in slices of 200 (`search/service.py:314`, `documents/service.py:635`) | **one text per call** (`nodes.py:644`) |
| `task_type` | `RETRIEVAL_DOCUMENT` on ingest (`search/service.py:317`), `RETRIEVAL_QUERY` on query (`service.py:173`) | `_call_embedding_fn` passes **no `task_type`** (`nodes.py:740`) — asymmetric embeddings vs. the search side |
| Caching | none | redis, key `kb:embedding:<sha256>`, TTL 24 h (`nodes.py:721,734`) |
| Normalization | **none on stored vectors** (`search/service.py:315`); query-side only in documents (`documents/service.py:829`) | `normalize_embedding` on every stored vector (`nodes.py:733`) |
| Retry | none | `retry_immediate(label="gemini_embedding")` (`nodes.py:729`) |

Note: documents does **not** have a separate client — it imports search's
(`documents/service.py:24`). The genuine second path is `ingestion_kb`'s duck-typed
`embedding_fn`.

Acceptance criteria for one unified embedder in `langchain_layer`: single-text **and**
batched document embedding; explicit `task_type` on both query and document calls;
dimension read from `settings.EMBEDDING_DIMENSION` not a literal; redis caching with a
documented key + TTL; `normalize_embedding` applied consistently (or never) so one
`chunks.embedding` column never mixes conventions; `retry_immediate` wrapping; and a
`Vector(settings.EMBEDDING_DIMENSION)` column declaration so model and client cannot drift.

## 7. The mount question — decision inputs (no recommendation)

`search/router.py:22` declares `APIRouter(prefix="/search", tags=["search"])`.
`src/app/api/v1.py:4-17` mounts auth, health, users, profile, documents, agent_saul —
**no search import, no `include_router`**. `src/app/api/v2.py:8-10` mounts health + billing
only. `search/__init__.py:14` does export `router`, so nothing but the mount is missing.

| Endpoint | `router.py` | Auth | Works against current schema? |
|---|---|---|---|
| `POST /search/ingest` | `:25` | **none** | yes — writes `search_documents` + queues outbox `tasks.search_ingest` (`service.py:110`) |
| `GET /search/ingest/{task_id}` | `:39` | **none** | yes — reads Celery `AsyncResult` (`service.py:131`) |
| `POST /search/hybrid` | `:50` | **none** | yes, if the DB has `search_chunks_bm25_idx`, diskann and pg_trgm (`migration :82-99`) |
| `POST /search/rag` | `:61` | **none** | yes (wraps `hybrid_search`) |
| `POST /search/ask` | `:72` | `UserIdDep` | **no** — see below |

Flags if mounted as-is:

1. **`UserIdDep` is broken repo-wide.** `search/dependencies.py:45` returns
   `request.state.user_id`; **nothing in the codebase ever assigns it** — no auth
   middleware exists (`src/app/middleware/` = api_versioning, global_exception_handler,
   health_check, otel, server_middleware; registrations at `main.py:77-94`), and
   `server_middleware.py:49` only seeds a *logging* dict with `"user_id": None`.
   `agent_saul/dependencies.py:62` carries the comment "Example: return
   `request.state.user_id` after auth middleware sets it." `POST /search/ask` would
   `AttributeError`. **`documents/dependencies.py:61-62` is identical, so the already-mounted
   documents router has the same defect** — outside my area but load-bearing here.
2. **Four of five endpoints have no authentication and no user scoping at all.**
   `search_documents` has no `user_id` column (`model.py:22-46`), so multi-tenant isolation
   is structurally impossible — any caller reads every other caller's chunks via
   `/search/hybrid`.
3. **Unmetered LLM/embedding cost.** `/search/ingest` accepts arbitrary `content` and
   `/search/hybrid` calls `aembed_query` per request (`service.py:171`) with **no rate limit**
   — `src/app/utils/rate_limit/dependencies.py` exists but is not referenced by
   `search/router.py`.
4. **`/search/ask` also depends on Graphiti and the `clauses` table**
   (`service.py:259-265` → `retrieval_kb`, which calls `legal_rrf_search` against `clauses`),
   so it fails wherever `clauses` or `clauses_bm25_idx` is absent.
5. **No client or test hits these paths.** `tests/integration/test_search.py` calls services
   directly (`:46`, `:72`, `:110`) — never an HTTP client. No OpenAPI/e2e reference found.

Also: `build_retrieval_graph` (`retrieval_kb/graph.py:28`) has exactly **one** caller —
`search/service.py:259`. Unmounting means the entire `retrieval_kb` graph is currently
unreachable. Deleting `ask_legal` deletes the only live entry point to it.

## 8. Full-text-search mechanics

- `content_tsv` (`model.py:75-79`) is a **PostgreSQL generated column**, not a trigger and
  not application code: `Computed("to_tsvector('english', content)", persisted=True)`,
  mirrored in the migration at `:52-57`. It is `STORED`, so it costs write amplification on
  every chunk insert.
- **There is a GIN index**: `ix_search_chunks_content_tsv_gin` (migration `:93-95`).
- **Nothing reads either.** No `@@`, `ts_rank`, `plainto_tsquery`, `websearch_to_tsquery`, or
  `content_tsv` reference exists outside `model.py` and the migration.
- Actual BM25 is the **VectorChord-BM25 extension**, not tsvector: `repository.py:415,417,419`
  use `c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')` against the index created
  at migration `:85-88` (`USING bm25(content) WITH (text_config='english')`), extension
  `pg_textsearch` (migration `:27`). Scores are negated to make higher-better
  (`repository.py:415`).
- RRF exists and is correct: `fusion.py:28-46`, `k=RRF_K=60` (`constants.py:8`), fusing the
  three branches assembled by `_run_parallel_search` (`service.py:367-396`:
  bm25 + vector + trigram). A second, **in-database weighted** RRF exists at
  `repository.py:308-405` over `clauses`.
- Re-ranking: `CrossEncoderReranker` exists in `retrieval_kb` and is imported by
  `documents/service.py:32` — but **search's own `hybrid_search` never re-ranks**
  (`service.py:161-211` goes straight from RRF to hydration).

For item 195: the existing machinery **is the foundation** — BM25 (VectorChord) and RRF are
already implemented and tuned. `content_tsv` + its GIN index are the part item 195 *replaces*,
and they are already dead, so item 185's removal is a pure subtraction with no reader to fix.
The genuinely missing third of item 195 is wiring re-ranking into the search path.

## Fog

1. **Whether `search_chunks` has rows in a deployed environment.** I proved no code path
   populates them and no seed exists, but I cannot inspect a running database. Settling it
   needs `SELECT count(*) FROM search_chunks` against staging/prod.
2. **Whether `chunks.embedding` has a diskann index.** `documents/model.py:73-79` declares
   four btree/GIN indexes but no vector index, while `documents/repository.py:15-19` imports
   DiskANN `SET LOCAL` tuning — implying one exists in a migration I did not open. Needs a
   grep of `src/alembic/versions/` for `diskann` on `chunks`.
3. **Whether `chunks` has a trigram index on `content`.** `TRIGRAM_SIMILARITY_THRESHOLD` is
   imported by `documents/repository.py:17`, so documents does trigram search too; if
   `ix_chunks_content_trgm` is missing, that branch already seq-scans.
4. **`chunk_metadata` → `metadata_` vs `custom_metadata`.** `UnifiedChunk` has two JSONB
   metadata columns (`:95`, `:96`); which one receives search's `chunk_metadata` changes
   which GIN index serves `@> CAST(:metadata_filter AS jsonb)` (`repository.py:418`).
   Only the documents-side query code can decide, and I did not read
   `build_search_filter_params`.
5. **`UnifiedChunk` has no `updated_at`.** Whether losing upsert-timestamp tracking
   (`search/repository.py:162`) is acceptable is a product question, not a code fact.
6. **Global vs per-user `content_hash` uniqueness.** Collapsing search's global unique
   (`model.py:34`) into `(user_id, content_hash)` (`documents/model.py:32`) changes dedup
   semantics; whether cross-user duplicates are desired is undetermined.
7. **`normalize_embedding` inconsistency.** Search stores raw vectors, `ingestion_kb`
   stores normalized ones. Cosine distance is scale-invariant so ranking is unaffected for
   `vector_cosine_ops`, but any switch to inner-product ops would silently mis-rank a mixed
   column. I could not determine whether normalized/unnormalized vectors already coexist.
8. **No openspec artifact covers this.** `openspec/specs/` has no search or document-schema
   spec; the only mentions of `search_documents` are in an archived change
   (`openspec/changes/archive/2026-06-22-result-pattern-eliminate-dual-method-part-2/`) and
   in-flight changes are `cognee-saul-memory-migration` and `mintlify-documentation` —
   neither touches this area. Prior art searched: "hybrid search", "BM25", "RRF", "tsvector",
   "unified document", "embedding client", "ingestion".
9. **`__pycache__` holds `dependency.cpython-312.pyc` and `handler.cpython-312.pyc`** for
   source files that no longer exist — evidence of an earlier rename I could not reconstruct.

---

**Orchestrator correction (2026-08-17), authoritative over the body above:** every reference in this report
to "VectorChord" / "VectorChord-BM25" is a **mis-attribution**. The extension providing the `bm25` access method
and `to_bm25query()` is **`pg_textsearch`** (Timescale/TigerData), confirmed available at 1.3.0 on the live
Timescale Cloud server. `vchord_bm25` is not available there at all. Additionally: this report describes
`search_documents`/`search_chunks` as existing tables — **they do not exist**; the migration that defines them is
stamped-but-never-applied. See `docs/relay/findings-database.md` §3 and §4.
