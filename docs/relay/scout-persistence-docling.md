# Scout — persistence (alembic, parallel document schemas) + docling parse/chunk

## 1. Persistence truth table

`alembic/env.py:11` imports `database.Base` and registers ONLY two model modules: `app.features.billing.models` (`env.py:23`) and `app.shared.outbox.model` (`env.py:24`). `database/__init__.py:3-4` re-exports `Base`, `ChatMessage`, `ChatSession`, `DocumentVector` — so those three ARE on `Base.metadata` transitively.

| table | model file:line | migration | visible to env.py | reads | writes | verdict |
|---|---|---|---|---|---|---|
| `chat_messages` | `src/database/schemas/chat_messages.py:15` | `c0c17c6eb1cc:26` | YES (via `database/__init__.py:4`) | — | — | live-but-unused |
| `chat_sessions` | `src/database/schemas/chat_messages.py:37` | `c0c17c6eb1cc:47` | YES | — | — | live-but-unused |
| `document_vectors` | `src/database/schemas/document_vectors.py:15` | `c0c17c6eb1cc:64` (**real `create_table`, `Vector(1536)` at :70**) + renamed by `2bc7726317f6:21` | YES | `strategies.py` | — | live (**CORRECTION: not "commented-out SQL only" — it is the only fully round-tripped table**) |
| `search_documents` | `features/search/model.py:25` | `8a7d9b1c2e3f:32` | **NO** | `features/search/repository.py` | `features/search/service.py` | live, invisible to autogenerate |
| `search_chunks` | `features/search/model.py:54` | `8a7d9b1c2e3f:45` (`Vector(768)` :50) | **NO** | search repo/rag | search service | live, invisible |
| `documents` | `features/documents/model.py:30` (`UnifiedDocument`) | `a71f0d7d9c12:29` | **NO** | `features/documents/repository.py` | documents service | live; **name collides with nothing else, but is a second "documents" concept alongside `parent_documents`** |
| `chunks` | `features/documents/model.py:72` (`UnifiedChunk`) | `a71f0d7d9c12:53` (`Vector(dim=768)` :63) | **NO** | documents repo | documents service | live, invisible |
| `parent_documents` | `memory_schema.py:154` (orphan `Base`) | `9f4a1b7c6d2e:29` **CREATE** | **NO** (orphan Base, `database/__init__.py` never imports it) | — | — | **table exists in DB, model orphaned** |
| `clauses` | `memory_schema.py:190` (orphan `Base`) | `9f4a1b7c6d2e:63-105` — **`batch_alter_table` + `add_column` + `alter_column`, NO `create_table`** | **NO** | graphiti/clause tools | `ingestion_kb/nodes.py` | **phantom base**: migration ALTERs a table nothing ever created → `9f4a1b7c6d2e` fails on a clean DB |
| `entities` | `memory_schema.py:56` | **none** | NO | — | — | orphan |
| `relationships` | `memory_schema.py:109` | **none** | NO | — | — | orphan |
| `events` | `memory_schema.py:250` | **none** | NO | — | — | orphan |
| `memory_versions` | `memory_schema.py:279` | **none** | NO | — | — | orphan |
| `statutes` | **none** | **none** | NO | `agents/tools/retrieve_statute_section.py` (raw SQL) | — | **phantom** |
| `match_chunks()` (fn) | n/a | **none** | n/a | `rag_agent_advanced.py` (raw asyncpg) | — | **phantom** |
| outbox (2 tables) | `app/shared/outbox/model.py` | `0001:23,43` | YES | — | — | live |
| billing (15 tables) | `features/billing/models/` | `0002:38..354`, `0003`, `0004` | YES | — | — | live |

**Resolution of the `clauses` contradiction:** the claim "clauses IS in migration 9f4a1b7c6d2e" is *half* right. `9f4a1b7c6d2e` only **adds columns to** `clauses` (`:63-99`), backfills (`:101-102`), and re-types `embedding` to `Vector(768)` (`:105`). No revision anywhere creates `clauses`. So `clauses` is in the same category as `statutes` — no DDL origin — but worse, because a migration *depends* on it existing.

## 2. Alembic revision graph

```
None → c0c17c6eb1cc (chat_*, document_vectors Vector(1536))
     → 2bc7726317f6 (rename document_vectors.metadata → meta_data)
        ├── 8a7d9b1c2e3f (search_documents, search_chunks)   [HEAD A branch]
        │      └── 9f4a1b7c6d2e (parent_documents CREATE; clauses ALTER)
        │             └── 0001 (outbox) → 0002 (billing) → 0003 → 0004  ← HEAD A
        └── a71f0d7d9c12 (documents, chunks)                            ← HEAD B
```
Confirmed `down_revision` values: `c0c17c6eb1cc:18=None`, `2bc7726317f6:15=c0c17c6eb1cc`, `8a7d9b1c2e3f:19=2bc7726317f6`, `a71f0d7d9c12:17=2bc7726317f6`, `9f4a1b7c6d2e:18=8a7d9b1c2e3f`, `0001:17=9f4a1b7c6d2e`, `0002:17=0001`, `0003:19=0002`, `0004:18=0003`. 10 files incl. `__init__.py` → **9 revisions, 2 heads: `0004` and `a71f0d7d9c12`**.

A merge revision must reconcile only *naming/extension* overlap, not column conflicts — the two branches touch disjoint tables. But both branches independently `CREATE EXTENSION vector` / `pg_textsearch` (`8a7d9b1c2e3f:25-29`, `a71f0d7d9c12:23-26`, `9f4a1b7c6d2e:24-26`); all use `IF NOT EXISTS`, so idempotent.

**Beyond the split:**
- `env.py` imports neither `features.documents.model`, `features.search.model`, nor `database.schemas.memory_schema` → autogenerate would emit `DROP TABLE documents/chunks/search_*` (they're in the DB but not in `Base.metadata`).
- `memory_schema.py:51` declares its **own** `DeclarativeBase` — even importing it into `env.py` would not register it on `database.Base.metadata`.
- `9f4a1b7c6d2e` is **not runnable on a fresh DB**: `batch_alter_table("clauses")` at `:63` and `op.execute("UPDATE clauses …")` at `:101` presuppose a table no revision creates. This blocks the whole `0001→0004` chain, i.e. outbox and billing cannot be migrated from scratch today.
- Revision-ID style is mixed: 4 sequential (`0001`–`0004`) vs 5 hash IDs; the sequential ones sort ahead of hashes lexically but the chain is explicit, so ordering is safe.

## 3. `Vector(768)` / dimension hardcode sweep

| location | value | must become |
|---|---|---|
| `features/documents/model.py:94` | `Vector(768)` | `Vector(settings.EMBEDDING_DIMENSION)` |
| `features/search/model.py:73` | `Vector(768)` | same |
| `src/database/schemas/memory_schema.py:218` | `Vector(768)` (`clauses`) | same |
| `alembic/versions/8a7d9b1c2e3f:50` | `Vector(768)` | migrations must stay frozen-literal (historical) |
| `alembic/versions/a71f0d7d9c12:63` | `Vector(dim=768)` | frozen |
| `alembic/versions/9f4a1b7c6d2e:105` | `alter_column(... Vector(768))` | frozen |
| `alembic/versions/c0c17c6eb1cc:70` | `Vector(1536)` + comment `# <-- add this, adjust dims` | **inconsistent with every other table** |
| `features/search/embeddings.py:16` | `output_dimensionality=768` | `settings.EMBEDDING_DIMENSION` |
| `document_processing/embedder.py:26-29` | `{"dimensions": 1536}` for `gemini-embedding-001`/`gemini-embedding`, default 1536 | **contradicts 768 columns** — a 1536-vec insert into `Vector(768)` raises `DataError` |
| `document_processing/embedder.py:167,177,228` | `[0.0] * config["dimensions"]` zero-vector fallback | same |
| `settings.py:50-52` | validator map `gemini-embedding-2-preview:768`, `text-embedding-004:768` | authoritative |
| `settings.py:208` | `PINECONE_DIMENSION=768` | second, independent dimension knob |
| `settings.py:212` | `EMBEDDING_DIMENSION=768` | authoritative |
| `memory_schema.py:8,188` | docstrings say "768-dim" | doc only |

**Trap:** pgvector's `vector(n)` typmod is not widenable in place — `ALTER TABLE … ALTER COLUMN embedding TYPE vector(1024)` fails whenever any row holds a differently-sized vector, and every pgvector index (HNSW/IVFFlat/diskann) on the column must be dropped first. A dimension-change migration therefore has to: DROP the vector index → either `USING NULL::vector(n)` (discarding all embeddings) or add a new column + re-embed + swap → recreate the index → and the model default must be read at import time, so `EMBEDDING_DIMENSION` must be settled before the app boots. Net: **changing the dimension is a re-embedding job, not a DDL migration.**

## 4. docling parse → chunk → persist chain

**CORRECTION on paths/lines** — there is no `document_processing/parser.py`. The parser is `src/app/features/documents/parser.py`.

1. `features/documents/parser.py:19` `parse_document(raw_bytes, filename, content_type)` — `content_type` is `del`'d at `:20` (ignored). `:24` builds a **fresh** `DocumentConverter` per call via `create_document_converter(gpu_available=False)`; `:25` calls **synchronous** `converter.convert()` inside `async def` with no `to_thread` → blocks the event loop for the whole OCR/layout pass. `:29` `export_to_markdown()` (also sync/CPU). Confirmed. `:34` hardcodes `tables=[]` — docling table structures are parsed and then thrown away.
2. `features/documents/classification.py:86` `segment_chunks` — **CORRECTION: `:86`, not `:91`.** Branch at `:91`: `legal_contract`/`legal_policy` → `_segment_legal_chunks` (`:141`), everything else → `_chunk_document` (`:115`) → `chunk_document_simple` (`chunker.py:189`).
3. Legal path `_segment_legal_chunks:146` is a bare `re.split(r"\n\s*\n", parsed.markdown)` — **CORRECTION: regex split is `:146`, truncation `blocks[:200]` is `:158`** (not `:141`/`:161`). It is *fully sync* and bypasses `HybridChunker` entirely, so for legal docs the heading hierarchy docling extracted, the token budget (`max_tokens=512`, `classification.py:125`), and `merge_peers=True` (`chunker.py:35`) are all discarded. `blocks[:200]` silently drops paragraph 201+ with **no warning emitted** — the only `QualityWarning` (`:148-156`) fires on the *opposite* condition (`len(blocks) <= 1`).
4. `chunker.py:23` `get_tokenizer` calls `AutoTokenizer.from_pretrained` (sync, network/disk on first use) — also uncached and not the model used for embedding.

**"Hierarchical chunking" for legal docs requires** what `chunker.py:29-36` already provides (heading-path-contextualised, token-bounded, peer-merged chunks keyed to `DoclingDocument` structure) plus clause-boundary awareness; what exists today for legal docs is a blank-line regex. The `HybridChunker` is reachable only for `document_kind == "generic"`.

**Duplicate write, confirmed and refined** (`features/documents/service.py`): `:520` upserts all chunk rows; `:553` then calls `_verify_legal_chunks` (`:663`), which loops every chunk through `write_and_verify_chunk` (`:673` → `graphiti_verifier.py:28`, one `graphiti.add_episode` at `:50` **plus** one `graphiti.search` at `:68` per chunk — 2 Graphiti round-trips/chunk, serial, no gather), mutates the dict in place (`:682-683`), and re-upserts **the entire set** at `:686`. `build_chunk_rows` (`repository.py:601-604`) is a pure dict-spread over `Sequence[dict]`, so this is not a type error — it is a genuine second full write of every embedding payload.

## 5. Embedder inventory (this area)

| location | provider / model | dim | batch | cache |
|---|---|---|---|---|
| `features/search/embeddings.py:10-17` `build_embedding_client()` | LangChain `GoogleGenerativeAIEmbeddings`, `settings.GEMINI_EMBEDDING_MODEL` | `output_dimensionality=768` hardcoded `:16` | n/a | **none — new client per call** |
| `features/search/service.py:170-171` | via `build_embedding_client()`, `aembed_query` | 768 | — | none |
| `features/search/service.py:299,315` | `aembed_documents` | 768 | list-sized | none |
| `features/documents/service.py:267-268` | `build_embedding_client()`, `aembed_query` | 768 | — | none |
| `features/documents/service.py:633,636` | `aembed_documents` | 768 | batched | none |
| `features/documents/service.py:406` → `_cached_embedding` at `:813-824` | `aembed_query(task_type="RETRIEVAL_QUERY")` | 768 | — | **Redis** |
| `langgraph_layer/ingestion_kb/nodes.py:644` → `_cached_embedding:716`, `_call_embedding_fn:738` | injected `EmbeddingFunction` | unspecified | — | **Redis** (second, parallel implementation of the same helper) |
| `document_processing/embedder.py:51,60,125` | **raw `google.genai` SDK** `genai.Client()`, `gemini-embedding-001` (`:19`), `task_type=GEMINI_TASK_TYPE` | **1536** (`:26-29`) | `batch_size=100` (`:76`) | `EmbeddingCache` in-memory (`:299`) |
| `document_processing/embedder.py:281` `create_embedder()` | wraps the above | 1536 | — | — |
| `rag_agent_advanced.py:129,201,270,380,444` | `embedder.embed_query` from a **phantom import** | — | — | — |
| `shared/rag/strategies.py:352-655` | all commented out | — | — | — |

Two independent `_cached_embedding` implementations exist (`documents/service.py:813` and `ingestion_kb/nodes.py:716`), two independent client constructors (LangChain at `search/embeddings.py:10`, raw SDK at `embedder.py:51`), and **two mutually incompatible dimensions** (768 vs 1536). A single `langchain_layer` embedder must cover: `aembed_query` with per-call `task_type` (RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT), `aembed_documents` with configurable batch size, `output_dimensionality` driven by `settings.EMBEDDING_DIMENSION`, Redis-backed caching keyed on (model, dim, task_type, text), a zero-vector-or-raise policy for failures (`embedder.py:167,177,228` currently silently substitutes zeros), and process-lifetime client reuse rather than per-call construction.

## 6. Phantom-import check

- `from ingestion.embedder import create_embedder` at `rag_agent_advanced.py:119,198,267,373` — **CONFIRMED PHANTOM.** `src/` top-level packages are `alembic, app, database, lynk, mcp_core, tasks`; no `ingestion`. All four are *function-local* imports, so the module still imports cleanly and the failure is deferred to call time (`ModuleNotFoundError`), which is why lint/type checks do not surface it.
- `from .embedder import embed_chunks` at `ingest_v2.py:17` — **VALID**; target is `document_processing/embedder.py:182`. (**CORRECTION: `:17`, not `:18`.**)
- `from .embedder import create_embedder` at `ingest.py:18` — **VALID**; target `embedder.py:281`.
- Repo-wide sweep for imports of undeclared top-level packages (`ingestion|agent|utils|tools|core|models` as roots) returned **only** the four `rag_agent_advanced.py` lines.
- `src/app/shared/rag/document_processing/todo_temp.py` does not parse, so it is invisible to both ruff's import checks and any AST sweep — its imports are unverifiable.

## 7. Confirmations of prior findings, and one correction

- `document_processing/models.py` is all Pydantic `BaseModel` (`Chunk` at `:120`, `IngestionConfig`) — **no `@dataclass` remains** in `document_processing/`, `features/documents/`, or `langgraph_layer/ingestion_kb/`. **todo (a)'s dataclass→pydantic conversion is already done** for this area (excluding the unparseable `todo_temp.py`).
- `document_vectors` is the one claim I must correct: it is **not** commented-out SQL. `c0c17c6eb1cc:63-70` creates it with `Vector(1536)`, `2bc7726317f6:21` renames a column on it, `document_vectors.py:15` models it, and `database/__init__.py:4` puts it on `Base.metadata` — the only table in this area with model + migration + env.py visibility all three.

## Fog

- **Could not settle live/orphan for `parent_documents`, `events`, `memory_versions`.** `parent_documents` has a migration but its model sits on the orphan `Base`; no reader/writer found. Settling it needs `\dt` against a real database, or the deploy history.
- **`clauses` writer path unverified.** `ingestion_kb/nodes.py` and `graphiti/write_clause_episodes.py` reference clause concepts, but whether any code issues SQL against the `clauses` *table* (vs. Graphiti/Neo4j nodes) I did not trace to a statement. Needs `graphify path "write_clause_episodes" "clauses"`.
- **Whether the deployed DB was built by `alembic upgrade` at all.** Given `9f4a1b7c6d2e` cannot run on a clean DB and `statutes`/`match_chunks()` exist in no revision, some tables were almost certainly created out-of-band (psql script, `Base.metadata.create_all`, or Supabase console). I found no such script under `src/`; a search of deploy/ops directories outside `src/` would settle it.
- **`openspec/`**: no spec under `openspec/specs/` (21 dirs, all MCP/typing/outbox topics) covers persistence, migrations, or embeddings. `openspec/changes/` has two in-flight (`cognee-saul-memory-migration`, `mintlify-documentation`) — `cognee-saul-memory-migration` may touch `memory_schema.py`; I did not open it.
- **`EMBEDDING_DIMENSION` has exactly two readers** (corrected): `src/app/utils/embedding.py:16` and its own validator at `settings.py:48`. Nothing in the ORM, migrations, or any embedding client reads it — and `embedding.py:16`'s only consumer path is broken by the `logger` submodule-shadowing bug at `embedding.py:5`. `PINECONE_DIMENSION` (`settings.py:208`) has **zero** readers.
