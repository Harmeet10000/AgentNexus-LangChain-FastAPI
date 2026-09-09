# Decisions — RAG cluster (todos 235, 240, 163, 164, 185, 162, 165, 195, 176)

Relay session 2026-09-09, branch `main` @ `7cca750`. Companion to `docs/relay/scout-rag-cluster.md`
(terrain) and `docs/relay/research-rag-external.md` (external research).

Extends `docs/relay/decisions.md` — it is **not** superseded. Its locked entries still bind, notably
`:55` (unified `langchain_layer` embedder), `:117-118` (BM25 + RRF via `pg_textsearch`, not tsvector),
`:388` (three RRF branches), and the extension probe at `:345`.

---

## Measured baseline (do not re-derive; re-measure only after step 0)

`import app.main` **fails** — `ModuleNotFoundError: app.shared.rag.document_processing`.

`uv run ruff check --no-cache src/` = **24 errors**, split by cause:

| Rule | N | Cause | Owned by |
|---|---|---|---|
| `INP001` implicit-namespace-package | 12 | `src/app/features/__init__.py` deleted in `8e25352` (August) | **pre-existing debt** |
| `PLC0415` import-outside-top-level | 7 | stale per-file-ignores at `pyproject.toml:539-553` naming the dead `document_processing/*` | change 0 |
| `PLC2701` import-private-name | 3 | `_build_chat_model` imported from `langchain_layer.models` by `documents/{dependencies,service}.py` | **pre-existing**; now in scope via the second #240 |
| `I001` unsorted-imports | 2 | `documents/{classification,parser}.py` — the files importing the dead path | change 0 |

**Change 0 owns ~9 of the 24.** The handover's "ruff clean at `1ce52ec`" measured a different branch
and must not be used as a baseline (`scout-rag-cluster.md` Fog #11, confirmed by
`git log -- src/app/features/__init__.py`).

`pytest` state unmeasured — collection cannot succeed while the tree does not import. Re-measure
immediately after step 0 and capture that number as the real baseline.

---

## D1 — The half-done rename is the user's, and gets finished

`src/app/shared/rag/document_processing/` (deleted, staged) → `src/app/shared/rag/docling/`
(untracked) is **the user's in-progress rename**. It is intended to survive.

Change 0 repoints, in this order:

1. `src/app/shared/rag/docling/__init__.py:3,9,21,27,37` — the new package imports the package it replaced
2. `src/app/features/documents/classification.py:13,14`, `parser.py:11,12`
3. `src/app/shared/langgraph_layer/ingestion_kb/nodes.py:31` (`table_markdown`)
4. `src/app/examples/policy_examples.py:36,40`
5. `pyproject.toml:539-553` — per-file-ignores repointed to `docling/*`
6. Tests: `tests/unit/test_auth_documents_feature_errors.py:19`,
   `tests/unit/shared/rag/test_chunker_tokenizer_cache.py:33,34,40`,
   `tests/unit/shared/rag/test_embedder_no_substitution.py:26,27,34`,
   `tests/unit/shared/rag/test_rag_agent_embedder_import.py:38` (string target, not an import)
7. Non-load-bearing doc refs: `src/app/utils/embedding.py:23`,
   `src/alembic/versions/0013_*.py:101`, `src/app/shared/rag/langextract/langextract_to_graph.py:1`

**Proof:** `uv run python -c "import app.main"` exits 0; `uv run ruff check --no-cache src/` drops from
24 to at most 15 (the pre-existing `INP001` × 12 + `PLC2701` × 3).

**Open:** whether change 0 also creates `src/app/features/__init__.py` to clear the 12 `INP001`.
One file, clears half the remaining errors, but is strictly outside the rename. Not yet ruled on.

---

## D2 — LangExtract and PageIndex are unfinished; change 5 wires them up

Both are real modules under `src/app/shared/rag/` with **zero runtime call sites**. PageIndex's only
construction is commented out at `lifespan.py:538`. They are **not** parked and **not** dead.

Consequence: todo 195 is the *largest* item in the cluster, not the smallest. Its Postgres half
(BM25 + RRF + trigram + graphiti) is already shipped; its `langextract`-before and
`pageindex`-parallel half is a genuine integration build.

---

## D3 — Todo 185 closes as verify-only

No application code contains `tsvector`. The ORM already uses `bm25` (`documents/model.py:114`),
`diskann` (`:122`) and `gin_trgm`; the SQL already uses the explicit `to_bm25query()` form with the
index named (`documents/repository.py:405`), exactly as `.github/skills/pg-textsearch-skill`
prescribes.

Surviving `TSVECTOR` is confined to migrations `0004:54-55` and `0014:79-81,208,218-219`, on
`search_chunks` — a table the archived `documents-unified-schema` change declared "not a retrieval
table". **No down-migration is written; applied migrations are historical record.**

What 185 actually contributes to change 3 is its *second* clause: "write correct SQL query for
documents/ taking skills for pgvector/pgvectorscale". The removal half is done.

**Deliverable:** a spec requirement asserting no `tsvector` in application code, plus a guard test.

---

## D4 — Deployment is Tiger Cloud (managed Timescale)

Consequences:

- `pg_textsearch`'s `shared_preload_libraries` requirement **does not apply** (self-hosted only, per
  the skill).
- `decisions.md:345` already records an executed, rolled-back probe: `tsdbadmin` is not superuser but
  holds `CREATEDB`/`CREATEROLE` and **can** create `pg_trgm` 1.6 and `pg_textsearch` 1.3.0.
  Do not re-run this probe.
- **`vectorscale` is pre-installed by luck, not by design** (`decisions.md:~350`): no revision in the
  chain creates it, and `CREATE EXTENSION` appears **only** in migrations — never in
  `docker-compose.yml`, `docker-compose.prod.yml`, or `infra/`. This is a requirement waiting to be
  written; it is invisible until a fresh environment is provisioned.
- `0013:62-67` — `CREATE EXTENSION IF NOT EXISTS` does **not** soften a missing extension. The
  `bm25`/`diskann` index branches fail hard. The hybrid stack either works or has never run, with no
  degraded middle state.

**Still open (assigned to research):** is `vectorscale` available on Tiger Cloud as a *declarable*
extension, and is `pg_textsearch` current there. Note `pg_textsearch` (Tiger/Timescale) is **not**
`pg_search` (ParadeDB) — the repo uses the former.

---

## D5 — Six OpenSpec changes, one capability each

| Change | Covers | Depends on |
|---|---|---|
| **0 — tree-repair** | the half-done rename; restores a valid baseline | — |
| **1 — graph-lifecycle** | 235 | 0 |
| **2 — ingestion-chunking** | 240, 176, 164, **second #240** | 0 |
| **3 — retrieval-sql** | 163, 185, 162 | 0 |
| **4 — agentic-retrieval** | 165 | 3 |
| **5 — knowledge-stack** | 195 | 2 |

Rejected: four-change merge (mixes a bug fix into a design change), three-change pipeline-stage cut
(units too large to review, ambiguous red gates), single change (violates the repo's own
small-dependency-ordered-tasks convention, all-or-nothing rollback).

`openspec/specs/` currently holds **28 capability specs, none retrieval/RAG-related**, and
`openspec/changes/` holds **only `archive/`** (`openspec list` → "No active changes found").
This territory is **spec-greenfield** — all deltas are `## ADDED Requirements`, no `MODIFIED` blocks,
which sidesteps the archive-semantics trap entirely.

---

## D6 — Adjacent todos: the second #240 is in scope; 190 and 136 are not

`tests/performance/todo.md` contains **two items numbered 240**. Always disambiguate by line.

- **`:351` — first #240** (the one the user quoted): document processing + crawler DB injection,
  docling improvement, legal chunking strategy. → change 2.
- **`:362` — second #240, NOW IN SCOPE**: *"remove build chat model from documents/ and review
  chunking strategy used here and in crawler and find out from where to add them."* This is literally
  the 3 `PLC2701` errors — `_build_chat_model` imported from `langchain_layer.models` by
  `documents/dependencies.py:13` and `documents/service.py:19,26`, and by
  `shared/crawler/processor.py:14`. → change 2.

**Out of scope:** `:347` #190 (move `documents/` into the ingestion pipeline) and `:348` #136
(LangExtract outputs → graph knowledge). They overlap change 5 but stay in `todo.md` for a later pass.

---

## D7 — Graph DI: inject per-invocation via `config["configurable"]`

`build_document_ingestion_graph` (`features/documents/ingestion_graph.py:47`, compiles at `:74` with
**no checkpointer parameter at all**) requires a `DocumentRepository` (`:51`) wrapping a job-scoped
`AsyncSession` (`documents/repository.py:62-63`). It is currently **recompiled per Celery job**
(codegraph edge `run_document_ingestion_task -> build_document_ingestion_graph`) — the real
"compiled in the wrong place" site behind todo 235.

**Decision:** compile once in lifespan into `app.state`; pass the job-scoped `DocumentRepository`
per-invocation through `config["configurable"]`.

Rejected: a session factory closed over at compile time (loses single-transaction semantics across the
graph run); carrying the repository in the graph `State` TypedDict (non-serialisable object in state —
the checkpointer would choke, and
`tests/unit/shared/langgraph_layer/test_ingestion_checkpoint_plumbing.py` already guards this).

### Correction to the todo's premise

`IngestionService` **already** takes the graph by constructor injection
(`features/ingestion/service.py:33-35`, invoked at `:71`). It does not build the graph. Todo 235's
"graph in the service" is **not literally true** for that service.

The genuine 235 work is:

- `app.state.ingestion_graph` — commented out, `lifespan.py:522-537` (with an in-place note forbidding
  restoration of an `embedding_fn=` argument)
- `app.state.pageindex_client` — commented out, `lifespan.py:538`
- the LangGraph checkpointer block — commented out, `lifespan.py:549-565` (a re-enable must use
  `get_database_url(flavour="plain")`; the saver is psycopg-backed and cannot parse the async dialect
  scheme)
- `app.state.saul_graph` — **never assigned anywhere**, so `features/agent_saul/dependencies.py:37-45`
  raises `ServiceUnavailableException` unconditionally. `get_saul_checkpointer` (`:48-53`) reads
  `app.state.langgraph_checkpointer`, also never set. **`agent_saul` 503s today.**
- three import-time module-global compiles in
  `shared/langgraph_layer/open_deep_search/graph.py:278,478,555`
- `build_document_ingestion_graph`, per D7 above

`build_ingestion_graph` takes `db_engine: AsyncEngine` (`ingestion_kb/graph.py:58-64`), which is
assigned at `lifespan.py:464` — **before** the commented block at `:528`. Ordering already permits
the hoist, and the engine is a handle, not a live connection: compile does no I/O.

---

## D8 — Change 0 also clears the pre-existing `INP001` debt

Change 0 adds `src/app/features/__init__.py`, clearing all 12 `INP001`. Ruff goes **24 → 3**, and the
only survivors are the 3 `PLC2701` that change `ingestion-chunking` owns via the second #240.

Caveat to carry into the change: the file was deleted in `8e25352`
(*"perf(imports): sever model-import to router-import coupling"* is `fd6f5ce`, an adjacent commit in
the same series). Restoring it must **not** reintroduce eager imports — the file stays empty. If a
proof shows import-time coupling returning, fall back to a scoped `per-file-ignores` for `INP001`.

## D9 — `agent_saul` is fixed in `graph-lifecycle`

`app.state.saul_graph` is never assigned, so `features/agent_saul/dependencies.py:37-45` raises
`ServiceUnavailableException` on every request; `get_saul_checkpointer` (`:48-53`) reads
`app.state.langgraph_checkpointer`, also never set. Todo 235 says *all* graphs in the lifespan, and
`saul_graph` is a graph that was never put there — in scope, and it turns a dead feature back on.

## D10 — Legal corpus is all four document families

Contracts/agreements, statutes/regulations, case law/judgments, **and** litigation filings.

This is the widest possible corpus and it forces specific things:

- **Heterogeneous structure** — clause numbering (contracts), `§` hierarchy (statutes), prose with
  pin-cites (case law), scanned/OCR-variable (filings). No single splitter serves all four; the
  strategy must dispatch on document kind. `features/documents/classification.py` already computes
  doc-kind and is the natural dispatch point.
- **Filings push docling harder** — heterogeneous scans mean OCR quality is a first-class concern, and
  the VLM pipeline becomes worth costing.
- **Statutes + amended contracts justify D12** (chunk versioning) on their own.
- `0017_scope_statute_identity_index` already exists, so statute identity is partly modelled.

## D11 — Token budget fix lands in `agentic-retrieval`

`features/documents/rag.py:79` budgets assembled context by `len(content.split())` — a **word** count.
The agentic loop is what needs an accurate budget to decide how much context to pull per hop, so it is
fixed where it is consumed. The tokenizer itself is chosen in `ingestion-chunking` (D14) and merely
used here.

## D12 — Chunk identity is versioned now: `(document_version, locus)`

`document_version` becomes part of chunk identity in the first migration of this cluster. Cheapest
decision to make now, most expensive to retrofit against populated tables. Retrieving a clause
superseded by Amendment No. 2, or a statute not in force on the date of breach, is a silent wrong
answer with professional-liability consequences — not a quality nit.

Chosen over a validity-range (`valid_from`/`valid_to`) table, which was the Postgres-native
alternative to Graphiti's bi-temporal model. Graphiti's temporal edges remain available on top.

## D13 — Evaluation harness is its own change, and it runs early

A golden-set + RAGAS-style harness ships as its own change, immediately after `tree-repair`.

Rationale (research §15): Uber's actual unlock in the EAg-RAG work was cutting evaluation from weeks
to minutes. **Every other recommendation in this cluster is a prior, not a measurement** — without an
eval you cannot tell which ones are wrong on this corpus. Building it after the retrieval work means
changes `ingestion-chunking` and `retrieval-sql` ship unmeasured.

Cost accepted: delays visible feature work, and needs SME time to build the golden set.

## D14 — Drop torch: hosted reranker + RapidOCR, keep `AutoTokenizer`

- **Drop** `sentence-transformers` (`pyproject.toml:51`) and torch — roughly 1–2 GB of image.
- **Replace** `shared/langgraph_layer/retrieval_kb/reranker.py`'s local `CrossEncoder` with a hosted
  reranker.
- **Move Docling's OCR off EasyOCR** to RapidOCR or Tesseract. This half is load-bearing: without it
  torch returns through the ingestion door and the drop buys nothing. The two decisions are coupled.
- **Keep** `transformers` / `AutoTokenizer` (`rag/docling/chunker.py:55,87`) — ~30 MB, pulls no torch,
  and `HybridChunker` needs a real tokenizer.
- **Drop** `langchain-docling>=2.0.0` (`pyproject.toml:43`) — declared, zero imports, and it flattens
  `DocMeta`, losing the bounding boxes pin-cites need. Calling `docling` directly is correct on the
  merits.

**Accepted consequences:** per-query reranker cost, a network dependency in the retrieval hot path,
and **late chunking is foreclosed** (it needs a self-hosted long-context encoder).

`tests/unit/shared/rag/test_chunker_tokenizer_cache.py` guards `AutoTokenizer` caching and must keep
passing.

## D15 — `knowledge-stack`: LangExtract wired, PageIndex as an idea, Graphiti stays

- **LangExtract — wire it.** It runs before chunking/embedding and feeds graphiti. Code exists at
  `shared/rag/langextract/` with no runtime call site; `LANGEXTRACT_API_KEY` already in
  `settings.py:24,252`.
- **PageIndex — implement the idea, drop the dependency.** Build reasoning-over-tree retrieval on the
  **stored `DoclingDocument` tree**, which is the same structure PageIndex constructs. Do not adopt
  the package. `shared/rag/pageindex/` is re-exported by `shared/rag/__init__.py:3` — that re-export
  and the commented construction at `lifespan.py:538` both need resolving.
- **Graphiti — stays.** Research recommended deferring it (second database, an LLM call per episode,
  no published ingestion cost figures), but that advice was uninformed about this repo: graphiti is
  **already built and wired** at `lifespan.py:205` with Neo4j in `app.state`, plus `setup_graphiti:90`,
  `setup_graphiti_indices:159`, `close_graphiti:179` and a full schema module. The second-database
  cost research warns about is **already sunk**.

## D16 — Postgres stack, settled by research

- **Tiger Cloud has `pgvector` and `pgvectorscale` enabled by default.** No extension decision.
- **`pg_textsearch` is Tiger Data's own first-party BM25 extension, available on Tiger Cloud.** It is
  **not** ParadeDB's `pg_search`, which is not on Tiger's extension list at all. Never conflate them.
- **Stay on `diskann`.** The repo already tunes `diskann.query_search_list_size` and
  `diskann.query_rescore` per query, which is the correct surface. `query_rescore` is *the* recall
  lever under `storage_layout='memory_optimized'`, because the graph walk runs over SBQ-compressed
  vectors. Note `diskann`'s native `labels` filtering and parallel index builds are **mutually
  exclusive** — pick one per index.
- **Do not adopt `PGVectorStore`.** It cannot express a `diskann` index (its index classes are
  `HNSWIndex`/`IVFFlatIndex`, both pgvector), cannot `SET LOCAL diskann.*` in its own query
  transaction, and its `hybrid_search_config` RRF is shaped for two-leg `tsvector` fusion, not a
  three-branch fusion over `<@>`. Adopting it is a **capability downgrade**. `PGVector` is formally
  deprecated as of v0.0.14. Keep LangChain's `Embeddings` interface and `Document`; wrap the existing
  repository query in a small `BaseRetriever` for LangGraph.
  **ADR expiry condition:** revisit if this project ever leaves Tiger Cloud for plain-pgvector managed
  Postgres — the capability gap closes there.
- **RRF: keep `k = 60`, add per-leg weights.** `k` only controls how sharply rank 1 beats rank 10;
  weights are the lever that matters. Cormack et al. (2009) used `k=60` untuned and never claimed it
  optimal. With **three** branches, unweighted RRF hands 2/3 of the influence to the text-derived
  legs — legal corpora argue for more lexical weight, but set it deliberately.
- **Build reranking in v1.** ~80–120 ms CPU for 20 candidates; context precision 0.71 → 0.79. It fixes
  a *different* failure than hybrid search (reranking barely moved recall, 0.83 → 0.84) — the two are
  not substitutes.
- **`<@>` returns NEGATIVE BM25 scores.** A `DESC` sort silently returns worst-first. Current code is
  correct (`repository.py:405` negates, filters `< 0`, sorts `ASC`) — this needs a **guard test**
  before someone "fixes" the sort direction.
- **Extensions must be created explicitly**, not relied on ambiently. `CREATE EXTENSION` appears only
  in migrations, never in `docker-compose*.yml` or `infra/`.

## D17 — Chunking, settled by research

- **`HybridChunker` as the engine, plus a clause-numbering post-pass.** Docling's layout model often
  misses contract clause numbers styled as inline bold runs rather than structural headings; regex the
  numbering and rebuild the tree. Budget for this rather than discovering it.
- **`merge_peers=True` is safe** — peer merging only combines chunks with the same headings &
  captions, so citation provenance survives.
- **Embed `contextualize(chunk)`, display and cite `chunk.text`, store both columns.** Docling's docs
  say `contextualize()` is "typically used to feed an embedding model". Embedding bare text makes a
  clause whose only topical signal is its heading ("Force Majeure") nearly unretrievable; displaying
  contextualized text pollutes every quote with heading boilerplate. ~15% storage overhead.
  **Verify `max_tokens` applies to the contextualized serialization, not the bare text** — if it does
  not, every contextualized chunk silently overflows.
- **Overlap 0 where clause boundaries are reliable**; 10–15% only on fallback splits that cut
  mid-clause, and flag those chunks as impure. Overlap across clause boundaries corrupts citation
  attribution — a correctness failure for legal work. Use parent-child retrieval and ±1 neighbour
  expansion instead; they do overlap's job without duplicating the index.
- **Ship `contextualize()` first; add LLM contextual prefixes later, selectively**, on the
  low-structure subset (case law, correspondence, exhibits) where headings are thin. Anthropic's
  technique reports ~35% failure-rate reduction but costs real money per chunk and is only economic
  with prompt caching.

---

## D18 — Evaluation is two-tier: a pure metric core now, a RAGAS layer later

`rag-eval-harness` ships **pure, dependency-free metrics** (recall@k, MRR, nDCG, context precision)
computed as functions over `(golden_set, retrieved_ids)`. No LLM, no network, no new dependency — so
the harness runs in CI on every change and its numbers are reproducible.

A **RAGAS-style layer lands later, as a separate concern**: faithfulness and answer-relevancy need a
judge model, which means cost, non-determinism, and an API key in CI. Splitting them means the
regression gate that other changes depend on cannot be blocked by a flaky judge. The generation-side
metrics still get specified — as a requirement the harness must be *extensible* to — so the later
layer is an addition, not a rewrite.

**Consequence for `ingestion-chunking` and `agentic-retrieval`:** both cite eval numbers as Proofs.
Those Proofs may only reference tier-1 metrics.

---

## D19 — The database is fully available for schema work

User ruling, verbatim: *"you can do anything in the DB as it has 0 data init currently and serving 0
users."*

This is **permissive, and it unblocks four things** the plans were routing around:

- `CREATE DATABASE` for a scratch instance, and `alembic upgrade head` against the live Tiger Cloud
  instance — so **O7 is settleable by direct measurement**, not inference from three disagreeing
  sources.
- `EXPLAIN (ANALYZE, BUFFERS)` capture against seeded data — so `retrieval-sql`'s index-reachability
  claims become measured, not argued. The CTE-materialisation defect in `legal_rrf_search` gets a
  plan, not a paragraph.
- Seeding a corpus for `rag-eval-harness`'s golden set.
- D12's chunk-identity migration needs no backfill design — the `search_text` generated column has
  zero rows to rewrite.

**It does not authorise destructive work against anything else.** The scope is this project's
database. The archived-tasks Proof rule that says "bring up a local scratch Postgres, never the
managed instance" is **superseded for this cluster** by this ruling.

---

## D20 — D17's display rule supersedes two prose sites, and they get rewritten

D17 stores contextualized text in `preamble` and bare text in `content`. Two existing prose sites
assert the opposite convention and would be left lying:

- `src/app/shared/langgraph_layer/checkpointer.py:11-17`
- `tests/unit/features/ingestion/test_unprovisioned_graph_fails_closed.py:3-8`

**Ruling: supersede — rewrite both.** Not "leave them, they're only comments." A stale docstring is
the exact failure mode this cluster keeps hitting (see the Corrections below: `graphiti/registry.py`
cost a planning round because its docstring named symbols that do not exist). Prose that contradicts
the schema is a defect with a file and a line number, so it gets a task and a Proof like any other.

---

## D21 — Saul's memory is exclusively cognee's, and it is already specified

User ruling: Saul memory is handled entirely by cognee, and the work was done in two archived
changes — `2026-09-07-cognee-agent-memory` and `2026-09-07-agent-tools-unification`.

Confirmed against source: `AgentMemoryService.__init__` (`agent_memory_service.py:136`) takes only
`partition_prefix` plus optional callables that default to `cognee.remember` / `.recall` /
`.improve`. **No client, no engine, no session.** The live spec
`openspec/specs/saul-memory-prefetch-and-retrieval/` already governs memory *behaviour*.

**Consequence: `graph-lifecycle` does not specify memory semantics at all.** It constructs
`AgentMemoryService` and passes it through the `configurable` seam (D7). Anything about *what*
memory does is out of scope and must not be re-specified — re-speccing shipped behaviour is the
failure D3 caught on todo 185.

---

## D22 — The spec tree was never synced; the seven changes attach to ten restored capabilities

**This is my assumption, not a user ruling.** It is recorded as an assumption because the
continuation instruction was to proceed without further questions, and this blocker had to be
resolved to name a single capability. It is the one decision in this ledger the user has not seen.

### The finding, in four measurements

1. **`openspec/specs/` holds 28 capabilities. The archives hold ~108.** Eighty exist only as deltas
   inside `openspec/changes/archive/`.
2. **The September 2026 archives were moved by hand, not by `openspec archive`.** Proof:
   `2026-09-07-ingestion-pipeline-unification/specs/typed-exception-handling/spec.md` adds four
   requirements ("Embedding failures SHALL raise a typed failure rather than substitute a placeholder
   value", the three retry-boundary ones). Live `typed-exception-handling` has thirteen requirements
   and **not one of the four is among them**. That capability is live only because of the earlier
   `2026-07-22-noqa-exception-handling-migration` archive. The delta form does not explain the gap —
   synced and unsynced capabilities use identical `## Purpose` + `## ADDED Requirements` shape.
3. **The work was declared complete.** Tick counts: cleanup-foundation 27/27, error-handling-foundation
   141/141, documents-unified-schema 27/27, ingestion-pipeline-unification 24/24,
   agent-tools-unification 45/46, cognee-agent-memory 45/46.
4. **But at least two archived requirements are knowingly unimplemented.**
   `src/app/shared/rag/docling/embedder.py:44-48` carries an explicit IOU — *"Reconciling the model,
   not just the width, belongs to B1, which collapses the four embedding paths into one"* — while
   `_PROVIDER_EMBEDDING_MODEL = "gemini-embedding-001"` diverges from configured
   `gemini-embedding-2-preview`. That is `unified-embedding`'s "Every embedding consumer resolves to
   the single path", specified and violated. Corroborating: `graphiti/registry.py`'s
   `AgentToolBundle` docstring says agents are "currently built with empty tool lists in factory.py",
   which is `agent-tool-registry`'s "Every agent role receives the tools assigned to it".

### The ruling

**Adopt the archived capability names, and restore exactly the ten the seven changes touch.**
`rag-tree-repair` gains a scoped spec-sync task group that copies these into `openspec/specs/`
verbatim from their archived deltas:

| Capability | Reqs | Scenarios | From |
|---|---|---|---|
| `celery-worker-deployment` | 8 | 19 | ingestion-pipeline-unification |
| `document-ingestion-pipeline` | 10 | 27 | ingestion-pipeline-unification |
| `graph-entity-canonicalisation` | 4 | 12 | ingestion-pipeline-unification |
| `hierarchical-document-chunking` | 7 | 16 | ingestion-pipeline-unification |
| `hybrid-retrieval-ranking` | 4 | 12 | ingestion-pipeline-unification |
| `langgraph-checkpointing` | 8 | 24 | ingestion-pipeline-unification |
| `unified-embedding` | 10 | 24 | ingestion-pipeline-unification |
| `document-retrieval-schema` | 10 | 27 | documents-unified-schema |
| `legal-corpus-retrieval` | 4 | 11 | agent-tools-unification |
| `agent-tool-registry` | 5 | 15 | agent-tools-unification |

All ten are grammar-conformant — every requirement carries at least one `#### Scenario:` — so the
restore is a file copy with the `## ADDED Requirements` header dropped, and `openspec validate
--strict` is the Proof.

**Not a bulk restore of all eighty.** That is a repo-hygiene project, not this cluster's work, and it
would put seventy capabilities into the baseline that no change here can defend.

**Two capabilities are deliberately excluded.** `graphiti-init-order` and `embedding-dimension-config`
(both `2026-06-22-quality-fixes-batch-2`, the latter also `2026-06-16-tech-debt-reliability`) predate
the requirement grammar entirely — they use `## Scope` / `## Problem` / `## Solution` / `## Verification`
and contain **zero `### Requirement:` blocks**. There is nothing to restore. Their content is
absorbed as ordinary source facts, not as a spec baseline.

### Why restore rather than coin new names

- **Mechanical.** A `## MODIFIED Requirements` block has no base to modify if the capability is
  absent from `openspec/specs/`. Without the restore the seven changes could only ever say `## ADDED`.
- **Honest.** The overlaps are *not* already-shipped (measurement 4), so the task groups do **not**
  shrink to verify-only. But MODIFIED-against-a-restored-base makes the regression legible;
  a fresh `## ADDED` of a near-duplicate name would hide it and fork the lineage.
- **Cheap to reverse.** If the user prefers new names, the restore is ten file copies to delete.

### Delta shape, per requirement

| Situation | Shape |
|---|---|
| Archived requirement already says what the change wants | **no delta** — tasks only, Proof re-verifies |
| The change tightens or narrows it | `## MODIFIED`, copying the **entire** existing requirement including every scenario |
| Genuinely new behaviour | `## ADDED` |

The MODIFIED rule is not stylistic: a MODIFIED block **replaces its requirement wholesale on
archive**, so an omitted scenario is silently deleted and `validate --strict` cannot detect it.

### Reconciling D22 with D5 ("one capability per change")

D5 ruled one capability per change. D22 puts ten restored capabilities in the baseline that several
changes touch. The tension is only apparent, and resolves as: **each change owns exactly one *new*
capability; restored capabilities are cited, and only amended when the change genuinely tightens
them.** Citing costs nothing and duplicates nothing; amending is the exception and must be justified
in that change's `design.md`.

| Change | New capability (owned) | Restored capabilities it touches | How |
|---|---|---|---|
| `rag-tree-repair` | `source-tree-integrity` | — (it *creates* the baseline) | restores all ten; amends none |
| `rag-eval-harness` | `retrieval-evaluation` | `hybrid-retrieval-ranking` | cite — it measures ranking, does not respecify it |
| `graph-lifecycle` | `compiled-graph-lifecycle` | `langgraph-checkpointing`, `celery-worker-deployment`, `document-ingestion-pipeline` | cite all three; the new capability holds only process-scoped compilation, the `configurable` seam, and Saul provisioning |
| `ingestion-chunking` | `legal-document-chunking` | `hierarchical-document-chunking`, `unified-embedding` | **MODIFY** "Every document kind is chunked structure-aware" (tighten to the four legal families + policy recorded per chunk); cite `unified-embedding` — its "Every embedding consumer resolves to the single path" already says what task group 6 does, so that group is a *repair of a violated requirement*, not a new one |
| `retrieval-sql` | `postgres-hybrid-retrieval` | `hybrid-retrieval-ranking`, `document-retrieval-schema`, `legal-corpus-retrieval` | **MODIFY** `legal-corpus-retrieval`'s "Ranked retrieval and fusion have a single implementation" (the `legal_rrf_search` CTE defeats it); cite the other two |
| `agentic-retrieval` | `agentic-retrieval-loop` | `agent-tool-registry`, `hybrid-retrieval-ranking` | cite; `agent-tool-registry`'s "Every agent role receives the tools assigned to it" is violated today (empty tool lists) and gets tasks, not a delta |
| `knowledge-stack` | `knowledge-extraction-stack` | `graph-entity-canonicalisation` | cite — canonicalisation is already fully specified; LangExtract feeds it |

**The pattern worth naming:** three of these changes exist because an archived requirement is
*specified and violated*, not because behaviour is missing. Those get task groups and Proofs but no
delta — writing an `## ADDED` for something already required would launder a regression as a feature.

---

## Corrections to earlier claims in this cluster

Recorded because each cost a planning round, and because a plan that repeats one will be wrong the
same way.

1. **"The repo is spec-greenfield for RAG" — false.** Eighty archived capabilities already specify
   large parts of this territory. Measurement, not assumption, settled it (D22).
2. **`shared/rag/graphiti/registry.py:11-34` is not a code recipe — it is a module docstring, and a
   stale one.** It names `build_tool_registry`; the real symbol is `build_tool_bundle` (`:92`). It
   names `app.state.saul_checkpointer`; the actual reader at `agent_saul/dependencies.py:49` uses
   `app.state.langgraph_checkpointer`. Planning off it produced a wiring step for a function that
   does not exist.
3. **`build_saul_graph`'s `checkpointer` parameter is not optional**, which forces the
   `graph-lifecycle` ordering: checkpointer → Saul → `configurable` seam → Celery → KB graph →
   globals. The plan cannot construct Saul before the checkpointer exists.

---

## Still open — routed, not dropped

- **O2 — phrase search.** Does `pg_textsearch` support phrase queries and boolean operators?
  `tsvector` has `phraseto_tsquery` and positional `<->`. For legal defined terms — *"reasonable
  efforts"*, *"Permitted Encumbrance"* — phrase search is a real capability. **Verify before
  concluding no `tsvector` is needed anywhere.** If phrase search is missing, a narrow `tsvector` may
  be justified **for that purpose only, never as a scoring leg in RRF** (double-counting the lexical
  signal). Note this does not reopen D3: no application code has `tsvector` today, and this would be
  an *addition* decision, not a removal one. → `retrieval-sql`
- **O4 — multi-tenant / jurisdiction isolation.** Partition (list or hash) with local indexes once
  there are more than a handful of distinct filter values; partial indexes for 3–5 stable categories.
  `pg_textsearch` supports partitioned tables, so both legs can follow the same scheme. The decisive
  reason is pgvector's own docs: a shared graph lets **one tenant's vectors degrade another tenant's
  recall** — a correctness issue, invisible until audited. `_FILTER_SQL`
  (`documents/repository.py:47-56`) already filters on `jurisdiction`, `contract_type`, `clause_type`.
  → `retrieval-sql`
- **O7 — live alembic revision unverified.** Disk carries `0001`–`0017`; the handover's claimed head
  `b3e7c41d92af` matches **no revision id on disk**; memory says the DB is stamped at `0004`. At least
  one of {handover, memory, disk} is wrong. Settling it needs `alembic current` against the live DB —
  and `alembic/env.py` is known to break migration commands. **Blocks D12's migration.** **D19
  unblocks the measurement**: the live instance is fully available, so this is a command to run, not
  a question to reason about. The `env.py` repair is in `rag-tree-repair`'s scope.
  → `rag-tree-repair` step 0, consumed by `ingestion-chunking` as a recorded value
- **O8 — `<@>` sign-convention guard test** does not exist yet. → `retrieval-sql`
- **O9 — does `HybridChunker`'s `max_tokens` measure `contextualize()` output or bare text?**
  → `ingestion-chunking`
- **O10 — `pg_textsearch` access-method name.** `decisions.md` F8: creating the extension is not the
  same as confirming it registers an AM literally named `bm25`, which `documents/model.py:114` and
  `repository.py:405` both depend on. Still unverified. → `retrieval-sql`

---

## Change set — final

Seven changes. Named, not numbered, because the numbers shifted during refinement.

| Change | Covers | Depends on |
|---|---|---|
| **`rag-tree-repair`** | the half-done rename + D8 + D22 spec-sync + O7 | — |
| **`rag-eval-harness`** | D13, D18 — golden set, tier-1 metrics, baseline capture | `rag-tree-repair` |
| **`graph-lifecycle`** | 235, D7, D9, D21 | `rag-tree-repair` |
| **`ingestion-chunking`** | 240, second #240, 176, 164, D10, D12, D14, D17 | `rag-tree-repair`, O7 |
| **`retrieval-sql`** | 163, 185, 162, D16, O2, O4, O8, O10 | `rag-tree-repair` |
| **`agentic-retrieval`** | 165, D11, reranking, the Uber EAg-RAG loop | `retrieval-sql`, `rag-eval-harness` |
| **`knowledge-stack`** | 195, D15 | `ingestion-chunking` |
