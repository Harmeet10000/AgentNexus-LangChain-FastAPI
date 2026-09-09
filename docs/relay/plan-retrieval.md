# Plan — RETRIEVAL group (`hybrid-retrieval-sql`, `agentic-retrieval-loop`)

Leg 2 of relay, 2026-09-09, `main` @ `7cca750`. Binds to `docs/relay/decisions-rag-cluster.md`
(authoritative) and `docs/relay/scout-rag-cluster.md`.

**Capability names, per D22.** Change `retrieval-sql` owns the new capability
**`postgres-hybrid-retrieval`** — deliberately *not* `hybrid-retrieval-sql`, because D22 restores an
archived capability called `hybrid-retrieval-ranking` and two near-identical names in one baseline is
how a lineage forks. This change **cites** `hybrid-retrieval-ranking` and `document-retrieval-schema`,
and carries one `## MODIFIED` against `legal-corpus-retrieval`'s *"Ranked retrieval and fusion have a
single implementation"* — which correction 1 below proves is violated today. Change `agentic-retrieval`
owns **`agentic-retrieval-loop`** and cites `agent-tool-registry`.

## Three ground-truth corrections that changed the plan

1. **There is a second, worse retrieval path.** `DocumentRepository.legal_rrf_search`
   (`src/app/features/documents/repository.py:596-728`) is what the retrieval graph actually calls
   (`retrieval_kb/nodes.py:237`). All three legs select `FROM candidate_chunks`, a CTE referenced
   **three times** (hence materialised), so `chunks_bm25_idx` and `chunks_embedding_idx` cannot be
   used; it sets no `SET LOCAL diskann.*`; its trigram CTE has `LIMIT 50` with no statement-level
   `ORDER BY` (arbitrary 50 rows, then ranked); it hard-codes `60.0` and a `0.15` trigram weight beside
   `constants.RRF_K`; and it honours a **narrower filter set** than `_FILTER_SQL` (no `chunk_kind`, no
   jsonb `@>`, no parties).
2. **`chunks.user_id` already exists** (`model.py:155`, `ix_chunks_user_document`). Every search leg
   still detours through `JOIN documents ... WHERE d.user_id`, which is what forces post-filtering
   around the ANN scan — **O4's mechanism is available today without a migration**.
3. **O2 is settled from the repo's own bundled skill**: `pg-textsearch-skill/SKILL.md:120` and
   `references/pg_textsearch.md:637-649` — **`pg_textsearch` has no phrase search**, prescribed remedy
   is BM25 over-fetch + `ILIKE` post-filter. The repo *already* does this (`repository.py:694`,
   `exact_phrase`) but unescaped and only on one path. **No `tsvector` addition is justified.** Same
   reference, `:672-674`: BM25 statistics on partitioned tables are **partition-local and not
   comparable across partitions** — that constrains O4.

---

## Change A — `hybrid-retrieval-sql`

**Why.** Retrieval exists as two divergent implementations of the same three-branch RRF: a Python-fused
path (`bm25_search:405`, `vector_search:452`, `trigram_search:507`, `fusion.py:28`,
`_fuse_search_branches:461`) and a monolithic weighted CTE (`legal_rrf_search:596`) reached from the
retrieval graph. The CTE path cannot use the `bm25` or `diskann` indexes it was written for, skips the
`diskann` query-time tuning the branch path sets, ranks trigram over an arbitrary 50 rows, and enforces
fewer filters — so the same query returns different results depending on which door it entered. This
change collapses to one leg-SQL per branch and one fusion, applies the pgvector/pg_textsearch skills to
each leg, and closes todo 185 as verify-only with a guard.

### Shape
Keep the three branch methods as the single source of truth for each leg's SQL (each over base
`chunks`, each with its own `ORDER BY … LIMIT` and tiebreaker, vector leg keeping `SET LOCAL
diskann.*`), delete the monolithic CTE, and route the graph's hybrid node through the same
`_fuse_search_branches`, with per-leg weights added to `reciprocal_rank_fusion` and named in
`constants.py`.

### Rejected
Make `legal_rrf_search` the single path and delete the Python fusion — lost because a single statement
fails all-or-nothing, whereas `_fuse_search_branches` already distinguishes "branch returned nothing"
from "branch raised" (`service.py:469-512`, guarded by
`tests/unit/documents/test_hybrid_search_failure.py`), and because a CTE-shaped query is the thing that
lost index access in the first place.

### Requirements

**Requirement: Retrieval SHALL expose exactly one fused search path**
- Scenario: WHEN the retrieval graph and the search endpoint issue the same query with the same
  filters, THEN both MUST return the same ranked chunk ids in the same order.
- Scenario: WHEN a retrieval branch raises, THEN the caller MUST receive a failure naming the branch,
  and MUST NOT receive results fused from the surviving branches.

**Requirement: Each retrieval leg SHALL be planned over its own index**
- Scenario: WHEN the query plan for the keyword leg is captured, THEN it MUST reference
  `chunks_bm25_idx`.
- Scenario: WHEN the query plan for the vector leg is captured, THEN it MUST reference
  `chunks_embedding_idx`.
- Scenario: WHEN a leg is evaluated, THEN it MUST carry its own `ORDER BY` and `LIMIT` in the same
  statement as its ranking expression.

**Requirement: BM25 scoring SHALL respect the inverted sign convention**
- Scenario: WHEN the keyword leg orders results, THEN it MUST sort the `<@>` expression ascending and
  MUST reject non-negative scores.
- Scenario: WHEN a change sorts a `<@>` expression descending, THEN the test suite MUST fail.

**Requirement: Fused ranking SHALL be deterministic and reproducible**
- Scenario: WHEN the same query runs twice over unchanged data, THEN the fused ordering MUST be
  identical, including among equally-scored rows.

**Requirement: Every retrieval branch SHALL honour the same filter surface**
- Scenario: WHEN a filter is supplied, THEN every branch MUST apply it, and no branch MAY silently
  ignore it.
- Scenario: WHEN a new filter predicate is added to the shared filter surface, THEN it MUST take effect
  on all branches without a per-branch edit.

**Requirement: Retrieval SHALL be tenant-scoped on the chunk relation**
- Scenario: WHEN a search runs for a user, THEN the tenant predicate MUST be applied on `chunks` and
  MUST NOT depend on a join to `documents` for correctness.
- Scenario: WHEN a user owns a small fraction of all chunks, THEN the vector leg MUST return the
  requested number of candidates or fewer only because none exist, not because the ANN candidate pool
  was exhausted by filtering.

**Requirement: Exact-phrase retrieval SHALL work without `tsvector`**
- Scenario: WHEN a query carries a quoted phrase, THEN results MUST contain that phrase literally.
- Scenario: WHEN a phrase contains `%` or `_`, THEN those characters MUST match literally and MUST NOT
  act as wildcards.

**Requirement: Application code SHALL NOT use `tsvector`**
- Scenario: WHEN application code under `src/app/` is scanned, THEN no `tsvector`, `tsquery`, or
  `to_tsvector` construct MUST appear.

**Requirement: Required extensions SHALL be declared, not assumed**
- Scenario: WHEN a fresh environment is provisioned, THEN `vector`, `vectorscale`, `pg_trgm` and
  `pg_textsearch` MUST be created by the schema chain before any index depending on them is built.
- Scenario: WHEN `pg_textsearch` is present, THEN an access method named `bm25` MUST be registered.

**Requirement: Fusion weights SHALL be explicit per leg**
- Scenario: WHEN three branches are fused, THEN each branch's weight MUST come from a named constant,
  and the fused score MUST change when a weight changes.

### Tasks

**## 1 Baseline (measure before touching anything)**
- [ ] 1.1 Capture gate baseline into `docs/relay/baseline-retrieval.md`. **Proof:**
  `uv run pytest tests/unit/documents tests/unit/shared/langgraph_layer -q`,
  `uv run ruff check --no-cache src/`, `uv run ty check src/` — record the three counts verbatim; every
  later proof compares to these, never to an absolute.
- [ ] 1.2 Create a scratch database on the live instance (`CREATE DATABASE retrieval_scratch;` —
  `tsdbadmin` holds `CREATEDB`, `decisions.md:345`); run
  `CREATE EXTENSION IF NOT EXISTS vector, vectorscale, pg_trgm, pg_textsearch`, then
  `SELECT name, installed_version FROM pg_available_extensions WHERE name IN (...)` and
  `SELECT amname FROM pg_am WHERE amname IN ('bm25','diskann')`. **Proof:** both `amname` rows present
  → O10 closed; record the version table. If `bm25` is absent, stop — `model.py:114` and
  `repository.py:418` are both wrong and the whole change re-scopes.
- [ ] 1.3 Build the scratch schema from the ORM, seed ~50k synthetic chunks across 20 users, and
  capture `EXPLAIN (ANALYZE, BUFFERS)` for `legal_rrf_search` and for each of the three branch methods
  into `docs/relay/baseline-retrieval.md`. **Proof:** the captured plans show whether
  `chunks_bm25_idx` / `chunks_embedding_idx` appear. Expect absent for `legal_rrf_search` (materialised
  CTE) and present for the branch methods; this file is the comparison target for tasks 3 and 4.
- [ ] 1.4 Run the three-branch path once against a real `AsyncSession` on the scratch DB. **Proof:** it
  either returns rows or raises `InterfaceError: another operation is in progress` — `service.py:490`
  gathers three `execute()` calls on one session. Record which. A raise means task 3.2 must serialise
  the branches and the fused path has never run in production.

**## 2 Guards (land before any behaviour moves)**
- [ ] 2.1 Add `tests/unit/documents/test_bm25_sign_convention.py`: static assertions over
  `repository.py` query text that the `<@>` expression is ordered `ASC` and filtered `< 0`, plus a
  fusion-level test that a more-negative raw score outranks a less-negative one. **Proof:**
  `uv run pytest tests/unit/documents/test_bm25_sign_convention.py -q` passes; flipping `ASC`→`DESC` in
  the source makes it fail. (O8)
- [ ] 2.2 Add `tests/unit/documents/test_no_tsvector_in_app_code.py` asserting no
  `tsvector|tsquery|to_tsvector|search_vector` under `src/app/` (migrations excluded and the exclusion
  documented as historical record, D3). **Proof:** the test passes now; adding `to_tsvector` to any
  `src/app/` file makes it fail. (D3 deliverable)
- [ ] 2.3 Add a test pinning the filter surface: every branch method's SQL contains the shared filter
  block, and every key in the built filter params is consumed by every branch. **Proof:**
  `uv run pytest tests/unit/documents -q` gains 1 test over the 1.1 baseline; removing one predicate
  from one branch fails it.

**## 3 Collapse to one fused path**
- [ ] 3.1 Add per-leg weights to `reciprocal_rank_fusion` as a keyword-only argument defaulting to
  unweighted, with `RRF_WEIGHT_VECTOR` / `RRF_WEIGHT_BM25` / `RRF_WEIGHT_TRIGRAM` in `constants.py`,
  weighted toward lexical per D16. **Proof:** `uv run pytest tests/unit/documents/test_fusion.py -q` —
  existing tests pass unchanged (default is unweighted, so `search_legal_precedents.py:188` is
  untouched), plus a new test showing the order changes when a weight changes.
- [ ] 3.2 Point `make_hybrid_retrieval_node` at the shared fused path instead of `legal_rrf_search`,
  mapping `QueryPlan.vector_weight`/`keyword_weight` onto the weight arguments and
  `exact_phrase`/`bm25_threshold` onto the branch inputs. **Proof:**
  `uv run pytest tests/unit/shared/langgraph_layer/test_retrieval_retry_shape.py -q` passes; a new test
  asserts the graph path and `DocumentQueryService.search` return identical chunk-id order for one
  query.
- [ ] 3.3 Delete `legal_rrf_search` and its stub in `src/app/examples/policy_examples.py:142`.
  **Proof:** `rg -n 'legal_rrf_search' src/ tests/` returns nothing;
  `uv run python -c "import app.main"` exits 0; `uv run pytest -q` count is ≥ the 1.1 baseline.

**## 4 Index and query tuning**
- [ ] 4.1 Give every leg a deterministic tiebreaker (`, c.id`) in both its window `ORDER BY` and its
  statement `ORDER BY`. **Proof:** the same query run twice on the scratch DB yields byte-identical
  chunk-id order.
- [ ] 4.2 Move `DISKANN_QUERY_SEARCH_LIST_SIZE` / `DISKANN_QUERY_RESCORE` onto the fused path so every
  vector leg sets them in its own transaction. **Proof:** `EXPLAIN (ANALYZE)` on the scratch DB with
  `query_rescore = 0` vs the configured value shows different recall against an exact-KNN ground truth
  computed once; record both against 1.3.
- [ ] 4.3 Re-capture the `EXPLAIN` set from 1.3. **Proof:** every leg's plan now names its index where
  the 1.3 capture did not.

**## 5 Tenant isolation (O4)**
- [ ] 5.1 Move the tenant predicate onto `chunks.user_id` in every leg, keeping the `documents` join
  only where a document column is projected. **Proof:** `EXPLAIN` shows the vector leg filtering before
  the ANN scan or using `ix_chunks_user_document`; recall for a user owning ~1% of chunks, measured
  against exact KNN, improves over the 1.3 capture.
- [ ] 5.2 Write `openspec/changes/<change>/adrs.md` recording the isolation ladder decision — partial
  indexes for the 3–5 stable `jurisdiction`/`document_kind` values, `diskann` `labels` **or** parallel
  builds (mutually exclusive) for many tenants, list partitioning last — and record that
  `pg_textsearch` statistics are **partition-local**, so a partitioned BM25 leg produces scores that are
  not comparable across partitions. **Proof:** the ADR names a measured trigger threshold from the 5.1
  recall numbers, not a guess.
- [ ] 5.3 Add the chosen partial index(es) in a migration that also carries
  `CREATE EXTENSION IF NOT EXISTS` for all four extensions, and register them in
  `model.py.__table_args__`. **Proof:** `uv run alembic check` proposes no diff; a fresh scratch
  database built from `alembic upgrade head` reaches head with no pre-installed extensions.

**## 6 Phrase search (O2, settled: no `tsvector`)**
- [ ] 6.1 Move the phrase post-filter into the keyword leg as over-fetch + escaped `ILIKE` (escape `%`,
  `_`, `\` in `exact_phrase`), per `pg_textsearch.md:637-649`, and expose it on the shared branch input
  so both callers get it. **Proof:** `uv run pytest tests/unit/documents -q` gains a test where a phrase
  containing `%` matches literally; a chunk containing the words separately but not the phrase is
  excluded.
- [ ] 6.2 Record in `design.md` that `pg_textsearch` supports neither phrase nor boolean query syntax,
  that the over-fetch + post-filter is the vendor-prescribed remedy, and that a `tsvector` column is
  therefore **not** added — it would double-count the lexical signal in a three-branch RRF. **Proof:**
  the citation `pg-textsearch-skill/references/pg_textsearch.md:637` appears in the design; the 2.2
  guard test keeps it true.

**## 7 Close-out**
- [ ] 7.1 Full gate sweep. **Proof:** `uv run ruff check --no-cache src/`, `uv run ty check src/`,
  `uv run pytest -q` — each equal or better than the 1.1 baseline.

### Dependencies and seams
Needs `rag-tree-repair` merged (the tree does not import today) and its post-repair `pytest` baseline.
**Do not touch:** the chunk-identity migration (`ingestion-chunking`, D12 — this change adds only
index/extension DDL, and delivers the *shared filter surface* so the version predicate lands in one
place later), the golden set (`rag-eval-harness`), lifespan compilation (`graph-lifecycle`),
`shared/rag/**` (`ingestion-chunking`, `knowledge-stack`), and the reranker (change B).

---

## Change B — `agentic-retrieval-loop`

**Why.** `retrieval_kb/graph.py:28-73` already ships more of EAg-RAG than Uber's article does: a query
analyzer that rewrites, decomposes, routes and sets per-leg weights; hybrid retrieval; a reranker funnel
at 20→5; a context grader with a **cyclic** edge back to the analyzer, capped at two iterations. Uber's
shipped pipeline is explicitly *acyclic* (research §A.1). What is genuinely missing is Uber's
highest-value idea — a source identifier that narrows the corpus **before** retrieval — a
post-processing step that dedupes and restores document order, an honest token budget, and a reranker
that survives D14's removal of torch.

### Shape
Add two nodes to the existing graph (source identifier before retrieval, post-processor between rerank
and generate), reuse `assemble_rag_context` as the post-processor, replace the word-count budget with
the tokenizer `ingestion-chunking` selects, and put the reranker behind a protocol with a hosted
implementation.

### Rejected
Build a fresh EAg-RAG graph beside `retrieval_kb/` — lost because it would duplicate a query planner,
grader and retry loop that already exist and are tested (`test_retrieval_retry_shape.py`), and would
leave two retrieval graphs for `graph-lifecycle` to hoist.

### EAg-RAG: portable vs Uber-specific

| Uber component | Verdict here |
|---|---|
| Query Optimizer (rewrite + decompose) | **Already built** — `make_query_analyzer_node`, `nodes.py:94`. No work. |
| Source Identifier (document-title allowlist) | **Portable, and the one real gap.** Legal analogue: jurisdiction / document-kind / matter router populating `doc_ids_filter`. Also the fix for the ANN recall cliff — a narrow allowlist turns filtered-ANN into small exact search. |
| Post-processing: dedupe + restore document order | **Portable, and missing on the graph path.** `assemble_rag_context` (`rag.py:37`) already groups by document and restores chunk order; the graph never calls it. |
| BM25 over LLM-generated metadata (summaries/FAQs/keywords) | **Portable but NOT this change.** It needs ingest-time enrichment feeding `search_text` (a generated column, `model.py:175`) — `ingestion-chunking`/`knowledge-stack` own that. Named as a seam only. |
| Google Docs migration, custom Docs API loader | **Uber-specific.** Corpus is PDFs via docling. |
| Offline feature store; Michelangelo; Langfx; Slack surface | **Uber-specific.** `app.state` + Redis already fill the artifact-cache role. |
| LLM-as-a-Judge harness, SME golden set | **Portable — owned by `rag-eval-harness`.** Not planned here. |
| "Agentic loop" | **Already exceeded.** Uber ships strictly sequential; this repo already cycles grader → analyzer. Do not rebuild. |

### Requirements

**Requirement: Retrieval SHALL narrow the corpus before searching it**
- Scenario: WHEN a query names a jurisdiction, document kind or matter, THEN the retrieval planner MUST
  produce a document allowlist and the search MUST be constrained to it.
- Scenario: WHEN the planner cannot narrow the corpus, THEN retrieval MUST proceed unconstrained rather
  than returning nothing.
- Scenario: WHEN the allowlist excludes every document containing the answer, THEN the grader MUST mark
  the context insufficient and the loop MUST retry with a widened allowlist before generating.

**Requirement: Assembled context SHALL be deduplicated and read in document order**
- Scenario: WHEN reranked chunks from one document reach generation, THEN they MUST appear in ascending
  in-document order.
- Scenario: WHEN the same chunk is returned by more than one branch, THEN it MUST appear once in the
  assembled context.

**Requirement: The context budget SHALL be measured in tokens**
- Scenario: WHEN context is assembled against a budget, THEN the accounting MUST use the
  embedding/generation tokenizer's count, not a whitespace word count.
- Scenario: WHEN assembled context would exceed the budget, THEN whole sections MUST be dropped from
  the tail and no section MUST be truncated mid-chunk.

**Requirement: Reranking SHALL run in the retrieval path and SHALL NOT require a local model**
- Scenario: WHEN candidates are retrieved, THEN a reranker MUST reorder them before generation and MUST
  pass fewer candidates onward than it received.
- Scenario: WHEN the reranking provider is unavailable, THEN retrieval MUST degrade to the fused order
  and MUST NOT fail the request.
- Scenario: WHEN the runtime environment has no local deep-learning framework installed, THEN reranking
  MUST still function.

**Requirement: The retrieval loop SHALL terminate**
- Scenario: WHEN the grader repeatedly reports insufficient context, THEN the loop MUST stop at its
  iteration cap and return the grounded fallback.

### Tasks

**## 1 Baseline**
- [ ] 1.1 Capture `uv run pytest tests/unit/shared/langgraph_layer -q` counts and record which tests
  cover `reranker`, `context_grader`, `generator` into `docs/relay/baseline-agentic.md`. **Proof:** the
  file lists `test_reranker_singleton.py` and `test_retrieval_retry_shape.py` with their current
  outcomes.
- [ ] 1.2 Record the current node/edge shape of `build_retrieval_graph` as a test fixture (node names
  and edge pairs). **Proof:** a new test asserting the exact node set passes today; it is the diff
  target for every later graph edit.

**## 2 Reranker seam (unblocks D14)**
- [ ] 2.1 Define a `Reranker` protocol matching the call already made at `nodes.py:263`
  (`rerank(query, chunks, limit)`), and make `make_reranker_node` accept it. **Proof:**
  `uv run ty check src/` clean;
  `uv run pytest tests/unit/shared/langgraph_layer/test_reranker_singleton.py -q` passes with the
  existing `CrossEncoderReranker` satisfying the protocol structurally.
- [ ] 2.2 Add a hosted reranker implementation behind the protocol, selected by settings, with the local
  one still available. **Proof:** a test injecting a stub hosted client asserts 20 candidates in →
  `limit` out, in the provider's returned order.
- [ ] 2.3 Add a degradation test: provider raises → node returns the fused order truncated to `limit`,
  no exception escapes. **Proof:** `uv run pytest tests/unit/shared/langgraph_layer -q` gains 1 test
  over 1.1.
- [ ] 2.4 Make the hosted implementation the configured default and note in `design.md` that deleting
  `sentence-transformers`, `torch` and `retrieval_kb/reranker.py` is `ingestion-chunking`'s task under
  D14, unblocked by 2.1-2.3. **Proof:** `rg -n 'sentence_transformers' src/app/shared/langgraph_layer/`
  still returns only `reranker.py` — this change removes nothing.

**## 3 Source identifier**
- [ ] 3.1 Extend `QueryPlan` with an allowlist field and add a `source_identifier` node between
  `query_analyzer` and retrieval, populating `doc_ids_filter` from cheap metadata (jurisdiction /
  document_kind / matter) with few-shot examples. **Proof:** the 1.2 graph-shape test is updated in the
  same commit and shows exactly one new node and two new edges; a test with a stub LLM asserts the
  allowlist reaches the branch filter params.
- [ ] 3.2 Widen the allowlist on retry: when the grader reports insufficient context, the next iteration
  MUST relax the allowlist before re-running. **Proof:** a test drives grader→insufficient once and
  asserts the second retrieval call carries a strictly larger (or empty) allowlist.

**## 4 Post-processing and budget (D11)**
- [ ] 4.1 Insert a `post_process` node between `reranker` and `context_grader` that dedupes by chunk id
  and calls `assemble_rag_context`. **Proof:** a test with two branches returning an overlapping chunk
  asserts it appears once and that chunks of one document are in ascending `chunk_index`.
- [ ] 4.2 Replace `len(content.split())` at `rag.py:79,87` with the tokenizer counter
  `ingestion-chunking` selects, injected rather than imported. **Proof:**
  `uv run pytest tests/unit/documents/test_rag.py -q` passes; a new test feeds text whose token count
  differs from its word count by >20% and asserts the section is dropped at the token boundary, not the
  word one.
- [ ] 4.3 Note in `design.md` that the tokenizer *choice* is D14/`ingestion-chunking`'s and this change
  only consumes it through an injected callable. **Proof:**
  `rg -n 'AutoTokenizer|tiktoken' src/app/features/documents/` returns nothing.

**## 5 Close-out**
- [ ] 5.1 Full gate sweep. **Proof:** `uv run ruff check --no-cache src/`, `uv run ty check src/`,
  `uv run pytest -q` each equal or better than the 1.1 baseline; the graph-shape test names all eight
  nodes.

### Dependencies and seams
Needs `hybrid-retrieval-sql` merged (task 3.2 there removes `legal_rrf_search`, which this change's
hybrid node calls) and `rag-eval-harness` for the quality numbers that justify the weights and the
allowlist. **Do not touch:** any SQL in `documents/repository.py`, the RRF weights themselves,
`sentence-transformers`/`torch` removal, the tokenizer choice, lifespan compilation of the graph, or
`shared/rag/**`.

---

## Blast radius to re-verify (both changes)

`tests/unit/documents/{test_hybrid_search_failure,test_fusion,test_rag,test_chunking,test_vector_width_configured}.py`;
`tests/unit/shared/langgraph_layer/{test_retrieval_retry_shape,test_reranker_singleton}.py`;
`tests/unit/test_feature_error_exhaustiveness.py`. Non-test call sites:
`src/app/examples/policy_examples.py:122-160` (stubs `bm25_search`, `vector_search`, `trigram_search`,
`exact_phrase_search`, `legal_rrf_search` and asserts branch names — it is executable and will break on
any `RETRIEVAL_BRANCHES` edit); `src/app/shared/langchain_layer/agents/tools/search_legal_precedents.py:188`
(a **second** consumer of `reciprocal_rank_fusion`, `RRF_K` and `HYBRID_CANDIDATE_LIMIT` — the weights
argument must default to unweighted or this call site changes too);
`documents/{router,dto,dependencies}.py`; `retrieval_kb/{graph,nodes,state}.py`.

## Risks

- **Concurrent `execute()` on one `AsyncSession`** — `service.py:490` gathers three branch coroutines
  against `self.repo`'s single session. If that raises against a real connection, the fused path has
  never run in production and change A's consolidation surfaces it. Earliest visible: task A-1.4.
- **The materialised-CTE claim is planner-dependent.** `candidate_chunks` is referenced three times, so
  Postgres will not inline it — but this is asserted, not measured. Earliest visible: task A-1.3's
  `EXPLAIN` capture; if the indexes *do* appear, tasks 3-4 shrink to the tuning and determinism fixes
  and the ordering is unaffected.
- **`bm25` access method may not exist under that name** (O10) — `model.py:114` and every query embed
  the literal. Earliest visible: task A-1.2; a negative result re-scopes the whole change.
- **Partitioning would break the lexical leg**: partition-local BM25 statistics make scores incomparable
  across partitions, so a cross-partition `ORDER BY <@> LIMIT` is mis-ordered. Earliest visible: task
  A-5.2, which is why the ADR must reach partitioning last and prefer partial indexes or `diskann`
  labels.
- **Two changes both migrate `chunks`** — A-5.3 (indexes/extensions) and `ingestion-chunking`'s D12
  (identity columns). Alembic will produce a branch if both are authored from the same head. Earliest
  visible: `uv run alembic check` in A-5.3; mitigation is that A adds no columns, so a rebase is
  mechanical.
