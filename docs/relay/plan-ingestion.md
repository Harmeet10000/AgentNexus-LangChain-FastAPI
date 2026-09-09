# Plan — INGESTION group (`ingestion-chunking`, `knowledge-stack`)

Leg 2 of relay, 2026-09-09, `main` @ `7cca750`. Binds to `docs/relay/decisions-rag-cluster.md`
(authoritative) and `docs/relay/scout-rag-cluster.md`.

**Capability names, per D22.** `ingestion-chunking` owns **`legal-document-chunking`** and carries one
`## MODIFIED` against restored `hierarchical-document-chunking`'s *"Every document kind is chunked
structure-aware"*, tightening it to the four legal families with the resolved policy recorded on each
chunk. It **cites** restored `unified-embedding`: its *"Every embedding consumer resolves to the single
path"* already requires what task group 6 does, so group 6 is a **repair of a violated requirement**,
not a new one — `src/app/shared/rag/docling/embedder.py:44-48` says so in an in-code IOU. No `## ADDED`
may be written for it. `knowledge-stack` owns **`knowledge-extraction-stack`** and cites
`graph-entity-canonicalisation`, which already fully specifies idempotent canonical graph writes.

## Two findings from source that changed the plan

- **`UnifiedChunk` (`src/app/features/documents/model.py:158-184`) already has the two-column shape
  D17 asks for**: `content`, `preamble`, and a generated
  `search_text = clause_type || preamble || content`. `retrieval_kb/reranker.py:56` already reranks
  `f"{chunk.preamble}\n\n{chunk.chunk_text}"`. **D17 needs no new text column** — it needs
  contextualization moved *out* of `content` and *into* `preamble`, and because `search_text`
  concatenates both, BM25/trigram input is unchanged by the move.
- **`src/app/shared/rag/docling/chunker.py:250-254` currently writes `content=contextualized_text.strip()`**
  — so citations quote heading boilerplate today. That is the exact D17 violation, live. And `:251`
  computes the token count *after* `contextualize()`, which turns **O9 into a runnable probe** rather
  than a research question.

---

## Change 1 — `ingestion-chunking` · capability `legal-document-chunking`

**Why.** Three unreconciled chunkers coexist and the one legal-grade path stores contextualized text in
the citation column. Chunk identity has no version, so a superseded clause is retrievable as current.
This change makes chunking dispatch on document kind, versions chunk identity, splits embed-text from
cite-text onto the columns that already exist, collapses ingestion onto the `langchain_layer` embedder,
and removes torch.

### Requirements

1. **Chunk identity is version-scoped** — WHEN a document is re-ingested at a new version, THEN chunks
   from prior versions MUST remain addressable and retrieval MUST scope to a single `document_version`.
2. **Chunking dispatches on document kind** — WHEN a document is classified as contract, statute,
   judgment, or filing, THEN the resolved chunk policy MUST match that kind and MUST be recorded on
   every chunk.
3. **Clause numbering survives layout misses** — WHEN clause numbers appear as inline bold runs rather
   than headings, THEN the post-pass MUST recover them into chunk locus.
4. **Embedded text and cited text are distinct** — WHEN a chunk is stored, THEN `content` MUST hold
   bare chunk text and `preamble` MUST hold the contextualization; embeddings MUST be computed over the
   concatenation.
5. **Fallback splits are declared impure** — WHEN a split cuts mid-clause, THEN the chunk MUST carry an
   impurity flag and MUST be the only path permitted non-zero overlap.
6. **One embedder serves ingestion** — WHEN ingestion embeds, THEN it MUST use
   `shared/langchain_layer/embeddings.py` and MUST NOT hardcode a model id.
7. **No torch in the runtime image** — WHEN dependencies resolve, THEN `sentence-transformers`, torch,
   and `langchain-docling` MUST be absent and Docling OCR MUST NOT use EasyOCR.
8. **No private cross-module imports** — WHEN a module needs chat-model construction, THEN it MUST use
   a public factory.

### Ordered tasks

**## 1 Baseline and preconditions**
- [ ] 1.1 Capture `uv lock --check`, `uv run ruff check --no-cache src/ 2>&1 | tail -1`,
  `uv run pytest -q 2>&1 | tail -1`, `uv run python -c "import app.main"` into the change's
  `baseline.md`. **Proof:** file exists, import exits 0.
- [ ] 1.2 Record the O7 resolution (`alembic current` output produced by `rag-tree-repair`) verbatim;
  **do not guess a head**. **Proof:** recorded value equals a `revision` literal present under
  `src/alembic/versions/`.

**## 2 Chunk identity (D12)**
- [ ] 2.1 Add `document_version: int` and `locus: str | None` to `UnifiedChunk`; replace
  `uq_chunks_document_chunk_index` with `(document_id, document_version, chunk_index)`.
  **Proof:** `uv run ty check src/` ≤ 1.1 baseline.
- [ ] 2.2 Migration with `down_revision` set from 1.2. **Proof:** `uv run alembic check` reports no
  pending autogenerate diff.
- [ ] 2.3 Write path populates both. **Proof:** new test asserting two versions of one document coexist.

**## 3 Kind dispatch (D10)**
- [ ] 3.1 Pure `resolve_chunk_policy(document_kind) -> ChunkPolicy` in
  `features/documents/chunking.py`, no I/O. **Proof:** unit test, four kinds, four distinct policies.
- [ ] 3.2 Call it from the docling path. **Proof:** chunk metadata carries the policy name.

**## 4 Clause post-pass (D17)**
- [ ] 4.1 Pure `recover_clause_numbering(chunks)` reusing `_CLAUSE_START_RE` (`classification.py:124`).
  **Proof:** unit test on inline-bold-numbered fixture recovers N loci.
- [ ] 4.2 Wire after `_hybrid_chunk_documents`. **Proof:** fixture ingestion populates `locus`.

**## 5 Contextualize split + O9**
- [ ] 5.1 **Settle O9 first**: test asserting every `HybridChunker` chunk's
  `len(tokenizer.encode(contextualize(chunk)))` ≤ `config.max_tokens`. **Proof:** test passes, or fails
  and pins the answer as a recorded xfail feeding 5.2.
- [ ] 5.2 Change `chunker.py:254` to `content=chunk.text`, `preamble=contextualized`; if 5.1 failed,
  budget `max_tokens` against the contextualized length. **Proof:** test asserting `preamble != ""` and
  `content` contains no heading prefix.
- [ ] 5.3 Flag fallback chunks impure and set overlap 0 on non-fallback policies.
  **Proof:** `_simple_fallback_chunk` output carries the flag; hybrid output has zero overlap.

**## 6 Embedder collapse**
- [ ] 6.1 Repoint `docling/ingest_v2.py:18` to `embed_texts`; delete `docling/embedder.py`'s
  `_PROVIDER_EMBEDDING_MODEL`. **Proof:** `rg -c 'gemini-embedding-001' src/` returns 0.
- [ ] 6.2 Update `tests/unit/shared/rag/test_embedder_no_substitution.py` and
  `test_rag_agent_embedder_import.py`. **Proof:** pytest ≥ 1.1 baseline pass count.

**## 7 Dependency drop (D14)**
- [ ] 7.1 Swap `docling_enhanced.py:63-71` `PdfPipelineOptions` to RapidOCR; audit bare
  `DocumentConverter()` at `ingest_v2.py:127` and `ingestion_kb/nodes.py:486` (they inherit the EasyOCR
  default). **Proof:** `rg -n 'DocumentConverter\(\)' src/` returns 0.
- [ ] 7.2 Drop `sentence-transformers` (`pyproject.toml:51`) and `langchain-docling` (`:43`).
  **Depends on `agentic-retrieval` having replaced `retrieval_kb/reranker.py:9`** — that is the seam;
  do not touch the reranker. **Proof:** `uv lock && uv run python -c "import torch"` exits non-zero.
- [ ] 7.3 `test_chunker_tokenizer_cache.py` still passes. **Proof:** named test green.

**## 8 Layering (164, second #240)**
- [ ] 8.1 Public `build_chat_model` wrapper; repoint `documents/dependencies.py:13`,
  `documents/service.py:19,26`, `crawler/processor.py:14`. **Proof:** `PLC2701` count 3 → 0 against 1.1.
- [ ] 8.2 Retire `features/documents/chunking.py::chunk_text`'s 5 callers onto the docling path; delete
  the whitespace splitter and `INGEST_CHUNK_SIZE/OVERLAP`. **Proof:** `rg -n 'INGEST_CHUNK_SIZE' src/`
  returns 0, `tests/unit/documents/test_chunking.py` retargeted.

**Rejected shape.** Dependency drop first (mechanical, gates image size), chunking after. Lost because
the OCR swap's only honest proof is an ingestion run, and ingestion output is not trustworthy until
5.2 lands.

---

## Change 2 — `knowledge-stack` · capability `knowledge-extraction-stack`

**Why.** LangExtract and PageIndex are complete modules with zero runtime call sites; graphiti is built
and wired. This change runs LangExtract as a pre-chunking extraction stage feeding graphiti episodes,
and re-implements PageIndex's reasoning-over-tree retrieval against the stored `DoclingDocument` tree so
the package dependency can go.

### Requirements

1. **Extraction precedes chunking** — WHEN a document is ingested, THEN LangExtract MUST run before
   chunking and its output MUST be available to the chunk writer.
2. **Extractions become graph episodes** — WHEN extraction yields clause entities, THEN they MUST be
   written as graphiti episodes idempotently under re-ingestion.
3. **Extraction failure is non-fatal** — WHEN the LangExtract provider is unavailable, THEN ingestion
   MUST complete and the document MUST be flagged as extraction-incomplete.
4. **Tree reasoning uses the stored tree** — WHEN a query needs structural navigation, THEN retrieval
   MUST reason over the persisted document tree and MUST NOT call an external PageIndex service.
5. **The PageIndex package surface is retired** — WHEN the package is imported, THEN no PageIndex
   client symbol is exported.

### Ordered tasks

**## 1 Baseline and the tree question**
- [ ] 1.1 Baselines as above.
- [ ] 1.2 **Determine whether the `DoclingDocument` tree is persisted at all** — not established, and
  will not be assumed. **Proof:** `rg -n 'DoclingDocument' src/app/features/documents/` plus a
  `UnifiedDocument.metadata_` inspection; record yes/no. If no, 2.x becomes a persistence task first.

**## 2 Persist the tree** (conditional on 1.2)
- [ ] 2.1 Store the serialized tree on the document row. **Proof:** round-trip test reconstructs the
  tree from a stored document.

**## 3 Retire the PageIndex surface**
- [ ] 3.1 Remove the re-export at `shared/rag/__init__.py:3` and delete the commented construction at
  `lifespan.py:538`. **Proof:**
  `uv run python -c "import app.shared.rag as r; assert not [n for n in dir(r) if 'ageindex' in n.lower()]"`
  exits 0.
- [ ] 3.2 Drop `shared/rag/pageindex/`. **Proof:** `uv run python -c "import app.main"` exits 0, ruff ≤
  baseline.

**## 4 Tree-reasoning retrieval**
- [ ] 4.1 Pure navigator over the stored tree returning node paths. **Proof:** unit test on a fixture
  tree returns the expected section path.
- [ ] 4.2 Expose it as a retrieval branch through the repository, not the router. **Proof:**
  service-level test; no new import of `repository` from `router.py`.

**## 5 LangExtract stage**
- [ ] 5.1 Async extraction service reading `LANGEXTRACT_API_KEY`, client in `app.state` per the lifespan
  convention. **Proof:** `rg -n 'langextract' src/app/lifecycle/lifespan.py` hits; ruff `ASYNC` clean.
- [ ] 5.2 Insert before chunking with `Result`-typed failure. **Proof:** test injecting a failing
  provider — ingestion succeeds, flag set.
- [ ] 5.3 Feed `graphiti/write_clause_episodes.py` (its `TYPE_CHECKING` import at `:42` becomes real).
  **Proof:** re-ingestion test creates no duplicate episodes.

**Rejected shape.** Uncomment `lifespan.py:538` and wire the existing PageIndex client first, deferring
the tree work. Lost to D15 — that adds the external dependency the decision removes.

---

## Dependencies and seams

- Both changes are blocked by **`rag-tree-repair`** (tree does not import today). `knowledge-stack` is
  additionally blocked by `ingestion-chunking`.
- Task 2.2's `down_revision` is blocked by **O7**, consumed as a recorded value from `rag-tree-repair`,
  never guessed.
- **Seam with `agentic-retrieval`**: this group owns `pyproject.toml:51` removal and the OCR swap; they
  own replacing `retrieval_kb/reranker.py:9`'s `CrossEncoder`. Task 7.2 cannot land before theirs. This
  group touches neither `reranker.py` nor `features/documents/rag.py:79` (D11 is theirs).
- **Not mine**: `retrieval-sql` owns `repository.py` search SQL, `fusion.py`, and O2/O4/O8/O10. Task 8.2
  edits `chunking.py` write-side only.
- Change 1 writes `chunks`, never `clauses` (`decisions.md:257-264`).

## Blast radius to re-verify

`tests/unit/shared/rag/test_chunker_tokenizer_cache.py` (D14 guard), `test_embedder_no_substitution.py`,
`test_rag_agent_embedder_import.py`, `tests/unit/documents/test_chunking.py`,
`tests/unit/test_feature_error_exhaustiveness.py`. Call sites: `docling/ingest_v2.py` (5
`chunk_document` callers), `shared/rag/docling/__init__.py`, `ingestion_kb/nodes.py:486,721`,
`features/documents/{parser,service,dependencies}.py`, `retrieval_kb/nodes.py:405`,
`crawler/processor.py:14`. Twelve pre-existing websocket fixture-drift failures are owned by no change
here.

## Risks

- **O9 comes back "bare text"** — every contextualized chunk overflows the embedder window. Shows at
  task 5.1, before any storage change, which is why 5.1 precedes 5.2.
- **Dropping EasyOCR degrades filing OCR quality** (D10's hardest family). Shows at 7.1; no eval harness
  exists yet to quantify it, so `rag-eval-harness` landing first would materially de-risk this task.
- **`search_text` is a `Computed(persisted=True)` column** — moving text between `content` and
  `preamble` rewrites every generated value. On an empty table this is free; if O7 reveals populated
  chunks, 5.2 needs a backfill. Shows at 1.2.
- **Task 1.2 in `knowledge-stack` may reveal the tree is never persisted**, converting change 2 from
  wiring into a storage build. Deliberately the first task.
