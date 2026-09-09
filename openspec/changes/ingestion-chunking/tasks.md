# Tasks — ingestion-chunking

## How to read the Proofs

1. **Never use a test-process exit code as a Proof of test outcome.** The coverage floor makes a green
   suite exit non-zero. Compare **summary pass and failure counts** against
   `docs/relay/baseline-pytest.txt`, re-measured immediately before the task.
2. **Never prove a schema fact by rendering migrations offline.** Migration tooling is repaired by
   `rag-tree-repair`; prove schema facts against the live database.
3. **Never make a Proof depend on a durable outbound event firing.** The outbox tables do not exist.
4. **The live database may be used** — zero data, zero users, ruled available.
5. **No Proof in this change may cite an answer-quality metric.** `rag-eval-harness` ships tier-1
   retrieval metrics only; faithfulness and answer relevancy do not exist yet.

**Blocked by `rag-tree-repair`** (the tree does not import today) and **`rag-eval-harness`** (task 7.1
degrades OCR quality on the hardest document family, and a before-number is the only way to see it).
**Task 7.2 is additionally blocked by `agentic-retrieval`**, which owns replacing the cross-encoder
reranker.

## 1 Baseline and preconditions

- [ ] 1.1 Capture `uv lock --check`, `uv run ruff check --no-cache src/ 2>&1 | tail -1`,
  `uv run pytest -q 2>&1 | tail -1`, and `uv run python -c "import app.main"` into the change's
  `baseline.md`.
  **Proof:** the file exists and the import exits `0`.
- [ ] 1.2 Record the migration head verbatim from the `alembic current` output that `rag-tree-repair`
  produced. **Do not guess a head.**
  **Proof:** the recorded value equals a `revision` literal present under `src/alembic/versions/`.
- [ ] 1.3 Record whether the `chunks` table is populated.
  **Proof:** a row count against the live database, recorded in `baseline.md`. A non-zero count turns
  task 5.2 into a backfill; a zero count makes the generated-column rewrite free.
- [ ] 1.4 Record the tier-1 retrieval baseline produced by `rag-eval-harness`.
  **Proof:** `baseline.md` names the golden-set version and the commit identifier of the report it
  cites.

## 2 Chunk identity

- [ ] 2.1 Add `document_version: int` and `locus: str | None` to `UnifiedChunk`
  (`src/app/features/documents/model.py:158-184`); replace `uq_chunks_document_chunk_index` with a
  constraint over `(document_id, document_version, chunk_index)`.
  **Proof:** `uv run ty check src/` diagnostic count ≤ the 1.1 baseline.
- [ ] 2.2 One migration, with `down_revision` set from the value recorded in 1.2.
  **Proof:** `uv run alembic check` reports no pending autogenerate diff.
- [ ] 2.3 The write path populates both fields.
  **Proof:** a test asserting two versions of one document coexist with distinguishable chunks, and
  that a `locus` of unknown structural position is stored as absent rather than as an empty string.

## 3 Kind dispatch

- [ ] 3.1 Pure `resolve_chunk_policy(document_kind) -> ChunkPolicy` in
  `src/app/features/documents/chunking.py`, performing no I/O.
  **Proof:** a unit test over four kinds returns four distinct policies; a fifth test asserts an
  unclassifiable kind resolves the default rather than raising.
- [ ] 3.2 Call it from the docling path and record the resolved policy on every chunk.
  **Proof:** a fixture ingestion asserts chunk metadata carries the policy name.

## 4 Clause post-pass

- [ ] 4.1 Pure `recover_clause_numbering(chunks)` reusing `_CLAUSE_START_RE`
  (`src/app/features/documents/classification.py:124`).
  **Proof:** a unit test on an inline-bold-numbered fixture recovers the expected loci, and a second
  asserts a chunk with no recoverable number keeps an absent locus.
- [ ] 4.2 Wire it after `_hybrid_chunk_documents`.
  **Proof:** a fixture ingestion populates `locus`.

## 5 The contextualize split

- [ ] 5.1 **Settle the token-budget question first.** Write a test asserting that for every
  `HybridChunker` chunk, the token count of `chunker.contextualize(chunk)` is within the configured
  bound. Today `chunker.py:251` counts tokens *after* contextualization, which makes this a runnable
  probe rather than a research question.
  **Proof:** the test passes; or it fails and is recorded as a pinned expected failure whose measured
  overflow feeds 5.2's budget.
- [ ] 5.2 Change `chunker.py:254` from `content=contextualized_text.strip()` to `content=chunk.text`
  with `preamble=contextualized_text`. If 5.1 failed, budget `max_tokens` against the contextualized
  length rather than the bare length.
  **Proof:** a test asserting `preamble` is non-empty and `content` carries no heading prefix; plus a
  test asserting the value of `search_text` for a fixture chunk is unchanged from before the move —
  this is what proves lexical retrieval input did not shift.
- [ ] 5.3 Flag fallback chunks impure; set overlap to zero on every non-fallback policy.
  **Proof:** `_simple_fallback_chunk` output carries the flag; hybrid output has zero overlap; a
  consumer-level test reads the flag without re-running chunking.

## 6 Embedder collapse — a repair, not a feature

- [ ] 6.0 **Read before editing.** `unified-embedding`'s requirement *"Every embedding consumer resolves
  to the single path"* already requires this. `src/app/shared/rag/docling/embedder.py:44-48` records an
  in-code IOU acknowledging the divergence. This task group closes a known violation; **no new
  requirement is written for it.**
  **Proof:** `rg -n 'unified-embedding' openspec/changes/ingestion-chunking/specs/; test $? -eq 1` →
  exit `0` (no ADDED requirement claims this ground).
- [ ] 6.1 Repoint `src/app/shared/rag/docling/ingest_v2.py:18` at `embed_texts`; delete
  `_PROVIDER_EMBEDDING_MODEL` from `docling/embedder.py`.
  **Proof:** `rg -c 'gemini-embedding-001' src/` returns `0`.
- [ ] 6.2 Update `tests/unit/shared/rag/test_embedder_no_substitution.py` and
  `test_rag_agent_embedder_import.py`.
  **Proof:** pytest summary pass count ≥ the 1.1 baseline.

## 7 Dependency drop

- [ ] 7.1 Swap `docling_enhanced.py:63-71`'s `PdfPipelineOptions` to RapidOCR. Audit the two bare
  `DocumentConverter()` constructions at `ingest_v2.py:127` and `ingestion_kb/nodes.py:486` — they
  inherit the EasyOCR default silently.
  **Proof:** `rg -n 'DocumentConverter\(\)' src/` returns `0`; and a re-run of the tier-1 retrieval
  eval over the filing family, compared against the 1.4 baseline, with the delta recorded in
  `review.md` whether it improved or regressed.
- [ ] 7.2 Drop `sentence-transformers` (`pyproject.toml:51`) and `langchain-docling` (`:43`).
  **This task cannot land before `agentic-retrieval` has replaced `retrieval_kb/reranker.py:9`.** Do
  not touch the reranker here.
  **Proof:** `uv lock && uv run python -c "import torch"` exits non-zero.
- [ ] 7.3 The token-counter cache guard still holds.
  **Proof:** `tests/unit/shared/rag/test_chunker_tokenizer_cache.py` is green.

## 8 Layering

- [ ] 8.1 Add a public `build_chat_model` factory; repoint `documents/dependencies.py:13`,
  `documents/service.py:19,26`, and `crawler/processor.py:14`.
  **Proof:** the `PLC2701` count goes from 3 to 0 against the 1.1 baseline.
- [ ] 8.2 Retire the five callers of `features/documents/chunking.py::chunk_text` onto the docling path;
  delete the whitespace splitter and the fixed chunk-size and overlap settings.
  **Proof:** `rg -n 'INGEST_CHUNK_SIZE' src/` returns `0`, and
  `tests/unit/documents/test_chunking.py` is retargeted rather than deleted.

## 9 Close out

- [ ] 9.1 **Proof:** `openspec validate ingestion-chunking --strict` exits `0`;
  `uv run ruff check --no-cache src/` line count ≤ `docs/relay/baseline-ruff-after.txt`;
  `uv run ty check src/` count ≤ `docs/relay/baseline-ty.txt`; `uv run pytest -q 2>&1 | tail -1`
  failure count ≤ `docs/relay/baseline-pytest.txt`.
- [ ] 9.2 Re-verify the named blast radius: `test_chunker_tokenizer_cache.py`,
  `test_embedder_no_substitution.py`, `test_rag_agent_embedder_import.py`,
  `tests/unit/documents/test_chunking.py`, `tests/unit/test_feature_error_exhaustiveness.py`.
  **Proof:** each named file's summary shows zero failures. The twelve pre-existing websocket
  fixture-drift failures are owned by no change in this cluster and are excluded by name.
