# Review — ingestion-chunking

Sections marked **accepted cost** are known limitations recorded deliberately. Sections marked
**pending** are filled during implementation by the task that names them.

---

## Measured before planning — the live citation defect

`src/app/shared/rag/docling/chunker.py:250-254` writes `content=contextualized_text.strip()` and leaves
`preamble` unpopulated. Since `content` is the field a citation quotes, **every citation this system
has produced quotes heading boilerplate prepended to the clause.** This is not a latent risk; it is
current behaviour.

The correction is two lines, and the schema needed no change — `UnifiedChunk` already carries
`content`, `preamble`, and a `search_text` generated from both, and `retrieval_kb/reranker.py:56`
already reranks over both. The design was correct and unhonoured.

---

## Accepted cost — the OCR swap has only an indirect proof

Replacing EasyOCR with RapidOCR changes extraction quality on scanned documents, which is the filing
family. The metric that would measure it directly is judged faithfulness, and tier 2 does not exist by
decision.

Tier-1 retrieval metrics see OCR quality through three layers of attenuation: worse character
recognition → worse chunk text → worse embeddings and lexical matches → lower recall. Over a
deliberately small golden set, a real regression may not clear the noise floor.

**This is accepted, not mitigated.** Task 7.1 requires the delta to be recorded whether it improved or
regressed, so a later decision — reinstating a heavier engine for scanned filings specifically — has an
input. What is rejected is blocking the dependency drop until tier 2 exists; the signal is attenuated,
not absent, and the image-size win is real.

**Named follow-up:** re-measure the filing family once the judged layer lands.

---

## Accepted cost — task 8.2 deletes a splitter that five call sites use

Retiring `features/documents/chunking.py::chunk_text` moves five callers onto the docling path. Those
callers currently get fast, dependency-free, structurally-ignorant chunking; they will get slower,
parser-backed, structure-aware chunking.

For legal documents that is the entire point. For any caller chunking something that is not a document
— a short text field, a crawler snippet — it is a real cost in latency and in a new dependency on the
parser being constructible.

Accepted because the alternative is keeping two chunking implementations and a standing invitation for
the wrong one to be used on a contract. The mitigation is that `tests/unit/documents/test_chunking.py`
is **retargeted rather than deleted**, so the behaviour of each of the five call sites is asserted
after the move rather than merely assumed.

---

## Note — why no `## ADDED` requirement covers the embedder

Recorded because a reader will look for one and its absence is deliberate.

`unified-embedding`'s *"Every embedding consumer resolves to the single path"* already requires what
task group 6 does, and `docling/embedder.py:44-48` carries an in-code IOU admitting the violation. The
behaviour is **specified and absent**, so group 6 is a repair, not a feature.

Writing an ADDED requirement for it would put a near-duplicate rule in a second capability, and the two
would drift the first time either was edited. Task 6.0's Proof greps this change's own spec directory
to enforce that no requirement claims that ground.

---

## Recorded — the token-budget probe (task 5.1)

- Chunks measured: the real HybridChunker fixture emits a non-empty set over eight legal paragraphs.
- Chunks whose contextualized token count exceeds the configured 32-token bound: **0**.
- Maximum observed overflow: **0 tokens**; no extra 5.2 budget was required.

A negative result here is a finding, not a failure. It is recorded as a pinned expected failure and
consumed by 5.2.

---

## Recorded — the `search_text` invariance check (task 5.2)

- Fixture `search_text` before: `Master Agreement\nIndemnity\n\nThe supplier shall indemnify the buyer.`
- Fixture `search_text` after: `Master Agreement\nIndemnity\n\nThe supplier shall indemnify the buyer.`
- Confirmation: **byte-identical**. Only citation storage moved to bare `content`.

This is the single assertion in this change that can catch a wrong concatenation order or separator. If
it is weakened to "both fields are populated", the change can ship having silently altered every
lexical retrieval input in the corpus.

---

## Recorded — the dependency drop (task 7.2)

- `sentence-transformers` and `langchain-docling`: absent from `pyproject.toml`, from all
  imports under `src/` and `tests/`, and from `uv.lock`. The remaining
  `sentence-transformers/...` strings are HuggingFace model IDs and commented-out code, not
  package dependencies. The last first-party `import torch`
  (`docling_enhanced.py::check_gpu_available`) is removed; `rg '^\s*(import torch|from torch)'`
  over `src/` and `tests/` returns nothing.
- The task's literal proof (`uv lock && uv run python -c "import torch"` exits non-zero)
  **does not pass**: `docling==2.87.0` hard-depends on `torch` and `torchvision`, so the
  package remains importable transitively. Removing `docling` would contradict task 8.2
  (done), which moved five callers onto the docling path. The CPU-wheel pins and the
  `pytorch-cpu` index in `pyproject.toml` are therefore kept deliberately — they hold the
  transitive runtime to CPU wheels instead of multi-gigabyte CUDA ones.
- Net effect of this change: no first-party code imports the tensor runtime; the image-size
  win is bounded by what `docling` itself declares. Revisit if `docling` ever drops the
  hard dependency.

## Recorded — the chunks table row count (task 1.3)

- Row count against the live database: **1**.

Zero makes the generated-column rewrite free. Non-zero turns task 5.2 into a backfill, because
`search_text` is `Computed(persisted=True)` and every stored value would need regenerating.

---

## Recorded — the OCR delta on the filing family (task 7.1)

- Proof part 1: `rg -n 'DocumentConverter\(\)' src/` returns `0`; no `EasyOCR`/`easyocr`
  reference remains under `src/`. `docling_enhanced.py` builds the converter with
  `RapidOcrOptions(backend="onnxruntime")`, and both audited constructions
  (`ingest_v2.py:127` region, `ingestion_kb/nodes.py:486` region) go through the
  `create_document_converter` factory rather than a bare constructor.
- Baseline tier-1 aggregates (task 1.4, `evals/reports/baseline.json`,
  `legal_retrieval_v1`, commit `0653503845c2`): recall 1.0, reciprocal rank 1.0,
  nDCG@1 1.0, precision@1 1.0.
- Post-swap filing row (`test_live_retrieval.py -m requires_db`, 1 passed,
  2026-09-14 re-run): recall@1 **1.0**, reciprocal rank **1.0**, nDCG@1 **1.0**,
  precision@1 **1.0**.
- Delta: **0.0 — no change**, whether improvement or regression. The corpus is the
  seeded fixture (one chunk per family), so this re-run proves wiring and
  non-regression on the harness path, not OCR quality on real scans; see the
  accepted cost above and the named follow-up to re-measure once the judged
  layer lands.
- Golden-set version: `legal_retrieval_v1` for both runs. The 1.4 report file was
  restored byte-identical after the re-run.
