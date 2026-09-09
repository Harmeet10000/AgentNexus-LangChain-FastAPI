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

## Pending — the token-budget probe (task 5.1)

- Chunks measured: _pending_
- Chunks whose contextualized token count exceeds the bound: _pending_
- If non-zero, the maximum observed overflow, which becomes 5.2's budget input: _pending_

A negative result here is a finding, not a failure. It is recorded as a pinned expected failure and
consumed by 5.2.

---

## Pending — the `search_text` invariance check (task 5.2)

- Fixture chunk's `search_text` before the move: _pending_
- Fixture chunk's `search_text` after the move: _pending_
- Confirmation the two are identical: _pending_

This is the single assertion in this change that can catch a wrong concatenation order or separator. If
it is weakened to "both fields are populated", the change can ship having silently altered every
lexical retrieval input in the corpus.

---

## Pending — the chunks table row count (task 1.3)

- Row count against the live database: _pending_

Zero makes the generated-column rewrite free. Non-zero turns task 5.2 into a backfill, because
`search_text` is `Computed(persisted=True)` and every stored value would need regenerating.

---

## Pending — the OCR delta on the filing family (task 7.1)

- Baseline tier-1 aggregates on the filing family: _pending_
- Post-swap aggregates: _pending_
- Delta, recorded regardless of direction: _pending_
- Golden-set version both were scored against: _pending_
