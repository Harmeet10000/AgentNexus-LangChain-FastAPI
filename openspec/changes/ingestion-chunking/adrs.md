# ADRs — ingestion-chunking

## ADR-001 — Chunk identity is versioned now, not deferred

**Status:** accepted (user ruling: "version now — `(document_version, locus)`").

**Context.** Chunk identity is `(document_id, chunk_index)`. Re-ingesting a document at a new version
either overwrites prior chunks or collides. For a legal corpus the consequence is that an amended
clause and the clause that replaced it are indistinguishable at retrieval time.

**Decision.** Add `document_version` and `locus` to `UnifiedChunk` and make uniqueness
`(document_id, document_version, chunk_index)`. Retrieval scopes to one version.

**Consequences.** Superseded clauses stop being retrievable as current, and "what did this say last
quarter" becomes answerable — a real requirement of the domain. The cost is a migration in a change
that would otherwise be code-only, and a `down_revision` that must be consumed as a recorded value from
`rag-tree-repair` rather than guessed. Soft-deleting superseded chunks was rejected because it makes
prior versions unreachable, which defeats half the purpose.

## ADR-002 — The embed-text / cite-text separation uses the existing columns

**Status:** accepted.

**Context.** The originally-planned shape was a new text column. Measurement showed `UnifiedChunk`
already has `content`, `preamble`, and a generated `search_text = clause_type || preamble || content`,
and that `retrieval_kb/reranker.py:56` already reranks over both fields.

**Decision.** Add no column. Change the write site at `chunker.py:254` to put bare chunk text in
`content` and the contextualization in `preamble`.

**Consequences.** The change shrinks from a schema migration with a backfill to a two-line correction.
Critically, because `search_text` is generated from both fields, **lexical retrieval input is
byte-identical** across the move — BM25 and trigram see no change. Only what a citation quotes changes.
The risk this introduces is getting the concatenation order or separator wrong, which is why task 5.2's
Proof asserts `search_text` is unchanged for a fixture chunk rather than merely asserting the two
fields are populated.

## ADR-003 — The token-budget probe runs before the storage change

**Status:** accepted.

**Context.** Whether contextualized chunks fit the embedding window was treated as an open research
question. It is not: `chunker.py:251` already counts tokens after `contextualize()`, so the current
tree can answer it.

**Decision.** Task 5.1 asserts the bound over contextualized text and runs before 5.2. If it fails, the
measured overflow becomes 5.2's budget input, recorded as a pinned expected failure rather than a
blocker.

**Consequences.** A negative result changes 5.2's implementation instead of surprising it. Without this
ordering, an overflow would appear as silent truncation at embedding time, diagnosed through two layers
of simultaneous change. The cost is one task that produces no product behaviour.

## ADR-004 — Torch leaves; OCR moves to a lightweight engine

**Status:** accepted (user ruling: "drop torch — hosted reranker + RapidOCR").

**Context.** `sentence-transformers` pulls torch into the runtime image for one cross-encoder call.
Docling's default OCR engine is EasyOCR, which also requires torch. Two bare `DocumentConverter()`
constructions inherit that default silently.

**Decision.** Swap the pipeline options to RapidOCR, supply pipeline options explicitly at every
converter construction, and drop `sentence-transformers` and `langchain-docling`.

**Consequences.** Several hundred megabytes leave the image and cold starts improve. Extraction quality
on scanned documents changes, and the filing family is where that lands hardest — see ADR-005. The
dependency drop cannot land before `agentic-retrieval` replaces the cross-encoder, which is a hard
ordering dependency between two changes rather than a preference.

## ADR-005 — The OCR quality risk is recorded, not mitigated

**Status:** accepted with a named residual risk.

**Context.** Replacing EasyOCR with RapidOCR is a change to extraction quality on the hardest document
family. The metric that would measure it directly is a judged faithfulness score, and tier 2 does not
exist — by decision in `rag-eval-harness`.

**Decision.** Require task 7.1 to record the measured tier-1 delta on the filing family against the
recorded baseline **whether it improves or regresses**. Do not require an improvement, and do not
gate the task on one.

**Consequences.** The change ships with a known, measured, indirect signal rather than a strong direct
one. This is honest and it is usable: a recorded regression is an input to a later decision about
whether to reinstate a heavier engine for scanned filings specifically. A silent regression would not
be. The alternative — blocking the dependency drop until tier 2 exists — was rejected as too high a
price for a signal that is attenuated rather than absent.

## ADR-006 — The embedder collapse is a repair against an existing requirement

**Status:** accepted.

**Context.** `unified-embedding` already requires that every embedding consumer resolves to the single
path. `src/app/shared/rag/docling/embedder.py:44-48` carries an in-code IOU acknowledging that it does
not, hardcoding a model id that diverges from the configured one.

**Decision.** Treat task group 6 as a repair of a violated requirement. Cite `unified-embedding`; write
no `## ADDED` requirement for the embedder.

**Consequences.** No near-duplicate rule appears in two capabilities to drift apart later. The general
principle this instantiates — **restore, cite, and repair rather than re-specify** — is what keeps the
seven changes in this cluster from collectively re-stating the archived baseline. Task 6.0's Proof
enforces it mechanically rather than by review.

## ADR-007 — The dependency drop goes last, not first

**Status:** accepted.

**Context.** The mechanical order is to drop dependencies first: it gates image size and touches
nothing behavioural.

**Decision.** Do the chunking and storage work first; drop dependencies last.

**Consequences.** The OCR swap's only honest proof is an ingestion run scored against a retrieval
baseline, and ingestion output is not trustworthy for scoring until the storage semantics are correct —
before task 5.2, the stored `content` is contextualized text, so any citation-level comparison measures
the wrong field. The cost is that the image-size win arrives at the end of the change rather than the
start.
