# Chunk legal documents by kind, version chunk identity, and drop the torch runtime

**Class: L.** A schema migration, a behavioural change to what text is stored in the citation column, a
dependency removal that changes the runtime image, and a layering repair — across the whole ingestion
path.

## Why

Three unreconciled chunkers coexist, and the one legal-grade path stores the wrong text in the column
citations quote from. `src/app/shared/rag/docling/chunker.py:250-254` writes
`content=contextualized_text.strip()`, so a citation today quotes heading boilerplate prepended to the
clause rather than the clause. `UnifiedChunk` already has the correct two-column shape — `content`,
`preamble`, and a generated `search_text` concatenating both — so this is a misuse of an existing
schema, not a missing one.

Chunk identity has no version. A superseded clause from an earlier ingestion of the same document is
indistinguishable from the current one, so retrieval can return withdrawn contract language as
authoritative. For a legal corpus this is the most consequential defect in the cluster.

And the ingestion path carries a second embedder that hardcodes a model id diverging from the
configured one, plus `sentence-transformers` and therefore torch — several hundred megabytes of runtime
image for one cross-encoder call.

## What Changes

- **Chunk identity becomes version-scoped.** `(document_id, document_version, chunk_index)` replaces
  the two-column uniqueness constraint, and every chunk carries a `locus`. Chunks from prior versions
  remain addressable; retrieval scopes to one version.
- **Chunking dispatches on document kind.** A pure `resolve_chunk_policy(kind) -> ChunkPolicy` maps
  contract, statute, judgment, and filing to four distinct policies, and the resolved policy name is
  recorded on every chunk. This **tightens** the existing structure-aware chunking requirement rather
  than replacing it.
- **Clause numbering is recovered after layout misses.** When clause numbers arrive as inline bold runs
  rather than headings, a pure post-pass recovers them into chunk locus, reusing the existing
  `_CLAUSE_START_RE` from `classification.py:124`.
- **Embedded text and cited text separate onto the columns that already exist.** `content` holds bare
  chunk text; `preamble` holds the contextualization. Because `search_text` is generated from both,
  lexical retrieval input is **unchanged** by the move — only what a citation quotes changes.
- **Fallback splits declare themselves impure.** A split that cuts mid-clause carries a flag and is the
  only path permitted non-zero overlap.
- **Ingestion collapses onto one embedder** and the torch-bearing dependencies leave.
  `sentence-transformers` and `langchain-docling` are dropped, and Docling's OCR moves from EasyOCR to
  RapidOCR.
- **Private cross-module imports are repaired** with a public chat-model factory.

## Capabilities

**New Capabilities**

- `legal-document-chunking` — version-scoped chunk identity, clause-locus recovery, the embed-text /
  cite-text split, impurity declaration, and the toolchain constraints the legal chunking path imposes.

**Modified Capabilities**

- `hierarchical-document-chunking` — the requirement *"Every document kind is chunked structure-aware"*
  is tightened. It currently requires structure-aware chunking for every kind uniformly; it now
  additionally requires that the four legal families each resolve their own policy and that the
  resolved policy is recorded on the chunk. Every existing scenario is preserved.

This change also **cites** `unified-embedding` without amending it. Its requirement *"Every embedding
consumer resolves to the single path"* already requires what the embedder collapse does — the second
embedder is a **live violation** of an existing requirement, recorded as an in-code IOU at
`src/app/shared/rag/docling/embedder.py:44-48`. Task group 6 is therefore a repair, and no new
requirement may be written for it.

## Impact

- **Schema:** `UnifiedChunk` gains `document_version` and `locus`; the uniqueness constraint changes.
  One migration, whose `down_revision` is consumed as a recorded value from `rag-tree-repair`, never
  guessed.
- **Behaviour:** citations stop quoting heading boilerplate. Retrieval scoped to a document version
  stops returning superseded clauses.
- **Dependencies:** torch leaves the runtime image. This is blocked on `agentic-retrieval` having
  replaced the `CrossEncoder` reranker — that is the seam, and this change does not touch
  `retrieval_kb/reranker.py`.
- **Risk:** `search_text` is a `Computed(persisted=True)` column. Moving text between `content` and
  `preamble` rewrites every generated value. On an empty table that is free.
- **Not touched:** retrieval SQL, fusion, the reranker, graph lifecycle. `retrieval-sql` owns
  `repository.py` search SQL; this change edits `chunking.py` write-side only.
