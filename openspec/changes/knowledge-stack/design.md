# Design — knowledge-stack

## Two complete modules with zero runtime callers

The finding that shapes this change: `shared/rag/langextract/` and `shared/rag/pageindex/` are both
finished, reviewable, importable code that **nothing calls**.

That is a specific and slightly unusual kind of technical debt. It is not half-built — it is built and
disconnected. Two consequences follow that a normal "unfinished feature" would not have:

- **The type system already believes the edge exists.** `graphiti/write_clause_episodes.py:42` imports
  the extraction types under `TYPE_CHECKING`. The type checker sees a dependency; the runtime does not.
  A type-level edge with no runtime edge beneath it reads, to anyone browsing, as a wired feature.
- **The cost of the two modules is opposite.** Wiring the extraction module is straightforwardly good.
  Wiring the external structural client would **add** an external service dependency and send document
  content to a third party — so for that one, the right move is deletion, not connection.

So this change connects one and deletes the other, and the asymmetry is the design.

## Why the external structural service is replaced rather than wired

The commented construction at `lifespan.py:538` is one uncomment away from working. It is rejected
anyway.

The capability it provides — reasoning over a document's structural tree to navigate to the right
section — does not require an external service. The document parser already produces that tree during
ingestion. Reasoning over a tree the system already holds is a local computation over local data.

Wiring the client instead would mean: an external dependency in the retrieval path, **document content
leaving the system** to a third party, a second copy of the document structure held elsewhere, and a
per-query cost for something computable in process. Against that, the only benefit is not writing a
navigator.

The navigator is specified as **pure** — a function from a tree and a query to node paths, with no I/O —
which is what makes it cheap to build and exact to test. A fixture tree in a test file, an expected
section path, no database.

## The measurement that decides half this change

**Whether the structural tree is persisted at all is not established.** The parser produces one during
ingestion; whether it survives into storage is a separate question, and nothing in the planning
established it.

Task 1.2 settles it, and the two branches are materially different work:

| 1.2 result | Task group 2 | This change is |
|---|---|---|
| tree **is** persisted | no-op, recorded explicitly | wiring |
| tree is **not** persisted | mandatory storage work plus a migration | a storage build |

This is deliberately the first substantive task, and the proposal states the conditional rather than
picking the likely branch. A plan that assumed *yes* and found *no* would discover a migration halfway
through a change scoped as wiring.

Task 2.3 exists for the *yes* branch specifically: it requires recording that the branch was taken, so
a later reader does not read an unticked task group as abandoned work.

## Extraction runs before chunking, and the reason is reconciliation

The ordering could go either way at first glance — extract from raw text, or extract from chunks.
Before chunking is correct, for one reason: **both stages then derive from the same parse.**

Extract afterward and you have entities located in one coordinate system (character offsets in the
document, say) and chunks in another (chunk index, locus), and every downstream consumer that wants
"the chunk containing this clause entity" has to reconcile them. Reconciliation code that maps between
two views of one document is exactly the kind of code that works on the fixtures and drifts on the
corpus.

Extracting first means the chunk writer receives the extraction output and can associate the two while
it still holds both.

## Failure semantics: three outcomes, not two

The natural implementation has two states — extracted, or not. This change requires three:

1. **Extraction succeeded and yielded entities.**
2. **Extraction succeeded and yielded nothing.** A document may genuinely contain no clause entities.
3. **Extraction failed.** The provider was unavailable, or errored.

Collapsing 2 and 3 into "no entities" is the failure mode worth designing against, because it is
silent and it is permanent: a document ingested during a provider outage looks, forever after,
identical to a document that had nothing to extract. Nothing later would prompt a re-run.

Hence the extraction-incomplete flag applies to case 3 only, and task 5.2 requires **two** tests — one
asserting the flag is set on failure, one asserting it stays unset on an empty success.

The failure is also required to be a **typed value** the ingestion path handles explicitly rather than
an exception caught and discarded, which is what makes case 3 representable in the first place.

## Why no canonicalisation semantics are written here

`graph-entity-canonicalisation` already requires that canonical graph writes are idempotent and that
re-ingestion does not duplicate entities. Graphiti is built, wired, and has a writer.

So this change routes its episodes through that writer and **specifies nothing about canonicalisation**.
Writing an idempotency requirement here would put a near-duplicate of an existing rule into a second
capability, and the two would drift the first time either was edited.

Task 5.4's Proof enforces it mechanically: this change's spec directory may not contain
canonicalisation or deduplication language except where a scenario explicitly defers to the existing
writer. This is the same discipline `ingestion-chunking` applies to the embedder — **restore, cite, and
repair rather than re-specify.**

## The seam with `graph-lifecycle`

`graph-lifecycle` deliberately left `app.state.pageindex_client` unwired, with task 6.2 there asserting
the commented line is **unchanged**. This change removes it.

That is a two-change handshake over a single line of commented code, which is more ceremony than the
line deserves — except that the alternative is `graph-lifecycle` wiring a client that this change then
deletes, which is churn plus a window in which document content is being sent to a third party for no
reason.

## What this change does not touch

Chunking policy, chunk identity, the embedder, retrieval SQL, fusion, the reranker, and graph
lifecycle. It inserts one stage into the ingestion path at a boundary recorded in task 1.3, and adds
one retrieval branch reached through the repository and the service layer.
