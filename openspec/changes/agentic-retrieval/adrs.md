# ADRs — agentic-retrieval

## ADR-001 — Extend the existing retrieval graph; do not build a second one

**Status:** accepted.

**Context.** The brief was to implement an enhanced agentic RAG pipeline after a published reference
architecture. Reading that architecture against `retrieval_kb/graph.py:28-73` shows the query optimizer,
hybrid retrieval, the reranking funnel, and a grader loop already exist — and that the existing loop is
**cyclic** where the reference pipeline is explicitly acyclic.

**Decision.** Add two nodes to the existing graph. Do not build a parallel one.

**Consequences.** No duplicated query planner, grader, or retry loop, and `graph-lifecycle` has one
retrieval graph to provision rather than two. The reference architecture's "agentic loop" is not
implemented, because implementing it would replace a cyclic design with a sequential one — a regression
dressed as an upgrade. The cost is that the new nodes must fit an existing state schema and an existing
edge topology rather than being designed freely.

## ADR-002 — The source identifier is the one genuinely missing component

**Status:** accepted.

**Context.** Of the reference architecture's components, most are already built, not applicable to this
corpus, or owned by another change. One is portable and absent: a source identifier producing a document
allowlist before retrieval.

**Decision.** Add a `source_identifier` node between the query analyzer and retrieval, populating a
document allowlist from cheap metadata — jurisdiction, document kind, matter. The allowlist is applied
as a search filter, never as a post-filter on returned results.

**Consequences.** Retrieval searches a smaller, more relevant corpus. It also addresses this stack's
approximate-nearest-neighbour recall cliff from the query axis: a sufficiently narrow allowlist makes
the filtered set small enough to search exactly, so approximate recall stops mattering.
`retrieval-sql`'s tenant-predicate move covers the same cliff from the tenant axis; neither alone is
sufficient. The cost is one model call per query in the retrieval path.

## ADR-003 — A narrowing that cannot widen is a new failure mode, so widening is required

**Status:** accepted.

**Context.** A source identifier that only narrows makes a query whose answer sits outside the
allowlist permanently unanswerable — and does so invisibly, returning confident, well-ranked,
wrong-corpus results.

**Decision.** When the grader reports insufficient context, the next iteration runs with a strictly
wider allowlist, or with none. Widening does not extend the iteration cap.

**Consequences.** A router mistake becomes recoverable within the existing loop rather than fatal. The
existing grader → analyzer cycle makes this cheap: widening is a state change on retry rather than new
control flow. The termination assertion is written separately from the widening (task 3.3), so that a
change to one cannot silently weaken the other — without that separation, widening could turn a bounded
loop into an unbounded one by a route the widening test would not notice.

## ADR-004 — The context budget is counted in tokens, and the tokenizer is injected

**Status:** accepted (user ruling: the token budget belongs to this change).

**Context.** `rag.py:79,87` counts budget with `len(content.split())`. For legal text — citations,
statute numbers, section symbols, irregular spacing — tokens per whitespace-delimited word diverge
substantially from one.

**Decision.** Count with the tokenizer. Drop whole sections from the tail on overflow; never truncate
mid-chunk. Receive the counter as an **injected callable**; do not choose or import a tokenizer here.

**Consequences.** The budget stops overshooting the model's real limit, and the overshoot's symptom —
a truncated or rejected generation request, far from the accounting that caused it — disappears.
Dropping whole sections rather than truncating matters specifically for this domain: a clause cut in
half is worse than an absent clause when a model reasons about obligations. The tokenizer choice stays
with `ingestion-chunking`, and task 4.3's Proof — no tokenizer library imported under
`features/documents/` — is what stops the seam collapsing into a direct import.

## ADR-005 — This change creates the reranker seam; `ingestion-chunking` removes the dependency

**Status:** accepted (user ruling: "drop torch — hosted reranker + RapidOCR").

**Context.** `retrieval_kb/reranker.py:9` imports a cross-encoder, which is the sole reason
`sentence-transformers` and torch are in the runtime image. Removing them requires a replacement
reranker; choosing a replacement reranker is a retrieval-quality decision.

**Decision.** Here: define a `Reranker` protocol matching the existing call shape, add a hosted
implementation, make it the configured default, and prove degradation to the fused order. The existing
local implementation must satisfy the protocol **structurally, with no modification**. There:
`ingestion-chunking` deletes the local implementation and drops the dependencies.

**Consequences.** Each change contains exactly one kind of decision. Task 2.4's Proof asserts that
`sentence_transformers` is **still present** after this change — an unusual shape, and the point of it:
this change removes nothing. The cost is a hard ordering dependency between two changes, recorded in
both, and a window in which two reranker implementations coexist.

## ADR-006 — Degradation, not failure, when the reranking provider is unavailable

**Status:** accepted.

**Context.** Moving from a local model to a hosted provider introduces a network dependency into the
retrieval path, where none existed.

**Decision.** When the provider is unavailable, return the fused order truncated to the requested count.
Do not fail the request.

**Consequences.** A reranker outage degrades result quality instead of taking retrieval down — and the
fused order is a genuinely reasonable fallback, since it is a three-branch reciprocal-rank fusion rather
than an arbitrary ordering. The cost is that a persistent provider outage is silent from the caller's
perspective: results keep arriving, slightly worse. The degradation path is therefore specified as
behaviour with its own test (task 2.3) rather than left as a `try`/`except`.

## ADR-007 — The graph's topology is asserted before it is changed

**Status:** accepted.

**Context.** A LangGraph topology change is a handful of `add_node` and `add_edge` calls — easy to
review past, invisible in a test summary.

**Decision.** Task 1.2 pins the current node names and edge pairs as a test **before** any node is
added. Each subsequent node addition updates that assertion in the same commit.

**Consequences.** "This change adds exactly one node and two edges" becomes an assertion rather than a
claim in a commit message. It also gives `graph-lifecycle` a foundation: a graph whose topology is
asserted can be compiled once per process without anyone wondering whether the compiled copy matches
the source. The cost is one test that must be updated by every future topology change — which is the
intended friction.

## ADR-008 — Keyword search over model-generated metadata is named as a seam, not built

**Status:** accepted, deferred.

**Context.** The reference architecture's remaining portable idea is running lexical search over
model-generated summaries, questions, and keywords rather than only over raw text.

**Decision.** Do not build it here. Record it as a named seam belonging to `ingestion-chunking` and
`knowledge-stack`.

**Consequences.** It requires ingest-time enrichment feeding the generated `search_text` column, which
is write-side work in a different change; building it here would mean this change reaching into
ingestion. Recording it explicitly means a later reader looking for the reference architecture's last
idea finds a pointer rather than an absence and concludes it was overlooked.
