# Design — agentic-retrieval

## What the reference architecture actually offers, component by component

The starting brief was to build an enhanced agentic RAG pipeline after a widely-cited engineering blog.
Reading it against this repository produces an unexpected result: **most of it is already here, and one
part of it is behind us.**

| Reference component | Verdict here |
|---|---|
| Query optimizer — rewrite and decompose | **Already built.** `make_query_analyzer_node`, `nodes.py:94`. No work. |
| Source identifier — a document allowlist before retrieval | **Portable, and the one real gap.** The legal analogue is a jurisdiction / document-kind / matter router. |
| Post-processing — dedupe and restore document order | **Portable, and missing on the graph path.** `assemble_rag_context` (`rag.py:37`) already does it; the graph never calls it. |
| Keyword search over model-generated metadata | **Portable, but not this change.** It needs ingest-time enrichment feeding the generated `search_text` column. Named as a seam. |
| Document-platform migration and a custom loader | **Not applicable.** This corpus is PDFs through a document parser. |
| Offline feature store, internal ML platform, chat surface | **Not applicable.** Application state plus Redis already fill the artifact-cache role. |
| Judged evaluation harness and expert golden set | **Portable — owned by `rag-eval-harness`.** Not planned here. |
| "The agentic loop" | **Already exceeded.** The reference pipeline is strictly sequential and explicitly acyclic; this graph already cycles grader → analyzer with a cap. |

The last row is the one worth dwelling on, because it inverts the framing of the original task. The
graph at `retrieval_kb/graph.py:28-73` has a **cyclic** edge from the context grader back to the query
analyzer. Rebuilding toward an acyclic reference pipeline would be a regression dressed as an upgrade.

So this change adds two nodes to a graph that mostly works, rather than building a graph.

## Rejected shape — a fresh graph beside the existing one

The alternative was to build a clean enhanced-agentic-RAG graph next to `retrieval_kb/`.

Rejected because it would duplicate a query planner, a grader, and a retry loop that already exist and
are tested (`test_retrieval_retry_shape.py`) — and because it would leave **two retrieval graphs** for
`graph-lifecycle` to provision, compile, and reason about. A second graph is not a second
implementation of a node; it is a second everything.

## The source identifier is also the ANN recall fix

The narrowing node is worth building for the reason the reference architecture gives — searching a
smaller corpus finds better evidence — but it has a second effect specific to this stack that is easy
to miss.

An approximate-nearest-neighbour index draws a fixed-size candidate pool of globally nearest vectors,
and filters afterward. A restrictive filter over a large corpus therefore hits a **recall cliff**: the
pool fills with documents the filter will discard, and the query returns fewer results than exist.

A narrow allowlist inverts that. When the allowlist is small enough, the filtered set is small enough
to search exactly, and approximate-search recall stops being a concern at all. So the same node that
implements the reference architecture's best idea also removes this stack's sharpest retrieval failure
mode.

`retrieval-sql`'s tenant-predicate move addresses the same cliff from the other direction — on the
tenant axis rather than the query axis. Together they cover it; separately, each leaves a gap.

## Why widening on retry is a requirement, not a refinement

A source identifier that narrows is an optimisation. A source identifier that narrows **and cannot
widen** is a new failure mode: a query whose answer sits in a document the router did not think to
include becomes permanently unanswerable, and the failure is invisible — retrieval returns confident,
well-ranked, wrong-corpus results, and the grader has no way to distinguish "the corpus does not
contain this" from "I looked in the wrong part of it".

The existing grader → analyzer cycle is what makes the fix cheap: the loop already exists, so widening
is a state change on retry rather than new control flow.

The bound matters too. Widening must not extend the iteration cap, or a query that keeps failing to
narrow correctly can loop until the cap is reached by a different route. Task 3.3 asserts termination
independently of task 3.2's widening.

## Words are not tokens, and legal text is where that bites

`rag.py:79,87` counts context budget with `len(content.split())`.

For prose, words and tokens track each other closely enough that the error is a rounding issue. Legal
text is the worst case for that assumption: citations, statute numbers, section symbols, and
inconsistent spacing all tokenize into several tokens per whitespace-delimited word. A budget that
believes it has room for a section can overshoot the model's real limit substantially — and the failure
appears at generation time, as a truncated or rejected request, far from the accounting that caused it.

Two constraints follow. The count comes from the tokenizer, and overflow drops **whole sections from
the tail** rather than truncating mid-chunk — because a chunk cut in half is a clause cut in half, and
a clause cut in half is worse than an absent clause when a model is reasoning about obligations.

The tokenizer **choice** is deliberately not made here. `ingestion-chunking` selects it, and this
change consumes it as an injected callable. Task 4.3's Proof — no tokenizer library imported under
`features/documents/` — is what keeps that seam from quietly collapsing into a direct import.

## The reranker seam: create it here, use it there

`retrieval_kb/reranker.py:9` imports a cross-encoder, and that import is the only reason
`sentence-transformers` — and therefore torch — is in the runtime image.

The split is deliberate and it runs in this order:

1. **Here:** define the protocol, add a hosted implementation, make it the default, prove degradation.
   The existing local implementation must satisfy the protocol **structurally, without modification** —
   which is what makes this step safe to land on its own.
2. **`ingestion-chunking`:** delete the local implementation and drop the dependencies.

Task 2.4's Proof is unusual in that it asserts something is **still present**:
`sentence_transformers` still appears in the tree after this change. That is the point. A change that
creates a seam and a change that removes a dependency are separable, and conflating them would put a
retrieval-quality decision (which reranker) inside a dependency change, or a dependency decision inside
a retrieval change. Either way, nobody would find it later.

## The graph-shape test is the diff target

Task 1.2 pins the current node names and edge pairs before anything is added. This is not coverage for
its own sake — a LangGraph topology change is a handful of `add_node` and `add_edge` calls, easy to
review past and impossible to see in a test summary. Pinning the shape means "this change adds exactly
one node and two edges" is an assertion rather than a claim in a commit message.

It also gives `graph-lifecycle` something to lean on: a graph whose topology is asserted is a graph
that can be compiled once per process without anyone wondering whether the compiled copy matches the
source.

## What this change does not touch

Any SQL in `documents/repository.py`, the fusion weights themselves, the removal of
`sentence-transformers` and torch, the tokenizer choice, lifespan compilation of the graph, and
`shared/rag/**`.

And keyword search over model-generated metadata — summaries, generated questions, extracted keywords
feeding the generated `search_text` column. It is portable and it is valuable and it belongs to
ingestion. Recorded here so that a later reader looking for the reference architecture's remaining idea
finds a pointer rather than an absence.
