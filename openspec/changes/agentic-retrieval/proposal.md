# Narrow the corpus before retrieving, assemble context honestly, and free the reranker from torch

**Class: M.** Two nodes added to an existing, tested graph; one protocol seam; one accounting fix.
No schema change.

## Why

The retrieval graph (`retrieval_kb/graph.py:28-73`) already ships more of the agentic-RAG pattern than
the reference architecture it would be measured against. It has a query analyzer that rewrites,
decomposes, routes, and sets per-leg weights; hybrid retrieval; a reranker funnel narrowing twenty
candidates to five; and a context grader with a **cyclic** edge back to the analyzer, capped at two
iterations. The reference pipeline is explicitly acyclic. Rebuilding it would be a downgrade.

What is genuinely missing is four things:

1. **The corpus is never narrowed before it is searched.** The reference architecture's highest-value
   idea is a source identifier that produces a document allowlist *before* retrieval. Its legal
   analogue — jurisdiction, document kind, matter — is also the direct fix for the
   approximate-nearest-neighbour recall cliff: a narrow allowlist turns a filtered approximate search
   into a small exact one.
2. **Assembled context is never deduplicated or restored to document order.**
   `assemble_rag_context` (`rag.py:37`) already groups by document and restores chunk order — and the
   graph never calls it. A chunk returned by two branches reaches generation twice, and clauses reach
   the model out of order.
3. **The context budget is counted in words.** `rag.py:79,87` uses `len(content.split())`. The model's
   limit is in tokens, and for legal text with citations and numbering the two diverge substantially.
4. **The reranker requires a local deep-learning framework.** `retrieval_kb/reranker.py:9` imports a
   cross-encoder, which is the sole reason `sentence-transformers` — and therefore torch — is in the
   runtime image.

## What Changes

- **A source-identifier node** runs between the query analyzer and retrieval, populating a document
  allowlist from cheap metadata. When it cannot narrow, retrieval proceeds unconstrained rather than
  returning nothing. When the grader reports insufficient context, the next iteration **widens** the
  allowlist before re-running — otherwise a wrong allowlist becomes an unrecoverable dead end.
- **A post-processing node** between reranking and grading dedupes by chunk identity and calls the
  existing `assemble_rag_context`.
- **The context budget is measured in tokens**, using a counter injected as a callable. Whole sections
  are dropped from the tail rather than truncated mid-chunk.
- **Reranking moves behind a protocol** with a hosted implementation as the configured default, and
  degrades to the fused order when the provider is unavailable. **This change removes nothing** — it
  creates the seam that lets `ingestion-chunking` remove `sentence-transformers` and torch.

## Capabilities

**New Capabilities**

- `agentic-retrieval-loop` — corpus narrowing before search, allowlist widening on retry, deduplicated
  context in document order, token-measured budgets, provider-independent reranking, and loop
  termination.

**Modified Capabilities**

None. This change **cites** two restored capabilities without amending them:

- `agent-tool-registry` already requires that every agent role receives the tools assigned to it. The
  nodes added here are graph nodes, not tools, and the registry contract is unchanged.
- `hybrid-retrieval-ranking` already governs how ranked lists are produced and fused. This change
  consumes that path and changes neither its SQL nor its weights.

## Impact

- **Code:** `src/app/shared/langgraph_layer/retrieval_kb/{graph,nodes,state}.py`,
  `src/app/features/documents/rag.py:79,87`, and a new reranker protocol with a hosted implementation.
- **Behaviour:** duplicate chunks stop reaching generation; clauses from one document reach the model
  in reading order; the context budget stops overshooting on citation-dense text; retrieval narrows
  before searching when the query names a jurisdiction, kind, or matter.
- **Graph shape:** two nodes and their edges. A shape test captured before the change is the diff
  target for both additions.
- **Not touched:** any SQL in `documents/repository.py`, the fusion weights themselves, the removal of
  `sentence-transformers` and torch, the tokenizer *choice*, lifespan compilation of the graph, and
  `shared/rag/**`.

## Explicitly out of scope

Keyword search over model-generated metadata — summaries, generated questions, keywords — is portable
and valuable, and it is **not this change**. It requires ingest-time enrichment feeding the generated
`search_text` column, which `ingestion-chunking` and `knowledge-stack` own. Named here as a seam so it
is not silently dropped.
