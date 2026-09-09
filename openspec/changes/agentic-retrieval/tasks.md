# Tasks — agentic-retrieval

## How to read the Proofs

1. **Never use a test-process exit code as a Proof of test outcome.** Compare **summary pass and
   failure counts** against the baseline captured in task 1.1.
2. **Never prove a schema fact by rendering migrations offline.** This change alters no schema.
3. **Never make a Proof depend on a durable outbound event firing.** The outbox tables do not exist.
4. **The live database may be used** — zero data, zero users, ruled available.
5. **No Proof in this change may cite an answer-quality metric.** Only tier-1 retrieval metrics exist,
   which constrains what task 3.1 can claim about the allowlist.
6. **This change removes nothing.** Task group 2 creates a seam; `ingestion-chunking` uses it to remove
   the dependency. A Proof here that asserts an absent package belongs in that change, not this one.

**Blocked by `retrieval-sql`** — its task 3.3 deletes `legal_rrf_search`, which this change's hybrid
node calls today — and by **`rag-eval-harness`**, which supplies the numbers that justify the allowlist.

## 1 Baseline

- [ ] 1.1 Capture `uv run pytest tests/unit/shared/langgraph_layer -q` counts into
  `docs/relay/baseline-agentic.md`, and record which tests currently cover the reranker, the context
  grader, and the generator.
  **Proof:** the file lists `test_reranker_singleton.py` and `test_retrieval_retry_shape.py` with their
  current outcomes.
- [ ] 1.2 **Pin the graph's current shape as a test.** Record `build_retrieval_graph`'s node names and
  edge pairs as an asserted fixture.
  **Proof:** the new test passes against the tree as it stands today. It is the diff target for every
  later graph edit, and the reason a topology change cannot land invisibly.
- [ ] 1.3 Record the tier-1 retrieval baseline from `rag-eval-harness`.
  **Proof:** `docs/relay/baseline-agentic.md` names the golden-set version and the commit identifier of
  the report it cites.

## 2 The reranker seam — creates the opening for the dependency drop

- [ ] 2.1 Define a `Reranker` protocol matching the call already made at `nodes.py:263`
  (`rerank(query, chunks, limit)`), and make `make_reranker_node` accept it.
  **Proof:** `uv run ty check src/` is clean, and
  `uv run pytest tests/unit/shared/langgraph_layer/test_reranker_singleton.py -q` passes with the
  **existing** cross-encoder implementation satisfying the protocol structurally — no edit to it.
- [ ] 2.2 Add a hosted implementation behind the protocol, selected by settings, with the local one
  still available.
  **Proof:** a test injecting a stub hosted client asserts twenty candidates in, `limit` out, in the
  provider's returned order.
- [ ] 2.3 Degradation test: the provider raises, the node returns the fused order truncated to `limit`,
  and no exception escapes.
  **Proof:** `uv run pytest tests/unit/shared/langgraph_layer -q` gains one test over the 1.1 baseline.
- [ ] 2.4 Make the hosted implementation the configured default. Record in `design.md` that deleting
  `sentence-transformers`, torch, and `retrieval_kb/reranker.py` is **`ingestion-chunking`'s** task,
  unblocked by 2.1 through 2.3.
  **Proof:** `rg -n 'sentence_transformers' src/app/shared/langgraph_layer/` still returns only
  `reranker.py` — this change removes nothing, and that is the assertion.

## 3 The source identifier

- [ ] 3.1 Extend `QueryPlan` with an allowlist field, and add a `source_identifier` node between
  `query_analyzer` and retrieval, populating the document filter from cheap metadata — jurisdiction,
  document kind, matter — with few-shot examples.
  **Proof:** the 1.2 graph-shape test is updated **in the same commit** and shows exactly one new node
  and two new edges; and a test with a stub language model asserts the allowlist reaches the branch
  filter parameters rather than being applied after results return.
- [ ] 3.2 **Widen the allowlist on retry.** When the grader reports insufficient context, the next
  iteration relaxes the allowlist before re-running.
  **Proof:** a test drives the grader to report insufficient once, then asserts the second retrieval
  call carries a strictly larger allowlist, or an empty one.
- [ ] 3.3 Assert the widening cannot extend the loop.
  **Proof:** a test drives the grader to report insufficient at every iteration and asserts the loop
  stops at its existing cap and returns the grounded fallback.

## 4 Post-processing and the token budget

- [ ] 4.1 Insert a `post_process` node between the reranker and the context grader that dedupes by chunk
  identity and calls the existing `assemble_rag_context`.
  **Proof:** a test with two branches returning an overlapping chunk asserts it appears once, and that
  chunks of one document appear in ascending in-document order rather than relevance order.
- [ ] 4.2 Replace `len(content.split())` at `rag.py:79,87` with the token counter, **injected as a
  callable rather than imported**.
  **Proof:** `uv run pytest tests/unit/documents/test_rag.py -q` passes; and a new test feeds text whose
  token count differs from its word count by more than twenty percent and asserts the section is dropped
  at the token boundary, not the word one.
- [ ] 4.3 Record in `design.md` that the tokenizer **choice** belongs to `ingestion-chunking`, and that
  this change only consumes it through an injected callable.
  **Proof:** `rg -n 'AutoTokenizer|tiktoken' src/app/features/documents/` returns nothing.

## 5 Close out

- [ ] 5.1 **Proof:** `openspec validate agentic-retrieval --strict` exits `0`;
  `uv run ruff check --no-cache src/`, `uv run ty check src/`, and `uv run pytest -q` are each equal to
  or better than the 1.1 baseline; and the graph-shape test names every node including the two added
  here.
- [ ] 5.2 Re-verify the named blast radius:
  `tests/unit/documents/{test_rag,test_fusion,test_hybrid_search_failure}.py`,
  `tests/unit/shared/langgraph_layer/{test_retrieval_retry_shape,test_reranker_singleton}.py`,
  `tests/unit/test_feature_error_exhaustiveness.py`.
  **Proof:** each shows zero failures.
- [ ] 5.3 Re-run the tier-1 retrieval eval and record the delta against the 1.3 baseline, **whether it
  improved or regressed**.
  **Proof:** `review.md` carries both numbers and the golden-set version they were scored against.
