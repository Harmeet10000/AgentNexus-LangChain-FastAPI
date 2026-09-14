# Review — agentic-retrieval

Sections marked **accepted cost** are known limitations recorded deliberately. Sections marked
**pending** are filled during implementation by the task that names them.

---

## Measured before planning — the repo is ahead of the reference architecture

The task named a published engineering blog as the target. Reading it against the tree found that the
graph already has a query analyzer that rewrites, decomposes, routes and sets per-leg weights; hybrid
retrieval; a reranking funnel at twenty into five; and a context grader with a **cyclic** edge back to
the analyzer, capped at two iterations.

The reference pipeline is explicitly **acyclic**.

So the honest scope of this change is much smaller than the brief implies, and one part of the brief —
"build the agentic loop" — would have been a regression. What remains is four specific gaps: no corpus
narrowing, no post-processing on the graph path, a word-count budget, and a reranker that requires
torch.

This is recorded prominently because the gap between the brief and the finding is the most useful thing
a later reader can know about this change.

---

## Accepted cost — the allowlist's benefit cannot be measured directly yet

Task 3.1 adds corpus narrowing. Its value is that retrieval searches a smaller, more relevant corpus —
which shows up in tier-1 recall only when the golden set contains queries whose answers sit in
documents a naive search would rank below the cut.

The seed golden set is deliberately small. It may not contain such a query, in which case the allowlist
will show **no measurable improvement** while still being correct.

Accepted, with two mitigations. Task 5.3 records the delta whether it improved or regressed, so an
absent signal is documented rather than interpreted. And the allowlist's *other* benefit — collapsing
the approximate-search recall cliff — is measurable independently on the `retrieval-sql` side, where
recall for a narrow filter is compared against exact nearest-neighbour ground truth.

---

## Accepted cost — a network dependency enters the retrieval path

The hosted reranker replaces a local model with a provider call. Retrieval acquires a network
dependency it did not have.

Accepted because degradation is specified as behaviour (ADR-006): the provider failing returns the
fused order truncated to the requested count, which is a three-branch reciprocal-rank fusion and
therefore a genuinely reasonable ordering, not an arbitrary one.

The residual risk is that the degradation is **silent from the caller's perspective** — a persistent
outage produces slightly worse results indefinitely, with no failure to notice. There is no metric in
this cluster that would catch it. A named follow-up: emit a counter on the degradation path so a
sustained fallback is observable.

---

## Accepted cost — two reranker implementations coexist for a window

This change adds the hosted implementation and makes it the default; `ingestion-chunking` deletes the
local one. Between those two changes landing, both exist in the tree.

Accepted because the alternative puts a retrieval-quality decision (which reranker) inside a dependency
change, or a dependency decision inside a retrieval change — and in either case nobody would find it
later. Task 2.4's Proof asserts the local implementation is **still present**, which is the unusual
shape that makes the split explicit rather than accidental.

---

## Note — why the widening requirement is separate from the narrowing one

Recorded because they look like one feature and are not.

Narrowing is an optimisation: worst case, it helps less than hoped. **Narrowing without widening is a
new failure mode**: a query whose answer sits outside the allowlist becomes permanently unanswerable,
and it fails invisibly — retrieval returns confident, well-ranked, wrong-corpus results, and the grader
cannot distinguish "the corpus does not contain this" from "I searched the wrong part of it".

That is why the spec carries them as two requirements, and why task 3.3 asserts termination separately
from task 3.2's widening. Without the separation, widening could turn a bounded loop into an unbounded
one by a route the widening test would not exercise.

---

## Recorded — the graph shape, before and after (tasks 1.2 and 5.1)

Pinned by `tests/unit/shared/langgraph_layer/test_retrieval_graph_shape.py`.

- Node set before: `__start__`, `__end__`, `query_analyzer`, `graph_neo4j`, `hybrid_postgres`,
  `reranker`, `context_grader`, `generate` (8 including sentinels)
- Edge pairs before (10): `__start__→query_analyzer`, `query_analyzer→graph_neo4j`,
  `query_analyzer→hybrid_postgres`, `query_analyzer→generate`, `graph_neo4j→hybrid_postgres`,
  `hybrid_postgres→reranker`, `reranker→context_grader`, `context_grader→query_analyzer`,
  `context_grader→generate`, `generate→__end__`
- Node set after: before, plus `source_identifier` and `post_process` (**+2 nodes**, as expected)
- Edge pairs after (12): `__start__→query_analyzer`, `query_analyzer→source_identifier`,
  `query_analyzer→generate`, `source_identifier→graph_neo4j`, `source_identifier→hybrid_postgres`,
  `graph_neo4j→hybrid_postgres`, `hybrid_postgres→reranker`, `reranker→post_process`,
  `post_process→context_grader`, `context_grader→query_analyzer`, `context_grader→generate`,
  `generate→__end__`
- Measured edge delta: **+2**, not the brief's "+4". The identifier mediates only the two retrieval
  arms (analyzer→generate stays direct), and post_process replaces one edge with two — net +2. The
  topology deviation is deliberate so cached/trivial queries skip identification.

---

## Recorded — the tokenizer-versus-word-count divergence (task 4.2)

From `tests/unit/documents/test_rag.py::test_assemble_rag_context_drops_sections_at_the_token_boundary`:

- Fixture text: two 10-word sections (`" ".join(f"w{i}" for i in range(...))`)
- Word count per section: 10
- Token count under the injected counter (`3 * len(text.split())`): 30
- Divergence: **200%** (well above the required >20%). Budget 45 keeps the first section and drops
  the second — a word-count budget of 45 would have kept both.

---

## Recorded — the reranker degradation evidence (task 2.3)

From `tests/unit/shared/langgraph_layer/test_hosted_reranker.py::test_hosted_reranker_failure_degrades_to_fused_order`:

- Candidates in: 20 (`c-0` … `c-19`)
- Candidates out when the provider raises: 5 (`c-0` … `c-4`) — `limit`, in fused order
- Confirmation no exception escaped: the await completed and returned the truncated list

---

## Recorded — the retrieval delta (task 5.3)

**Verification correction (2026-09-13): this delta is invalid and must be rerun.** The referenced
`rag-eval-harness` liveness test used a fake service that echoed identifiers read from the database;
it never invoked production retrieval. Consequently the before and after zeros measure the same fake
path. Tasks 1.3, 5.3, and the aggregate gate in 5.1 are reopened until the real
`DocumentQueryService` integration test produces a valid baseline and post-change report.

The `No local framework is required` scenario is also only provisionally satisfied: this change keeps
`sentence_transformers` as required by task 2.4, and `reranker.py` imports it eagerly. The scenario
becomes true only when `ingestion-chunking` removes the local implementation and dependency; verify an
import and hosted rerank after that removal before closing 5.1.

Re-ran `uv run pytest -m requires_db tests/integration/evaluation/test_live_retrieval.py -q`
(1 passed, 30.45s). Report rewritten to `evals/reports/baseline.json`.

- Baseline tier-1 aggregates (task 1.3 / `rag-eval-harness` 4.1, commit `0653503845c2`):
  recall_at_k 0.0, reciprocal_rank 0.0, ndcg_at_k 0.0, precision_at_k 0.0
- Post-change aggregates (timestamp `2026-09-13T10:11:18Z`): recall_at_k 0.0, reciprocal_rank 0.0,
  ndcg_at_k 0.0, precision_at_k 0.0
- Delta, recorded regardless of direction: **0.0 on every metric**
- Golden-set version both were scored against: `legal_retrieval_v1`

## Re-recorded — the retrieval delta on the real service (task 5.3, 2026-09-14)

The zeros above measured the retired fake-echo path and are superseded (kept for audit).
Re-ran `uv run pytest -m requires_db tests/integration/evaluation/test_live_retrieval.py -q`
(1 passed, 39.09s) through the real `DocumentQueryService`:

- Baseline tier-1 aggregates (`docs/relay/baseline-agentic.md`, commit `0653503845c2`):
  recall_at_k 1.0, reciprocal_rank 1.0, ndcg_at_k 1.0, precision_at_k 1.0
- Post-change aggregates (timestamp `2026-09-14T05:38:14Z`): recall_at_k 1.0,
  reciprocal_rank 1.0, ndcg_at_k 1.0, precision_at_k 1.0
- Delta, recorded regardless of direction: **0.0 on every metric** — the graph changes
  (identifier, post_process, hosted reranker) preserve tier-1 retrieval exactly
- Golden-set version both were scored against: `legal_retrieval_v1`
- The pre-run report file was restored byte-identical after the re-run.

An absent improvement here is a finding about the golden set, not a failure of the change. Read it
alongside the accepted cost at the top of this file before drawing a conclusion from it.
