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

## Pending — the graph shape, before and after (tasks 1.2 and 5.1)

- Node set before: _pending_
- Edge pairs before: _pending_
- Node set after: _pending_ (expected: two additional nodes)
- Edge pairs after: _pending_ (expected: four additional edges)

---

## Pending — the tokenizer-versus-word-count divergence (task 4.2)

- Fixture text: _pending_
- Word count: _pending_
- Token count: _pending_
- Divergence: _pending_ (the test requires more than twenty percent, to prove the accounting change is
  observable rather than theoretical)

---

## Pending — the reranker degradation evidence (task 2.3)

- Candidates in: _pending_
- Candidates out when the provider raises: _pending_ (expected: `limit`, in fused order)
- Confirmation no exception escaped: _pending_

---

## Pending — the retrieval delta (task 5.3)

- Baseline tier-1 aggregates: _pending_
- Post-change aggregates: _pending_
- Delta, recorded regardless of direction: _pending_
- Golden-set version both were scored against: _pending_

An absent improvement here is a finding about the golden set, not a failure of the change. Read it
alongside the accepted cost at the top of this file before drawing a conclusion from it.
