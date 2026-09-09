# ADRs — rag-eval-harness

## ADR-001 — Evaluation is two-tier: pure retrieval metrics now, judged generation metrics later

**Status:** accepted (user ruling: "both — pure core now, RAGAS layer later").

**Context.** Retrieval metrics are computable from ranked identifier lists. Generation metrics require
a judge model, and therefore cost, non-determinism, and a credential in continuous integration.

**Decision.** Ship tier 1 — recall@k, reciprocal rank, nDCG@k, precision@k — as pure functions in the
default gate. Specify the seam for tier 2 and defer its implementation to a separate change.

**Consequences.** The regression gate the other four changes depend on cannot be blocked by a flaky
judge or a missing key. Tier-2 metrics are unavailable until that later change, so no Proof in this
cluster may cite faithfulness or answer relevancy. The seam requirement is what makes the later layer
an addition rather than a rewrite.

## ADR-002 — `ragas` is deferred, not rejected

**Status:** accepted.

**Context.** `ragas` is the obvious off-the-shelf choice and was the originally-locked wording.

**Decision.** Do not adopt it for tier 1. Reconsider it for tier 2.

**Consequences.** Four metrics are implemented here as pure arithmetic with property tests, which is a
few dozen lines. In exchange, the fast gate stays offline and deterministic, and no second evaluation
stack enters the dependency set — consistent with this cluster's removal of torch. The cost is that
the tier-2 layer will have to reconcile its own metric definitions with these when it lands; the
report's separate-section requirement is what keeps that reconciliation from silently changing a
tier-1 number.

## ADR-003 — The runner takes an injected retriever protocol

**Status:** accepted.

**Context.** The harness must be testable without infrastructure, and must be able to score two
different retrieval implementations in the same run so `retrieval-sql` can produce a genuine
before/after.

**Decision.** `run_retrieval_eval(*, queries, retrieve)` with `retrieve` as an async `Protocol`.
`src/app/shared/evaluation/` imports nothing from `features/`, and this is enforced by a Proof.

**Consequences.** The metric path never learns what retrieval is, so unit tests are exact rather than
approximate. Comparing two implementations is passing two callables. A single convenience import from
`features/` would collapse the seam, which is why the ban is proven rather than documented.

## ADR-004 — An empty live result is a wiring failure, not a score of zero

**Status:** accepted.

**Context.** Every offline proof runs against a fake retriever. A green offline suite is exactly
consistent with a harness connected to nothing, and that failure would survive to the first change
that cites a number from it.

**Decision.** Specify a liveness probe as behaviour. A live run must return identifiers present in the
chunk store. An empty identifier set across all queries is classified as a wiring failure and fails
the task; it is never reported as a metric of zero.

**Consequences.** The harness cannot be declared done on offline evidence alone. It requires the live
database, which the user has ruled available. The distinction matters because a score of zero is a
legitimate publishable measurement of poor retrieval, and conflating the two would let a disconnected
harness masquerade as a bad-retrieval finding.

## ADR-005 — The golden set carries a version, and the report names it

**Status:** accepted.

**Context.** The seed corpus ships deliberately small and will grow as subject-matter expertise is
added.

**Decision.** `GoldenSet` carries a version string; every report names the version it scored.

**Consequences.** A before/after comparison across different golden-set versions is detectable rather
than silently wrong — recall dropping because harder questions were added is no longer
indistinguishable from retrieval regressing. Cost: one field, and the discipline of bumping it.
