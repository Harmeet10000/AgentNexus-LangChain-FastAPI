# Design — rag-eval-harness

## Two tiers, and why the split is the design

Evaluation of a RAG system splits cleanly along one line: **can the metric be computed from
identifiers alone?**

| Tier | Metrics | Inputs | Cost | Determinism |
|---|---|---|---|---|
| 1 — retrieval | recall@k, MRR, nDCG@k, precision@k | ranked id list + expected id set | none | exact |
| 2 — generation | faithfulness, answer relevancy, context precision as judged | generated answer + retrieved context + a judge model | per-call, real | approximate, drifts with the judge |

Tier 1 is arithmetic. It can run on every commit, in the default gate, with no key and no network, and
two runs on the same input produce identical numbers. Tier 2 cannot do any of that.

The decision is to **build tier 1 now and specify tier 2's seam**, rather than adopting a framework
that routes both through a model. The user ruled for both, in that order.

### Why not adopt `ragas` as the primary gate

Rejected for the *primary* gate; not rejected as a later layer.

- It routes even retrieval metrics through an LLM, which makes the fast gate network-bound and
  non-deterministic — the opposite of what a regression gate needs.
- It is a second evaluation stack, against this cluster's explicit posture on dependency weight (the
  same posture that removes torch in `ingestion-chunking`).
- The four rank metrics needed here are a few dozen lines of pure arithmetic with property tests. The
  dependency buys nothing at tier 1.

The seam requirement — "Answer-quality judging is separable from retrieval scoring" — is what keeps
this a deferral rather than a rejection. When the judged layer lands, it attaches as a report section
and changes no retrieval number.

## Why the runner takes an injected retriever

`run_retrieval_eval(*, queries, retrieve)` where `retrieve` is an async `Protocol`, not a service, not
a session, not a repository.

This is the whole reason tier 1 can be pure. The metric path never learns what retrieval is. It makes
the unit tests exact — a fake retriever with a hand-written ranking produces aggregates you can
compute on paper — and it means the harness can score `retrieval-sql`'s new path and the old one in
the same run, by passing two different callables.

The constraint that makes it real is the import ban: `src/app/shared/evaluation/` imports nothing from
`features/`. A single convenience import would collapse the seam.

## The liveness probe, and why it is a requirement rather than a nicety

Every offline proof in this change runs against a fake. That is the design working correctly — and it
means a fully green suite is *exactly consistent* with a harness that reaches nothing at all. The
failure would be silent and would survive to the first change that cites a number from it.

So the probe is specified as behaviour, not left as a task: a live run must return identifiers that
exist in the chunk store, and an empty result is classified as a **wiring failure rather than a score
of zero**. That distinction is the point. A score of zero is a legitimate, publishable measurement of
bad retrieval. An empty identifier set means nothing ran.

## Why the golden set is versioned

Two reports are comparable only if they scored the same questions. A golden set that grows — and this
one will, it ships deliberately small pending subject-matter expansion — makes an unversioned
before/after comparison quietly wrong: recall can drop because harder questions were added, not
because retrieval regressed.

Carrying the version in the report is one field and removes the whole class of error.

## What the seed corpus is honest about

It is small and it is not expert-authored. The requirement is coverage of all four legal document
families, not statistical power. A four-family stub that exists beats a comprehensive set that is
still being negotiated, because the changes that need a before-number are ready now.

`review.md` records this as an accepted cost with a named follow-up, rather than presenting the stub
as a finished artifact.

## Ordering: why this runs second, immediately after the tree repair

Nothing collects until the tree imports, so it cannot run first. Everything that follows it makes
retrieval claims, so it cannot run later. That is the entire ordering argument.
