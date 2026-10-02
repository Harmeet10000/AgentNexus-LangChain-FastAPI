# A deterministic retrieval evaluation harness

**Class: M.** New, self-contained package plus a golden-set artifact. No existing behaviour changes.

## Why

Every retrieval recommendation in this cluster is a prior, not a measurement — chunk sizes, fusion
weights, index parameters, reranking. Without a harness, the four changes that follow would ship
unmeasured and no later reader could tell which prior was wrong on this corpus. Building it first is
cheap; retrofitting it after the retrieval work means the retrieval work has no before-number.

## What Changes

- A **pure metric core** — recall@k, reciprocal rank, nDCG@k, precision@k — computed from ranked
  identifier lists alone. No model, no network, no database, no new dependency. It runs in the default
  test gate on every change.
- A **Pydantic-validated golden set** with a seed corpus covering all four legal document families
  (contracts, statutes, judgments, filings), flagged as awaiting expansion by a subject-matter expert.
  A malformed row is an expected failure carrying the offending row index; an absent file raises a
  typed exception.
- A **runner** taking an injected async retriever protocol, so the metric path is testable without any
  infrastructure and the harness imports nothing from `features/`.
- A **machine-readable report** carrying per-query rows, aggregates, and the commit identifier of the
  tree it measured.
- A **liveness probe**: an integration check, marked `requires_db`, that the harness actually reaches
  live retrieval. Without it a fully green suite is consistent with a harness wired to nothing.
- A **recorded first baseline**, produced against the live database.
- A declared **seam for judged answer-quality metrics** — faithfulness, answer relevancy — which need
  a judge model and are deliberately deferred to a later change. Specifying the seam now means that
  layer is an addition rather than a rewrite.

## Capabilities

**New Capabilities**

- `retrieval-evaluation` — a deterministic, dependency-free retrieval scoring surface with a
  versioned golden set, a recorded baseline, and a separable judged layer.

**Modified Capabilities**

None. This change **cites** restored `hybrid-retrieval-ranking` — it measures fused ranking and does
not respecify it.

## Impact

- **New code:** `src/app/shared/evaluation/` (`metrics.py`, `schema.py`, `runner.py`, `report.py`).
- **New artifacts:** `evals/golden/legal_retrieval_v1.jsonl`, `evals/reports/baseline.json`.
- **New tests:** unit and property tests under `tests/unit/shared/evaluation/` and
  `tests/property/`; one `requires_db` integration test.
- **Dependencies:** none added. `hypothesis` and the `property` marker already exist.
- **Downstream:** `agentic-retrieval` depends on this change and cites its reports. `retrieval-sql`
  and `ingestion-chunking` may cite tier-1 metrics only.
- **Not touched:** retrieval SQL, fusion, chunking, the embedder, the reranker. The harness only calls
  the existing service-level search seam.
