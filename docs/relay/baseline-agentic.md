# Baseline — agentic-retrieval tier-1 (task 1.3, recorded 2026-09-14)

Tier-1 retrieval baseline from `rag-eval-harness`, scored by the real
`DocumentQueryService` integration test
(`tests/integration/evaluation/test_live_retrieval.py`, requires_db) — not the
retired fake-echo path described in `review.md`'s verification correction.

- Golden-set version: `legal_retrieval_v1`
- Commit identifier of the cited report: `0653503845c2`
- Baseline aggregates (`evals/reports/baseline.json` before the 5.3 re-run):
  recall_at_k 1.0, reciprocal_rank 1.0, ndcg_at_k 1.0, precision_at_k 1.0
  (4/4 golden queries retrieve their expected chunk at rank 1)
