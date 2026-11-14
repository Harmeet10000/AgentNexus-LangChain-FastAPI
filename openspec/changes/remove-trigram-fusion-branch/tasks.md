> Change class: **S**. Tasks are the implementation record; the change is complete.

- [x] 1.1 Remove `_run_trigram_branch` + the trigram entry from `RETRIEVAL_BRANCHES` (bm25, vector remain). Verify: `rg trigram src/app/features/documents/application/queries/documents.py` → prose only.
- [x] 1.2 Hybrid node weights → two-element (`[plan.keyword_weight, plan.vector_weight]`), both call sites; `RRF_WEIGHT_TRIGRAM` deleted from constants + the example. Verify: `rg RRF_WEIGHT_TRIGRAM src/` → zero hits.
- [x] 1.3 `repo.trigram_search` docstring: unfused note with the E4 evidence.
- [x] 2.1 Parity re-record: stub loses trigram; weights [0.6, 0.4]; expectation → `["b", "a", "c"]`. Verify: parity suite 55 green.
- [x] 2.2 Normalization drift fixed: `_normalize_plan` total-normalizes only (query_type forcing removed) — the two-door divergence the trigram constant masked. Verify: parity suite green after the fix (the trigram removal unmasked it).
- [x] 3.1 Gates: ruff/format clean; ty (2 pre-existing diagnostics only); pytest tests/unit/ green; lint-imports 5/5.
