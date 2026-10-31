> Change class: **M**. Tasks are ordered by dependency: corpus → gate → attribution → branch registration. No production feature implementation beyond the registered branch (default-off) is included in this proposal.

## 1. Golden corpus growth (G1)

- [ ] 1.1 Define the coverage matrix from the corpus's recorded families/jurisdictions/difficulties/query styles (including SRC-04 multi-contract and SRC-05 scanned-OCR profiles) and record it beside the manifest; verify every existing row maps to a matrix cell and the matrix names the target 150–300.
- [ ] 1.2 Grow `evals/golden/legal_retrieval_v2.jsonl` in annotated batches of 10–25 rows mined from real corpus uploads, each row carrying the full provenance/annotation schema (dual-engineer for non-deterministic rows; SME-pending marked per `evals/ADJUDICATION-RUNBOOK.md`); verify `uv run python -m app.ai.evaluation.validate evals/golden/legal_retrieval_v2.jsonl` stays green after every batch and the corpus version is bumped.
- [ ] 1.3 Reach at least 150 labeled queries with every matrix cell covered by at least one query; verify the manifest declares the count and the composition check (families, multi-contract, scanned-OCR) passes.

## 2. CI recall/nDCG gate (G1)

- [ ] 2.1 Produce a recorded baseline report for the current corpus version into `evals/reports/` (recall@k, MRR, nDCG, per-query rows, commit id, corpus version); verify two runs on the same tree produce identical values.
- [ ] 2.2 Add the CI step in `.github/workflows/test.yml` after the provenance-validation step (line ~95) that runs the deterministic harness and fails below the recorded thresholds, naming metric, observed value, and threshold on failure; verify a deliberately degraded expectation in a scratch commit fails the gate, then revert.
- [ ] 2.3 Cite the baseline report from the workflow step (comment or threshold file) so a threshold change must confront the report that produced it; verify the provenance-only step is unchanged.

## 3. Per-branch attribution in the harness (G1, supporting G4)

- [ ] 3.1 Extend the runner/report (`src/app/ai/evaluation/`) with per-branch attribution rows (branch name, ranked identifiers contributed, aggregate metric deltas with/without the branch) without altering the existing aggregate fields; verify an unchanged input produces unchanged retrieval metrics plus the new attribution section.

## 4. Graphiti as the fourth fused branch (G4, default-off)

- [ ] 4.1 Add a retrieval-shaped Graphiti read in `src/app/retrieval/graphiti/` returning ranked, tenant-scoped chunk identifiers in the branch output shape (distinct from the evidence-policy filter `_graphiti_filter_chunk_ids`, which stays untouched); verify the read returns the same identifier space the other branches fuse over and contributes nothing when the graph has no reachable relationships for the query.
- [ ] 4.2 Register the branch in the ask path's branch registry (name paired with runnable; per `documents.py:~120-129` the gather/zip/fusion requires no change) guarded by a settings flag defaulting off, with the fusion weight as a named constant sourced from a harness report; verify the default-off path runs the three mandatory modes only, the disabled state does not surface as a provisioning failure, and no second fusion implementation was introduced.
- [ ] 4.3 Add the enabled-path behaviours: a branch failure fails the request naming the graph branch, and execution diagnostics record the branch's contribution (executed flag, identifiers contributed, weight applied); verify both with fakes and that all four baseline gates (`uv run ruff check --fix src/`, `uv run ty check src/`, `uv run pytest`, `ast-grep scan src/`) are green.
- [ ] 4.4 Score the gated branch on the grown corpus with the harness and record the measured weight and contribution deltas in a report under `evals/reports/`; verify the report exists, is cited from the branch's constant definition, and the branch remains default-off — the flip is a future change citing this report.
