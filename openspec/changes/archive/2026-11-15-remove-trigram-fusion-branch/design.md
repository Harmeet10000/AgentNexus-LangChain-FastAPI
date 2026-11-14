> Change class: **S**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

The branch registry lives in one tuple (`RETRIEVAL_BRANCHES`) consumed by `retrieve_fused` → `_run_branches` on every door; the served graph's hybrid node passes per-plan weights alongside the branch set. The E4 sweep is the measurement; the owner directive is the decision.

## Goals / Non-Goals

**Goals:** a two-branch served fusion with one recorded parity contract; the normalization drift (parity item 9) fixed in the same pass because the removal unmasked it.

**Non-Goals:** deleting `repo.trigram_search` (retained, unfused); touching the offline harness's trigram sweep; any vector-leg work.

## Decisions

### Decision 1 — remove from the registry, keep the repository method
The branch dies at the registry; `trigram_search` stays as an unfused capability with a docstring note. The offline harness's sweep and the live scorer's trigram branch still exercise it — deleting the method would break the measurement tooling for a capability the E4 evidence says to keep measurable.

### Decision 2 — re-record the parity expectation rather than force the old order
The parity suite's real contract is two-door agreement; the recorded order was the three-branch artifact. The new recorded order (`["b","a","c"]` on the fixture) is derived from the two-branch RRF math (b tops both legs' intersection).

### Decision 3 — fix the normalization drift the removal unmasked
The trigram constant had masked a real two-door divergence: the graph's `_normalize_plan` forced weights by query type while the inline path total-normalized only. Aligned to total-only; plan steering belongs in the plan, not the fusion door.

## Risks / Trade-offs

- Recall@10 may shift marginally in production (offline evidence: −0.32 pts for bm25-only vs the fused pair; the live bm25+vector delta is measured at the next embedding-quota window) — the assertion suite (8.8) gates any degradation.

## Open Questions

None.
