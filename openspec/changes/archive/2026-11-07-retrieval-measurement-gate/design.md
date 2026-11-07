> Change class: **M**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

Verified on disk (2026-10-31):

- `evals/golden/legal_retrieval_v2.jsonl` holds 105 labeled rows (plus manifest and chunking notes); the retired plan's G1 targets 150–300. Wave-3 annotation infrastructure (`evals/annotations/`, `ADJUDICATION-RUNBOOK.md`) and the provenance validator (`app/ai/evaluation/validate`) already exist — the engine has fuel lines, just not enough fuel.
- `.github/workflows/test.yml:95` validates provenance only. The harness computes `recall_at_k`, `precision_at_k`, `reciprocal_rank`, `ndcg_at_k` (`src/app/ai/evaluation/runner.py` importing from `.metrics`) and emits per-query rows and aggregates — nothing gates on them.
- The ask path's branch registry (`src/app/features/documents/application/queries/documents.py:~120-129`) documents that "Adding branch #4 is a registration below — the gather/zip/fusion does not change", with per-leg weights travelling on `retrieve_fused`'s `weights` argument; the current third-leg call site passes `[plan.keyword_weight, plan.vector_weight, RRF_WEIGHT_TRIGRAM]` (~980).
- Graphiti reads live in `src/app/retrieval/graphiti/` (`fetch_relationship_edges`, grouping, client); today retrieval uses Graphiti only as an evidence-policy filter, not as a ranked branch.
- The corpus records multi-contract (SRC-04 with 5 boundaries) and scanned-OCR (SRC-05) profiles that G1's matrix must cover.

## Goals / Non-Goals

**Goals:**

- Retrieval claims become measurements: CI fails on a recall/nDCG regression.
- The corpus covers the matrix the plan named, with provenance intact.
- Graphiti enters the fusion as a registration, default-off, with a weight the harness produced.

**Non-Goals:**

- No change to the three mandatory fused modes or their provisioning-failure contract (`document-retrieval-schema`).
- No default-on flip for the graph branch in this change.
- No reranking, contextual retrieval, small-to-big, or feedback work (G2/G5/G7/G8 stay unowned here).
- No serving-implementation change to the ask path (`mount-retrieval-graph` owns that; this change's registration is expressed against the branch registry contract, which survives the cutover).

## Decisions

### 1. Gate on the offline harness first; live scoring stays explicit

The CI gate runs the deterministic path (recorded retrieval artefacts, no database/provider), matching the `retrieval-evaluation` requirement that the default selection stays offline. The live-run proof requirement ("a green offline suite is not evidence the harness is connected") stays satisfied by the existing explicit live harness invocation; the CI gate is a *quality* gate on top, not a wiring proof. Alternative rejected: gating CI on a live seeded-corpus run — flaky infrastructure in the default path, and the offline gate already moves the project from provenance-only to quality-gated.

### 2. The weight is measured once, then stored — not recomputed per deploy

The graph branch's RRF weight lands as a named constant produced by a harness report on the grown corpus (report cited in the change that flips default-on). Deriving weights at runtime from corpus statistics was rejected: it makes the fusion non-deterministic across deploys and untestable.

### 3. Registration against the registry contract, cutover-agnostic

The branch registers wherever the ask path's branch registry lives at implementation time (inline today, graph nodes after `mount-retrieval-graph` cuts over — that change reserves the registration point explicitly). The spec binds to the registry contract (name + runnable + weights plumbing), not to a file location, so the cutover does not invalidate this change.

### 4. Corpus growth follows the existing annotation discipline

New rows carry the same provenance/annotation fields the validator enforces; wave-3 style dual-engineer annotation for non-deterministic rows; SME-pending rows marked per the runbook. Quantity never bypasses provenance — a 150-query corpus with `awaiting_sme_expansion` flags is worth more than 300 unreviewed rows.

## Risks / Trade-offs

- [Offline gate measures stale artefacts, not the live path] → Accepted trade-off; the live-run proof remains an explicit, regularly-run harness invocation, and the baseline requirement forces improvement claims to cite reports either way.
- [Corpus growth stalls below 150] → The gate threshold is set against whatever corpus version is current; the requirement is the target and the version discipline, not a hard CI dependency on the full 150 on day one.
- [Graph branch returns poor rankings] → That is what measurement is for: default-off means the risk is a measured negative result recorded in a report, not a user-facing regression.
- [Per-branch attribution leaks query semantics into logs] → Attribution rows carry counts and identifiers, not query text; tenant scoping applies to the branch read like every other branch.

## Migration Plan

1. Grow the corpus in annotated batches (matrix first, bulk second); bump the corpus version; provenance validator green throughout.
2. Add the CI gate step after the provenance step; set initial thresholds from a recorded baseline report of the current corpus version.
3. Add per-branch attribution rows to the runner/report.
4. Register the graph branch behind the default-off setting; add the harness scoring of the gated branch; record the measured weight in the report.

## Open Questions

None blocking. The initial recall/nDCG threshold values are deliberately set by a baseline report at implementation time rather than invented here — inventing them would repeat the folklore the gate exists to kill.
