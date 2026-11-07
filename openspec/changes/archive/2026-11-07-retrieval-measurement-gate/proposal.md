> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**M** — evaluation-corpus growth plus a CI quality gate (evals + CI wiring) and one opt-in retrieval branch registration in the documents ask path. No public API change, no migration; the new fused branch ships default-off behind a setting.

## Why

The retrieval architecture (3 fused branches, cross-encoder reranking, adaptive grade loop, span-grounded citations) is sophisticated but unmeasured: the golden corpus sits at 105 labeled queries toward the 150–300 target, CI validates only provenance (`test.yml:95`) — recall/nDCG exist in the harness (`app/ai/evaluation/runner.py`) but gate nothing — and the fusion constants (`k=60`, trigram weight) are folklore because no metric can move. Meanwhile Graphiti is a live knowledge graph with typed clause relationships that retrieval never fuses; the branch registry in `documents.py` (~120-129) explicitly reserves branch #4 as a registration. This change makes retrieval quality measurable and puts the graph into the measurement.

## What Changes

- **G1 — grow the golden corpus.** `evals/golden/legal_retrieval_v2.jsonl` grows from 105 toward 150–300 labeled queries, mined from the corpus's real uploads and covering the documented matrix: document kind × jurisdiction × difficulty × query style (obligation lookup, cross-reference, "does X allow Y", temporal), including the multi-contract and scanned-OCR profiles the corpus already flags.
- **G1 — recall/nDCG CI gate.** CI moves beyond provenance-only validation: the harness's recall@k, MRR, and nDCG metrics run against the golden corpus in CI with an explicit threshold gate, so a retrieval-affecting change that degrades measured quality fails the build instead of shipping silently.
- **G4 — Graphiti as the fourth fused branch.** The relationship-graph branch registers in the documents ask path's branch registry (the reserved registration point — gather/zip/fusion unchanged, per the registry's own contract), fused into the RRF with an explicit measured weight, behind a setting that defaults off until its weight is measured on the G1 corpus. The existing `_graphiti_filter_chunk_ids` evidence-policy filtering is unchanged for evidence policy; the new branch is additive retrieval, not a filter replacement.
- **Measured before default-on.** The branch's fusion weight is set from a harness report on the grown corpus; the setting flips to default-on only in a later change that cites that report, per the `retrieval-evaluation` baseline requirement.

## Scope / Non-Goals

In scope: corpus growth with annotation provenance intact, the CI metric gate, the Graphiti branch registration with a measured weight and a default-off setting, and the harness support for scoring the gated branch.

Out of scope (explicit): G5 small-to-big retrieval; G6 per-stage tracing/budgets (first slice owned by `mount-retrieval-graph`); G7 feedback capture; contextual retrieval (G2); the re-ingestion preconditions (the `re-ingestion-readiness` change); the HTTP cutover and qna engine work (`mount-retrieval-graph` owns the ask path's serving implementation — this change's registration must land compatibly with that cutover but does not depend on it); flipping the Graphiti branch default-on; agentic multi-hop retrieval.

## Capabilities

### New Capabilities

- `graph-fusion-branch`: the relationship-graph retrieval branch as an optional fourth fused branch in the ask path — registration in the branch registry, a measured RRF weight, default-off gating by setting, and observability of the branch's contribution.

### Modified Capabilities

- `retrieval-evaluation`: the golden corpus grows to the documented 150–300 target with coverage of the documented query matrix, and retrieval metrics become a CI gate (recall/nDCG thresholds) rather than a provenance-only check.

## Impact

Affected code: `evals/golden/legal_retrieval_v2.jsonl` + manifest (growth; provenance fields preserved per the existing validator), `.github/workflows/test.yml` (new gate step after the provenance step at line 95), `src/app/ai/evaluation/` (runner/report: per-branch attribution rows so the gated branch's contribution is measurable), `src/app/features/documents/application/queries/documents.py` (branch registry registration ~120-129, `_run_branches`/`retrieve_fused` weights plumbing), settings (`src/app/config/` or the settings module — a feature flag defaulting off), and the Graphiti read path in `src/app/retrieval/graphiti/` (a retrieval-shaped read returning ranked chunk identifiers, distinct from the evidence-policy filter). No request/response schema change. Normative input: `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md` (RETIRED — G1 and G4 definitions and the "measure before keeping" rule); `PLAN-agent-tools.md` §D6 for ask-path context.
