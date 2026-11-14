> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**S** — one branch removed from the served fusion registry, its weight constant deleted, and the recorded two-door parity expectation re-recorded to the new contract. Owner directive 2026-11-14.

## Why

The owner directed the trigram leg's removal from the served RRF fusion. The E4 branch sweep (evals/reports/phase8_experiments.json, bm25-force-merge-e4) is the evidence: trigram is the weakest leg solo (recall@10 0.5097 vs bm25 0.5806), and the fused bm25+trigram pair traded −0.32 pts recall for +0.61 MRR against bm25-only — a marginal profile the live three-leg system no longer needs, with the graphiti leg (measured +2.58 recall) as the designated fourth-leg candidate and the vector leg returning on the Gemini quota reset.

## What Changes

- The served fusion comprises the **bm25 and vector branches**; the trigram leg is **unfused** (removed from `RETRIEVAl_BRANCHES` and the hybrid node's weight plumbing; `RRF_WEIGHT_TRIGRAM` deleted).
- `repo.trigram_search` is retained as an unfused repository capability (the offline harness sweep and the live scorer's trigram branch still exercise it); its docstring records the removal.
- The two-door parity expectation is re-recorded to the two-branch fused order (`["b", "a", "c"]` on the parity fixture) — both doors agree, which was and remains the suite's actual contract.
- The plan-weight normalization drift surfaced by the removal is fixed: the graph's `_normalize_plan` no longer forces weights by `query_type` — it total-normalizes only, identical to the inline plan contract (parity drift item 9, unmasked when the trigram constant stopped masking it).

## Scope / Non-Goals

In scope: the fusion-registry removal, the weight plumbing, the parity re-record, the hybrid-retrieval-ranking spec delta.

Out of scope (explicit): the offline harness's trigram sweep support (it stays — the sweep is the measurement that justified the removal); `repo.trigram_search` itself (retained, unfused); any vector-leg work (quota-gated); the graphiti leg (measured, registration unchanged).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `hybrid-retrieval-ranking`: the served fusion comprises the bm25 and vector branches; the trigram leg is unfused with the measured E4 evidence recorded; per-plan adaptive weights stay, normalized by total on both doors.

## Impact

Files: `application/queries/documents.py` (branch registry), `retrieval/graph/nodes.py` (weights + normalization alignment), `domain/constants.py`, `examples/policy_examples.py`, `tests/unit/shared/langgraph_layer/test_hybrid_fused_parity.py` (re-record). Gates: 2492 unit tests passing (2 pre-existing failures now fixed — 2499 at last full run); parity suite 55 green.
