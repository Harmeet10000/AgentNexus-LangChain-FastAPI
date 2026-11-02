> Change class: **M** (normal checklist). Role: reviewer, not author. Read proposal.md, specs/, design.md before completing anything.

## Completeness

- [x] Each spec requirement has at least one scenario (portfolio-traversal-tools: 3 requirements / 6 scenarios; agent-saul-delegation: 1 requirement / 1 scenario)
- [x] Edge and error cases named: binding before gate, no-result vs empty-success, scope-widening arguments, missing runtime context, cap overflow
- [x] Delta operations correct: new `portfolio-traversal-tools` capability; `agent-saul-delegation` delta is an ADDED requirement consistent with the D3/D5 deltas already pending on that capability

## Correctness

- [x] Requirements observably testable — binding failure, envelope bounds, scope disclosure, cap arithmetic; no Cypher or class names in specs
- [x] The gate requirement is phrased as observable binding behaviour, not as an internal flag
- [x] "Raw clause text SHALL NOT be returned" is testable at the envelope boundary

## Verification matrix

| Requirement | Proof |
|---|---|
| Gated binding | graph-build test: portfolio binding without edges fails naming the gate |
| Honest empty states | no-result test asserting distinct status vs successful traversal |
| Handle-bound envelopes | envelope test: matches carry provenance + run-scoped handle; size bound holds |
| Server-owned scope | scope-widening argument test; no-context refusal per executor contract |
| Swap under cap | toolset-identity test before/after a swap; projection-consistency suite green |

## Decisions to defend in review

1. **Binding-time gate over runtime condition** (design Decision 2) — challenge with "what if edges arrive later per-document?" The gate is corpus-level; per-document incompleteness is the `ingestion_incomplete`-style disclosure, not the gate.
2. **Three named tools over one generic query tool** (Decision 4) — the model-flexibility argument should lose to the enforceable-contract argument.
3. **Swap-don't-add as a test** (Decision 3) — confirm the reviewer accepts that the specific tool swapped out is an apply-time, reviewed decision rather than a spec constant.

## Cross-change interactions

- Gate dependency: retired plan Phase D / harden Phase 4 relationship work (the unblocking change must reference this change's gate check).
- `deep-research-bounded-subgraph`: run-scoped handle contract consumed unchanged.
- `tool-capability-catalog`: registration, projection-consistency tests, and (future) egress classification consumed; no portfolio tool is egress-classified.
- `mount-retrieval-graph`: no overlap — that engine answers from retrieved context; traversal supplies structure for it to reason over.
