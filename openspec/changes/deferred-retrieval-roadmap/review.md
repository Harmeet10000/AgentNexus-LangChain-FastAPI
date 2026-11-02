> Change class: **M** (normal checklist). Role: reviewer, not author. Read proposal.md, specs/, design.md before completing anything.

## Completeness

- [x] Each spec requirement has at least one scenario (contextual-chunk-retrieval: 2 req / 4 scenarios; retrieval-feedback-loop: 2 req / 3 scenarios; hybrid-retrieval-ranking: 2 req / 3 scenarios; agentic-retrieval-loop: 3 req / 4 scenarios)
- [x] Edge and error cases named: disabled-vs-empty preamble, provider-call proportionality, graphless corpus, participating-but-empty branch, mid-ask budget exhaustion, alert-before-kill, unreviewed signals, clause-precise citations under hydration
- [x] Delta operations correct: two new capabilities; two existing capabilities receiving ADDED requirements consistent with the deltas `mount-retrieval-graph` and `retrieval-measurement-gate` place on the same capabilities

## Correctness

- [x] Requirements observably testable — call-count proportionality, ranking equality for graphless corpora, decision-event naming, ledger-charge-style budget attribution, golden-set immutability under unreviewed signals
- [x] Every trigger is a landed artifact named in the proposal, not a date or a mood
- [x] "Measured before default-on" is phrased as an observable adoption decision citing a recorded comparison

## Verification matrix

| Item | Gate (checkable) | Proof at implementation |
|---|---|---|
| G2 contextual retrieval | `retrieval-measurement-gate` baseline exists | preamble-fill test + provider-call proportionality + measured-default test |
| G4 fourth branch | edges exist (harden 4.1 ✅) + baseline | registration-point test; graphless-corpus equality; weight-cites-baseline test |
| G5 small-to-big | baseline attributes context starvation | citation-precision-under-hydration test + measured adoption |
| G6 remainder | `mount-retrieval-graph` spans slice lands | budget-exhaustion degradation test + standing-alert-before-kill test |
| G7 feedback | console signals + harness ingestion | capture-completeness test + unreviewed-signals-immutability test |

## Decisions to defend in review

1. **One change, five gates** (design Decision 1) — challenge with "won't this rot?" The answer is the ledger rows plus artifact-triggered gates; a rotting roadmap is exactly what the next audit detects.
2. **G4's weight must cite a baseline** (Decision 3) — the convenience argument ("just reserve the weight constant") should lose; the whole point of G1 was to make weights measurements.
3. **G7 review-before-promotion** (Decision 5) — confirm no shortcut exists where signals alter gates without review; this is the poisoning defence.

## Cross-change interactions

- `retrieval-measurement-gate`: the harness all three retrieval gates read; this change adds no measurement of its own.
- `mount-retrieval-graph`: consumes its spans slice and reserved registration point; its non-goal deferrals (G4/G5/G7) are resolved here.
- `harden-agent-saul-evidence-pipeline`: 4.1's edges unblock G4; 9.1/9.8's events and kill criteria bound G6's alert (alert first, kill last).
- `ag-ui-copilotkit-console`: the signal source for G7.
- `docs/quarantine-ledger.md`: two rows added (preamble column, fusion registration point) — resolving either row happens through this change's implementation tasks.
