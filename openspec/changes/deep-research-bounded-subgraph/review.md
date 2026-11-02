> Change class: **M** (normal checklist). Role: reviewer, not author. Read proposal.md, specs/, design.md before completing anything.

## Completeness

- [x] Each spec requirement has at least one scenario (deep-research-subgraph: 5 requirements / 8 scenarios; agent-saul-delegation: 1 requirement / 2 scenarios)
- [x] Edge and error cases are named: no runtime context, model-supplied scope, budget exhaustion, no-evidence, stale handle, interrupt attempt
- [x] Delta operations correct: `deep-research-subgraph` is a new capability; `agent-saul-delegation` deltas are ADDED requirements (its existing handoff/HITL contract untouched)

## Correctness

- [x] Requirements are observably testable — every scenario names an observable envelope, refusal, or propagation assertion
- [x] No implementation details in specs (`DeepResearchState`, ledger class names live in design/tasks only)
- [x] The "one visible tool" requirement is phrased as toolset identity, not as a binding mechanism

## Verification matrix

| Requirement | Proof |
|---|---|
| Server-owned context only | V6 propagation test (model-supplied scope never enters subgraph state) + deny-without-runtime test |
| Hard-bounded, typed statuses | budget-exhaustion test returns `budget_exhausted` with partial handles; no-evidence test returns `no_evidence` |
| Run-scoped handles | cross-run handle-membership refusal test, including on a resumed thread |
| No interrupt inside | full-run test asserts exactly one terminal envelope, zero interrupts |
| One visible tool | toolset-identity assertion in the plan-level execution test |
| Risk disclosure parity | factory binding test + disclosure assertion on risk output with an incomplete document |

## Decisions to defend in review

1. **No checkpointer inside the subgraph** (design Decision 1) — challenge with "what does the user lose on resume?" The answer should be "a re-invocation, bounded and cheap", not "the research".
2. **Budget via Saul's ledger, not a local counter** (Decision 3) — a local counter would mint a fresh allowance per call; confirm the ledger is handed in.
3. **Superseding harden 6.9 rather than forking it** (Decision 6) — the harden change's umbrella status must stay intact; this change's tasks must visibly map onto 6.9's intent.

## Cross-change interactions

- `harden-agent-saul-evidence-pipeline`: task 6.9 refined (search stage), task 6.11's ledger consumed, task 6.10's memory-writer gating explicitly out of scope here.
- `agent-tool-capability-enforcement`: the wrapper's deny-without-context refusal reuses the executor-enforcement semantics; authority tags mark the wrapper read-only.
- `mount-retrieval-graph`: pattern precedent (compiled subgraph, no checkpointer, explicit state mapping); no shared code.
- `saul-validation-adoption` (D7): the subgraph's synthesize stage is a future validation-seam candidate, deliberately untouched here.
