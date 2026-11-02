> Change class: **M** (normal checklist). Role: reviewer, not author. Read proposal.md, specs/, design.md before completing anything.

## Completeness

- [x] Each spec requirement has at least one scenario (4 requirements / 7 scenarios)
- [x] Edge and error cases named: both decisions reachable, interrupt-through-sink, mid-pipeline exhaustion, middleware-stack drift
- [x] Delta operations correct: ADDED requirements on the existing `agent-saul-delegation` capability, consistent with the D3/D5 deltas pending on the same capability

## Correctness

- [x] Requirements observably testable — suite-green-unmodified, sink-delivered interrupt, ledger charge arithmetic, middleware configuration equality
- [x] "Unchanged behaviour" is stated as a testable invariant (existing suite unmodified), not as an intention
- [x] No implementation details in specs (adapter shapes live in design only)

## Verification matrix

| Requirement | Proof |
|---|---|
| Shared-vocabulary outcomes, unchanged behaviour | existing verdict-path suite green unmodified + type-level ACCEPT/ESCALATE reachability tests |
| Findings in shared shape | finding-locator mapping test |
| ESCALATE through the sink | interrupt-through-sink test driving the real HITL flow |
| Ledger honesty | charge-arithmetic test; exhaustion → `budget_exhausted`; no fresh allowance |
| Middleware boundary | stack-configuration equality assertion test |

## Decisions to defend in review

1. **Type-level only; gate code does not move** (design Decision 1) — challenge any "while we're here" normalization; the unmodified-suite gate is the argument.
2. **The middleware boundary is a spec requirement with a test** (Decision 4) — confirm the reviewer treats it as architectural, not cosmetic: folding transport retry into the semantic layer would silently double-count attempts.
3. **Sequencing after the qna refactor** (proposal Why) — adopting the shared vocabulary before `mount-retrieval-graph`'s parity refactor would type the same seam twice.

## Cross-change interactions

- `post-llm-validation-pipeline`: hard gate — this change cannot start before the core lands (tasks reflect it).
- `mount-retrieval-graph` + `validation-seam-adoption`: sequenced after; Saul's qna inherits the refactored loop.
- `deep-research-bounded-subgraph`: its synthesize stage is noted as a future validation seam — deliberately not adopted here.
- `harden-agent-saul-evidence-pipeline`: task 6.5's gates are the fence this change maps types onto; 6.7's review semantics are consumed unchanged by the sink.
