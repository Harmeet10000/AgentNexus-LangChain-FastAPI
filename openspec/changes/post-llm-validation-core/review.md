> Change class: **M** (normal checklist). Role: reviewer, not author. Read proposal.md, specs/, design.md before completing anything.

## Completeness

- [x] Each spec requirement has at least one scenario (9 requirements / 10 scenarios — the plan's seven contract rules plus the vocabulary and routing requirements)
- [x] Edge and error cases named: unconfigured evaluator, raising evaluator, non-retryable rejection, mid-pipeline budget exhaustion, schema-only failure, mechanical short-circuit
- [x] Delta operations correct: single new capability; no existing capability touched (adoption is deliberately elsewhere)

## Correctness

- [x] Requirements observably testable — outcome shape, invocation-or-not of the evaluator, feedback contents, ledger charges, event count
- [x] No implementation details in specs (generics, protocol names, module layout live in design only)
- [x] "Fail-closed" is stated as an observable decision (ESCALATE), not as an implementation attitude

## Verification matrix

| Requirement | Proof |
|---|---|
| Vocabulary / ACCEPT-only value | property test: no path returns ACCEPT without both gates passing |
| Schema gate first | schema-failure short-circuit test (evaluator-not-invoked assertion) |
| Unavailability fail-closed | unconfigured + raising evaluator tests → ESCALATE; the plan's DoD requires this path reachable in tests |
| Feedback-carrying retry | producer-args test (errors + findings + attempt number present) |
| Retryable-only | non-retryable immediate-escalation test |
| Budget honesty | per-reinvocation charge assertion; exhaustion → `budget_exhausted`; no-fresh-allowance test |
| Mechanical first | short-circuit test (no model call on mechanical failure) |
| One decision event | event-count == 1 assertion per outcome |
| Caller-routed sinks | two-sink equivalence test |

## Decisions to defend in review

1. **Generic frozen outcome with a validator-enforced ACCEPT-only invariant** (design Decision 1) — a convention would be violated by the first tired caller; confirm the reviewer prefers the model-level guard.
2. **Budget as an optional protocol port** (Decision 2) — the alternative (mandatory ledger) would break zero-infrastructure importability for seams that legitimately need no budget.
3. **Findings carry seam codes; the core reason set stays closed** (Decision 5) — the alternative (open reason enum) is how three vocabularies grew last time.

## Cross-change interactions

- `validation-seam-adoption`: first consumer (relationship pass + legal metadata + ask loop); its parity proofs depend on this module's contract rules being final.
- `saul-validation-adoption` (D7): gated on this change; consumes CompositeEvaluator shape, `HumanReviewSink`, and the ledger port; its tool-calling seams explicitly stay on middleware.
- `feature-error-contract`: typed-failure vocabulary consumed (sink outputs), not modified.
- lint-imports: the module's import boundary is part of this change's gate (5/5).
