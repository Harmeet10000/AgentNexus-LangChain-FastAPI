> Change class: **L**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

The core pipeline's contract rules are final (the core change precedes this one). Each seam arrives with existing validation of its own — the relationship pass has the most complete mechanical validation in the repo, the ask loop has the only feedback-carrying retry, and legal metadata has none beyond the schema. The adoption order (relationships → metadata → ask) is risk-ordered; each step's parity proof is its own gate.

## Goals / Non-Goals

**Goals:**
- Three seams on one vocabulary with provably unchanged observable behaviour, except the additions the specs name.
- Hand-rolled retry logic deleted, not duplicated (the ask loop is the explicit case).

**Non-Goals:**
- Any pipeline change (if a seam proves a rule wrong, the core change is amended first).
- Saul (D7's change) and the retrieval-engine cutover (`mount-retrieval-graph`).
- Quality improvement as a goal — this change makes validation *uniform and honest*; quality moves come from the evaluation work elsewhere.

## Decisions

### Decision 1 — parity by recorded behaviour, not by re-derivation
Each seam's parity suite replays recorded cases (existing grade-loop tests; drop-reason fixtures; metadata fixtures) against the pipeline-backed implementation and asserts byte-identical results. Re-deriving "equivalent" behaviour would let drift hide inside equivalence arguments.

### Decision 2 — existing validators become predicates without logic movement
The relationship validators move behind the `MechanicalEvaluator` predicate signature; their code does not change shape. This keeps the diff reviewable as *wiring* and makes the drop-reason regression test the real gate.

### Decision 3 — the metadata gate is deliberately shallow and mechanical
Parties/jurisdiction occurrence in boundary text is checkable without a model call and catches the dominant fabrication mode. Deeper semantic metadata validation is a future evaluator addition, not a reason to skip the mechanical gate.

### Decision 4 — the metadata failure state is persisted, not defaulted
`metadata_validation_failed` is a new honest terminal state for the metadata record. Consumers that today read "no metadata" must distinguish "not extracted" from "extracted but failed validation" — the state exists so the fidelity story stays truthful (the plan's "honest absence").

### Decision 5 — the ask loop adopts without touching serving order
The refactor lands behind the parity suite while the inline path is still served (`mount-retrieval-graph` cuts over separately). If the cutover lands first, the parity suite runs against the compiled graph's engine instead — the contract being parity with *the served behaviour*, not with a particular implementation.

## Risks / Trade-offs

- **Three seams, three parity suites** → the suite cost is the price of byte-identical guarantees; each is scoped to that seam's existing tests.
- **The retry in the relationship pass changes drop counts** → allowed *only* for the declared retryable classes and inside the existing share; the regression test pins the non-retryable drops as identical.
- **`metadata_validation_failed` is a new state consumers must learn** → the delta spec makes the distinction observable; consumer updates ride the same change.

## Migration Plan

Seam-by-seam, lowest risk first: relationships → metadata → ask loop. Each lands with its parity suite green before the next starts. No flag: pre-launch, and each seam's parity gate is its own rollback boundary.

## Open Questions

None.
