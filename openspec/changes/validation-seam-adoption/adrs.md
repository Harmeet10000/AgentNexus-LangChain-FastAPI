# ADRs — validation-seam-adoption

## ADR-1: Parity is recorded behaviour, byte-identical

**Decision:** each seam's parity suite replays recorded cases and asserts byte-identical outputs.
**Alternatives:** equivalence arguments per case (rejected: drift hides inside "equivalent"), snapshot-everything (rejected: unreviewable noise).
**Consequence:** the guarantee is checkable; any behavioural change requires a spec delta, not a test edit.

## ADR-2: Adoption is wiring, not rewrites

**Decision:** existing validators become pipeline predicates with logic movement minimized; the diff reads as re-plumbing.
**Alternatives:** rewriting validators to "pipeline-native" shapes (rejected: rewrites void the parity guarantee they were supposed to accompany).
**Consequence:** reviewable diffs; regression tests carry the correctness burden.

## ADR-3: The heuristic metadata merge dies, honestly

**Decision:** failed metadata validation persists an explicit state; no unvalidated merge fallback.
**Alternatives:** keep the heuristic merge as a degraded mode (rejected: it launders unvalidated facts into the record — the exact dishonesty the fidelity work exists to remove).
**Consequence:** downstream consumers gain a truthful distinction; absence becomes information.

## ADR-4: The ask loop is adopted, not redesigned

**Decision:** grading becomes the semantic evaluator and the rewrite loop becomes bounded feedback-carrying retry; the two-iteration bound and every observable remain fixed.
**Alternatives:** "improve the loop while we're here" (rejected: quality work without an eval gate is guesswork; the parity contract exists precisely to forbid it here).
**Consequence:** the loop gains vocabulary and uniformity; quality changes belong to the measurement work.
