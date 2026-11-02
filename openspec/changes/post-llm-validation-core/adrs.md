# ADRs — post-llm-validation-core

## ADR-1: Unavailability is ESCALATE, never a pass

**Decision:** an evaluator that is unconfigured or raises yields `verdict="unavailable"` and decision ESCALATE.
**Alternatives:** skip-the-gate-on-outage (rejected: the vacuous-success defect class PRs #109/#110 removed), degrade-to-schema-only ACCEPT (rejected: semantic grounding silently lost).
**Consequence:** an evaluator outage degrades to human/terminal handling, never to unexamined output.

## ADR-2: The outcome is frozen, generic, and validator-guarded

**Decision:** `ValidationOutcome[T]` is a frozen Pydantic model enforcing value-iffy-ACCEPT at the model level.
**Alternatives:** a dataclass with conventions (rejected: the invariant would be a comment), a non-generic dict (rejected: seams would re-type it, rebuilding the fragmentation this layer exists to end).
**Consequence:** outcomes can be persisted next to seam records; the invariant cannot be quietly violated.

## ADR-3: Budget is a port, not a dependency

**Decision:** the pipeline accepts an optional ledger protocol and charges it; it never creates or extends one.
**Alternatives:** importing a concrete ledger (rejected: breaks the `app/ai` boundary), pipeline-internal counting (rejected: the fresh-allowance minting failure mode is exactly what a port prevents).
**Consequence:** documents' extraction ledger and Saul's budget ledger both plug in; exhaustion is always a typed ESCALATE.

## ADR-4: Routing is a caller decision via a one-method sink protocol

**Decision:** three stock sinks (terminal failure, human review, dead letter) are thin adapters the caller wires; the pipeline only decides ESCALATE + reason.
**Alternatives:** routing inside the pipeline by reason code (rejected: invents a fourth escalation mechanism — the fragmentation this layer removes).
**Consequence:** the same ESCALATE goes to HITL in one seam and to a terminal state in another, on one vocabulary.

## ADR-5: Mechanical predicates are pure functions composed into evaluators

**Decision:** `MechanicalEvaluator` composes pure predicate callables; `CompositeEvaluator` gates the model call behind them.
**Alternatives:** monolithic per-seam evaluator classes (rejected: the existing validators wouldn't map without rewrites).
**Consequence:** adoption is a mapping exercise (D7: "vocabulary, not rewiring"), and the happy path gains zero model calls.
