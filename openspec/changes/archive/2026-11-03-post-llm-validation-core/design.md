> Change class: **M**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

The plan §3 fixes the module's shape (types, pipeline, evaluators, escalation) and its §1 boundary constraint (`app/ai` may not import `app.features`, enforced by lint-imports). The design decisions below are the ones the plan leaves open: generics discipline, budget integration shape, evaluator composition, and how the sinks stay infrastructure-free.

## Goals / Non-Goals

**Goals:**
- A pipeline whose contract rules are each provable by a named test, and whose two property invariants (no ACCEPT without both gates; no RETRY without attempts + retryable finding) hold by construction.
- Zero infrastructure dependencies: the module is importable by every feature.

**Non-Goals:**
- Adopting any seam (the two adoption changes own that, with their parity proofs).
- Prompt engineering: the pipeline appends structured feedback; wording belongs to seam prompts.
- Any telemetry beyond the one decision event (per-stage tracing is `mount-retrieval-graph`'s slice of G6).

## Decisions

### Decision 1 — generic over the validated type, frozen outcome
`ValidationPipeline[T]` / `ValidationOutcome[T]` are generic; the outcome is a frozen Pydantic model so it can be persisted next to a seam's record without mutation risk. The value rides as `T | None` with the ACCEPT-only invariant enforced in the model (a validator, not a convention — see the spec's first requirement).

### Decision 2 — the budget is an optional port, mirrored on the existing ledger shape
`BudgetLedger | None` is a protocol the existing extraction ledger (and later Saul's `domain/budget.py` ledger) satisfies. The pipeline charges per `produce` re-invocation and never constructs a ledger. This keeps `app/ai/validation` free of feature imports while making the PR #110 lesson a structural rule.

### Decision 3 — evaluators are composed, not inherited
`MechanicalEvaluator` takes pure predicates `(value, context) -> SemanticFinding | None`; `CompositeEvaluator` runs mechanicals and consults the `LLMEvaluator` only on mechanical pass. Existing seam validators (relationship span checks, quote containment) become predicate callables at adoption time with no logic movement — the vocabulary maps, the code stays.

### Decision 4 — sinks are caller-constructed ports with three stock shapes
`EscalationSink` is a protocol; the three stock sinks are thin adapters the caller wires. The pipeline never imports a sink implementation beyond the protocol — routing stays a caller decision (spec's final requirement), which is what lets agent_saul's HITL and documents' dead-letter coexist on one vocabulary.

### Decision 5 — the reason vocabulary is a closed enum plus seam-scoped codes
Stable core reasons (`budget_exhausted`, `evaluator_unavailable`, `non_retryable_rejection`, `attempts_exhausted`, `accepted`) live in the module; seam-specific finding codes extend the findings, not the core enum. D7's "every finding code Saul emits enters the layer's stable vocabulary" is satisfied by findings carrying codes, not by growing the decision enum.

## Risks / Trade-offs

- **A shared pipeline could become a bottleneck abstraction** → the protocol seams (evaluator, sink, ledger) are narrow; seams that need more adopt partially (type vocabulary only, as D7 does for Saul).
- **Generic frozen models + Pydantic v2** → runtime cost is one model construction per attempt; acceptable and measurable via the decision event's attempts field.

## Migration Plan

Pure addition. The module lands with its suite; the adoption changes land after, each with its own parity proof. No deprecation, no flag.

## Open Questions

None — the plan's integration order fixes sequencing; the decisions above fix the remaining shape.
