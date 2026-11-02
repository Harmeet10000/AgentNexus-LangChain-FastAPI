> Change class: **M**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

Saul's verdict path already embodies the validation pattern (mechanical gates, fail-closed entailment, budget attribution); D7 is a vocabulary and routing adoption. The gates were built under harden task 6.5 and their tests are the regression fence. Gating: the core module must exist first; sequencing after `mount-retrieval-graph` means the qna grade loop is refactored once (in the seam-adoption change), and Saul's qna inherits it.

## Goals / Non-Goals

**Goals:**
- One vocabulary across every seam in the application, with Saul's — the most security-sensitive path — proving that adoption can be zero-behaviour-change.
- ESCALATE routing unified: the review node is reached through the sink contract, closing the "fourth escalation mechanism" risk permanently.

**Non-Goals:**
- Any gate-logic or quality change in the verdict path.
- Adopting the tool-calling seams (middleware boundary stays — that is a spec requirement here, not a compromise).
- Deep-research synthesis validation (future seam; noted, not designed).

## Decisions

### Decision 1 — mapping is type-level; gate code does not move
The mechanical predicates adopt the `MechanicalEvaluator` predicate *signature* and the findings adopt the report shape; the gate bodies stay where they are. The existing suite passing unmodified is the acceptance gate, which only holds if the logic truly doesn't move.

### Decision 2 — the sink wraps the existing node, not a new one
`HumanReviewSink` is implemented as a thin adapter over `make_human_review_node`'s existing entry. The HITL interrupt semantics (approval, resume, role checks from harden 6.7) are unchanged; the sink only standardizes *how an outcome arrives*.

### Decision 3 — ledger conformance is an adapter, not a pipeline fork
Saul's `domain/budget.py` ledger gains conformance to the pipeline's budget port. Charging happens per re-invocation exactly as the core contract requires; the entailment pass's existing attribution remains the source of truth for its own spend.

### Decision 4 — the middleware boundary gets a regression test, not a comment
The tool-calling seams' middleware stack is asserted in a test so a future refactor cannot quietly fold transport retry into the semantic layer. The reference-14 boundary (transport retry = middleware; semantic gates = validation layer) becomes executable.

## Risks / Trade-offs

- **Zero-behaviour-change adoption can feel like ceremony** → its value is the shared vocabulary the next three seams inherit for free; the acceptance tests prove the cost is types only.
- **The sink adapter could drift from the node's real entry** → the interrupt-through-sink test drives the actual HITL flow, so drift fails loudly.

## Migration Plan

Single landing after its two gates (core exists; qna refactor landed). No flag needed: behaviour is provably unchanged.

## Open Questions

None.
