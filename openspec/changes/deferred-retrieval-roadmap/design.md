> Change class: **M**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

Five roadmap items were deferred by explicit non-goal statements across three active changes, and the audit found nothing owns them. This change converts deferrals into gated contracts. The pattern is the one `agent-portfolio-traversal-tools` established: author the contract while the capability is unexposed, make the gate checkable, and let the trigger — not a calendar — open the work.

## Goals / Non-Goals

**Goals:**
- Every deferred G-item has exactly one owner, one trigger, and one measurable adoption standard.
- The two surface-level deferrals (empty `preamble` column, reserved fusion registration point) become ledger rows so the quarantine ledger stays the single place disconnection is recorded.

**Non-Goals:**
- Implementing anything (all tasks are trigger-deferred).
- G8 niceties (late chunking, multi-query expansion, semantic cache) — permanently out of scope until an eval demands one; named here so the exclusion is recorded.
- Reopening decisions that are already correct (pageindex stays deleted; the D-then-E sequencing stays abandoned).

## Decisions

### Decision 1 — one change, five gates, not five changes
Five premature changes would each carry more ceremony than substance and none would age well. One change with per-item triggers keeps the roadmap in one reviewable place; if an item's implementation later needs an L-class change of its own (G2 most likely), it *forks from* this contract rather than replacing it.

### Decision 2 — every trigger is a landed artifact, never a date
G2 and G5 gate on the measurement baseline existing and speaking; G4 gates on edges *and* baseline; G6 gates on the spans slice; G7 gates on UI signals *and* the harness. A date would make this a calendar; a landed artifact makes it a dependency.

### Decision 3 — G4's weight is measured, not configured
The reserved registration point makes the branch easy to add; a configured weight would make it easy to add *wrongly*. The delta requires a recorded baseline comparison supporting the weight — same evidence standard as G2/G5, so all three retrieval changes share one adoption discipline.

### Decision 4 — G6-remainder is scoped to the loop, not the platform
The harden change already owns platform observability (9.1 events, 9.8 kill criteria) and `mount-retrieval-graph` owns the spans slice. What was genuinely orphaned is the *loop's own* budget enforcement and the standing alert between "spans exist" and "kill the pilot" — this change owns exactly that slice and no more.

### Decision 5 — feedback promotion requires review by construction
G7's value collapses if unreviewed signals move the gate — a poisoned signal is a poisoned baseline. The delta makes promotion a reviewed, provenance-carrying act; unreviewed accumulation changes nothing measurable.

## Risks / Trade-offs

- **A roadmap change can rot if triggers never fire** → the ledger rows give the two surface deferrals a physical presence, and the triggers name artifacts whose landing is already tracked elsewhere; a stale roadmap here is visible in the same audit that found these gaps.
- **One change owning five items could bottleneck a single item's momentum** → acceptable: forking from this contract is one review, and the contracts are small.

## Migration Plan

None — pure planning. Each item's implementation task names its trigger and lands as its own commit against this change's contracts.

## Open Questions

None. (Whether G2's implementation deserves its own L-class change is decided at trigger time, per Decision 1.)
