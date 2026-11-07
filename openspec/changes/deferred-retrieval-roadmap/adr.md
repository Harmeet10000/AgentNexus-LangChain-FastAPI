# ADRs — deferred-retrieval-roadmap

## ADR-1: Deferrals become gated contracts, not non-goal prose

**Decision:** each deferred G-item gains an owning requirement with a checkable trigger, in one roadmap change.
**Alternatives:** leave them in non-goal sections (rejected: the audit found them only there — prose nobody owns), one change per item now (rejected: five premature ceremonies).
**Consequence:** the next audit finds owners, not orphans; an item forks from this contract when its trigger fires.

## ADR-2: Triggers are landed artifacts

**Decision:** every gate names an artifact whose landing is tracked in another change (baseline, edges, spans, UI signals).
**Alternatives:** dates or effort estimates (rejected: calendars don't prove readiness).
**Consequence:** opening a gate is a verifiable fact, not a judgement call.

## ADR-3: Every weight and default is a measurement

**Decision:** G2's default-on, G4's branch weight, and G5's hydration all cite a recorded baseline comparison; regression reverts.
**Alternatives:** configured weights and enthusiastic defaults (rejected: the measurement harness exists precisely so these stop being opinions).
**Consequence:** the retrieval path only grows on evidence, and shrinkage is as easy as growth.

## ADR-4: Feedback promotion is reviewed by construction

**Decision:** captured signals are candidates; only review moves them into the golden set, with provenance.
**Alternatives:** auto-promotion of rejected answers (rejected: a poisoned signal becomes a poisoned gate).
**Consequence:** the feedback loop compounds the baseline's trustworthiness instead of gambling with it.
