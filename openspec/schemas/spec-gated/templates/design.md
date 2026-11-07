# Design: <change>

## Context

<!-- Only current state + constraints needed to explain the approach. Reference proposal (Why) instead of restating. See class S/M/L classification in proposal.md. -->

## Goals / Non-Goals

**Goals:**
-

**Non-Goals:**
- <!-- Explicitly excluded; don't restate proposal scope - design-level boundaries only -->

## Decisions

### D1: <choice>
**Choose:** <X> over <Y>.
**Why:** <rationale; alternatives considered with why rejected>

<!-- UpskillLift layering: thin router → service orchestrates → repo persists → connections adapt; kernel pure; clients live on app.state (lifespan), composed via Annotated Depends; async-only I/O. -->

## Risks / Trade-offs

- [<Risk>] → [Mitigation]

## Migration Plan

<!-- Deploy steps + rollback; omit for non-migrating changes -->

## Open Questions

<!-- Genuinely deferrable unknowns ONLY - not skipped decisions. If answering would change specs/approach/task breakdown, resolve with user NOW. If this design supersedes a decision from another change, name `<prior-change>#ADR-NNN` so the adr step can record it. Omit if none. -->

## Blast Radius

<!-- For every touched symbol/module: callers + dependents from the code index, not memory. -->

| Symbol/module | Callers | Dependents |
|---|---|---|

## Async Semantics
<!-- For async flows only - these decisions feed contracts/asyncapi.yaml -->

| Question | Decision |
|---|---|
| Delivery semantics | at-least-once / effectively-once |
| Ordering guarantees | |
| Expiry / retry policy | |
| Payload schema versioning | |
| Read models while stale | |
