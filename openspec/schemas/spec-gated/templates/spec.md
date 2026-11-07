# <capability-path>

## Purpose
<!-- NEW capabilities only: 1-2 sentences, 50+ chars (strict validate checks length). -->
<!-- Do NOT add Purpose to a delta for an existing capability - it already has one. -->

<What this capability is for.>

## ADDED Requirements

### Requirement: <name>
<!-- SHALL/MUST only (no should/may). Observable behavior - external contract, not implementation. Quick test: if implementation can change without changing externally visible behavior, it does not belong here. -->
<!-- Measurable attribute (latency/budget/tolerance/capacity)? Use Planguage - Scale | Meter | MUST. Bare adjectives (fast, small) are not requirements. -->

The system SHALL ...

#### Scenario: <name>
- **WHEN** <trigger>
- **THEN** <observable outcome>

<!-- Event-driven: write scenarios against discovery.md domain events (WHEN <event> THEN <reaction>), one scenario per event per requirement where the event matters. Identity-creating/renaming changes: add a requirement covering stability across rename/republish, rename handling, and orphan behavior. -->
<!-- Status machines / multi-condition logic: use a decision table (columns = conditions, last column = outcome). Every row is a test case and MUST be covered. -->

## MODIFIED Requirements

<!-- Copy the ENTIRE existing requirement block from openspec/specs/<capability-path>/spec.md (### Requirement: through all scenarios), then edit. Header must match exactly (whitespace-insensitive). Partial content loses detail at archive. -->

## REMOVED Requirements

### Requirement: <name removed>
**Reason**: <why>
**Migration**: <what callers do instead>

## RENAMED Requirements

### Requirement: FROM: <old name> / TO: <new name>
