# ingestion-delegation Specification

## Purpose
Define how stage handoffs in the ingestion graph appear in the shared transcript: the same linked
message pairs as every other graph, so ingestion delegations are traceable under the same
budget rules as the rest of the system.

## ADDED Requirements
### Requirement: Stage handoffs are linked message pairs

Every ingestion stage handoff SHALL appear in the shared transcript as an assistant message with
one tool call immediately followed by the tool message bearing the same call id.

#### Scenario: Stage transition is visible

- **WHEN** ingestion advances from one stage to the next via delegation
- **THEN** the transcript SHALL contain the pair naming the transition and its reason

### Requirement: Pairs use the shared constructor

All pairs in this graph SHALL be constructed with the constructor defined in
`agent-saul-delegation`. This graph SHALL NOT define its own pair shape.

#### Scenario: Cross-graph pair uniformity

- **WHEN** an ingestion delegation pair is compared with a Saul delegation pair for the same
  reason shape
- **THEN** both SHALL satisfy the same linkage and single-source-reason rules

### Requirement: Budget trimming preserves pair integrity

Ingestion context budgeting SHALL NOT leave half a pair: trimming that reaches a delegation pair
SHALL retain both messages together, drop both together, or replace both with one summary entry.

#### Scenario: Long documents keep whole pairs

- **WHEN** ingestion context pressure requires dropping transcript content containing a
  delegation pair
- **THEN** no orphaned assistant or tool message SHALL remain
