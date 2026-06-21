## ADDED Requirements

### Requirement: Cognee reconciliation is separate from Graphiti reconciliation
The system SHALL use a separate reconciliation workflow for Cognee memory and SHALL NOT reuse the existing Graphiti/entity reconciliation graph for Cognee semantics.

#### Scenario: Cognee reconciliation has its own workflow
- **WHEN** Cognee memory needs dedupe or promotion cleanup
- **THEN** the system SHALL invoke the Cognee reconciliation workflow

#### Scenario: Graphiti reconciliation remains independent
- **WHEN** Graphiti entity reconciliation runs
- **THEN** it SHALL remain focused on KB/entity relationship cleanup

### Requirement: Cognee reconciliation SHALL handle curated memory drift
The Cognee reconciliation workflow SHALL resolve duplicate observations, stale preferences, and conflicting summaries.

#### Scenario: Duplicate observations are merged or dropped
- **WHEN** two Cognee observations describe the same long-term memory fact
- **THEN** reconciliation SHALL merge or remove the duplicate according to policy

#### Scenario: Stale preferences are decayed or replaced
- **WHEN** a newer user preference supersedes an older one
- **THEN** reconciliation SHALL promote the newer preference and decay the older one

### Requirement: Cognee reconciliation SHALL not mutate approved final reports
The reconciliation workflow SHALL preserve approved final reports as immutable source artifacts.

#### Scenario: Approved report remains intact
- **WHEN** reconciliation processes memory derived from an approved final report
- **THEN** the original report artifact SHALL remain unchanged

### Requirement: Cognee reconciliation SHALL use idempotency keys
The workflow SHALL use a deterministic idempotency key derived from the report or observation content and run identity.

#### Scenario: Repeat reconciliation run is safe
- **WHEN** the same reconciliation input is processed more than once
- **THEN** the workflow SHALL produce the same effective state without duplicate side effects
