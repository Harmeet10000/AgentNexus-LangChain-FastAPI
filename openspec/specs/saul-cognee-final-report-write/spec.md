# saul-cognee-final-report-write Specification

## Purpose
TBD - created by archiving change cognee-saul-memory-migration. Update Purpose after archive.
## Requirements
### Requirement: Saul persist_memory writes approved final reports to Cognee
The Saul `persist_memory` node SHALL write the approved final report directly to Cognee after human approval.

#### Scenario: Approved final report is persisted
- **WHEN** the Saul graph reaches `persist_memory` after human approval
- **THEN** the node SHALL persist the final report to Cognee

#### Scenario: Final report write does not wait for async maintenance
- **WHEN** the final report is ready for persistence
- **THEN** the direct Cognee write SHALL occur in the `persist_memory` node before graph completion

### Requirement: Saul persist_memory does not write final reports to Graphiti
The Saul `persist_memory` node SHALL NOT write final reports to Graphiti.

#### Scenario: Final report storage is Cognee-only
- **WHEN** Saul completes a run and persists memory
- **THEN** no Graphiti final-report write path SHALL be invoked

### Requirement: Cognee final-report persistence is gated by approved output
The system SHALL only persist final reports that passed the approved output gate.

#### Scenario: Unapproved run does not persist final report
- **WHEN** human approval has not been granted
- **THEN** the system SHALL NOT store the final report in Cognee

