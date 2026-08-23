# saul-memory-prefetch-and-retrieval Specification

## Purpose
TBD - created by archiving change cognee-saul-memory-migration. Update Purpose after archive.
## Requirements
### Requirement: Saul SHALL prefetch memory after qna
The Saul graph SHALL execute a memory prefetch step after `qna` and before deeper reasoning.

#### Scenario: Prefetch runs after qna
- **WHEN** `qna` produces the clarified intent
- **THEN** the system SHALL prefetch memory before invoking deeper reasoning nodes

### Requirement: Prefetch SHALL be Cognee-first
The prefetch step SHALL retrieve Cognee memory first and MAY add a small Graphiti supplement for grounding.

#### Scenario: Cognee is primary recall source
- **WHEN** the prefetch step runs
- **THEN** Cognee memory SHALL be queried first

#### Scenario: Graphiti supplement remains small
- **WHEN** the prefetch step adds non-Cognee context
- **THEN** the Graphiti supplement SHALL remain limited to matter/document grounding context

### Requirement: Deep memory retrieval is limited to selected reasoning nodes
The system SHALL expose the deeper `retrieve_from_memory` tool only to `risk_analysis` and `compliance`.

#### Scenario: Risk analysis can retrieve deeper memory
- **WHEN** `risk_analysis` needs prior memory context
- **THEN** it SHALL be allowed to call the deeper memory retrieval tool

#### Scenario: Compliance can retrieve deeper memory
- **WHEN** `compliance` needs prior memory context
- **THEN** it SHALL be allowed to call the deeper memory retrieval tool

#### Scenario: Orchestrator cannot retrieve deeper memory
- **WHEN** `orchestrator` runs
- **THEN** it SHALL NOT have access to the deeper memory retrieval tool

### Requirement: Memory retrieval failures fail open
Memory retrieval failures SHALL fail open and allow the Saul run to continue with available context.

#### Scenario: Cognee retrieval failure degrades gracefully
- **WHEN** Cognee retrieval fails during prefetch or deep retrieval
- **THEN** the graph SHALL continue using current-run context and any available Graphiti supplement

