## Purpose

Extend the delegation contract for two toolset facts: deep research's projection is a
single subgraph-backed tool, and risk's conclusions disclose ingestion completeness
exactly as compliance's do.

## ADDED Requirements

### Requirement: Risk discloses ingestion completeness

The risk role's toolset SHALL include the scoped `get_ingestion_state` capability, and
a risk conclusion produced while any in-scope document is not ingestion-complete SHALL
disclose that state in its output rather than reading as a complete analysis.

#### Scenario: Risk parity with compliance on readiness

- **WHEN** a risk analysis runs over a document set containing an incomplete document
- **THEN** the role can retrieve the document's ingestion state through the same
  scoped capability compliance uses
- **AND** the analysis output reflects the incompleteness instead of silently
  proceeding

#### Scenario: No capability without scope

- **WHEN** `get_ingestion_state` is invoked without a server-owned runtime context
- **THEN** it is refused per the tool-registry execution-authority contract
