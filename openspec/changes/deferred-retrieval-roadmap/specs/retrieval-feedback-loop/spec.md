## Purpose

Define the capture-and-promote contract that turns user signals on retrieval quality
into reviewed additions to the golden set — the compounding mechanism that keeps the
measurement baseline honest as usage grows.

## ADDED Requirements

### Requirement: Retrieval signals are captured with query context

The system SHALL capture user signals on ask results (accept/reject and citation
interactions) together with the query, the served chunk/document identifiers, and the
evidence policy in effect. Signals SHALL be captured without legal text content
leaving the trust boundary of the existing evidence and audit contracts.

#### Scenario: A signal identifies what was served, not just that something was

- **WHEN** a user accepts or rejects an ask result
- **THEN** the captured record names the query, the ranked identifiers served, and
  the evidence policy
- **AND** the record contains no content beyond what the audit contracts already
  retain

### Requirement: Signals promote to the golden set only under review

A captured signal SHALL be promotable into the retrieval golden set only as a
*candidate* that a review step accepts or rejects; promotion SHALL record its
provenance signal. Unreviewed signals SHALL NOT alter the golden set or any gate
computed from it.

#### Scenario: A rejected answer becomes a measured query

- **WHEN** a user rejects an answer and review accepts the derived query as a golden
  case
- **THEN** the golden set gains the query with its expected identifiers and a
  provenance note linking the originating signal
- **AND** the measurement gate's next run includes it

#### Scenario: Unreviewed signals never move the gate

- **WHEN** signals accumulate without review
- **THEN** the golden set and every CI gate computed from it are unchanged
