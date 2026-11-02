## Purpose

Define cross-document traversal as a tool capability: what the tools may ask the graph,
what they must return (handle-bound envelopes), and the gate that keeps them unexposed
until the graph can answer.

## ADDED Requirements

### Requirement: Portfolio tools are unexposed until relationship edges exist

The portfolio traversal tools SHALL NOT be bound to any role's toolset while the
knowledge graph lacks typed relationship edges for the in-scope corpus. The gate
SHALL be checkable: a test asserts the graph state the binding depends on, and
binding without it fails loudly rather than exposing tools that return empty
successes.

#### Scenario: Binding before the gate is a loud failure

- **WHEN** the graph factory attempts to bind a portfolio tool while the relationship
  edge check reports no edges for the corpus
- **THEN** binding fails with an error naming the gate
- **AND** no role's toolset contains a portfolio tool

#### Scenario: Empty results are honest after the gate

- **WHEN** a portfolio tool runs after the gate with a query the graph cannot answer
- **THEN** the envelope reports its no-result status distinctly from a successful
  traversal, never as an empty success

### Requirement: Traversal results are handle-bound envelopes

Every portfolio tool SHALL return a bounded envelope containing structured results
(clause/party/relationship identifiers with document provenance) plus evidence
handles served through the existing handle read path. Raw clause text SHALL NOT be
returned by traversal; content is materialized only through the evidence seam's
scoped read.

#### Scenario: Traversal finds where, evidence serves what

- **WHEN** a portfolio tool returns matches
- **THEN** each match carries document provenance and an evidence handle resolvable
  by the run-scoped read contract
- **AND** the envelope size is bounded regardless of match count

### Requirement: Scope is server-owned

Portfolio tools SHALL take only model-visible query arguments (party, clause class,
conflict notion). Tenant and document scope SHALL come exclusively from the injected
server-owned runtime context, and a traversal SHALL NOT cross the authorized document
scope under any query argument.

#### Scenario: A query cannot widen scope

- **WHEN** a portfolio tool is invoked with arguments naming documents outside the
  authorized scope
- **THEN** the traversal runs only inside the authorized scope and the result
  discloses the boundary

#### Scenario: No runtime context refuses

- **WHEN** a portfolio tool is invoked without a valid server-owned runtime context
- **THEN** it is refused per the tool-registry execution-authority contract
