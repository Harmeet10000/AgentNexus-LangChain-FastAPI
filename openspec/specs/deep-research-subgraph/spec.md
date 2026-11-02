# deep-research-subgraph Specification

## Purpose
Define the bounded deep-research machine that sits behind exactly one tool: how it is
entered (server-owned context only), how it is bounded (hard step budget, typed
statuses), and what its outputs promise (handles, coverage, no confident partials).

## Requirements

### Requirement: Deep research is entered only through one tool with server-owned context

The deep-research subgraph SHALL be reachable only through its single tool wrapper.
The wrapper's model-visible arguments SHALL be exactly the research objective (plus an
optional focus), and the subgraph's initial state SHALL be constructed by the wrapper
from the injected server-owned runtime context — tenant, document scope, and identity
SHALL NOT be model arguments and SHALL NOT be read from ambient state.

#### Scenario: Model-supplied scope never reaches the subgraph

- **WHEN** the wrapper is invoked during a graph run whose state carries tenant or
  document-scope values, and the model supplies conflicting scope-like arguments
- **THEN** the subgraph executes only against the scope from the server-owned runtime
  context
- **AND** a propagation assertion proves no model-supplied scope value entered the
  subgraph state

#### Scenario: Deny-by-default without runtime context

- **WHEN** the wrapper is invoked with no valid server-owned runtime context
- **THEN** the result is a typed refusal naming the missing authorization, and the
  subgraph does not start

### Requirement: The research loop is hard-bounded with typed terminal statuses

The subgraph SHALL enforce a hard step/recursion budget across its plan, search,
read, and synthesize stages. It SHALL terminate with exactly one of: `completed`,
`budget_exhausted` (partial evidence handles included), `no_evidence`,
`ingestion_incomplete`. Budget exhaustion SHALL be indistinguishable-in-kind from
success at the envelope level (same shape, different status) and SHALL NOT be rendered
as a confident partial answer.

#### Scenario: Budget exhaustion is a typed result, not a truncation

- **WHEN** the step budget is consumed before synthesis completes
- **THEN** the tool returns `status: budget_exhausted` with the evidence handles
  gathered so far and no synthesized answer presented as complete

#### Scenario: No evidence is a status, not an empty success

- **WHEN** the search stage returns no admissible evidence for the objective
- **THEN** the tool returns `status: no_evidence` rather than a synthesized answer
  from nothing

### Requirement: Evidence handles are run-scoped

Every evidence handle the subgraph issues SHALL be valid only within the run that
produced it. A handle from run N SHALL fail handle-membership when presented in run
N+1, including when the run is resumed on the same thread.

#### Scenario: A handle does not survive its run

- **WHEN** `read_evidence_by_id` inside a later run receives a handle issued by an
  earlier run on the same thread
- **THEN** the read is refused as out-of-scope and the refusal names the run-scoping
  rule

### Requirement: No interrupt crosses the tool boundary

The subgraph SHALL NOT pause for human approval; any approval the orchestration graph
requires SHALL happen on Saul's graph, outside the tool. The subgraph's runs SHALL
complete, exhaust, or fail without emitting an interrupt.

#### Scenario: Approval never fires inside deep research

- **WHEN** the subgraph executes a full research run
- **THEN** no interrupt is emitted and the tool call returns exactly one terminal
  envelope

### Requirement: The projection stays at one visible tool

The deep_research role's model-visible toolset SHALL be exactly the subgraph wrapper
plus its transfer tool. Rebuilding the internals SHALL NOT add, remove, or rename any
model-visible tool.

#### Scenario: Internals can change without a toolset change

- **WHEN** the subgraph's internal stages are reorganized
- **THEN** the role's toolset identity is unchanged and no tool-schema diff reaches
  the model
