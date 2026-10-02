## Purpose

Define how the retrieval loop behaves as an agent rather than as a pipeline: it narrows the corpus
before searching it, widens that narrowing when the evidence proves insufficient, assembles context
that is deduplicated and readable in document order, accounts for its budget in the units the model
actually consumes, reranks without requiring a local model, and always terminates.

## ADDED Requirements

### Requirement: Retrieval narrows the corpus before searching it

The retrieval planner SHALL attempt to constrain the searchable corpus before search runs, and SHALL
proceed unconstrained when it cannot.

#### Scenario: A query naming a jurisdiction narrows the search

- **WHEN** a query names a jurisdiction, a document kind, or a matter
- **THEN** the planner SHALL produce a document allowlist and the search SHALL be constrained to it

#### Scenario: An unnarrowable query still retrieves

- **WHEN** the planner cannot narrow the corpus
- **THEN** retrieval SHALL proceed unconstrained rather than returning nothing

#### Scenario: The allowlist reaches the search itself

- **WHEN** an allowlist is produced
- **THEN** it SHALL be applied as a filter on the search, and SHALL NOT be applied only after results
  are returned

### Requirement: A narrowing that excludes the answer is recoverable

An allowlist SHALL NOT be able to make an answerable question unanswerable.

#### Scenario: Insufficient context widens the allowlist

- **WHEN** the grader reports the retrieved context insufficient
- **THEN** the next retrieval iteration SHALL run with a strictly wider allowlist, or with none, before
  generation is attempted

#### Scenario: Widening precedes generation

- **WHEN** the allowlist excludes every document containing the answer
- **THEN** the loop SHALL retry with the widened allowlist rather than generating from the insufficient
  context

### Requirement: Assembled context is deduplicated and read in document order

Context reaching generation SHALL contain each chunk once, ordered as the document reads.

#### Scenario: A chunk found by two branches appears once

- **WHEN** the same chunk is returned by more than one retrieval branch
- **THEN** it SHALL appear exactly once in the assembled context

#### Scenario: Chunks of one document are in reading order

- **WHEN** reranked chunks from one document reach generation
- **THEN** they SHALL appear in ascending in-document order, not in relevance order

#### Scenario: Assembly reuses the existing implementation

- **WHEN** context is assembled on the graph path
- **THEN** it SHALL use the application's existing context-assembly implementation rather than a second
  one written for the graph

### Requirement: The context budget is measured in tokens

Budget accounting SHALL use the tokenizer's count, not a word count.

#### Scenario: Accounting uses the tokenizer

- **WHEN** context is assembled against a budget
- **THEN** the accounting SHALL use a token count obtained from the tokenizer, and SHALL NOT use a
  whitespace-separated word count

#### Scenario: Overflow drops whole sections from the tail

- **WHEN** assembled context would exceed the budget
- **THEN** whole sections SHALL be dropped from the tail, and no section SHALL be truncated mid-chunk

#### Scenario: The counter is supplied, not chosen here

- **WHEN** the budget is counted
- **THEN** the counter SHALL be supplied to the assembly path as a callable, so that the choice of
  tokenizer is made once elsewhere and consumed here

### Requirement: Reranking runs without requiring a local model

Reranking SHALL be defined by an interface, satisfied by at least one implementation that needs no
local deep-learning framework, and SHALL degrade rather than fail.

#### Scenario: Reranking narrows the candidate set

- **WHEN** candidates are retrieved
- **THEN** a reranker SHALL reorder them before generation and SHALL pass fewer candidates onward than
  it received

#### Scenario: An unavailable provider degrades to the fused order

- **WHEN** the reranking provider is unavailable
- **THEN** retrieval SHALL fall back to the fused order truncated to the requested count, and SHALL NOT
  fail the request

#### Scenario: No local framework is required

- **WHEN** the runtime environment has no local deep-learning framework installed
- **THEN** reranking SHALL still function

#### Scenario: The existing implementation still satisfies the interface

- **WHEN** the reranking interface is introduced
- **THEN** the existing local implementation SHALL satisfy it without modification, so the seam can be
  created before anything is removed

### Requirement: The retrieval loop terminates

The loop SHALL have a bounded number of iterations and a defined outcome when that bound is reached.

#### Scenario: A repeatedly insufficient context stops at the cap

- **WHEN** the grader repeatedly reports insufficient context
- **THEN** the loop SHALL stop at its iteration cap and return the grounded fallback rather than
  iterating further

#### Scenario: Widening does not extend the cap

- **WHEN** the allowlist is widened on retry
- **THEN** the iteration cap SHALL be unchanged by the widening, so that widening cannot produce an
  unbounded loop

### Requirement: The graph's node and edge shape is asserted

A change to the retrieval graph's topology SHALL be visible as a test change rather than only as a
source diff.

#### Scenario: The node set is pinned

- **WHEN** the retrieval graph is constructed
- **THEN** its node names and edge pairs SHALL match an asserted expected shape

#### Scenario: Adding a node updates the assertion in the same change

- **WHEN** a node is added to the graph
- **THEN** the asserted shape SHALL be updated in the same change that adds it
