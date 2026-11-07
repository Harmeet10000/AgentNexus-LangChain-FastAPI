# agentic-retrieval-loop Specification

## Purpose
Define how the retrieval loop behaves as an agent rather than as a pipeline: it narrows the corpus
before searching it, widens that narrowing when the evidence proves insufficient, assembles context
that is deduplicated and readable in document order, accounts for its budget in the units the model
actually consumes, reranks without requiring a local model, and always terminates.

## Requirements

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

### Requirement: The grade loop is a pipeline validation with identical observable behaviour

The ask path's context grading and rewrite loop SHALL run through the shared
validation pipeline (grading as the semantic evaluator, the rewrite loop as bounded
feedback-carrying retry). The ask endpoints' observable contract — response shapes,
fallback answer behaviour, warning envelope, cache-key derivation, iteration bound —
SHALL be byte-identical to the pre-pipeline behaviour, proven by a parity suite over
the existing grade-loop tests.

#### Scenario: Parity holds across the existing grade-loop cases

- **WHEN** the parity suite replays every existing grade-loop test case through the
  pipeline-backed loop
- **THEN** every response is byte-identical to the recorded inline behaviour
  (sufficient/insufficient decisions, rewrite usage, fallback, coverage, warnings)

#### Scenario: The rewrite arrives as structured feedback

- **WHEN** the grader returns insufficient with a rewrite suggestion and attempts
  remain
- **THEN** the retry runs with that suggestion carried as pipeline feedback, and the
  two-iteration bound is unchanged

### Requirement: Loop decisions carry the shared vocabulary

Each ask-loop validation SHALL emit one seam-named decision event, and the loop's
terminal outcomes SHALL be expressible as shared-vocabulary decisions (ACCEPT for
sufficient+generated, ESCALATE to the fallback answer for exhausted attempts) without
new escalation mechanisms.

#### Scenario: The fallback is an ESCALATE outcome, not a bespoke path

- **WHEN** both iterations complete with insufficient context
- **THEN** the loop's outcome is an ESCALATE decision whose sink produces the existing
  fallback answer behaviour, byte-identical to today's

### Requirement: Edge-aware grounding supplements the served loop

After a sufficient grade, the loop SHALL attempt edge-aware grounding whenever a healthy relationship
graph and an authorized document scope exist: typed relationship edges for the query are fetched and
the retrieved context is expanded along them, bounded per seed. The expansion SHALL fail closed — a
missing graph, tenant, scope, or a transport failure degrades to the exact edge-free contract. An
edge citation SHALL point at the hydrated passage the edge's grounded span actually falls in, with
document identity, version, and contract taken from that hydrated row — never from edge text or any
model output — and an expansion whose span resolves to no hydrated passage SHALL earn no citation.

#### Scenario: A grounded edge span cites the passage it falls in

- **WHEN** an expansion block's grounded span resolves to a hydrated retrieved row
- **THEN** the citation SHALL identify that row's chunk, document version, and contract, with the
  claim authored deterministically from the edge's grounded span

#### Scenario: An unresolvable span is not cited

- **WHEN** an expansion block carries no grounded span or its span resolves to no hydrated passage
- **THEN** the expansion text SHALL NOT produce a citation, whether or not its text informed
  generation

#### Scenario: An edge-free query keeps the pre-edge contract

- **WHEN** the relationship graph is unavailable, the scope is unauthorized for edge reads, or no
  typed edge touches the reranked chunks
- **THEN** the response SHALL be identical to the edge-free contract, with no error surfaced for the
  absent expansion

### Requirement: The compiled loop is parity-gated against the inline contract

The compiled graph's served behaviour SHALL be pinned by a parity suite that runs the same queries
through the compiled graph and through recorded fakes of the inline contract it replaces, asserting
identical response shapes, citation provenance, fallback behaviour, and answer cache-key derivation.
The suite SHALL remain in the test suite after cutover and SHALL fail on any behavioural drift. The
inline ask implementation SHALL be deleted only once this suite is green.

#### Scenario: The same query produces the same response shape

- **WHEN** the parity suite runs a query through the compiled graph and the recorded inline contract
- **THEN** the response shape, citation provenance, and fallback behaviour SHALL be identical

#### Scenario: The grade-failure fallback is identical

- **WHEN** grading reports insufficient context through the iteration cap in both paths
- **THEN** both SHALL return the same grounded fallback answer with no citations

#### Scenario: Drift fails the suite

- **WHEN** a change to either the graph nodes or the recorded contract alters any asserted behaviour
- **THEN** the parity suite SHALL fail, naming the divergent behaviour

### Requirement: Per-stage execution is observable

Each stage of the served loop — plan, retrieve, fusion, rerank, grade, generate — SHALL emit a span
attributed to that stage with its duration, and every grade outcome SHALL emit a decision log event
recording the outcome, the iteration it occurred on, and whether the loop exhausted its cap. The
events SHALL be structured so a grade-loop escape rate can be derived without re-parsing free text.

#### Scenario: Every stage leaves a span

- **WHEN** an ask completes through the compiled graph
- **THEN** a span SHALL exist for each executed stage with that stage's attribution

#### Scenario: A grade outcome is logged

- **WHEN** the grader returns a verdict
- **THEN** a decision log event SHALL record sufficient or insufficient, the iteration, and, on cap
  exhaustion, that the loop escaped to the fallback
