## ADDED Requirements

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
