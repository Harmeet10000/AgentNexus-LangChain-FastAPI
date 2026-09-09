## Purpose

Retrieval quality SHALL be measurable deterministically, from a versioned golden set, without a model
provider or a database, so that a change claiming a retrieval improvement can cite a measurement
rather than an assertion.

## ADDED Requirements

### Requirement: The golden set is a versioned, validated artifact

Golden-set rows SHALL be schema-validated on load, and a malformed row SHALL be reported as an
expected failure rather than raised as an exception.

#### Scenario: A malformed row is an expected failure

- **WHEN** a row omits a required field
- **THEN** the loader SHALL return a failure identifying the offending row index

#### Scenario: An absent file is an exception

- **WHEN** the golden-set file is absent or unreadable
- **THEN** the loader SHALL raise a typed exception from the project hierarchy

#### Scenario: All four document families are covered

- **WHEN** the golden set is loaded
- **THEN** it SHALL contain at least one query for each of contracts, statutes, judgments, and filings

#### Scenario: The set carries its own version

- **WHEN** a report is produced
- **THEN** it SHALL name the golden-set version it scored, so two reports are only comparable when
  they scored the same set

### Requirement: Deterministic retrieval metrics

Retrieval metrics SHALL be computed from ranked identifier lists alone, with no model or database
access.

#### Scenario: A hit at rank one scores one

- **WHEN** an expected chunk appears at rank one
- **THEN** recall-at-k and reciprocal rank SHALL both be one for that query

#### Scenario: A miss scores zero

- **WHEN** no expected chunk appears within the top k
- **THEN** recall-at-k and reciprocal rank SHALL both be zero

#### Scenario: Scoring is reproducible

- **WHEN** the same inputs are scored twice
- **THEN** identical values SHALL be produced

#### Scenario: Every metric is bounded

- **WHEN** any metric is computed over any input
- **THEN** its value SHALL lie between zero and one inclusive, and recall-at-k SHALL NOT decrease as k
  increases

### Requirement: The harness never blocks the default test gate

Scoring SHALL be usable in the default test selection, and infrastructure-dependent scenarios SHALL be
excluded from it.

#### Scenario: The default selection stays offline

- **WHEN** the default test selection runs
- **THEN** no scenario requiring a live database or a model provider SHALL execute

#### Scenario: An explicit invocation emits a report

- **WHEN** the harness is invoked explicitly
- **THEN** it SHALL emit a machine-readable report carrying per-query rows, aggregates, and the commit
  identifier of the tree it measured

### Requirement: The harness is proven to reach live retrieval

A green offline suite SHALL NOT be accepted as evidence that the harness is connected to anything.

#### Scenario: A live run returns real identifiers

- **WHEN** the harness runs against the live retrieval path over a seeded corpus
- **THEN** the emitted report's retrieved identifiers SHALL be a non-empty subset of the identifiers
  present in the chunk store

#### Scenario: An empty result is a wiring failure, not a score

- **WHEN** a live run returns no identifiers for every query
- **THEN** it SHALL be reported as a wiring failure rather than as a score of zero

#### Scenario: Retrieval is reached through the service layer

- **WHEN** the harness invokes retrieval
- **THEN** it SHALL do so through the service layer and SHALL NOT import a repository directly

### Requirement: A recorded retrieval baseline exists before retrieval changes

A durable baseline report SHALL exist before any retrieval-affecting change is implemented, and a
later claim of improvement SHALL be measured against it.

#### Scenario: An improvement claim cites a report

- **WHEN** a retrieval-affecting change reports an improvement
- **THEN** it SHALL cite a report produced by this harness against the recorded baseline report

#### Scenario: A failed baseline records its reason verbatim

- **WHEN** a baseline cannot be produced
- **THEN** the baseline artifact SHALL record the verbatim failure of the command that could not run,
  rather than an absence

### Requirement: Answer-quality judging is separable from retrieval scoring

Judged generation metrics SHALL be addable without altering the retrieval scoring path.

#### Scenario: Retrieval-only mode makes no provider call

- **WHEN** the harness runs in retrieval-only mode
- **THEN** no model-provider call SHALL be made

#### Scenario: The judged layer attaches without changing retrieval scores

- **WHEN** a judged answer-quality layer is added later
- **THEN** the retrieval metrics for an unchanged input SHALL be unchanged, and the judged metrics
  SHALL be reported as a separate section of the report
