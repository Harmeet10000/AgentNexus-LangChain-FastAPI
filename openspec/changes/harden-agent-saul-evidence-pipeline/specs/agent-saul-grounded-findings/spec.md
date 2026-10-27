## Purpose

Defines truthful, typed Agent Saul findings and the evidence/coverage contract that prevents unsupported legal claims
from reaching approval or finalization.

## ADDED Requirements

### Requirement: Findings are atomic and cite typed evidence

Every risk or compliance finding MUST be typed and MUST carry an atomic claim, severity/category, status, and zero or
more evidence references. Each reference MUST identify the evidence, chunk, document version, pipeline version,
`contract_index` when the source contains multiple contracts, and page when available; a quoted span MUST be retrievable
from the canonical chunk. Risk output MUST remain
`not_assessed` until supported analysis completes. Compliance output MUST be tri-state: `compliant`,
`non_compliant`, or `not_assessable`; `not_assessable` MUST carry one or more of the reason codes
`statute_not_in_corpus`, `insufficient_coverage`, and `ingestion_incomplete`. `overall_compliant=true` MUST NOT be
emitted without at least one grounded supporting finding and no unresolved or contradicted finding.

#### Scenario: Structured agent output is preserved
- **WHEN** a reasoning node emits structured findings
- **THEN** parsing MUST preserve the findings and evidence references rather than reconstructing a summary from free text

#### Scenario: Placeholder compliance cannot pass
- **WHEN** no supported finding establishes compliance
- **THEN** the result MUST be `not_assessed` or `not_assessable` with a reason and MUST NOT render as compliant

#### Scenario: Compliance state and reason are explicit
- **WHEN** compliance cannot be established because authority is missing, coverage is insufficient, or ingestion is incomplete
- **THEN** the result MUST be `not_assessable` with the applicable reason code and MUST NOT emit `overall_compliant=true`

### Requirement: Findings preserve contract identity

When a source document contains multiple contracts, every chunk, evidence reference, finding, and final report MUST
carry the originating `contract_index`. A finding MUST NOT name a party, date, or governing law without identifying the
contract from which that metadata was derived.

#### Scenario: Two-contract metadata remains separated
- **WHEN** a finding cites a chunk from a multi-contract file
- **THEN** the finding and report MUST carry the same contract index and MUST not attribute the party to another contract

#### Scenario: Contract identity is unavailable
- **WHEN** a multi-contract boundary cannot be established with sufficient fidelity
- **THEN** the affected finding MUST be downgraded or marked `not_assessable` and MUST disclose ingestion incompleteness

### Requirement: Grounding verification is a blocking gate

The system MUST verify each finding before the graph can advance. Deterministic checks MUST confirm that the handle was
issued in the current run, the document is in scope, the version matches, and the quoted text is a substring of the
stored chunk. Surviving claims MUST undergo entailment assessment; unsupported or contradicted claims MUST be rejected
or downgraded.

#### Scenario: Invalid quote blocks advancement
- **WHEN** a quoted span is absent from its referenced chunk
- **THEN** the finding MUST be marked unsupported and the graph MUST NOT advance it as verified

#### Scenario: Non-indexed but canonical evidence is disclosed
- **WHEN** a chunk is not Graphiti-indexed but passes scope, version, substring, and entailment checks
- **THEN** it MAY support a finding and the report MUST disclose graph coverage

### Requirement: Coverage distinguishes absence from incompleteness

Every final report MUST carry coverage describing searched documents, pending or failed documents, instruments,
sections, jurisdictions, graph availability, and missing expected sources. The system MUST distinguish `no_evidence`,
`index_not_ready`, `ingestion_failed`, `unauthorized`, `transient_failure`, `timeout`, and `budget_exhausted`.

#### Scenario: Failed ingestion is not a negative answer
- **WHEN** an in-scope document has failed ingestion
- **THEN** the result MUST disclose `ingestion_failed` and MUST NOT claim that the corpus contains no such authority

#### Scenario: Graphiti outage is visible
- **WHEN** Graphiti is unavailable during retrieval
- **THEN** the report MUST mark graph availability degraded while preserving canonical corpus results

### Requirement: Human review sees unresolved evidence and controls persistence

Human review MUST display unsupported, contradicted, and not-assessed items, require the current reviewer role, and
record overrides and self-review. Finalization and memory persistence MUST consume approved atomic findings, grounding
status, unresolved claims, overrides, coverage, and ingestion state; unapproved output MUST NOT be persisted.

#### Scenario: Non-reviewer cannot approve
- **WHEN** a user without reviewer role resumes at human review
- **THEN** approval MUST be denied and the report MUST remain unapproved

#### Scenario: Override is explicit
- **WHEN** a reviewer overrides an unsupported finding
- **THEN** the override reason, reviewer identity, and resulting status MUST be recorded in the report and audit trail

### Requirement: Per-run budgets are enforced

Each run MUST track model calls, retrieval calls, fan-out, tokens, and wall-clock budget. Corrective retrieval MUST be a
bounded evidence search under the same ledger; exceeding any budget MUST return `budget_exhausted` and stop expansion.

#### Scenario: Corrective loop reaches its budget
- **WHEN** a corrective retrieval would exceed the run budget
- **THEN** no further retrieval MUST occur and the report MUST disclose budget exhaustion
