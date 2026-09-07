## Purpose

Defines what an agent is allowed to emit: a declared, validated output shape for every reasoning role, an absolute
requirement that a legal assertion carries its citation, and retention of the usage accounting that structured
extraction otherwise discards.

## ADDED Requirements

### Requirement: Every reasoning agent declares its output shape

Each agent role that produces structured findings SHALL declare the schema of its output at construction, and its
output SHALL be validated against that schema before any consumer reads it.

#### Scenario: The orchestrator role declares an output schema

- **WHEN** the orchestrating agent is constructed
- **THEN** it SHALL be constructed with a declared output schema

#### Scenario: The risk role declares an output schema

- **WHEN** the risk analysis agent is constructed
- **THEN** it SHALL be constructed with its risk output schema

#### Scenario: The compliance role declares an output schema

- **WHEN** the compliance agent is constructed
- **THEN** it SHALL be constructed with its compliance output schema

#### Scenario: Output that does not satisfy the declared schema is rejected

- **WHEN** an agent produces output that does not satisfy its declared schema
- **THEN** the output SHALL be rejected rather than passed on unvalidated

### Requirement: An assertion without a citation is rejected

Every output model that carries a legal or factual assertion SHALL require at least one citation for that assertion,
and each citation SHALL identify the claim it supports, the source it rests on, and a bounded confidence value. An
uncited assertion SHALL fail validation; it SHALL NOT be accepted with a warning.

#### Scenario: A risk finding with no citations fails validation

- **WHEN** a risk finding carrying an assertion is constructed with an empty citation list
- **THEN** validation SHALL fail

#### Scenario: A compliance finding with no citations fails validation

- **WHEN** a compliance finding carrying an assertion is constructed with an empty citation list
- **THEN** validation SHALL fail

#### Scenario: A cited finding is accepted

- **WHEN** a finding is constructed with a citation identifying its claim, source, and confidence
- **THEN** validation SHALL succeed

#### Scenario: Confidence is bounded

- **WHEN** a citation is constructed with a confidence value outside the permitted range
- **THEN** validation SHALL fail

#### Scenario: The citation obligation is stated to the model

- **WHEN** a prompt is assembled for an agent whose output requires citations
- **THEN** the assembled prompt SHALL state that every assertion must carry a citation

### Requirement: Usage accounting is retained where it is captured

Where an agent or chain obtains model usage accounting alongside its structured result, that accounting SHALL be
retained and made available rather than discarded. Where a chain deliberately does not capture it, the reason SHALL be
recorded at the call site.

#### Scenario: A structured chain that captures usage exposes it

- **WHEN** a chain requests both the structured result and the raw model response
- **THEN** the usage accounting SHALL be retained alongside the structured result

#### Scenario: The structured result remains typed for consumers

- **WHEN** a chain retains usage accounting alongside its structured result
- **THEN** its consumers SHALL still receive the structured result under its declared type
