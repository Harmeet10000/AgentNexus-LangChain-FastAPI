## ADDED Requirements

### Requirement: Cognee maintenance SHALL run through Celery
Cognee curation, decay, and promotion jobs for Saul memory SHALL run through Celery.

#### Scenario: Maintenance jobs are enqueued to Celery
- **WHEN** a Saul run completes with memory work pending
- **THEN** the system SHALL enqueue maintenance work to Celery

### Requirement: Maintenance jobs operate on approved memory artifacts
The maintenance worker SHALL process approved final reports, curated observations, and user preferences only.

#### Scenario: Approved report can be curated
- **WHEN** a final report has already been approved and persisted
- **THEN** the maintenance worker MAY derive curated observations from it

#### Scenario: Unapproved memory is ignored
- **WHEN** a memory artifact has not passed the approval gate
- **THEN** the worker SHALL NOT promote it into long-term Cognee memory

### Requirement: Maintenance jobs SHALL be idempotent
The maintenance workflow SHALL be idempotent for repeated deliveries of the same report or observation set.

#### Scenario: Duplicate job delivery does not duplicate memory
- **WHEN** the same maintenance job is delivered more than once
- **THEN** Cognee memory SHALL not gain duplicate promoted artifacts

### Requirement: Maintenance SHALL support scheduled sweeps
The system SHALL support scheduled sweeps for decay, deduplication, and promotion.

#### Scenario: Scheduled sweep runs without user interaction
- **WHEN** the scheduler triggers a maintenance sweep
- **THEN** the worker SHALL process eligible memory artifacts without a live Saul session
