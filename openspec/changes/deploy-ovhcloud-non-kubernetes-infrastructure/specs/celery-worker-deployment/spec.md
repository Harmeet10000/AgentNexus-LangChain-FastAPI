# celery-worker-deployment

## ADDED Requirements

### Requirement: Every queue role has an isolated deployed consumer
The deployment SHALL run independently selectable default, ingestion, and crawler worker roles whose aggregate consumed queues cover every routed application queue and exclude the dead-letter queue. Resource pressure or rollout of one role SHALL NOT remove consumers for another role.

#### Scenario: Deployment queue coverage is checked
- **WHEN** all deployed worker roles report their registered tasks and consumed queues
- **THEN** their union covers every routed non-dead-letter queue exactly through the intended role assignments

#### Scenario: Ingestion worker is saturated
- **WHEN** all ingestion concurrency slots are processing long-running document work
- **THEN** default and crawler workers remain independently available for their routed tasks

### Requirement: Scheduler leadership is singular and observable
Exactly one scheduler SHALL publish periodic tasks for an environment, and scheduler replacement SHALL prevent overlapping active publishers.

#### Scenario: Scheduler is replaced during deployment
- **WHEN** a new scheduler revision is started
- **THEN** the prior scheduler is stopped or fenced before the replacement can publish the same schedule

### Requirement: Worker shutdown respects the maximum task boundary
Worker deployment shutdown SHALL preserve truthful task outcomes by allowing active tasks to finish within their configured hard time limit or by causing broker-visible redelivery without acknowledging false completion.

Scale: worker termination grace interval per role | Meter: configured grace interval minus configured hard task time limit | MUST: at least 30 seconds of positive margin unless a tested redelivery mechanism is used.

#### Scenario: Maximum-duration task overlaps a rollout
- **WHEN** a task remains active at worker replacement and has not crossed its hard time limit
- **THEN** the worker is not forcibly terminated before the required grace margin or the task is observably redelivered
