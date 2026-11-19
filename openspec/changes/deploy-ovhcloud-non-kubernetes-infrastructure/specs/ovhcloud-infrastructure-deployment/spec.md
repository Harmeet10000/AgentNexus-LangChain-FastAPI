# ovhcloud-infrastructure-deployment

## Purpose

Define the observable provisioning, security, compatibility, release, recovery, and operational guarantees required to run the complete application on OVHcloud without Kubernetes.

## ADDED Requirements

### Requirement: Test and production environments are isolated and reproducible
The deployment SHALL provision test and production as independently addressable environments with separate infrastructure state, credentials, private networks, data services, object namespaces, and destruction boundaries. Reapplying an unchanged infrastructure plan SHALL produce no mutations.

#### Scenario: Unchanged environment is planned again
- **WHEN** the declared infrastructure for an existing environment is planned without input changes
- **THEN** the plan reports no resource changes and does not rotate credentials or replace persistent resources

#### Scenario: Test environment is destroyed
- **WHEN** an authorized operator destroys the test environment
- **THEN** no production resource, state object, credential, network, DNS record, data service, or backup is selected for deletion

### Requirement: The deployment inventory covers every runtime role
An environment SHALL declare an API role, default worker role, ingestion worker role, crawler worker role, exactly one active scheduler, message broker, PostgreSQL, MongoDB, Redis-compatible state store, Neo4j with required procedures, object storage, crawler egress boundary, telemetry collector, migration job, and backup/restore jobs. A role omitted from the inventory SHALL make deployment acceptance fail.

#### Scenario: Complete inventory is accepted
- **WHEN** deployment acceptance enumerates running roles and configured dependencies
- **THEN** every required role has an owner, health signal, resource budget, network policy, configuration source, and recovery procedure

#### Scenario: Quiet subsystem is missing
- **WHEN** the scheduler, a worker role, a relay, a compatibility probe, or a backup job is absent while ordinary API health remains green
- **THEN** environment acceptance fails and identifies the missing subsystem

### Requirement: Public ingress terminates at a controlled private boundary
Only intended public HTTP, HTTPS, and WebSocket entry points SHALL be reachable from the internet. Application instances and stateful services SHALL use private addresses, and administrative endpoints SHALL require a separately authenticated management path.

#### Scenario: User request reaches a healthy API instance
- **WHEN** a client connects to the public application hostname over HTTPS or WebSocket
- **THEN** traffic crosses the public load-balancing boundary and reaches only a healthy private API instance with forwarding identity preserved

#### Scenario: Stateful endpoint is scanned publicly
- **WHEN** an internet client attempts to reach PostgreSQL, MongoDB, Valkey/Redis, RabbitMQ, Neo4j, telemetry, proxy metrics, or management ports
- **THEN** the connection is denied before application authentication is involved

### Requirement: Dependency capability gates fail before traffic activation
The exact provisioned service versions SHALL pass executable compatibility gates for PostgreSQL extensions and access methods, Redis-compatible commands and TLS behavior, Neo4j APOC procedures, RabbitMQ topology, object-storage operations, and crawler browser startup before an environment can receive application traffic.

#### Scenario: PostgreSQL keyword access method is unavailable
- **WHEN** the target PostgreSQL service cannot create every migration-required extension or expose the required vector and keyword access methods
- **THEN** migration and release activation stop before API traffic is switched to the release

#### Scenario: Redis-compatible service lacks a required active command
- **WHEN** a command exercised by a currently reachable application path is unavailable or semantically incompatible
- **THEN** deployment acceptance fails with the command family identified and no production endpoint is changed

#### Scenario: Browser cannot launch in the final image
- **WHEN** the final crawler image is executed with production user, filesystem, sandbox, proxy, and certificate settings
- **THEN** acceptance fails unless a pinned browser launches and completes the controlled navigation contract

### Requirement: Secrets are environment-scoped and absent from source and plans
Secret values SHALL NOT be committed, rendered in infrastructure plans, exposed through nonsensitive outputs, embedded in images, or shared across test and production. Runtime roles SHALL receive only the credentials required for their declared dependencies.

#### Scenario: Infrastructure plan is inspected
- **WHEN** an operator reviews a saved plan and its ordinary outputs
- **THEN** database passwords, API keys, TLS private keys, broker credentials, signing secrets, and object-storage secrets are absent in plaintext

#### Scenario: Crawler credential is compromised
- **WHEN** the crawler worker's runtime credentials are used outside the crawler role
- **THEN** they cannot administer the broker, select another Redis upstream, access application databases, read arbitrary object storage, or retrieve unrelated secrets

### Requirement: Releases are immutable, ordered, and reversible
Each release SHALL identify an immutable application image and configuration revision. Compatibility checks and one migration execution SHALL precede traffic activation; API instances SHALL roll without dropping all capacity; workers SHALL drain within their declared task boundary; and rollback SHALL restore the prior image without rerunning destructive migrations.

#### Scenario: Release succeeds
- **WHEN** a release candidate passes image, dependency, migration, readiness, queue-coverage, and smoke gates
- **THEN** API traffic is shifted incrementally, workers are replaced by role, exactly one scheduler remains active, and the deployed revision is observable

#### Scenario: New API revision fails readiness
- **WHEN** a replacement API instance fails readiness before receiving traffic
- **THEN** the healthy prior revision remains active and the failed revision is removed without changing persistent data

#### Scenario: Worker is replaced with an active long-running task
- **WHEN** a worker receives the deployment termination signal while a task is within its configured hard time limit
- **THEN** the deployment grants a drain interval greater than that hard limit or explicitly requeues the task without reporting false success

### Requirement: Persistent state has independent backup and verified restore paths
PostgreSQL, MongoDB, Redis-compatible durable state, RabbitMQ definitions and durable queues, Neo4j, object metadata, deployment configuration, and required certificate material SHALL have documented recovery mechanisms. A backup SHALL NOT count as recoverable until a restore has been verified in an isolated target.

Scale: each persistent system and environment | Meter: age of the newest successful isolated restore verification | MUST: production verification age is at most 31 days and test verification age is at most 93 days.

#### Scenario: Scheduled recovery drill completes
- **WHEN** a recovery drill restores the newest eligible backups into an isolated environment
- **THEN** schema checks, representative reads, ownership boundaries, queue topology, graph procedures, object access, and application smoke tests pass before the drill is recorded successful

#### Scenario: Provider-managed backup cannot be downloaded directly
- **WHEN** a provider backup supports only in-provider restoration
- **THEN** the recovery plan also defines an application-level export or an independently restorable copy for regional or account-level recovery

### Requirement: Observability distinguishes component and flow failures
The environment SHALL collect structured logs, metrics and traces with environment, service role, deployed revision and correlation identity. Alerts SHALL distinguish ingress, API, queue producer, broker, each worker role, scheduler, crawler boundary, dependency, backup, and restore failures.

Scale: production telemetry for critical services | Meter: interval between a synthetic failure and an actionable alert | MUST: alert delivery is verified within 10 minutes.

#### Scenario: Accepted job never starts
- **WHEN** a job is accepted but does not begin within its queue-specific threshold
- **THEN** telemetry identifies whether publishing, broker routing, consumer presence, worker execution, or result persistence failed

#### Scenario: Telemetry destination is unavailable
- **WHEN** the remote telemetry destination cannot be reached
- **THEN** application work remains bounded, local buffering is bounded, and the outage does not exhaust application disk or memory

### Requirement: The test environment proves production-sensitive boundaries
The test environment SHALL support a cost-reduced profile, but it SHALL exercise the same image digests, migrations, queue names, role commands, private dependency protocols, crawler policy decisions, secret-injection mechanism, deployment ordering, and restore procedures used by production.

#### Scenario: Cost-reduced topology is used
- **WHEN** multiple low-risk roles are colocated in test
- **THEN** role commands and network identities remain distinct and the acceptance report identifies every production separation that colocation does not prove

#### Scenario: Release is promoted
- **WHEN** a revision is proposed for production
- **THEN** the exact immutable images have passed the production-sensitive test gates and promotion does not rebuild them

### Requirement: Infrastructure cost and capacity are reviewable before apply
Every environment plan SHALL expose nonsecret resource counts, instance classes, storage allocations, managed-service plans, public IPs, and estimated recurring cost. A production change exceeding the configured budget threshold SHALL require an explicit review record.

Scale: estimated monthly OVHcloud infrastructure spend excluding variable third-party model/API usage | Meter: plan-time cost estimate versus configured environment budget | MUST: no unreviewed production plan exceeds its budget by more than 10 percent.

#### Scenario: Capacity change exceeds budget threshold
- **WHEN** a plan raises estimated production infrastructure cost by more than the allowed threshold
- **THEN** automated apply is blocked until the cost delta and capacity rationale are reviewed

### Requirement: Legacy provider infrastructure has an explicit disposition
The existing GCP infrastructure SHALL remain isolated from OVHcloud state and tooling until an authorized migration task either archives it as historical reference or removes it after proving that no active environment depends on it.

#### Scenario: OVHcloud infrastructure is applied
- **WHEN** an operator initializes or applies the OVHcloud environment
- **THEN** no GCP provider, state, credential, resource, or deployment script is invoked or modified

#### Scenario: Legacy GCP disposition is recorded
- **WHEN** OVHcloud production acceptance and the active-use audit are complete
- **THEN** the GCP tree is explicitly retained as isolated legacy reference or a separate archive/removal change is proposed with repository-reference and active-environment evidence; this change does not silently delete it

### Requirement: Edge health and host resources fail safely
The public load balancer SHALL use the versioned liveness route for backend membership and SHALL NOT expose application, metrics, data, or administration host ports directly. Every runtime SHALL declare CPU, memory, process, log-storage, database-connection, and shutdown budgets that fit its host with documented headroom.

#### Scenario: Optional dependency is degraded
- **WHEN** readiness reports a noncritical dependency or host-resource warning while the API process remains live
- **THEN** the load balancer does not remove the backend solely because aggregate readiness is degraded, and the condition remains visible to release automation and alerts

#### Scenario: Capacity envelope is exceeded
- **WHEN** declared service limits plus operating-system and failure headroom exceed host capacity or aggregate database pools exceed the connection budget
- **THEN** deployment acceptance fails before the affected services are started

### Requirement: Runtime artifacts are complete and role-specific
The release SHALL provide digest-pinned artifacts for API, default worker, ingestion worker, crawler worker, migration, and policy-proxy responsibilities. The API artifact SHALL NOT start a browser, the crawler SHALL own its compatible browser, the ingestion artifact SHALL provide its required model assets without startup downloads, and the migrator SHALL contain all migration and reconciliation inputs.

#### Scenario: Fresh database is deployed
- **WHEN** the migration artifact runs against a clean compatible PostgreSQL target
- **THEN** it upgrades to the single Alembic head, reconciles personal tenants, passes the reconciliation check, and exits successfully before API activation

#### Scenario: API process starts
- **WHEN** the API artifact completes lifespan startup
- **THEN** no Chromium or crawler-browser process is created by any Uvicorn worker

### Requirement: Production configuration is capability-aware
Production startup SHALL require strong nonplaceholder core secrets and SHALL require integration secrets only when the owning feature is enabled. OTLP enablement and endpoint SHALL be explicit for every deployed role.

#### Scenario: Optional provider is disabled
- **WHEN** an environment disables an optional email, search, reranking, extraction, or external-vector integration
- **THEN** that provider's absent credential does not prevent unrelated API and worker roles from starting

#### Scenario: Telemetry collector is absent
- **WHEN** OTLP is enabled but the configured local collector is unavailable
- **THEN** the deployment fails its telemetry preflight instead of silently retrying a default localhost endpoint
