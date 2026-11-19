# ADRs: deploy-ovhcloud-non-kubernetes-infrastructure

## ADR-001: Separate cloud provisioning, host configuration, and application release

- **Status:** accepted
- **Date:** 2026-11-16
- **Supersedes:** none

### Context

OVH resources, operating-system state, and application revisions change at different rates and have different rollback semantics. The application needs multiple fixed runtime roles but not a cluster scheduler.

### Decision

Terraform/OpenTofu owns OVH/OpenStack resources, cloud-init performs minimal first boot, Ansible owns durable host state, and systemd-managed Docker Compose projects own digest-pinned application releases. An application rollback does not run Terraform.

### Alternatives

- Kubernetes was rejected because its control plane and operational surface are disproportionate to the planned scale.
- A monolithic Terraform `remote-exec` or shell deployment was rejected because it couples replacement, configuration, secrets and release lifecycles.
- Docker Swarm and Nomad were rejected because the fixed placement model does not yet justify another scheduler.

### Consequences

The ownership boundary is inspectable and each layer is independently repeatable. Operators must maintain three intentionally small tool surfaces and explicit host inventory because Compose does not reschedule across machines.

## ADR-002: Self-host PostgreSQL for the hybrid retrieval extension contract

- **Status:** accepted
- **Date:** 2026-11-16
- **Supersedes:** none

### Context

The migration chain requires `pg_textsearch`, `vector`, `vectorscale`, `pg_trgm`, `unaccent`, `diskann`, and `bm25`. OVH Managed PostgreSQL does not expose the complete contract and cannot accept arbitrary extensions.

### Decision

Run a custom, digest-pinned PostgreSQL image on OVH compute. Select its major version only after a clean executable gate proves extensions, preload configuration, access methods, all migrations and representative retrieval. Use pgBackRest/WAL archival for baseline recovery and Patroni/etcd only in the HA profile.

### Alternatives

- OVH Managed PostgreSQL was rejected because a fresh schema cannot be proven compatible.
- Removing keyword/vector access methods or adding a separate search engine was rejected as an application and relevance redesign outside this deployment change.
- Pinning PostgreSQL 17 or 18 by assumption was rejected; compatibility evidence owns the selection.

### Consequences

Current retrieval behavior remains intact, but the team owns PostgreSQL patching, failover, backups and extension-image supply chain. A future managed migration requires a new change and retrieval-quality evidence.

## ADR-003: Enforce crawler non-bypass at the host network boundary

- **Status:** accepted
- **Date:** 2026-11-16
- **Supersedes:** none

### Context

Docker internal networks are local to one host, provider security groups may retain implicit egress, and a proxy environment variable is bypassable by compromised code. The crawler must reach a broker, DNS, an HTTP(S) policy proxy and a fixed-destination mTLS Valkey relay but nothing else.

### Decision

Place crawler workers on private-only VMs and proxy/relay processes on separate dual-homed VMs. Make default-drop host nftables rules the authoritative crawler enforcement point, with exact destination/port exceptions. Keep Smokescreen's actual policy explicit: public destinations are open through the proxy, while private, reserved, link-local and metadata destinations are denied. Prove non-bypass from inside the released crawler container.

### Alternatives

- Docker `internal: true` alone was rejected because it does not extend across hosts.
- Provider security groups alone were rejected because they cannot reliably distinguish container identities and may contain provider-managed egress.
- Co-locating worker and proxy is allowed only in cost-reduced test and does not satisfy production evidence.

### Consequences

The boundary survives proxy-variable removal and multi-host placement. nftables rendering, Docker-chain ordering and in-container negative tests become release-critical operational code.

## ADR-004: Use managed MongoDB and Valkey but self-host RabbitMQ and Neo4j

- **Status:** accepted
- **Date:** 2026-11-16
- **Supersedes:** none

### Context

Managed MongoDB and Valkey remove undifferentiated HA/backup work, while OVH has no managed RabbitMQ or Neo4j service matching the application. Valkey module helpers exist but are not on reachable production paths; Neo4j Community cannot truthfully provide clustering.

### Decision

Use private OVH Managed MongoDB and Valkey after exact command/TLS/failover probes. Run RabbitMQ 4 on compute with quorum queues and per-role credentials. Run one private Neo4j Community instance with APOC, treat graph paths as degradable, and back it up. HA RabbitMQ uses three nodes; Neo4j Enterprise or an external graph service is a later licensing/trust decision.

### Alternatives

- Self-hosting every datastore was rejected due to needless operational burden.
- Treating Redis modules as required was rejected because reachable-call analysis found no active callers; activation later changes the compatibility contract.
- Pretending Community Neo4j is HA was rejected.

### Consequences

The data plane has explicit ownership and truthful SPOFs. Managed-service SKU/version drift is blocked by executable probes, and graph-critical availability requires a later ADR.

## ADR-005: Build role-complete immutable artifacts and promote digests

- **Status:** accepted
- **Date:** 2026-11-16
- **Supersedes:** none

### Context

The current common image lacks a browser and migration inputs, while the API eagerly initializes Crawl4AI and ingestion downloads models at runtime. Tags and environment rebuilds do not prove that test and production run identical bits.

### Decision

Publish separate API, default-worker, ingestion-worker, crawler-worker and migrator images plus digest-pinned Smokescreen and PostgreSQL artifacts. The API never starts a browser; the crawler contains its pinned browser; ingestion contains required model assets; the migrator contains Alembic and tenant reconciliation inputs. Build once, scan/sign, test, and promote identical digests.

### Alternatives

- One universal image was rejected because it expands attack surface, size and runtime coupling while still omitting role-specific prerequisites.
- Runtime browser/model downloads were rejected because they make rollout depend on mutable external endpoints.
- Tags alone were rejected because they are mutable identities.

### Consequences

Build pipelines become more complex but failures move to CI and environment parity becomes verifiable. Dependency layers should be shared to control build time and registry cost.

## ADR-006: Terminate public traffic at the OVH load balancer and use liveness for membership

- **Status:** accepted
- **Date:** 2026-11-16
- **Supersedes:** none

### Context

The current Caddy production route is inconsistent, direct port 5000 can bypass it, and aggregate readiness returns 503 for optional dependency and host warnings. Using that readiness result for load-balancer membership can remove an otherwise live fleet during a dependency incident.

### Decision

Terminate TLS at the OVH load balancer, keep API/metrics/administration ports private, use `/api/v1/health/live` for backend membership, and retain readiness/deep health as release and alerting signals. Set forwarding depth and WebSocket idle timeouts for exactly one public proxy hop.

### Alternatives

- Caddy in the production path was rejected as an unnecessary second edge with currently invalid configuration.
- Aggregate readiness as an LB probe was rejected because it conflates optional degradation with process death.
- Disabling health probes was rejected because it hides real process failure.

### Consequences

Ingress has one authority and optional dependency incidents do not create a fleet-wide eviction loop. The load balancer becomes a critical provisioned component and release automation must inspect both liveness and readiness.

## ADR-007: Isolate test and production by project, state, identity, and network

- **Status:** accepted
- **Date:** 2026-11-16
- **Supersedes:** none

### Context

Terraform workspaces alone do not prevent credential, network, bucket or destroy-scope mistakes. The functional test profile is intentionally cheaper and may colocate roles, but must not share authority with production.

### Decision

Use separate OVH Public Cloud projects, remote states, IAM credentials, private networks, registry namespaces, secret roots, object namespaces and backup roots for test and production. Test may colocate roles, but production promotion requires topology-faithful crawler and exact managed-service gates.

### Alternatives

- A single project split only by names/workspaces was rejected because the blast radius still crosses environments.
- Making test identical at all times was rejected as unnecessarily expensive; temporary production-sensitive gates provide the required evidence.

### Consequences

Accidental cross-environment mutation is substantially harder and test can be destroyed safely. Duplicate foundations cost more and require explicit promotion rather than shared mutable resources.

## ADR-008: Optional integrations are enabled explicitly, never inferred from secrets

- **Status:** accepted
- **Date:** 2026-11-16
- **Supersedes:** none

### Context

Production currently requires every provider credential even when a deployment does not use that provider. Inferring enablement from credential presence makes operator intent ambiguous and can accidentally activate reachable code.

### Decision

Add a typed `ENABLED_OPTIONAL_INTEGRATIONS` set to settings. Core capabilities remain mandatory. Each optional provider has one enum member, its credential is required exactly when enabled, unknown members fail startup, and credential presence alone never enables it. Role manifests declare the allowed set and receive only matching secrets. Invocation of a disabled adapter fails closed with an observable unavailable result.

### Alternatives

- Credential-presence enablement was rejected because a mounted secret can silently change behavior.
- A separate Boolean for every provider was rejected as a growing, contradictory configuration surface.
- Continuing to require all credentials was rejected because disabled integrations should not block unrelated roles.

### Consequences

Operator intent is explicit and testable, but settings and adapter factories need a shared capability registry and disabled-path tests. Adding an optional integration requires an enum member, ownership metadata and secret mapping.

## ADR-009: Fence Celery Beat with an environment-scoped lease

- **Status:** accepted
- **Date:** 2026-11-16
- **Supersedes:** none

### Context

Compose/systemd ordering can serialize a normal deployment but cannot prevent duplicate schedule publishers during host partitions, stale processes or operator error.

### Decision

Require Beat to acquire one environment-scoped Valkey lease with a cryptographically random owner token. Acquisition is atomic; renewal and release compare the stored owner token atomically before mutation. Beat proves/renews ownership before every publication cycle and emits nothing after renewal failure. Release automation stops the old Beat, observes its token release/expiry, and then starts the replacement. A stale owner cannot release a successor's lease. Scheduled tasks remain idempotent.

### Alternatives

- Stop/start ordering alone was rejected because it is not a distributed fence.
- Two active Beat replicas were rejected because Celery Beat does not deduplicate schedule emission.
- A PostgreSQL advisory-lock implementation was rejected to avoid coupling scheduler liveness to the primary database session.

### Consequences

Duplicate publication is observable and prevented across hosts. Valkey availability becomes a Beat leadership dependency, so lease loss intentionally produces no scheduler rather than two schedulers.

## ADR Review Verdict

- [x] Every durable decision in design.md is captured here, or no durable decision exists
- [x] Every superseded decision names its prior change and ADR id
- [x] No repository-level ADR file was created
