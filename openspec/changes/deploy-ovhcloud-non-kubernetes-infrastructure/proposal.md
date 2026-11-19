# Change: Deploy the complete application on OVHcloud without Kubernetes

**Class:** L

## Why

The repository has no current deployment definition that can run its complete API, three role-isolated worker pools, scheduler, crawler security boundary, databases, broker, object storage, and observability stack. The existing `infra/gcp/` tree models a stale single-service Cloud Run deployment, while `docker-compose.prod.yml` is an incomplete single-host contract and cannot be treated as a production OVHcloud architecture.

## What Changes

- Introduce a non-Kubernetes OVHcloud deployment capability with separate test and production projects, private networking, public-to-private ingress, role-specific compute, stateful-service placement, and explicit failure domains.
- Define Terraform/OpenTofu as the owner of OVHcloud resources, Ansible and cloud-init as host-configuration owners, and digest- or commit-pinned Docker Compose/systemd deployments as the application-release mechanism.
- Add a new `infra/ovhcloud/` implementation surface and classify `infra/gcp/` as legacy provider-specific infrastructure pending an explicit archive-or-delete decision; the implementation SHALL NOT silently mutate the GCP state.
- Deploy the API, default worker, ingestion worker, crawler worker, and exactly one Celery Beat scheduler as distinct runtime roles with queue-coverage, readiness, resource, graceful-shutdown, and rollback evidence.
- Deploy the crawler as a route- and firewall-denied security cell whose worker can reach only its broker, mutually authenticated fixed-destination Redis relay, DNS resolver, and policy-enforcing HTTP(S) proxy; a production profile SHALL enforce the boundary with host networking controls independently of proxy environment variables or provider security-group defaults.
- Define authoritative OVHcloud mappings for PostgreSQL, MongoDB, Valkey/Redis, RabbitMQ, Neo4j with APOC, S3-compatible object storage, OpenTelemetry collection, secrets, DNS, certificates, block volumes, backups, and restore evidence.
- Make PostgreSQL extension compatibility, Valkey command/module compatibility, Neo4j plugin availability, and browser-image readiness fail-fast deployment gates rather than post-release discoveries.
- Add one-shot migration, rolling API deployment, warm worker drain, singleton Beat, rollback, disaster-recovery, observability, cost-budget, and environment-destruction procedures.
- Preserve the existing HTTP, WebSocket, MCP, Celery task names, queue names, payloads, result schemas, ownership rules, and crawler public API.
- **BREAKING**: the current production Compose layout and `infra/gcp/` deployment commands SHALL no longer be documented as a complete or supported production deployment for this application.

## Capabilities

### New Capabilities

- `ovhcloud-infrastructure-deployment` — provision, configure, release, observe, back up, restore, and verify the complete non-Kubernetes OVHcloud test and production environments.

### Modified Capabilities

- `celery-worker-deployment` — require all three role-isolated worker pools, singleton Beat, aggregate queue coverage, runtime sizing, and task-safe deployment draining in the OVHcloud deployment.
- `crawler-egress-policy` — require OVHcloud acceptance evidence that the worker cannot bypass the proxy or reach unapproved internal/public destinations while broker and Redis-relay control paths remain available.
- `crawler-worker-execution` — require the final crawler image to contain and launch its pinned browser/runtime dependencies before a deployment is accepted.

## Impact

- **Code**: `infra/ovhcloud/` (new Terraform/OpenTofu, Ansible, Compose overlays, systemd and runbooks), `infra/gcp/` disposition, deployment CI, capability-specific application/migrator/crawler images, `docker-compose.prod.yml` decomposition, feature-conditional production settings, API lifespan/health behavior, environment templates, migration/compatibility probes, and architecture/deployment documentation.
- **APIs**: no intended public HTTP, WebSocket, MCP, task-name, queue-name, or payload changes; health and deployment-readiness probes gain environment-level acceptance coverage.
- **Dependencies**: OVHcloud Public Cloud projects, vRack/private networks, Gateway, Floating IP, Load Balancer, VMs, block and object storage, Secret Manager/IAM, managed or self-hosted data services selected by capability gates, Terraform/OpenTofu, Ansible, Docker Compose, RabbitMQ, Neo4j/APOC, OpenTelemetry Collector, Chromium/Playwright, and existing external AI/search/email/payment providers.
- **Async flows**: the existing `default`, `ingestion`, `crawler`, and dead-letter queue contracts and Beat timers are preserved, but their deployment ownership, network paths, readiness evidence, graceful drain, and failure diagnostics become explicit; discovery is therefore mandatory for this L-class change.
