# Design: deploy-ovhcloud-non-kubernetes-infrastructure

## Context

See `proposal.md` for motivation. The current repository has three incompatible deployment stories: a stale GCP Cloud Run tree that models one web process, a Railway planning document that deliberately leaves the crawler security cell elsewhere, and a production Compose file that places selected processes on one Docker host while relying on undeclared external MongoDB, Redis and Neo4j endpoints. None is authoritative for a complete OVHcloud environment.

The code establishes hard deployment facts:

- PostgreSQL is startup-critical; MongoDB, Redis and Neo4j may degrade at API startup but are required by reachable features.
- Celery selects exactly one of `default`, `ingestion`, or `crawler` per worker process, and Beat is a separate process.
- Default, ingestion and crawler concurrency defaults are 8, 2 and 2; each prefork child has a 1,000,000 KiB recycle threshold.
- Global Celery soft/hard defaults are 270/300 seconds, but ingestion/search tasks override them to 1,500/1,620 seconds and crawler tasks to 330/360 seconds; the current 45-second Compose stop grace is unsafe for every role.
- PostgreSQL migrations require `vector`, `vectorscale`, `pg_textsearch`, `pg_trgm`, and `unaccent`, plus `diskann` and `bm25` access methods.
- The final production image installs no Chromium/Playwright operating-system runtime, while the API lifespan currently attempts to start Crawl4AI in every Uvicorn worker.
- The current production image omits `alembic.ini` and migration/reconciliation scripts, so it cannot be the release migrator.
- The current Compose liveness path is `/health/live`, but the actual API route is `/api/v1/health/live`; aggregate readiness can return 503 for optional dependencies and host-resource warnings and is unsafe as a load-balancer membership probe.
- Production settings currently require credentials for integrations that may be disabled and default OTLP to `localhost:4317`.
- Production enables OTLP by default but the current Compose file declares no collector.
- The crawler must have no direct public route; proxy environment variables are defense in depth, not the boundary.

The exact OVHcloud region is a deployment input selected only after the target project proves service/SKU availability, latency and data-residency fit. Whichever primary region is selected is treated as one regional failure domain. Regional disaster recovery uses independently restorable exports and object copies; this design does not pretend that multiple VMs in one region are multi-region resilience.

## Goals / Non-Goals

**Goals:**

- Make `infra/ovhcloud/` the reproducible, reviewable deployment authority for test, baseline production, and high-availability production profiles.
- Preserve current application and async wire contracts while making every runtime and dependency operationally owned.
- Provide a low-cost functional test path and a topology-faithful promotion gate.
- Enforce the crawler boundary without Kubernetes network policy.
- Resolve database/service compatibility before traffic activation.
- Make rollback, backup, restore, cost and failure evidence first-class release artifacts.

**Non-Goals:**

- Introduce Kubernetes, Nomad, Docker Swarm, service-mesh sidecars, or automatic horizontal autoscaling.
- Refactor hybrid retrieval to fit OVH Managed PostgreSQL in this change.
- Make Neo4j Community highly available or purchase an Enterprise licence automatically.
- Move Gemini, Tavily, reranking, Resend, Razorpay, OAuth or LangSmith provider services into OVHcloud.
- Change public API schemas, Celery task names, queue names, routing keys, job payloads or retry semantics.
- Delete `infra/gcp/` before an explicit active-use audit and user-approved disposition.

## Decisions

### D1: Separate infrastructure provisioning from application release

**Choose:** Terraform/OpenTofu for OVH/OpenStack resources, cloud-init for first boot, Ansible for host state, and systemd-managed Docker Compose projects for runtime releases.

**Why:** Infrastructure resources and application revisions have different lifecycles. Terraform owns projects, networks, security groups, gateways, load balancers, instances, volumes, buckets, managed databases, registry and DNS. Ansible owns Docker, nftables, users, filesystem permissions, certificates, collectors and systemd units. Release automation changes only immutable image references and rendered nonsecret configuration. Terraform `remote-exec`, a monolithic shell deployer, and Kubernetes were rejected because they conflate lifecycles or add an orchestration control plane the current scale does not require.

The implementation layout is:

```text
infra/ovhcloud/
├── README.md
├── terraform/
│   ├── modules/{project,network,security,compute,load-balancer,block-storage,object-storage,registry,managed-database,dns}/
│   └── environments/{test,production}/
├── ansible/
│   ├── inventories/{test,production}/
│   ├── roles/{base,docker,firewall,secrets,api,worker,crawler,proxy,postgres,rabbitmq,neo4j,observability}/
│   └── playbooks/{bootstrap,configure,deploy,restore-drill}.yml
├── compose/{api,workers,crawler,proxy,postgres,rabbitmq,neo4j,observability}.yml
├── systemd/
├── scripts/{plan,apply,deploy,rollback,acceptance,backup,restore-drill}
└── runbooks/
```

### D2: Isolate test and production at the OVH project and state boundary

**Choose:** one OVH Public Cloud project, Terraform state, IAM scope, vRack/private network, bucket namespace, registry namespace and secret namespace per environment.

**Why:** Workspace-only separation makes accidental destroy, credential reuse and state collision too easy. Production CIDRs use `10.20.0.0/16`; test uses `10.30.0.0/16`. Each is divided into edge/app, worker, data, crawler-control and management subnets. Only the public load balancer receives a Floating IP. Ordinary private workloads use the Gateway for outbound access. The crawler worker subnet has no general SNAT route.

Terraform state uses an OVH S3-compatible bucket with versioning, encryption, separate state credentials, `use_lockfile` when conditional-write compatibility passes, and serialized CI concurrency as a second guard. If lockfile compatibility fails, the backend moves to a locking remote-state service without changing resource topology; unlocked shared state is forbidden.

### D3: Use three capacity profiles instead of pretending one topology fits every stage

**Choose:** `test-functional`, `production-baseline`, and `production-ha` profiles driven by reviewed variables.

| Role | test-functional | production-baseline | production-ha |
|---|---|---|---|
| API | 1 shared core VM | 2 × 4 vCPU / 16 GB | 2+ × 4 vCPU / 16 GB |
| Default + ingestion workers | shared core, concurrency 2/1 | 1 × 8 vCPU / 32 GB, isolated containers | 2 × 8 vCPU / 32 GB, role containers on both |
| Beat | one container on core | one fenced container on worker host | one fenced container with replacement lock |
| Crawler worker | shared host allowed, concurrency 1 | 1 × 4 vCPU / 16 GB private-only | 2 × 4 vCPU / 16 GB private-only |
| Egress proxy + Redis relay | shared crawler host, marked non-equivalent | 1 × 2 vCPU / 8 GB separate VM | 2 × 2 vCPU / 8 GB behind private failover address |
| PostgreSQL | single custom compatible PostgreSQL on core/block volume | single custom compatible PostgreSQL with continuous WAL archive | 3 × 4 vCPU / 16 GB Patroni/etcd cluster |
| RabbitMQ | single node on core | single durable node | 3 × 2 vCPU / 8 GB quorum cluster |
| MongoDB | managed compatible test plan or local replica | OVH Managed Production | OVH Managed Production |
| Valkey | managed compatible test plan or local replica | OVH Managed Production | OVH Managed Production |
| Neo4j | single local node | 1 × 8 vCPU / 32 GB, APOC | same; Enterprise clustering is a separate decision |
| Observability | collector + short retention on core | 1 × 4 vCPU / 16 GB | collector replicas plus dedicated storage |

`test-functional` reduces concurrency and co-locates components to make the first environment affordable. Before production promotion, temporary topology-faithful crawler/proxy hosts and exact managed-service SKUs are provisioned for acceptance. `production-baseline` is explicitly allowed to have PostgreSQL, RabbitMQ, Neo4j and observability single points of failure with documented RTO; it is not labelled highly available. `production-ha` removes the PostgreSQL and broker host SPOFs but does not claim regional or Neo4j Community HA.

### D4: Self-host the PostgreSQL extension contract; select the major by executable proof

**Choose:** a digest-pinned custom PostgreSQL image containing the migration-required extension set. The currently pinned PostgreSQL 18/Timescale image is the first candidate, but no major version is accepted until a clean build proves extension creation, required preload libraries, `diskann`/`bm25` index creation, the entire Alembic chain, and representative retrieval. If it fails, build and prove PostgreSQL 17 rather than silently dropping an access method. Baseline runs one node with continuous WAL archive; HA runs three Patroni nodes with co-located etcd quorum and per-host local HAProxy routing based on Patroni leader health.

**Why:** OVH Managed PostgreSQL documents `pgvector`/`pgvectorscale` but does not list `pg_textsearch`, and managed users cannot install arbitrary extensions. Treating it as compatible would make a fresh Alembic migration fail. Refactoring retrieval or adding OpenSearch is a separate application architecture change. A repository image tag is not compatibility evidence, so the major-version choice is made by an executable contract rather than optimism.

The PostgreSQL image build must run `CREATE EXTENSION` and access-method probes in CI. Production data lives on encrypted block volumes. pgBackRest archives WAL continuously and writes full/differential backups to a separate OVH Object Storage bucket. Application hosts run a local private HAProxy endpoint that routes only to the current Patroni primary; SQLAlchemy retains application pooling. A failover drill must prove reconnection and migration safety.

Future migration to OVH Managed PostgreSQL requires a separate change that removes or replaces `pg_textsearch` and proves retrieval-quality parity.

### D5: Use managed MongoDB and Valkey, but gate exact behavior

**Choose:** OVH Managed MongoDB Production and OVH Managed Valkey Production on private networking for production; test uses the cheapest private-compatible SKU available or an ephemeral local substitute followed by an exact-SKU promotion probe.

**Why:** MongoDB and Valkey operational HA, patching and backups are not differentiating work. Valkey is accepted only for active application commands, TLS, ACL, Streams, pub/sub, Lua/functions if active, TTL and failover semantics. The repository's unused `FT.*` and `BF.*` helpers do not make module support a day-one requirement; activating a caller makes those commands part of the compatibility gate. Redis state is classified as coordination/session/ephemeral job state, not the sole durable business authority. Regional Redis recovery may invalidate sessions and mark in-flight jobs for truthful retry rather than inventing success.

### D6: Run RabbitMQ and Neo4j on OVH compute with explicit criticality

**Choose:** RabbitMQ 4 with TLS, per-role users/vhosts, durable definitions and quorum queues in HA; Neo4j Community on one private VM with APOC required and GDS optional.

**Why:** OVH has no managed RabbitMQ or Neo4j offering matching the application. Broker credentials are least privilege: producers publish declared exchanges, each worker consumes only its role queue, and the crawler cannot administer the broker. Baseline RabbitMQ is a documented SPOF; HA uses three nodes and quorum queues. Neo4j is an enrichment/graph dependency that the application can degrade around. A single backed-up Community node is preferred over pretending that unlicensed clustering exists. If graph availability becomes business-critical, Neo4j Enterprise clustering or a managed external service requires a separate cost/licensing decision.

Rabbit definitions are exported after topology changes. Durable message data uses block volumes, but volume snapshots do not replace quorum replication or application reconciliation.

### D7: Enforce crawler egress with separate machine identities

**Choose:** a private-only crawler VM and a separate dual-homed proxy/relay VM in production, with the crawler host's nftables policy as the enforcement point.

**Why:** Docker `internal` networks are host-local and cease to prove isolation when RabbitMQ or the relay moves to another VM. A security group also cannot distinguish a compromised crawler container from an allowed proxy container, and provider-injected egress defaults must not be assumed removable or represented in Terraform state. The crawler VM has no public IP. Its default-drop nftables policy, keyed to container bridge/source identity, allows only internal DNS, RabbitMQ TLS, the proxy listener, and the relay mTLS listener; all other forwarding and host-output paths from the crawler identity are denied. The proxy/relay VM alone can reach public TCP 80/443 through the Gateway and exactly one Valkey upstream. Smokescreen's current policy is accurately described as public-destination-open with private/link-local/metadata denial, not as a domain allowlist. HAProxy exposes no caller-selected upstream for Redis.

Acceptance originates inside the final crawler container and proves direct public, metadata, private subnet and database-port denial. Proxy success alone is never accepted as non-bypass evidence.

### D8: Split images by capability and promote without rebuilding

**Choose:** five immutable application image artifacts: API, default worker, ingestion worker, crawler worker, and migrator. Beat uses the default-worker artifact with a different command. Smokescreen and the custom PostgreSQL image are independently built, scanned and digest-pinned infrastructure artifacts.

**Why:** The current common image forces API, billing and crawler processes to share heavy Torch/Docling/Crawl4AI dependencies and still lacks Chromium. The API lifespan must stop eagerly starting a browser; crawler construction belongs only to the crawler capability. The crawler image pins Chromium and installs runtime libraries during build. The ingestion image owns CPU-only Torch, document parsers, and pre-baked Hugging Face/Docling model assets so production startup does not download model content. The migrator contains `alembic.ini`, every revision and the tenant reconciliation scripts. All artifacts are scanned, signed, labelled with Git SHA/build date, pushed once to the OVH private registry, and referenced by digest. Test-to-production promotion copies or references identical digests; it never rebuilds source.

### D9: Fetch runtime secrets outside Terraform state

**Choose:** Terraform creates Secret Manager/IAM objects and nonsecret references, while an authenticated deploy step writes secret versions. Hosts fetch only role-scoped values into root-owned tmpfs paths before Compose starts.

**Why:** `random_password`, secret values in Terraform variables, cloud-init user data, ordinary outputs and committed `.env.production` files leak values into state or logs. Compose consumes files from `/run/application-secrets/<role>/`; systemd orders secret fetch before runtime start and removes files on stop. Production settings expose an explicit typed `ENABLED_OPTIONAL_INTEGRATIONS` set; credentials never enable a feature by their mere presence. Core capabilities and their secrets are always required, an optional adapter's secret is required exactly when its enum member is enabled, unknown members fail validation, and a disabled adapter fails closed with an observable unavailable result if invoked. The deployment capability matrix declares this set per role and mounts only the corresponding secrets. Crawler relay certificates use a dedicated CA, short-lived leaf certificates and rotation overlap. Test and production roots never overlap.

### D10: Release in a fixed transactional order

**Choose:** one serialized environment release with preflight, backup checkpoint, migration, canary API, worker-role rollout, singleton Beat handoff, smoke, and recorded completion.

Order:

1. Resolve immutable digests and render nonsecret configuration.
2. Verify signatures/digests, browser startup and dependency capability probes.
3. Confirm newest backups and create the configured pre-release checkpoint.
4. Run `alembic upgrade head` exactly once from a one-shot migration container.
5. Start one new API instance out of load-balancer rotation; pass liveness, readiness, deep probes and smoke tests; then add it and drain one old instance at a time. OVH load-balancer membership uses `/api/v1/health/live`; readiness remains an operator/release signal and may truthfully report optional degradation.
6. Roll default, ingestion and crawler workers independently. Both Compose `stop_grace_period` and systemd `TimeoutStopSec` are derived from the maximum registered hard limit per role plus at least 30 seconds: currently at least 330 seconds for default, 1,650 seconds for ingestion, and 390 seconds for crawler, unless a role-specific tested redelivery contract permits a shorter value.
7. Fence old Beat through an environment-scoped Valkey lease with a random owner token and atomic acquire, compare-and-renew and compare-and-release operations. Beat proves the lease before every publication cycle and stops publication immediately when renewal fails. The old process exits, its owner token releases or expires, and only then may the replacement publish; prove stale or partitioned owners cannot publish or release a successor's lease.
8. Run queue coverage, crawler boundary, WebSocket, upload/ingestion, object, graph and billing-schedule synthetic checks.
9. Record revision, infrastructure plan, compatibility report, cost delta and rollback pointer.

Rollback returns API/workers/Beat to prior digests. Database rollback is forward-only: reversible migrations get explicit downgrade rehearsal; irreversible migrations require expand/migrate/contract releases and restore is the emergency path. Terraform is not invoked for an ordinary application rollback.

### D11: Run observability entirely as an owned OVH workload initially

**Choose:** one OpenTelemetry Collector agent per host exporting to a central private collector, with Prometheus, Alertmanager, Loki, Tempo and Grafana on the observability VM for the initial deployment.

**Why:** Production settings enable OTLP to localhost, which maps naturally to a host agent. The central stack keeps the initial controlled infrastructure in OVH and can later export to a managed destination. Host agents batch with bounded memory and disk queues. Alerts cover public availability, API readiness, dependency probes, Rabbit queue age/depth/consumer count, each worker heartbeat, Beat singleton, crawler proxy denials/errors, PostgreSQL replication/WAL archive, managed database health, disk, backup age and restore-verification age.

Every production role sets `OTEL_ENABLED` and its local collector endpoint explicitly; no environment relies on the production default. `/metrics` and collector administration remain private.

Logs default to 14 days in test and 30 days in production; metrics default to 30/90 days; traces are sampled and retained for 7/14 days. Retention is a capacity input, not an unbounded default.

### D12: Define data recovery by authority, not by container volume

**Choose:** recovery mechanisms per state authority, with monthly production restore drills.

| State | Authority / recovery |
|---|---|
| PostgreSQL | streaming replica in HA; pgBackRest full/differential + continuous WAL to versioned object storage |
| MongoDB | provider backups plus scheduled logical export for account/region independence |
| Valkey | provider replication/backups; recovery invalidates sessions and reconciles ephemeral jobs |
| RabbitMQ | quorum replication in HA, definitions export, volume backup, producer/read-model reconciliation |
| Neo4j | scheduled database dump where licence permits plus block snapshot; rebuildable graph data documented |
| S3 documents | versioning, lifecycle retention, cross-region replication or independent backup bucket |
| Secrets/PKI | encrypted offline root/recovery material; Secret Manager metadata is not the only copy |
| Terraform state | versioned encrypted remote state with locking and separate recovery credentials |

Baseline targets are PostgreSQL RPO 5 minutes/RTO 60 minutes, managed Mongo/Valkey according to selected plan plus logical-export RPO 24 hours for regional recovery, Rabbit baseline RTO 4 hours/HA RTO 15 minutes, Neo4j RPO 24 hours/RTO 4 hours, and object storage RPO according to replication lag with a 24-hour maximum. Restore drills measure rather than assume these values.

### D13: Keep legacy GCP infrastructure until OVH acceptance

**Choose:** add `infra/ovhcloud/` without reusing or rewriting `infra/gcp/`. Mark GCP documentation legacy after OVH test acceptance; archive or delete it only after an explicit reference and active-environment audit.

**Why:** Reusing provider-specific Cloud Run modules would preserve wrong assumptions about a single stateless process. Immediate deletion is destructive and the repository cannot prove whether an external GCP environment still consumes that state.

### D14: Make edge, connection and host budgets explicit

**Choose:** terminate TLS at the OVH load balancer, remove Caddy from the production request path, expose no API host port publicly, and configure WebSocket idle timeouts to exceed two ping intervals plus margin. Each Compose service receives CPU/memory/PID/log limits and a systemd slice; state services receive protective OOM priority. Database pool totals are computed from API workers, Celery processes, migration jobs and operator headroom before PostgreSQL `max_connections` is set.

**Why:** The current Caddy configuration does not provide a valid production TLS route, direct port `5000` bypasses the edge, and unconstrained co-resident processes let the kernel choose the database as an OOM victim. Four Uvicorn workers can multiply configured SQLAlchemy pools, so per-process settings cannot be sized independently. PgBouncer transaction pooling is not introduced casually because the application uses session-sensitive facilities such as LISTEN/NOTIFY; any pooler requires a separate compatibility proof.

## Risks / Trade-offs

- [Self-hosted PostgreSQL adds Patroni, etcd, extension image and backup operations] → [Start with baseline single-node test, automate image/restore/failover evidence, and do not claim HA until the three-node profile passes drills.]
- [The selected primary region is a single regional failure domain] → [Keep independent object copies/logical exports and a documented rebuild path in a second region; do not advertise multi-region availability.]
- [The functional test profile co-locates components] → [Require a temporary topology-faithful crawler/proxy and exact managed-service acceptance stage before production.]
- [Managed Valkey may not support future FT/BF helpers] → [Generate the command manifest from reachable callers and gate the exact target; activation of a new command family requires infrastructure compatibility evidence.]
- [Neo4j Community remains a SPOF] → [Treat graph paths as degradable, back up and restore, alert clearly, and require a separate licensing decision before making graph availability critical.]
- [Local observability consumes compute and storage] → [Bound retention and queues, monitor its own disk, and retain the option to export to a managed backend.]
- [Secrets fetched onto hosts are still accessible to root] → [Harden host access, use per-role IAM and short-lived credentials/certificates, keep secrets in tmpfs, and audit retrieval.]
- [Compose has no native cross-host scheduler] → [Make inventory and placement explicit in Ansible, use idempotent systemd units, and scale by adding reviewed hosts rather than hidden rescheduling.]
- [Forward-only database changes can block binary rollback] → [Require expand/migrate/contract sequencing and migration rollback classification before activation.]
- [Current dirty worktree contains unrelated user changes] → [Implementation stages and commits touch only declared infra/deployment files and never sweep unrelated changes.]
- [The API currently starts Crawl4AI at lifespan startup] → [Make browser initialization crawler-role-only and add an API startup test that fails if a browser process is created.]
- [Readiness aggregates optional dependencies and resource warnings] → [Use liveness for load-balancer membership and readiness/deep health only for release gates and alerts until a separately specified readiness model exists.]
- [Production secrets are globally mandatory today] → [Implement a tested feature-to-secret matrix; never satisfy disabled integrations with dummy production credentials.]

## Migration Plan

1. Capture current deployment/resource inventory, external dependency ownership, active GCP-use evidence, and environment budgets.
2. Create OVH test project, remote state, IAM, registry, vRack subnets, Gateway and security groups.
3. Build base Ansible roles and systemd/Compose ownership; prove idempotent second apply.
4. Correct API browser ownership, health paths, conditional secret validation and explicit OTLP configuration; build split images, including the migrator, offline crawler-browser smoke, pre-baked ingestion models and extension-compatible PostgreSQL image.
5. Provision the functional test profile with new isolated data; never import production data before sanitization rules exist.
6. Deploy state services, run compatibility gates, then migrate schema and start application roles.
7. Run API, queue, worker, Beat, ingestion, WebSocket, crawler non-bypass, object, graph, telemetry and billing synthetic tests.
8. Run backup and isolated restore drills; record achieved RPO/RTO and repair misses.
9. Provision temporary topology-faithful crawler/proxy and exact production managed-service SKUs; repeat boundary and compatibility evidence.
10. After test acceptance, emit a signed promotion approval binding the selected region, profile, exact SKUs, PostgreSQL digest/major and release digests. Regenerate the production Terraform plan after that approval and bind its hash to the signed region-fit, PostgreSQL-compatibility and release manifests. A pre-apply negative gate rejects missing, stale, mismatched or tampered evidence and emits a plan-specific binding attestation only on success; apply requires and validates that attestation before any mutation. Early production previews are explicitly non-applyable.
11. Release canary API, workers by role and leased Beat; run complete production acceptance and retain prior digests and configuration.
12. Restore production backups into an isolated scratch target, record achieved RPO/RTO, then observe a minimum 24-hour production soak with no unresolved critical alert before marking the release accepted.
13. Mark Railway/GCP plans non-authoritative and decide `infra/gcp/` archive/removal separately.

If any pre-traffic gate fails, destroy or repair the test candidate without changing production. If production canary fails, remove it and retain the prior release. If a post-migration failure occurs, use the prior compatible binary when schema permits; otherwise execute the documented forward repair or restore procedure.

## Open Questions

- Exact flavor IDs, block-volume performance classes and managed-database SKU identifiers are selected from the target OVH project at implementation time without changing topology or requirements.
- The final alert delivery receiver and public DNS zone are deployment inputs; absence of a verified receiver or domain blocks production acceptance.
- The `infra/gcp/` archive-versus-delete decision is deferred until external active-use evidence is collected; OVH implementation does not depend on the outcome.

## Blast Radius

| Symbol/module | Callers | Dependents |
|---|---|---|
| `docker/prod.Dockerfile` and image workflow | API, every worker role, Beat, migration jobs | startup, browser availability, document parsing, release identity, health checks |
| `docker-compose.prod.yml` | local/prod operators and deployment documentation | all declared roles, volumes, networks, secrets and stop behavior |
| `src/app/server.py` / `app.main` | API container command | Uvicorn worker count, WebSocket settings, graceful shutdown and load-balancer health |
| `src/app/lifecycle/lifespan.py:lifespan` | FastAPI application factory | PostgreSQL hard startup; MongoDB, Redis, Neo4j, graph, object storage and producer state; removal of API-owned browser startup |
| `src/app/worker/celery/profiles.py:profile_for_role` | CLI argument construction, worker app, completeness checks and tests | exact queue, concurrency, process pool, task modules and recycle budgets per worker |
| `src/app/worker/celery/app.py:create_worker_celery_app` | worker CLI and Beat CLI | one-role-per-process registration, queue completeness, Beat schedule ownership |
| `src/app/worker/celery/schedules.py:BEAT_SCHEDULE` | Beat composition and completeness checks | billing, credit, auth, transcript, memory and document timers |
| `src/app/worker/capabilities/documents.py` | ingestion task adapters | PostgreSQL, Redis, Neo4j/Graphiti lifecycle and shutdown order |
| `src/app/worker/capabilities/crawler.py` | crawler task adapter and worker signals | egress preflight, proxy environment, browser creation and fail-closed availability |
| crawler proxy/relay assets under `docker/crawler-egress/` | crawler production Compose and evidence tests | connection-time policy, Redis fixed-upstream mTLS and metrics |
| Alembic migrations | one-shot migration job and fresh-environment tests | extension/access-method availability and all relational/vector/search schema |
| `src/app/features/health/health_check.py` and health router | load balancer, acceptance and operator probes | dependency truth, APOC/GDS visibility and release gating |
| `src/app/config/settings.py` | API, workers, Beat and probes | connection endpoints, pool budgets, task limits, image behavior, secrets and OTLP defaults |
| `infra/gcp/` | legacy GCP scripts and possible external state | no OVH module; disposition only after active-use audit |

## Async Semantics

| Question | Decision |
|---|---|
| Delivery semantics | Preserve Celery at-least-once delivery and existing idempotency locks/status. Deployment never claims exactly-once execution. |
| Ordering guarantees | Preserve per-queue broker behavior; no global ordering across default, ingestion and crawler queues. Beat singleton prevents duplicate schedule publishers but tasks remain idempotent. |
| Expiry / retry policy | Preserve current retry/backoff policy, global 270/300-second defaults, per-task overrides (including ingestion/search 1,500/1,620 and crawler 330/360), 3,600-second task-result expiry, 86,400-second idempotency and crawler-job TTLs, and existing per-message expiry. Per-role deployment grace is derived from the maximum registered hard limit plus at least 30 seconds in both Compose and systemd. |
| Payload schema versioning | Existing typed job names and Pydantic payloads remain the compatibility boundary. No channel or payload version changes in this change; producer and consumer images are compatibility-tested before mixed-version rollout. |
| Read models while stale | Existing status/result endpoints remain truthful. Queue lag, dependency outage or lost ephemeral Redis state yields pending/retry/failure semantics already defined by the feature; deployment never synthesizes success. |
