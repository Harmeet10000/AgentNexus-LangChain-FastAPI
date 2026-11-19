# Tasks: deploy-ovhcloud-non-kubernetes-infrastructure

**Class:** L

The AsyncAPI artifact records no new async interface; all Celery, Beat and WebSocket tasks below preserve existing identities and therefore cite no new channel/operation.

## 1. Evidence and deployment inputs

- [ ] 1.1 Record the target OVH project regions, available VM flavors, managed MongoDB/Valkey plans, load-balancer features, block-storage classes, object endpoints, latency, residency, quotas and monthly prices for test, baseline production and HA production.
      Implements: Infrastructure cost and capacity are reviewable before apply / Capacity change exceeds budget threshold
      verify: `infra/ovhcloud/scripts/region-fit --all-candidates --require-services --measure-latency --check-residency --out dist/region-fit.json && infra/ovhcloud/scripts/plan-cost test && infra/ovhcloud/scripts/plan-cost production` — at least one region has signed pass evidence for required services/SKUs, latency, residency and quotas, and each profile emits a recurring estimate within budget
- [ ] 1.2 Audit repository references and known external environments for active use of `infra/gcp/`, and write a legacy/active disposition without changing GCP state or credentials.
      Implements: Legacy provider infrastructure has an explicit disposition / OVHcloud infrastructure is applied
      verify: `infra/ovhcloud/scripts/audit-legacy-gcp` — report names every reference and exits 0 without invoking a GCP provider
- [ ] 1.3 Build a disposable PostgreSQL compatibility matrix for the pinned PostgreSQL 18 candidate and PostgreSQL 17 fallback, covering preload libraries, all required extensions/access methods, a zero-to-head migration and representative hybrid retrieval.
      Implements: Dependency capability gates fail before traffic activation / PostgreSQL keyword access method is unavailable
      verify: `infra/ovhcloud/scripts/probe-postgres-image --all-candidates --out dist/postgres-compatibility.json --sign` — a signed report selects exactly one digest/major that passes extension, preload, `diskann`, `bm25`, Alembic-head and retrieval checks
- [ ] 1.4 Define the typed `ENABLED_OPTIONAL_INTEGRATIONS` registry and generate the runtime capability matrix mapping every API/worker/Beat capability to core-versus-optional status, required secrets, endpoints, network paths, data authority and disabled behavior; credential presence SHALL NOT enable an integration.
      Implements: Production configuration is capability-aware / Optional provider is disabled
      verify: `uv run pytest tests/unit/config/test_production_capability_matrix.py -q` — enabled features reject missing secrets and disabled features start without dummy credentials

## 2. Application startup and configuration corrections

- [ ] 2.1 Remove eager Crawl4AI/browser creation from API lifespan while preserving crawler-worker capability startup and fail-closed egress preflight.
      Implements: Runtime artifacts are complete and role-specific / API process starts; crawler-worker-execution / Browser smoke testing uses the production security settings
      verify: `uv run pytest tests/unit/lifecycle/test_api_has_no_browser.py tests/unit/worker/test_crawler_capability.py -q` — API starts no browser and crawler startup still fails closed when policy is absent
- [ ] 2.2 Implement `ENABLED_OPTIONAL_INTEGRATIONS` validation and adapter gating with `SecretStr`: core secrets remain mandatory, optional secrets are required iff their enum member is enabled, unknown members fail, role manifests reject undeclared capabilities, and disabled adapters fail closed without affecting unrelated roles.
      Implements: Production configuration is capability-aware / Optional provider is disabled; Secrets are environment-scoped and absent from source and plans / Infrastructure plan is inspected
      verify: `uv run pytest tests/unit/config/test_production_capability_matrix.py tests/unit/config/test_secret_redaction.py -q` — configuration cases pass and logs/reprs expose no secret value
- [ ] 2.3 Correct container and Compose liveness checks to `/api/v1/health/live`, preserve aggregate readiness/deep health for release diagnostics, and test their distinct failure semantics.
      Implements: Edge health and host resources fail safely / Optional dependency is degraded
      verify: `uv run pytest tests/unit/features/health tests/integration/test_health_probe_semantics.py -q` — live remains 200 during optional degradation while readiness reports it truthfully
- [ ] 2.4 Make OTLP enablement and collector endpoints explicit per runtime role and fail deployment preflight when enabled telemetry has no collector.
      Implements: Production configuration is capability-aware / Telemetry collector is absent; Observability distinguishes component and flow failures / Telemetry destination is unavailable
      verify: `uv run pytest tests/unit/config/test_otel_role_settings.py -q && infra/ovhcloud/scripts/probe-telemetry test` — role settings have no implicit endpoint and the probe detects a missing collector
- [ ] 2.5 Parameterize API worker count and calculate aggregate PostgreSQL pools, Valkey clients, and RabbitMQ connections/channels from API/Celery processes with at least 20 percent operator/migration headroom against every server-side limit.
      Implements: Edge health and host resources fail safely / Capacity envelope is exceeded
      verify: `uv run pytest tests/unit/config/test_connection_budget.py -q` — PostgreSQL, Valkey and RabbitMQ overcommit combinations fail and accepted profiles retain the configured headroom

## 3. Immutable runtime artifacts and supply chain

- [ ] 3.1 Create minimal API and default-worker/Beat image targets with role-specific commands, health checks and non-root filesystem ownership.
      Implements: Runtime artifacts are complete and role-specific / API process starts; The deployment inventory covers every runtime role / Complete inventory is accepted
      verify: `infra/ovhcloud/scripts/test-image api && infra/ovhcloud/scripts/test-image default-worker` — both run as the declared non-root user and API starts without Chromium
- [ ] 3.2 Create an ingestion-worker image that pre-bakes exact, revision-pinned Hugging Face/Docling assets, records their identities in image metadata, and completes a controlled extraction with download endpoints blocked.
      Implements: Runtime artifacts are complete and role-specific / Complete inventory is accepted
      verify: `infra/ovhcloud/scripts/test-image ingestion-worker --offline` — representative ingestion succeeds with network downloads denied
- [ ] 3.3 Create a crawler-worker image with a library-compatible pinned browser and all OS libraries, then run positive and deliberately mismatched browser/library contracts under production user, sandbox, limits, proxy, certificates and network policy.
      Implements: The released crawler artifact owns its browser runtime / Final artifact is tested offline from package registries; The released crawler artifact owns its browser runtime / Browser and crawler library drift; Browser smoke testing uses the production security settings / Test-only privilege hides a production failure; Dependency capability gates fail before traffic activation / Browser cannot launch in the final image
      verify: `infra/ovhcloud/scripts/test-image crawler-worker --production-security --offline && ! infra/ovhcloud/scripts/test-image crawler-worker --inject-browser-drift` — controlled navigation succeeds offline and incompatible browser/library revisions fail before promotion
- [ ] 3.4 Create a one-shot migrator image containing `alembic.ini`, all revisions and tenant reconciliation/check scripts.
      Implements: Runtime artifacts are complete and role-specific / Fresh database is deployed
      verify: `infra/ovhcloud/scripts/test-migrator-image --fresh-database` — zero-to-head upgrade, reconciliation and check exit 0
- [ ] 3.5 Build and test digest-pinned Smokescreen and selected PostgreSQL artifacts, including health checks and configuration-contract probes.
      Implements: Dependency capability gates fail before traffic activation / PostgreSQL keyword access method is unavailable; crawler-egress-policy / Cloud metadata is requested
      verify: `infra/ovhcloud/scripts/test-infrastructure-images` — selected PostgreSQL and Smokescreen digests pass their executable contracts
- [ ] 3.6 Extend CI to build, SBOM, vulnerability-scan, sign and publish every artifact once, then emit immutable digest manifests for promotion.
      Implements: Releases are immutable, ordered, and reversible / Release succeeds; The test environment proves production-sensitive boundaries / Release is promoted
      verify: `infra/ovhcloud/scripts/verify-release-manifest dist/release-manifest.json` — all required roles have verified signatures, SBOMs and digests with no mutable tag deployment reference

## 4. Terraform/OpenTofu foundations

- [ ] 4.1 Create `infra/ovhcloud/terraform` modules and separate test/production roots with pinned providers, variable validation, formatting, linting and documentation.
      Implements: Test and production environments are isolated and reproducible / Unchanged environment is planned again
      verify: `infra/ovhcloud/scripts/terraform-check` — format, validate, lint and module tests pass for both roots
- [ ] 4.2 Configure versioned encrypted remote state, separate state credentials, compatible locking and serialized CI concurrency; prohibit local/unlocked shared state.
      Implements: Test and production environments are isolated and reproducible / Test environment is destroyed; Secrets are environment-scoped and absent from source and plans / Infrastructure plan is inspected
      verify: `infra/ovhcloud/scripts/test-state-backend` — concurrent mutation is rejected and test/prod state credentials cannot read each other's objects
- [ ] 4.3 Implement separate project/IAM, vRack/private-network, subnet, router/gateway, security-group and authenticated-management modules for test and production; apply only test while producing a reviewed, non-applied production plan.
      Implements: Test and production environments are isolated and reproducible / Test environment is destroyed; Public ingress terminates at a controlled private boundary / Stateful endpoint is scanned publicly
      verify: `infra/ovhcloud/scripts/acceptance network-isolation --environment test && infra/ovhcloud/scripts/assert-production-not-applied` — test isolation passes and no production resource exists before promotion approval
- [ ] 4.4 Provision the OVH load balancer, TLS listener, private API pools, liveness monitor, WebSocket timeouts, DNS records and certificate lifecycle without a public API-host port.
      Implements: Public ingress terminates at a controlled private boundary / User request reaches a healthy API instance; Edge health and host resources fail safely / Optional dependency is degraded
      verify: `infra/ovhcloud/scripts/acceptance ingress` — HTTPS/WS reach private healthy backends, direct host/metrics ports fail, and optional readiness degradation does not evict a live backend
- [ ] 4.5 Implement role-host, encrypted-volume, registry, object-storage and managed MongoDB/Valkey modules for all three profiles; apply the test profile only and render explicitly non-applyable baseline/HA production preview plans.
      Implements: The deployment inventory covers every runtime role / Complete inventory is accepted; Persistent state has independent backup and verified restore paths / Provider-managed backup cannot be downloaded directly
      verify: `infra/ovhcloud/scripts/inventory --environment test --assert-complete && infra/ovhcloud/scripts/plan production-baseline --preview --out dist/production-preview-plan.json && ! infra/ovhcloud/scripts/apply production --approved-plan dist/production-preview-plan.json` — test inventory is complete and the early production preview cannot be applied
- [ ] 4.6 Add plan-time destructive-change, environment-scope, secret-leak and cost-budget policy checks and prove an unchanged second plan is empty.
      Implements: Test and production environments are isolated and reproducible / Unchanged environment is planned again; Infrastructure cost and capacity are reviewable before apply / Capacity change exceeds budget threshold
      verify: `infra/ovhcloud/scripts/policy-check dist/saved-test-plan.json && infra/ovhcloud/scripts/assert-noop-plan test` — policies pass and unchanged test plan reports zero mutations

## 5. Host configuration and service ownership

- [ ] 5.1 Implement idempotent cloud-init/Ansible base roles for users, SSH, patching, time sync, Docker/Compose, directories, volume mounts and audit/log rotation.
      Implements: Test and production environments are isolated and reproducible / Unchanged environment is planned again
      verify: `infra/ovhcloud/scripts/test-ansible-idempotence base` — second apply reports zero changes
- [ ] 5.2 Implement role-scoped secret retrieval into root-owned tmpfs, mTLS PKI issuance/overlap/expiry alerting, and cleanup on unit stop without secret values in Terraform or logs.
      Implements: Secrets are environment-scoped and absent from source and plans / Infrastructure plan is inspected; crawler-egress-policy / Application data plane is scanned
      verify: `infra/ovhcloud/scripts/acceptance secrets` — permissions/IAM/rotation pass and secret scanners find no plaintext in plans, images, journals or ordinary outputs
- [ ] 5.3 Implement systemd units and slices for each Compose project with dependency ordering, restart policy, CPU/memory/PID limits, OOM priorities and role-derived `TimeoutStopSec`: at least 330 seconds default, 1,650 seconds ingestion, and 390 seconds crawler unless tested redelivery permits less.
      Implements: Edge health and host resources fail safely / Capacity envelope is exceeded; Worker shutdown respects the maximum task boundary / Maximum-duration task overlaps a rollout
      verify: `infra/ovhcloud/scripts/acceptance service-budgets --assert-role-hard-limits` — unit timeouts exceed the maximum registered hard task limit per role by at least 30 seconds
- [ ] 5.4 Implement default-drop inbound host firewalling and private-only administration, metrics, database, broker, graph, proxy and telemetry ports.
      Implements: Public ingress terminates at a controlled private boundary / Stateful endpoint is scanned publicly
      verify: `infra/ovhcloud/scripts/acceptance port-scan` — only intended public 443 is reachable externally and private roles see only declared ports

## 6. Compose service topology

- [ ] 6.1 Decompose production Compose into API, workers/Beat, crawler, proxy/relay, PostgreSQL, RabbitMQ, Neo4j and observability projects using only digest references and explicit private networks.
      Implements: The deployment inventory covers every runtime role / Complete inventory is accepted; Releases are immutable, ordered, and reversible / Release succeeds
      verify: `infra/ovhcloud/scripts/compose-check --all-profiles` — rendered projects contain every required role, no mutable image reference and no undeclared public port
- [ ] 6.2 Declare distinct default, ingestion and crawler worker commands/queues plus exactly one lease-fenced Beat process; assert aggregate queue and schedule coverage.
      Implements: Every queue role has an isolated deployed consumer / Deployment queue coverage is checked; Scheduler leadership is singular and observable / Scheduler is replaced during deployment
      verify: `infra/ovhcloud/scripts/acceptance celery-topology` — each declared queue has the intended consumer and exactly one Beat publisher is observed
- [ ] 6.3 Apply per-service resource/log limits, health checks, persistent-volume mappings, role-scoped environment/secret mounts, and Compose `stop_grace_period` values of at least 330 seconds default, 1,650 seconds ingestion, and 390 seconds crawler unless role-specific redelivery is proven.
      Implements: Worker shutdown respects the maximum task boundary / Maximum-duration task overlaps a rollout; Edge health and host resources fail safely / Capacity envelope is exceeded; Observability distinguishes component and flow failures / Telemetry destination is unavailable
      verify: `infra/ovhcloud/scripts/acceptance compose-runtime-contracts --assert-role-hard-limits` — inspect output matches every budget/mount/log/health contract and Compose grace exceeds each role's registered hard limit by 30 seconds

- [ ] 6.4 Saturate every ingestion concurrency slot with long-running document work and prove default and crawler workers continue consuming and completing their own routed tasks within their SLOs.
      Implements: Every queue role has an isolated deployed consumer / Ingestion worker is saturated
      verify: `infra/ovhcloud/scripts/acceptance worker-isolation --saturate ingestion` — ingestion slots remain busy while controlled default and crawler jobs complete on their independent consumers

- [ ] 6.5 Remove every worker role and quiet subsystem one at a time (default worker, ingestion worker, crawler worker, Beat, relay, compatibility probe and backup job) and prove inventory acceptance fails with the missing subsystem named.
      Implements: The deployment inventory covers every runtime role / Quiet subsystem is missing
      verify: `infra/ovhcloud/scripts/acceptance inventory-negative --each-required-role-and-quiet-subsystem` — every worker-role and quiet-subsystem omission produces a nonzero result naming the absent owner and health signal

## 7. Data services and compatibility gates

- [ ] 7.1 Deploy the selected custom PostgreSQL artifact in the test environment using the baseline topology contract, encrypted block storage, explicit memory/connection tuning, pgBackRest and WAL archival; reserve local HAProxy/Patroni routing for the non-applied HA production plan.
      Implements: Dependency capability gates fail before traffic activation / PostgreSQL keyword access method is unavailable; Persistent state has independent backup and verified restore paths / Scheduled recovery drill completes
      verify: `infra/ovhcloud/scripts/acceptance postgres --profile production-baseline` — extension, migration, pool-headroom, backup and restore prechecks pass
- [ ] 7.2 Deploy managed MongoDB and Valkey with private endpoints, TLS/CA validation, least-privilege users, exact command/stream/pubsub/TTL probes and documented failover behavior.
      Implements: Dependency capability gates fail before traffic activation / Redis-compatible service lacks a required active command; Secrets are environment-scoped and absent from source and plans / Crawler credential is compromised
      verify: `infra/ovhcloud/scripts/acceptance managed-data` — active command and TLS tests pass and crawler credentials cannot access unrelated data/admin operations
- [ ] 7.3 Deploy RabbitMQ with TLS, role-specific users/vhosts, durable definitions, existing quorum work queues/classic DLQs, alarms and baseline/HA profile topology.
      Implements: The deployment inventory covers every runtime role / Complete inventory is accepted; celery-worker-deployment / Every queue role has an isolated deployed consumer
      verify: `infra/ovhcloud/scripts/acceptance rabbitmq` — definitions match expected exchange/queue/DLQ types and unauthorized role operations fail
- [ ] 7.4 Deploy private Neo4j Community with APOC, optional GDS detection, bounded memory, backup jobs and explicit degraded application behavior.
      Implements: Dependency capability gates fail before traffic activation / Complete inventory is accepted; Persistent state has independent backup and verified restore paths / Scheduled recovery drill completes
      verify: `infra/ovhcloud/scripts/acceptance neo4j` — APOC probe and representative graph path pass, GDS state is reported, and a dump restores on scratch storage
- [ ] 7.5 Configure OVH S3-compatible object storage with SigV4/path-style compatibility, browser-reachable presigned URLs, versioning, lifecycle and independent recovery copies.
      Implements: Dependency capability gates fail before traffic activation / Complete inventory is accepted; Persistent state has independent backup and verified restore paths / Provider-managed backup cannot be downloaded directly
      verify: `infra/ovhcloud/scripts/acceptance object-storage` — private put/get/delete, presigned browser GET, version restore and recovery-copy tests pass

## 8. Crawler security cell

- [ ] 8.1 Render and install Docker-aware default-drop nftables rules on crawler hosts, allowing only exact DNS, RabbitMQ TLS, proxy and mTLS relay destinations independently of provider egress rules.
      Implements: crawler-egress-policy / Provider security group contains implicit egress; Provider deployment proves the non-bypass boundary / Direct public bypass is attempted in OVHcloud
      verify: `infra/ovhcloud/scripts/acceptance crawler-direct-egress` — raw-IP and DNS direct public/private/metadata/database attempts fail from the final crawler container while exact control paths succeed
- [ ] 8.2 Deploy Smokescreen on the separate proxy/relay host with public destinations allowed only for HTTP(S) on TCP 80/443, private/reserved/link-local/metadata denied, DNS rebinding defenses and bounded logs/metrics.
      Implements: crawler-egress-policy / Cloud metadata is requested; Provider deployment proves the non-bypass boundary / Application data plane is scanned
      verify: `infra/ovhcloud/scripts/acceptance crawler-proxy` — controlled public HTTP(S) succeeds and loopback, link-local, metadata, reserved, private, unsupported scheme and port cases are denied
- [ ] 8.3 Deploy the HAProxy Valkey relay with mutual TLS, one fixed DNS upstream, certificate/SNI verification, DNS refresh and no caller-selectable destination.
      Implements: crawler-egress-policy / Application data plane is scanned; Secrets are environment-scoped and absent from source and plans / Crawler credential is compromised
      verify: `infra/ovhcloud/scripts/acceptance crawler-relay` — no-client-cert access fails, valid client PING succeeds, upstream substitution fails and DNS rotation recovers
- [ ] 8.4 Mark colocated test evidence non-equivalent, provision the temporary topology-faithful crawler/proxy hosts, and run the complete production non-bypass suite before promotion.
      Implements: Test colocation does not masquerade as production isolation / Test uses one host; The test environment proves production-sensitive boundaries / Cost-reduced topology is used
      verify: `infra/ovhcloud/scripts/acceptance crawler-boundary --topology-faithful --emit-report` — report passes and contains no colocation waiver for production evidence

## 9. Observability and alerting

- [ ] 9.1 Deploy bounded per-host OpenTelemetry collectors and the private central Prometheus/Alertmanager/Loki/Tempo/Grafana stack with explicit retention and disk queues.
      Implements: Observability distinguishes component and flow failures / Telemetry destination is unavailable
      verify: `infra/ovhcloud/scripts/acceptance telemetry-pipeline` — logs, metrics and traces arrive with environment/role/revision/correlation labels and destination loss remains within disk/memory caps
- [ ] 9.2 Add dashboards and alerts for ingress, API liveness/readiness, dependency probes, DB pools, broker queue depth/age, each worker role, Beat singleton, crawler denials, certificate expiry, disk, backup and restore age.
      Implements: Observability distinguishes component and flow failures / Accepted job never starts
      verify: `infra/ovhcloud/scripts/acceptance alerts --inject-failures` — each synthetic failure reaches the verified receiver with component attribution within 10 minutes
- [ ] 9.3 Add observability-stack self-monitoring, log-cardinality/secret-redaction checks and retention-capacity alarms.
      Implements: Observability distinguishes component and flow failures / Telemetry destination is unavailable; Secrets are environment-scoped and absent from source and plans / Infrastructure plan is inspected
      verify: `infra/ovhcloud/scripts/acceptance telemetry-safety` — bounded-storage and redaction tests pass and no secret fixture reaches stored telemetry

## 10. Release, rollback and recovery automation

- [ ] 10.1 Implement serialized preflight and release-manifest generation covering signatures, compatibility gates, inventory, backup freshness, cost delta and rollback pointer.
      Implements: Releases are immutable, ordered, and reversible / Release succeeds
      verify: `infra/ovhcloud/scripts/release preflight test --manifest dist/release-manifest.json` — all gates are recorded and any failed gate prevents mutation
- [ ] 10.2 Implement exactly-once-per-release migration/reconciliation locking and one-shot migrator execution before any new API receives traffic.
      Implements: Runtime artifacts are complete and role-specific / Fresh database is deployed; Releases are immutable, ordered, and reversible / Release succeeds
      verify: `infra/ovhcloud/scripts/acceptance concurrent-migration` — two release attempts produce one migration owner, one head and a successful tenant check
- [ ] 10.3 Implement out-of-rotation API canary, liveness/readiness/deep/synthetic gates, rolling load-balancer membership and WebSocket-aware drain without all-capacity loss.
      Implements: Releases are immutable, ordered, and reversible / New API revision fails readiness; Public ingress terminates at a controlled private boundary / User request reaches a healthy API instance
      verify: `infra/ovhcloud/scripts/acceptance api-rollout --inject-bad-canary` — prior traffic remains healthy and the failed canary never joins the pool
- [ ] 10.4 Implement role-by-role worker rollout using the maximum registered hard limit per role (default 300, ingestion 1,620, crawler 360) plus 30 seconds across Compose and systemd, with observable redelivery/idempotency and queue-coverage checks.
      Implements: Worker shutdown respects the maximum task boundary / Maximum-duration task overlaps a rollout; Releases are immutable, ordered, and reversible / Worker is replaced with an active long-running task
      verify: `infra/ovhcloud/scripts/acceptance worker-rollout --exercise-registered-maxima` — controlled tasks exercising 300/1,620/360-second role boundaries finish or visibly redeliver without false success and all queues retain consumers
- [ ] 10.5 Implement an environment-scoped Valkey lease for Beat using a random owner token, atomic acquire, compare-and-renew and compare-and-release operations; require a live lease before every publication cycle, stop publication immediately on renewal failure, and hand off only after old-token release/expiry.
      Implements: Scheduler leadership is singular and observable / Scheduler is replaced during deployment; Releases are immutable, ordered, and reversible / Release succeeds
      verify: `infra/ovhcloud/scripts/acceptance beat-handoff --inject-stale-partitioned-and-wrong-owner` — a stale/wrong owner cannot renew or release, lease loss prevents the next publication cycle, no publishers overlap and duplicate timer dispatch is absent
- [ ] 10.6 Implement binary/configuration rollback by prior digest, classify migration reversibility, and require expand/migrate/contract or restore for incompatible schemas.
      Implements: Releases are immutable, ordered, and reversible / New API revision fails readiness
      verify: `infra/ovhcloud/scripts/acceptance rollback` — prior compatible digest returns to service without rerunning migrations and incompatible rollback is blocked with restore guidance

## 11. Backup and disaster-recovery proof

- [ ] 11.1 Automate PostgreSQL full/differential/WAL, Neo4j dump, Mongo logical export, Valkey recovery policy, RabbitMQ definitions, object-version and PKI/state recovery jobs with age metrics.
      Implements: Persistent state has independent backup and verified restore paths / Provider-managed backup cannot be downloaded directly
      verify: `infra/ovhcloud/scripts/backup --all && infra/ovhcloud/scripts/backup-status --assert-fresh` — every authority produces an independently addressable recovery artifact or documented provider restore plus export
- [ ] 11.2 Implement isolated scratch restoration and application smoke verification for every state authority, recording measured RPO/RTO and evidence age.
      Implements: Persistent state has independent backup and verified restore paths / Scheduled recovery drill completes
      verify: `infra/ovhcloud/scripts/restore-drill test --all --emit-report` — schemas, tenant check, representative reads, queue topology, APOC, objects and application smoke pass
- [ ] 11.3 Schedule production monthly and test quarterly restore drills and alert before verification ages exceed 31 and 93 days respectively.
      Implements: Persistent state has independent backup and verified restore paths / Scheduled recovery drill completes; Observability distinguishes component and flow failures / Telemetry destination is unavailable
      verify: `infra/ovhcloud/scripts/acceptance restore-schedule` — schedules and pre-expiry alerts exist with the specified environment thresholds

## 12. Environment rollout and documentation

- [ ] 12.1 Provision and deploy the isolated `test-functional` profile, run the full acceptance suite, destroy/recreate it, and prove an unchanged post-recreate plan is empty.
      Implements: Test and production environments are isolated and reproducible / Test environment is destroyed; The test environment proves production-sensitive boundaries / Cost-reduced topology is used
      verify: `infra/ovhcloud/scripts/acceptance full --environment test --recreate` — complete suite passes with no production selection and final plan is no-op
- [ ] 12.2 Run exact production managed-service and topology-faithful promotion gates against the same digests, require signed region-fit and PostgreSQL digest/major compatibility evidence, select baseline or HA, then emit a signed promotion approval binding region, profile, SKUs and release digests.
      Implements: The test environment proves production-sensitive boundaries / Release is promoted; Infrastructure cost and capacity are reviewable before apply / Capacity change exceeds budget threshold
      verify: `infra/ovhcloud/scripts/promotion-report test production --require-region-fit dist/region-fit.json --require-postgres-compatibility dist/postgres-compatibility.json --out dist/promotion-approval.json --sign` — signed approval binds the selected region/profile/SKUs/PostgreSQL digest/release digests after every gate, cost and SPOF approval passes

- [ ] 12.3 Regenerate a fresh production Terraform plan from the signed promotion approval, bind its hash to the region-fit, PostgreSQL compatibility and release manifests, reject stale/mismatched/tampered evidence, then apply it with production-only credentials and configure hosts without deploying the application release.
      Implements: Test and production environments are isolated and reproducible / Unchanged environment is planned again; Legacy provider infrastructure has an explicit disposition / OVHcloud infrastructure is applied
      verify: `infra/ovhcloud/scripts/plan-approved-production --promotion dist/promotion-approval.json --region-fit dist/region-fit.json --postgres-compatibility dist/postgres-compatibility.json --release-manifest dist/release-manifest.json --fresh --out dist/approved-production-plan && infra/ovhcloud/scripts/acceptance evidence-binding --plan dist/approved-production-plan --reject-stale-mismatch-tamper --out dist/evidence-binding.attestation && infra/ovhcloud/scripts/apply production --approved-plan dist/approved-production-plan --require-binding-attestation dist/evidence-binding.attestation && infra/ovhcloud/scripts/inventory --environment production --assert-complete` — stale/mismatched/tampered evidence is rejected before mutation, apply refuses a missing/invalid attestation, only a fresh evidence-bound plan applies, and no GCP provider is invoked

- [ ] 12.4 Deploy the promoted digests to production through migration, API canary, role-by-role worker rollout and lease-fenced Beat, then run the complete production acceptance suite.
      Implements: Releases are immutable, ordered, and reversible / Release succeeds; The test environment proves production-sensitive boundaries / Release is promoted
      verify: `infra/ovhcloud/scripts/release deploy production --manifest dist/release-manifest.json && infra/ovhcloud/scripts/acceptance full --environment production` — the approved digests, one migration, healthy API pool, every worker role and one Beat are observed in production

- [ ] 12.5 Restore the newly deployed production backups into an isolated production-scoped scratch target, run application smoke, and record measured RPO/RTO plus restore-verification age.
      Implements: Persistent state has independent backup and verified restore paths / Scheduled recovery drill completes
      verify: `infra/ovhcloud/scripts/restore-drill production --all --isolated --emit-report` — every authority restores, application smoke passes, measured objectives are recorded and verification age starts below 31 days

- [ ] 12.6 Observe a minimum 24-hour production soak covering public HTTPS/WS, queue latency, every worker heartbeat, Beat lease, crawler denials, database health, backups and alerts; block acceptance on any unresolved critical signal.
      Implements: Observability distinguishes component and flow failures / Accepted job never starts; Releases are immutable, ordered, and reversible / Release succeeds
      verify: `infra/ovhcloud/scripts/soak production --minimum 24h --require-zero-unresolved-critical` — signed soak report covers the full interval and exits 0

- [ ] 12.7 Update `docs/architecture.md`, deployment documentation and runbooks for first deploy, release, rollback, restore, credential/certificate rotation, scaling, incident response and the authoritative OVH topology.
      Implements: The deployment inventory covers every runtime role / Complete inventory is accepted; Persistent state has independent backup and verified restore paths / Scheduled recovery drill completes
      verify: `infra/ovhcloud/scripts/docs-check` — every runtime/port/owner/health/recovery/runbook link in generated inventory is represented and link checks pass

- [ ] 12.8 Mark Railway and GCP documents non-authoritative only after production soak acceptance, and record whether `infra/gcp/` is retained as legacy or proposed for removal in a separate reviewable change.
      Implements: Legacy provider infrastructure has an explicit disposition / Legacy GCP disposition is recorded
      verify: `infra/ovhcloud/scripts/audit-deployment-authority --require-gcp-disposition` — exactly one authoritative production plan is named, GCP is either explicitly retained and isolated or linked to a separate evidence-backed archive/removal proposal, and this change performs no silent deletion

## 13. Acceptance gate

- [ ] 13.1 Run application formatting, lint, typing, unit/integration tests and structural rules after all behavior changes.
      Implements: all requirements and scenarios affected by application startup, settings, health, workers and artifact behavior
      verify: `uv run ruff format src/ tests/ && uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest && ast-grep scan src/` — every command exits 0
- [ ] 13.2 Run infrastructure formatting, validation, lint, security/policy, Ansible, Compose, image and documentation checks in a clean CI-equivalent environment.
      Implements: all ovhcloud-infrastructure-deployment requirements and scenarios
      verify: `infra/ovhcloud/scripts/ci-check` — every Terraform/OpenTofu, Ansible, Compose, shell, image, secret, policy and docs gate exits 0
- [ ] 13.3 Validate the change's spec deltas strictly and reconcile implementation evidence against every checkbox before archive.
      Implements: all requirements and scenarios in this change
      verify: `openspec validate deploy-ovhcloud-non-kubernetes-infrastructure --type change --strict` — all deltas pass, exit 0

## Task Dependency Graph

- **Wave 1:** 1.1, 1.2, 1.3, 1.4
- **Wave 2:** 2.1, 2.2, 2.3, 2.4, 2.5, 4.1
- **Wave 3:** 3.1, 3.2, 3.3, 3.4, 3.5, 4.2, 4.3
- **Wave 4:** 3.6, 4.4, 4.5, 4.6, 5.1
- **Wave 5:** 5.2, 5.3, 5.4, 6.1
- **Wave 6:** 6.2, 6.3, 7.1, 7.2, 7.3, 7.4, 7.5
- **Wave 7:** 6.4, 6.5, 8.1, 8.2, 8.3, 9.1
- **Wave 8:** 8.4, 9.2, 9.3, 10.1, 11.1
- **Wave 9:** 10.2, 10.3, 10.4, 10.5, 10.6, 11.2
- **Wave 10:** 11.3, 12.1
- **Wave 11:** 12.2
- **Wave 12:** 12.3
- **Wave 13:** 12.4
- **Wave 14:** 12.5
- **Wave 15:** 12.6
- **Wave 16:** 12.7, 12.8, 13.1, 13.2
- **Wave 17:** 13.3
