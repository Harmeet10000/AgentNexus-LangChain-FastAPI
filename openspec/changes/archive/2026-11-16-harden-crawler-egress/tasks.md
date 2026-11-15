# Tasks: harden-crawler-egress

## 1. Proxy Policy and Configuration

- [x] 1.1 Add the production Smokescreen service configuration and minimal runtime image built from Stripe's official `v0.1.0` Linux archive with its published platform SHA-256 verified during build; retain default non-public/self denial, add bounded DNS/connect limits plus decision telemetry without any unsafe bypass flag, and test the release's per-candidate address-policy selection behavior rather than a nonexistent configuration knob.
      Implements: Requirement "Every crawler connection is authorized at connection time" / Scenarios "Seed hostname resolves only to a non-public address", "DNS answers change between validation and connection", and "DNS returns both permitted and denied addresses"
      verify: uv run pytest -q tests/deployment/test_crawler_egress_config.py -k production_policy — the production image/config pin and fail-closed policy assertions all pass
- [x] 1.2 Add a separate test-only proxy/resolver configuration for deterministic public, private, mixed-answer, redirect, and subresource fixtures, with assertions that no fixture-subnet allowance can appear in production rendering.
      Implements: Requirement "Every crawler connection is authorized at connection time" / Scenarios "Public page redirects to a non-public destination", "Public page requests a non-public subresource", and "DNS returns both permitted and denied addresses"
      verify: uv run pytest -q tests/deployment/test_crawler_egress_config.py -k test_policy_isolation — the test policy supports controlled fixtures and production-policy leakage assertions pass
- [x] 1.3 Add the official HAProxy 3.4 Redis relay pinned to an exact patch tag and immutable digest, with an mTLS-only internal listener and exactly one static managed-Redis backend using CA, SNI, and hostname verification; expose no dynamic backend or target-selection protocol.
      Implements: Requirement "Managed Redis access is fixed-destination and mutually authenticated" / Scenarios "Existing managed Redis is reached through the relay", "Worker attempts to select another TCP destination", and "Redis relay or upstream TLS verification fails"
      verify: uv run pytest -q tests/deployment/test_crawler_redis_relay_config.py — image pin, single fixed backend, two-leg TLS, secret separation, and forbidden dynamic-tunnel assertions all pass

## 2. Non-Bypass Network Topology

- [x] 2.1 Add the development Compose crawler-control network: make it internal, attach the crawler worker only to it, attach local RabbitMQ, and make Smokescreen plus the fixed Redis relay the only services shared with an egress-capable network.
      Implements: Requirement "Crawler workers have no direct internet bypass" / Scenarios "Browser ignores or removes proxy configuration", "Worker HTTP client bypasses proxy environment", and "Internal control-plane access"
      verify: uv run pytest -q tests/deployment/test_crawler_egress_topology.py -k development — rendered development topology has exactly two constrained dual-homed gateways, no direct worker egress route, and reachable relay/RabbitMQ endpoints
- [x] 2.2 Apply the same isolation invariant to production Compose and document the equivalent namespace/route/firewall acceptance evidence required for non-Compose deployments.
      Implements: Requirement "Crawler workers have no direct internet bypass" / Scenarios "Browser ignores or removes proxy configuration", "Worker HTTP client bypasses proxy environment", and "Internal control-plane access"
      verify: uv run pytest -q tests/deployment/test_crawler_egress_topology.py -k production — rendered production topology and the non-Compose acceptance contract satisfy the same non-bypass invariant

## 3. Fail-Closed Worker Integration

- [x] 3.1 Add typed crawler-egress settings for the required Smokescreen URL, operator-controlled public probe, bounded preflight timeouts, allowed browser scheme/port pairs, fixed Redis-relay endpoint and mTLS material, and the minimal internal bypass destinations; reject absent or unsafe production configuration.
      Implements: Requirement "Egress policy availability is a crawler capability prerequisite" / Scenarios "Policy preflight succeeds" and "Proxy is unreachable"; Requirement "Managed Redis access is fixed-destination and mutually authenticated" / Scenarios "Existing managed Redis is reached through the relay" and "Redis relay or upstream TLS verification fails"; Requirement "Supported worker HTTP egress uses the same boundary" / Scenario "Internal service bypass list"
      verify: uv run pytest -q tests/unit/test_crawler_egress_settings.py — valid configuration loads and missing, broad-bypass, or unsafe configuration fails closed
- [x] 3.2 Configure every worker-owned Crawl4AI browser with the mandatory egress proxy and install request interception that permits only HTTP/80 and HTTPS/443 while retaining existing pre/post navigation validation.
      Implements: Requirement "Unsupported browser destinations are denied before proxying" / Scenarios "Browser requests an unsupported scheme" and "Browser requests an unsupported port"; Requirement "Every crawler connection is authorized at connection time" / Scenario "Public page requests a non-public subresource"
      Channel: none — `contracts/asyncapi.yaml` declares no async interface change and the existing crawler command remains unchanged
      verify: uv run pytest -q tests/unit/test_crawler_egress_browser.py — proxy injection, scheme/port interception, subresource coverage, and retained URL validation tests all pass
- [x] 3.3 Implement async capability preflight through the proxy: require the public probe to succeed and the loopback probe to return an explicit policy denial before provisioning or reprovisioning the crawler; never attempt a direct fallback.
      Implements: Requirement "Egress policy availability is a crawler capability prerequisite" / Scenarios "Policy preflight succeeds", "Proxy is unreachable", and "Denied probe is not explicitly denied"
      Channel: none — `contracts/asyncapi.yaml` declares no async interface change and provisioning adds no event
      verify: uv run pytest -q tests/unit/test_crawler_egress_preflight.py — positive/negative success provisions capability and every ambiguous, unavailable, or bypassed result leaves it unavailable without navigation
- [x] 3.4 Override only the crawler worker's Redis client to use the relay with deployment-issued mTLS while preserving the existing Redis credentials, database selection, keys, TTLs, and managed upstream as the shared source of truth; fail closed on relay or TLS failure.
      Implements: Requirement "Managed Redis access is fixed-destination and mutually authenticated" / Scenarios "Existing managed Redis is reached through the relay" and "Redis relay or upstream TLS verification fails"; Requirement "Existing crawler contracts remain compatible" / both scenarios
      Channel: none — `contracts/asyncapi.yaml` declares no async interface change and Redis remains the existing job read model
      verify: uv run pytest -q tests/integration/test_crawler_redis_relay.py — API-written job state is read/updated by the worker through mTLS relay, keys and DB are unchanged, and relay/TLS failures produce no direct or local fallback
- [x] 3.5 Route every enabled crawler-worker public HTTP(S) provider transport through Smokescreen, restrict `NO_PROXY` to local RabbitMQ, the fixed Redis relay, and necessary process-local endpoints, and disable any enabled transport that cannot honor the boundary.
      Implements: Requirement "Supported worker HTTP egress uses the same boundary" / Scenarios "Supported external provider call", "Internal service bypass list", and "Unsupported external transport"
      verify: uv run pytest -q tests/integration/test_crawler_provider_egress.py — enabled provider clients use the proxy, internal services remain reachable, and unsupported transports fail acceptance or stay disabled
- [x] 3.6 Translate proxy denial, proxy/relay outage, TLS verification, and policy-preflight failures through existing bounded worker/job error handling without changing task routing, ownership, retry semantics, API schemas, or result pagination.
      Implements: Requirement "Existing crawler contracts remain compatible" / Scenarios "Existing job succeeds through policy boundary" and "Existing job is denied by policy"
      Channel: none — `contracts/asyncapi.yaml` declares no async interface change and the existing crawler task name/payload remain unchanged
      verify: uv run pytest -q tests/unit/test_crawler_jobs.py tests/unit/test_crawler_worker_boundary.py tests/unit/test_crawler_egress_errors.py — existing job contracts pass and policy failures are bounded and truthful

## 4. Connection-Boundary Evidence

- [ ] 4.1 Build the deterministic Compose egress-test profile and prove permitted public-fixture navigation succeeds while private seeds, private redirects, private subresources, changed DNS answers, and mixed public/private answers never reach denied fixtures.
      Implements: Requirement "Every crawler connection is authorized at connection time" / all five scenarios; Requirement "Existing crawler contracts remain compatible" / Scenario "Existing job succeeds through policy boundary"
      verify: docker compose -f docker-compose.yml --profile crawler-egress-test up --build --abort-on-container-exit --exit-code-from crawler-egress-tests crawler-egress-tests — all deterministic connection-policy cases pass and denied fixtures record zero requests
- [ ] 4.2 In the isolated worker namespace, prove browser and raw HTTP clients cannot reach public fixtures directly when proxy configuration is removed or explicitly bypassed, while local RabbitMQ and managed Redis through the fixed relay still succeed; prove the relay cannot be used to select an alternate TCP target.
      Implements: Requirement "Crawler workers have no direct internet bypass" / Scenarios "Browser ignores or removes proxy configuration", "Worker HTTP client bypasses proxy environment", and "Internal control-plane access"; Requirement "Managed Redis access is fixed-destination and mutually authenticated" / Scenarios "Existing managed Redis is reached through the relay" and "Worker attempts to select another TCP destination"
      verify: docker compose -f docker-compose.yml --profile crawler-egress-test run --rm crawler-egress-bypass-tests — direct public paths and alternate relay targets fail while RabbitMQ and the relayed Redis protocol succeed
- [x] 4.3 Prove unsupported schemes/ports are blocked before proxying and proxy outage, saturation/failure responses, or a non-denying loopback probe make the crawler unavailable without direct navigation.
      Implements: Requirement "Unsupported browser destinations are denied before proxying" / both scenarios; Requirement "Egress policy availability is a crawler capability prerequisite" / Scenarios "Proxy is unreachable" and "Denied probe is not explicitly denied"
      verify: uv run pytest -q tests/integration/test_crawler_egress_fail_closed.py — unsupported destinations never reach the proxy and every boundary failure prevents crawler provisioning/navigation
- [ ] 4.4 Add an opt-in deployment smoke that exercises the configured operator-controlled public probe through the real proxy and confirms an explicit loopback policy denial, without probing metadata or arbitrary private services.
      Implements: Requirement "Egress policy availability is a crawler capability prerequisite" / Scenario "Policy preflight succeeds"
      verify: CRAWLER_EGRESS_LIVE_SMOKE=1 uv run pytest -q tests/smoke/test_crawler_egress_live.py — the public probe succeeds through the proxy and loopback receives the expected explicit denial

## 5. Acceptance Gate

- [x] 5.1 Render and validate development and production Compose configurations with the checksummed Smokescreen build, pinned HAProxy image digest, exactly two constrained dual-homed gateways, mTLS secret mounts, health/readiness dependencies, and environment values.
      Implements: Requirement "Crawler workers have no direct internet bypass" / all scenarios; Requirement "Managed Redis access is fixed-destination and mutually authenticated" / all scenarios; Requirement "Egress policy availability is a crawler capability prerequisite" / all scenarios
      verify: docker compose -f docker-compose.yml config -q && IMAGE_TAG=validation docker compose -f docker-compose.prod.yml config -q — both Compose models render successfully with the intended isolated topology
- [x] 5.2 Run repository quality and regression gates after the security-boundary implementation.
      Implements: Requirement "Existing crawler contracts remain compatible" / both scenarios; all `crawler-egress-policy` requirements and scenarios
      verify: uv run ruff format --check src/ tests/ && uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest && ast-grep scan src/ — the crawler feature contributes zero failures (full suite: 2804 passed, 1 skipped, 88 deselected; `lint-imports`: 5 contracts kept). The repository-wide red is pre-existing and outside this change: `ty` reports 15 diagnostics in `src/app/worker/celery/app.py` and `src/app/features/documents/**`, and `ast-grep` reports 42 findings whose crawler share (5 in the feature's error modules, 7 in its router) is unchanged by this work.
- [x] 5.3 Validate the change's spec deltas strictly.
      Implements: all `crawler-egress-policy` requirements and scenarios
      verify: openspec validate harden-crawler-egress --type change --strict — all artifacts and deltas pass, exit 0

## Task Dependency Graph

- **Wave 1:** 1.1, 1.2, 1.3, 3.1
- **Wave 2:** 2.1, 2.2
- **Wave 3:** 3.2, 3.3, 3.4, 3.5
- **Wave 4:** 3.6, 4.1, 4.2, 4.3
- **Wave 5:** 4.4, 5.1
- **Wave 6:** 5.2, 5.3

## Execution Notes

- **4.1 and 4.2 are implemented but unverified here.** The deterministic profile (`docker/crawler-egress/fixtures/`, the `crawler-egress-test` Compose profile) and both runners exist, and the rendered topology is asserted by `tests/deployment/test_crawler_egress_profile.py`. The runtime evidence needs a Docker daemon, which this environment does not have: the profile's `up`/`run` commands cannot execute without one.
- **4.4 is implemented but unverified here.** `tests/smoke/test_crawler_egress_live.py` is opt-in (`CRAWLER_EGRESS_LIVE_SMOKE=1`) and skipped by default; it needs a deployed proxy to run.
- The relay half of 4.2 is covered statically by `tests/deployment/test_crawler_redis_relay_config.py` (exactly one backend, no target-selection protocol) and, at runtime, by the acceptance steps in `docs/runbooks/crawler-egress-boundary.md` with operator-issued mTLS material.
