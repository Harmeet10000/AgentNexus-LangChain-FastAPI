# Design: harden-crawler-egress

## Context

See `proposal.md` for motivation and `specs/crawler-egress-policy/spec.md` for the behavior contract. The crawler currently validates seed and browser-reported URLs, but the browser owns the actual socket and both Compose workers share an ordinary network with direct egress. The worker also needs Redis/RabbitMQ control-plane connectivity and may make public HTTP(S) provider calls for optional processing.

## Goals / Non-Goals

**Goals:**
- Make connection-time destination classification and dialing one indivisible proxy operation.
- Make proxy bypass impossible at the worker network layer.
- Keep local RabbitMQ and externally managed Redis connectivity while forcing supported public HTTP(S) traffic through Smokescreen and Redis through a non-programmable fixed-destination relay.
- Produce repeatable Compose, integration, and deployment evidence for allowed, denied, unavailable, and bypass paths.

**Non-Goals:**
- Treat browser interception, environment variables, or URL validation as the primary SSRF boundary.
- Add hostname allowlists for the open-web crawler; hostname policy remains open while address policy remains mandatory.
- Enable unsafe private-range, self-connection, or hostname bypass switches.
- Change crawler endpoints, payloads, ownership, task names, result schemas, authentication, circuit breaking, caching, chunking, or telemetry outside proxy security evidence.

## Decisions

### D1: Use checksummed Stripe Smokescreen v0.1.0
**Choose:** Build a minimal repository-owned runtime image from Stripe's official `v0.1.0` Linux release archive and verify its published SHA-256 during the image build (`linux/amd64`: `4e1e447449e3cc653dae956208588722b5c8fe8ce894f633adc756a851d1ec46`; `linux/arm64`: `f15ddd0dc2b2cbc6d7318f441928939eb9bedfab2a1881508af3096a1e24b7c9`). The official release does not publish a usable public container image. Run that binary as the dedicated HTTP CONNECT boundary with open hostname policy for arbitrary public sites, default private-address and self-connection denial, bounded DNS/connect timeouts, and no unsafe bypass flags. Smokescreen `v0.1.0` evaluates each returned address against IP policy before selecting the address to dial; it has no `most_specific_ip_rules` setting.
**Why:** The checksummed release documents that DNS candidates are evaluated against IP policy and an approved address is selected before dialing. Verifying Stripe's published archive digest keeps the build reproducible without inventing a nonexistent official image. Envoy dynamic-forward-proxy configurations require more policy composition and have active mode-specific SSRF caveats; Squid ACL behavior would require proving the resolver-to-dial invariant ourselves. A custom proxy would make this repository responsible for CONNECT parsing, tunneling, address selection, backpressure, and security updates.

### D2: Enforce non-bypass with an internal-only worker network
**Choose:** Add an internal crawler-control network and attach the crawler worker only to it. Attach local RabbitMQ to its existing network plus crawler-control. Attach Smokescreen and the fixed Redis relay to crawler-control plus the ordinary egress-capable network; these are the only dual-homed egress components and neither gives the worker a general raw route.
**Why:** Chromium flags and `HTTP_PROXY` are advisory if the container still has a default route. Docker's internal network makes missing or ignored proxy configuration fail closed. The current Compose deployment does not contain Redis and uses a managed external Redis, so pretending Redis can merely join crawler-control would break the job store. Non-Compose deployments MUST provide equivalent routing/firewall evidence: the worker namespace has no direct public route and can reach only Smokescreen, the fixed Redis relay, and required internal endpoints.

### D3: Preserve managed Redis through a fixed-destination HAProxy relay
**Choose:** Deploy the official HAProxy 3.4 image, pinned to an exact patch tag and immutable digest during implementation, in TCP mode with exactly one static backend derived from operator-controlled Redis host/port settings. It exposes no CONNECT, SOCKS, dynamic backend, DNS name from client input, or other target-selection protocol. The internal listener requires deployment-issued mTLS from the crawler worker. Its external backend requires TLS with system/operator CA validation, explicit SNI, and hostname verification. Redis username/password and database selection remain inside the Redis protocol and never enter HAProxy configuration. Override only the crawler worker's Redis endpoint to the relay; API and other services retain the existing managed endpoint.
**Why:** Redis is a raw TCP protocol and cannot use Smokescreen through the existing client. A static relay preserves the same managed Redis read model without giving the worker a general TCP tunnel. TLS on both legs prevents credentials from crossing either network in cleartext. A local replacement Redis would create an unplanned state migration and split API/worker job state.

### D4: Make proxy policy mandatory in crawler configuration
**Choose:** Replace optional crawler proxy semantics for worker execution with a required egress-proxy setting. Configure Crawl4AI's browser proxy unconditionally for worker-owned browsers. Set `HTTP_PROXY` and `HTTPS_PROXY` for supported public HTTP clients, and set a minimal `NO_PROXY` containing only local RabbitMQ, the fixed Redis relay, and loopback needed by the process itself.
**Why:** Optional `use_proxy` request flags cannot define a security boundary. One process network namespace means browser and provider traffic share the same routing truth; clients that ignore proxy settings simply lose public connectivity rather than bypassing policy.

### D5: Preflight the full allow-and-deny contract
**Choose:** During crawler capability provisioning, send requests through the configured proxy to an operator-controlled public probe URL and a loopback probe. Provision only when the public probe succeeds and the loopback probe yields Smokescreen's explicit policy-denial response. Bound both operations with short timeouts and repeat the preflight after capability reset.
**Why:** A TCP-open proxy can still be misconfigured with unsafe private ranges. A positive and negative probe validates the behavior the crawler depends on. Loopback is a safe negative target: a broken policy can at worst attempt the proxy host's own loopback rather than cloud metadata or another internal service.
**Alternatives rejected:** Probe cloud metadata (the probe could become the exploit), rely only on container health (does not prove policy), or silently continue after probe failure.

### D6: Keep browser request interception as defense in depth
**Choose:** Install a browser request hook that permits only HTTP on port 80 and HTTPS on port 443 before proxying. Permit only the exact browser-internal bootstrap target `about:blank`, which creates no network request, and deny every other `about:` URL. Keep existing pre/post URL validation.
**Why:** This provides early, explainable denial for unsupported schemes and ports and covers browser subresources at the request layer. It does not select the socket address and therefore cannot replace Smokescreen plus network isolation.

### D7: Separate deterministic integration evidence from opt-in live smoke
**Choose:** Add a Compose integration profile with Smokescreen, a test-only resolver, and controlled HTTP fixtures. The test proxy config may explicitly allow only its fixture subnet while production config contains no private-range exception. Separately run an opt-in live smoke against an operator-controlled public probe. Test direct bypass from the worker namespace, redirects, subresources, DNS answer changes, policy outage, and proxy saturation/failure responses.
**Why:** Deterministic CI cannot depend on an arbitrary public site, while production acceptance still needs proof over the real external route. Test-only address exceptions must be isolated in separate mounted configuration and rejected by production-config assertions.

### D8: Preserve wire and job semantics
**Choose:** Keep the existing crawler task command, stable `crawl_id`, Redis model, endpoints, retries, TTL, and output bounds. Translate proxy denial/unavailability into bounded truthful failures through existing worker error handling.
**Why:** Network enforcement changes where connections can go, not how clients submit or retrieve jobs.

## Risks / Trade-offs

- [Pinned release archive becomes unavailable or vulnerable] → Verify Stripe's published release SHA-256 during the image build, retain the version/checksum pair in source, document an upgrade cadence, and fail the build if the archive cannot be verified.
- [Optional model/provider client ignores HTTP proxy settings] → The internal network blocks direct egress; acceptance tests every enabled transport and disables unsupported crawler processing modes.
- [Fixed Redis relay becomes an arbitrary tunnel] → Use a static HAProxy TCP backend with no client-selected target protocol; assert the rendered config contains exactly one backend destination and test alternate-target attempts.
- [Redis credentials or traffic cross a network in cleartext] → Require worker-to-relay mTLS and relay-to-upstream TLS with CA/SNI/hostname verification; fail closed on either validation error.
- [Redis relay outage splits API and worker state] → Keep the existing managed Redis authoritative, probe the relay path before crawler readiness, and disable crawler execution rather than creating a local fallback store.
- [Proxy becomes a throughput bottleneck] → Configure bounded connection/request limits, expose proxy metrics, and load-test before raising crawler concurrency; never bypass the proxy to recover capacity.
- [Test-only private fixture allowance leaks into production] → Use separate files/profiles, assert production config contains no allow/private/self/bypass switch, and validate rendered Compose configuration.
- [Mixed DNS answers create nondeterministic selection] → Rely only on an individually approved selected address; test mixed public/private and all-private answer sets.
- [Rollback pressure encourages direct egress] → Roll back by disabling crawler acceptance or stopping crawler workers, not by restoring an external route.

## Migration Plan

1. Add pinned Smokescreen production and test configuration, rendered-config assertions, and decision logging/metrics.
2. Add the pinned fixed-destination Redis relay, deployment-issued mTLS material, and proof that it reaches the existing managed Redis with verified upstream TLS while exposing no target-selection mechanism.
3. Add the crawler-control network and attach local RabbitMQ, Smokescreen, and the Redis relay while the crawler worker remains disabled.
4. Add mandatory proxy settings, browser interception, capability preflight, and the crawler-only Redis endpoint override; validate every enabled worker transport.
5. Run deterministic integration tests, direct-bypass proof, Redis-relay proof, Compose validation, and the opt-in public-probe smoke.
6. Move the crawler worker exclusively onto crawler-control and enable job acceptance only after all gates pass.
7. Roll back by disabling acceptance/stopping the worker; retain network isolation, Smokescreen, and the relay rather than adding a direct route or local fallback Redis.

## Blast Radius

| Symbol/module | Callers | Dependents |
|---|---|---|
| `CrawlerConfig.get_proxy_dict` / `to_browser_config` | crawler connection factory; `WebCrawler.crawl` and `crawl_recursive` | Every worker-owned browser navigation |
| `validate_navigation_destination` and browser request hook | `WebCrawler` navigation paths | Seed, redirect, and subresource defense in depth |
| crawler capability provisioning | Celery crawler task boundary | Worker availability, browser lifecycle, retries |
| `crawl_job` / `_run_crawl_job` | Celery worker runtime | Durable job status and result writes |
| crawler settings | worker/browser construction and preflight | Deployment readiness and fail-closed behavior |
| crawler worker Redis client settings / TLS context | job store, cache, cancellation checks | Existing managed Redis reached through the fixed relay without changing keys or databases |
| `docker-compose.yml` / `docker-compose.prod.yml` | deployment operator | Worker, Redis, RabbitMQ, proxy reachability and isolation |
| Smokescreen config and ACL | proxy service | Address selection, policy denial, metrics/log evidence |
| HAProxy Redis relay config and mTLS secrets | crawler worker job-store path | Exactly one managed Redis upstream and no general TCP egress |

## Async Semantics

| Question | Decision |
|---|---|
| Delivery semantics | Preserve existing at-least-once Celery delivery and bounded retries; policy failures do not create a second command. |
| Ordering guarantees | No cross-job ordering guarantee; stable `crawl_id` continues across retries. |
| Expiry / retry policy | Preserve task retry/backoff and Redis TTL; capability provisioning fails before navigation, and persistent policy failure reaches the existing terminal path. |
| Payload schema versioning | No crawler command or public API payload change. |
| Read models while stale | Existing pending/running states remain possible until a worker attempt writes the bounded terminal outcome. Proxy telemetry is operational evidence, not a job-state authority. |
