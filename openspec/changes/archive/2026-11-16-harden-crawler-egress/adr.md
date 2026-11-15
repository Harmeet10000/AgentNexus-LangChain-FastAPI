# ADRs: harden-crawler-egress

## ADR-001: Use Stripe Smokescreen as the crawler connection boundary

- **Status:** accepted
- **Date:** 2026-11-15
- **Supersedes:** none

### Context

The crawler needs an HTTP CONNECT boundary that resolves destinations, applies address policy to DNS candidates, selects an approved address before dialing, and receives security fixes independently of the Python application.

### Decision

Build a minimal runtime image from Stripe's stable `v0.1.0` Linux release archive and verify the platform-specific SHA-256 published in the release's `SHA256SUMS` asset. Stripe does not publish a usable public official container image for this release. Keep hostname policy open for arbitrary public crawling while retaining default non-public-address and self-connection denial. Enable no private-range, self-connection, or hostname IP-filter bypass option. Upgrades require a new official checksum plus the same security scenarios and rendered-configuration assertions before rollout.

### Alternatives

- **Application URL/DNS validation:** rejected because the browser owns the connected socket and DNS can change after validation.
- **Browser interception alone:** rejected because it does not independently bind policy to the address actually dialed.
- **Envoy dynamic forward proxy:** rejected for this change because its SSRF posture depends on more mode-specific RBAC and DNS-cache composition.
- **Squid or a custom proxy:** rejected because this repository would need to prove or own the resolver-to-dial invariant and CONNECT implementation.

### Consequences

- The repository-owned minimal runtime image and Smokescreen release checksum become production dependencies and security update responsibilities.
- Proxy unavailability makes crawler capability unavailable by design.
- Open-web hostname access remains possible, but only public addresses selected by the proxy can be dialed.

## ADR-002: Isolate crawler workers behind constrained dual-homed gateways

- **Status:** accepted
- **Date:** 2026-11-15
- **Supersedes:** none

### Context

Proxy environment variables and Chromium flags are bypassable while the worker container retains a direct public route. The worker still requires internal Redis and RabbitMQ connectivity.

### Decision

Run crawler workers on an internal-only control network. Smokescreen and a fixed-destination managed-Redis relay are the only components attached to both that network and an egress-capable network. Local RabbitMQ also attaches to crawler-control; the crawler worker must not attach to the ordinary application/egress network. Neither dual-homed component exposes a general raw route: Smokescreen applies destination policy, and the Redis relay has exactly one static upstream. Non-Compose deployments must provide equivalent namespace, route, relay, or firewall evidence.

### Alternatives

- **Proxy configuration without network isolation:** rejected because ignored or removed proxy settings restore direct egress.
- **Host-only URL firewalling:** rejected because browser subresources and process-level bypass traffic remain possible.
- **Per-request proxy selection:** rejected because a security boundary cannot depend on caller-controlled `use_proxy` behavior.

### Consequences

- Public HTTP(S) provider clients in the crawler worker must work through the proxy or remain disabled.
- Internal destinations are limited to local RabbitMQ and the fixed Redis relay; the worker retains no public route.
- Rollback disables crawler execution; it never reconnects the worker directly to the internet.

## ADR-003: Provision crawler capability only after positive and negative policy probes

- **Status:** accepted
- **Date:** 2026-11-15
- **Supersedes:** none

### Context

A listening proxy can still be unsafe or misconfigured. Startup needs behavioral evidence without probing sensitive metadata services.

### Decision

Crawler capability provisioning requires a bounded proxy request to an operator-controlled public URL and an explicit Smokescreen denial for a loopback URL. Missing, ambiguous, or unexpected responses fail provisioning. The probe repeats after capability reset and never triggers a direct-network fallback.

### Alternatives

- **TCP health only:** rejected because it proves listener availability, not destination policy.
- **Cloud metadata negative probe:** rejected because a broken policy could turn the health check into the exploit.
- **Log-only detection after jobs run:** rejected because post-hoc observability cannot prevent the first unsafe connection.

### Consequences

- Operators must supply and maintain a public probe endpoint.
- Proxy startup and crawler readiness are coupled to the behavior contract.
- A policy regression removes crawler capacity instead of degrading security.

## ADR-004: Preserve managed Redis through a mutually authenticated fixed-destination relay

- **Status:** accepted
- **Date:** 2026-11-15
- **Supersedes:** none

### Context

The current Compose files provide local RabbitMQ but no Redis service; the configured Redis is an external managed endpoint. An internal-only crawler worker cannot reach that raw TCP endpoint through Smokescreen, and silently replacing it with a local Redis would split the API and worker job read model.

### Decision

Deploy official HAProxy 3.4, pinned to an exact patch tag and immutable image digest during implementation, as a TCP relay with exactly one operator-configured managed Redis backend. The relay exposes no CONNECT, SOCKS, dynamic backend, or client-selected hostname/port. Require deployment-issued mTLS on the worker-facing listener and verified TLS with CA, explicit SNI, and hostname checking on the managed-Redis leg. Keep Redis credentials and database selection in the application protocol, not relay configuration. Override only the crawler worker's Redis endpoint to use the relay.

### Alternatives

- **Send Redis through Smokescreen:** rejected because the current Redis client is not an HTTP CONNECT client and Smokescreen is the HTTP(S) destination-policy boundary.
- **Give the worker a narrow direct firewall rule:** rejected for Compose because the worker would need an egress-capable interface and rotating managed-service addresses weaken static network proof.
- **Add a local Redis service:** rejected because it creates an unplanned availability/state-migration decision and would split job state unless every producer/reader moved together.
- **Unauthenticated or plaintext relay:** rejected because Redis credentials and payloads would cross the internal bridge without transport protection.

### Consequences

- The relay and its mTLS material become additional deployment dependencies and readiness gates.
- The existing managed Redis remains the single source for crawler job state, cache, and cancellation.
- Relay failure removes crawler capacity; it never enables a direct route or local fallback.

## ADR Review Verdict

- [x] Every durable decision in design.md is captured here, or no durable decision exists
- [x] Every superseded decision names its prior change and ADR id
- [x] No repository-level ADR file was created
