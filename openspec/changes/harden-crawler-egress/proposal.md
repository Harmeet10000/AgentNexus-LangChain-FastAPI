# Change: Enforce crawler egress at the connection boundary

**Class:** L

## Why

The crawler accepts caller-supplied URLs and runs a browser, while current URL/DNS checks occur before or after navigation and cannot prove that the socket actually connected only to public infrastructure. A separate deployment boundary is required before arbitrary public crawling can be considered safe against redirects, subresources, DNS changes, or browser bypass traffic.

## What Changes

- Add Stripe Smokescreen `v0.1.0` as the crawler worker's mandatory HTTP CONNECT gateway. Build the runtime image from Stripe's checksummed official release binaries because the release does not publish a usable public official container image; retain default private-address and self-connection protections.
- Place the crawler worker on an internal-only network with local RabbitMQ access and no direct internet route. Preserve the repository's managed Redis deployment through a separate fixed-destination HAProxy TCP relay; only Smokescreen and that non-programmable relay are dual-homed.
- Fail crawler capability provisioning closed unless the configured proxy is present and a policy probe proves allowed public egress plus denied private egress.
- Route browser traffic and supported HTTP(S) worker control-plane traffic through Smokescreen. Limit `NO_PROXY` to RabbitMQ, the fixed Redis relay, and required process-local endpoints.
- Add browser request interception as defense in depth for unsupported schemes and ports; it is not the primary IP/DNS boundary.
- Add deployment and integration proofs for seed navigation, redirects, subresources, DNS-answer changes, unavailable proxy, and direct-egress bypass attempts.
- Preserve existing crawler API, task, payload, ownership, result, and retry contracts.

## Capabilities

### New Capabilities

- `crawler-egress-policy` — crawler workers can reach public HTTP(S) destinations only through a pinned, policy-enforcing proxy, can reach the existing managed Redis only through a fixed-destination authenticated relay, and cannot directly reach non-public or unrestricted network destinations.

### Modified Capabilities

## Impact

- **Code**: crawler capability provisioning, browser configuration/interception, settings, focused integration tests, and deployment probes.
- **APIs**: no public endpoint, request, response, or task payload changes.
- **Dependencies**: new repository-built Smokescreen runtime pinned to the official `v0.1.0` release archive SHA-256, plus official HAProxy `3.4.7-alpine` pinned to immutable multi-architecture digest `sha256:b9fe316f718d5015c5cfebd8ca019591f3e44656f61d93cb8b59a1028961e530`; no new Python runtime package.
- **Async flows**: existing crawler task command is unchanged; capability startup gains a fail-closed proxy-policy preflight, and proxy metrics/logs become operational signals rather than domain events.
