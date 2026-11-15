# Crawler Egress Policy Specification

## Purpose

This capability ensures crawler workers reach public HTTP(S) destinations only through a fail-closed connection boundary that prevents direct or proxied access to non-public infrastructure.

## Requirements

### Requirement: Every crawler connection is authorized at connection time
The crawler egress boundary MUST resolve each requested destination, apply address policy to the address selected for that connection, and reject loopback, private, link-local, multicast, reserved, unspecified, self, or otherwise non-public destinations before dialing them. Pre-navigation and post-navigation URL checks MUST remain defense in depth and MUST NOT be the sole enforcement.

#### Scenario: Seed hostname resolves only to a non-public address
- **WHEN** a crawler seed is authorized syntactically but its connection-time DNS result contains no permitted address
- **THEN** the boundary denies the connection before the destination receives traffic and the job records a bounded truthful failure

#### Scenario: Public page redirects to a non-public destination
- **WHEN** a permitted public navigation redirects to a hostname or literal address selected as non-public
- **THEN** the redirect connection is denied before the target receives traffic

#### Scenario: Public page requests a non-public subresource
- **WHEN** a permitted page requests an HTTP(S) subresource whose selected destination is non-public
- **THEN** the subresource connection is denied and no request reaches that destination

#### Scenario: DNS answers change between validation and connection
- **WHEN** a hostname passed application validation but the address selected at connection time is non-public
- **THEN** the connection-time boundary denies that address without reusing the earlier validation result

#### Scenario: DNS returns both permitted and denied addresses
- **WHEN** one DNS response contains both public and non-public candidates
- **THEN** only an address independently classified as permitted can be selected and dialed

### Requirement: Unsupported browser destinations are denied before proxying
The crawler MUST restrict browser network requests to HTTP on port 80 or HTTPS on port 443 before handing them to the egress boundary. The exact browser-internal bootstrap URL `about:blank` MAY be allowed because it creates no network request; all other `about:` URLs remain denied. This browser-level rule MUST NOT weaken or replace connection-time address enforcement.

#### Scenario: Browser requests an unsupported scheme
- **WHEN** navigation or a subresource uses a scheme other than HTTP or HTTPS
- **THEN** the browser denies the request before proxying or accessing the destination

#### Scenario: Browser requests an unsupported port
- **WHEN** navigation or a subresource uses a port other than 80 for HTTP or 443 for HTTPS
- **THEN** the browser denies the request before proxying or accessing the destination

### Requirement: Crawler workers have no direct internet bypass
The deployed crawler worker MUST have no general network route to public destinations except through the policy-enforcing HTTP(S) boundary. A separately specified fixed-destination managed-Redis relay MAY provide the only non-HTTP external control-plane path. Proxy settings alone MUST NOT be treated as the non-bypass control.

#### Scenario: Browser ignores or removes proxy configuration
- **WHEN** browser code attempts a direct public connection without the proxy
- **THEN** the worker network prevents the connection

#### Scenario: Worker HTTP client bypasses proxy environment
- **WHEN** worker code attempts public HTTP(S) egress while explicitly bypassing proxy configuration
- **THEN** the worker network prevents the connection

#### Scenario: Internal control-plane access
- **WHEN** the isolated worker connects to its internal message broker or the internal listener of its fixed-destination managed-Redis relay
- **THEN** those connections remain available without traversing the HTTP(S) policy boundary and without giving the worker a general public route

### Requirement: Managed Redis access is fixed-destination and mutually authenticated
When the configured Redis is outside the worker's internal network, the worker MUST connect through a relay that exposes no caller-selectable destination protocol and can connect only to one operator-configured Redis host and port. The worker-to-relay leg MUST use authenticated TLS, and the relay-to-Redis leg MUST require TLS with CA, SNI, and hostname verification. Redis credentials and database selection MUST remain end-to-end application data and MUST NOT be embedded in relay configuration. Relay failure MUST NOT enable a direct fallback.

#### Scenario: Existing managed Redis is reached through the relay
- **WHEN** the crawler worker reads or writes its existing Redis job state
- **THEN** it authenticates to the internal relay, which forwards the Redis protocol only to the configured managed Redis host and port over verified TLS

#### Scenario: Worker attempts to select another TCP destination
- **WHEN** worker-controlled bytes or configuration attempt to encode a destination other than the configured Redis upstream
- **THEN** the relay provides no target-selection mechanism and cannot open that connection

#### Scenario: Redis relay or upstream TLS verification fails
- **WHEN** the relay is unavailable or the upstream certificate, SNI, or hostname check fails
- **THEN** Redis-dependent crawler execution fails closed without a direct Redis connection or a general TCP route

### Requirement: Egress policy availability is a crawler capability prerequisite
The crawler capability MUST fail closed unless it can verify the configured boundary permits an operator-controlled public probe and explicitly denies a safe loopback probe. It MUST NOT fall back to unrestricted networking when the boundary is absent, unhealthy, or misconfigured.

#### Scenario: Policy preflight succeeds
- **WHEN** the proxy permits the configured public probe and returns an explicit policy denial for the loopback probe
- **THEN** crawler capability provisioning can complete

#### Scenario: Proxy is unreachable
- **WHEN** the configured proxy cannot be reached during capability provisioning or execution
- **THEN** the crawler remains unavailable and no direct navigation is attempted

#### Scenario: Denied probe is not explicitly denied
- **WHEN** the loopback probe is permitted, bypassed, or fails without a verifiable policy-denial response
- **THEN** capability provisioning fails closed

### Requirement: Supported worker HTTP egress uses the same boundary
Public HTTP(S) calls made by supported crawler-worker control-plane clients MUST traverse the same egress boundary, while internal service destinations MUST use an explicit minimal bypass list. A client transport that cannot honor the boundary MUST fail acceptance rather than regain direct egress.

#### Scenario: Supported external provider call
- **WHEN** a crawler job invokes a supported HTTP(S) model or provider client
- **THEN** the call traverses the policy boundary and retains the same non-public-address protections

#### Scenario: Internal service bypass list
- **WHEN** the worker connects to its internal message broker or the fixed Redis relay's internal hostname
- **THEN** only those exact internal destinations bypass the HTTP proxy and the worker still has no general public route

#### Scenario: Unsupported external transport
- **WHEN** a required provider transport cannot operate through the boundary
- **THEN** deployment acceptance fails or that crawler feature remains disabled

### Requirement: Existing crawler contracts remain compatible
Egress enforcement MUST preserve the existing public API, task name, queue routing, request payload, job ownership, retry behavior, and bounded result retrieval contract.

#### Scenario: Existing job succeeds through policy boundary
- **WHEN** a valid job reaches a permitted public destination through the boundary
- **THEN** existing status and result retrieval operations expose the outcome without a schema migration

#### Scenario: Existing job is denied by policy
- **WHEN** policy blocks a required destination
- **THEN** the existing job contract exposes a bounded truthful terminal outcome without leaking internal network details
