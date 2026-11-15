# Review: harden-crawler-egress

**Class:** L

## Completeness
- [x] Every requirement has at least one four-hash WHEN/THEN scenario
- [x] Private seed, redirect, subresource, mixed/changing DNS, unsupported scheme/port, proxy outage, bad negative probe, direct bypass, local RabbitMQ, managed Redis relay, provider transport, and compatibility cases are explicit
- [x] The only port policy is quantified as HTTP/80 and HTTPS/443; no unmeasured latency or capacity adjective is presented as a requirement
- [x] No status machine or multi-condition decision table is required; capability provisioning remains binary and each failure cause has an independent scenario

## Correctness
- [x] Requirements are observable behavior contracts and use MUST; Smokescreen and Compose details remain in design/ADR rather than the spec
- [x] `crawler-egress-policy` is a genuinely new capability and correctly uses an ADDED delta
- [x] Connection-time address enforcement, browser defense in depth, and network non-bypass are distinct requirements rather than conflated controls
- [x] `openspec validate harden-crawler-egress --type change --strict` passes

## Standards
- [x] The crawler feature and worker capability retain orchestration ownership; no HTTP or deployment policy is moved into persistence code
- [x] Proxy probing and browser lifecycle remain async and capability-owned rather than request-scoped globals
- [x] No new secret is introduced; future proxy credentials or probe credentials, if added, must use `SecretStr`
- [x] Tasks must run format, lint, type, full tests, ast-grep, Compose rendering, and live network-policy evidence
- [x] Durable proxy, topology, and preflight decisions are recorded in this change's `adr.md`

## Risk
- [x] The design surfaces dependency pinning, proxy availability, provider-client proxy support, throughput limits, test-only policy leakage, mixed DNS answers, and fail-closed rollback
- [x] The worker has no direct route; browser flags and environment variables are not misrepresented as the security boundary
- [x] The current external managed Redis is not mistaken for a Compose service: a fixed-destination HAProxy relay preserves the shared read model without providing a general TCP route
- [x] Both Redis transport legs require authenticated/verified TLS, and Redis credentials remain application data rather than relay configuration
- [x] Test-only private fixture allowances are separated from production configuration and require rendered-config assertions
- [x] Existing API/task/result compatibility and bounded error disclosure are preserved

## Contracts
- [x] Discovery adds no domain event or payload and the contracts artifact contains the permitted one-line no-async declaration
- [x] The existing crawler command remains unchanged, so no subscriber migration or duplicate AsyncAPI channel is invented

## VERDICT

**APPROVED** — The artifacts resolve the previous implementation blockers by choosing a pinned HTTP(S) connection boundary, preserving the existing managed Redis through a mutually authenticated fixed-destination relay, defining non-bypass topology and fail-closed readiness, and separating deterministic integration evidence from opt-in external smoke evidence.
