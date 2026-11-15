# ADRs: harden-crawler-runtime

## ADR-001: Separate worker compatibility from crawler egress enforcement

- **Status:** proposed
- **Date:** 2026-11-15
- **Supersedes:** none

### Context

The live failure is a Crawl4AI configuration incompatibility. Connection-time SSRF enforcement is a separate cross-deployment security boundary with its own dependency, topology, rollout, and rollback risks.

### Decision

This change repairs and verifies worker execution only. Its integration test constructs the real Crawl4AI run configuration and crosses real service and persistence seams with controlled I/O adapters. The independent `harden-crawler-egress` change owns proxy selection, network isolation, and browser egress proofs.

### Alternatives

- **Keep one combined change:** rejected because an availability adapter fix and a deployment security boundary cannot be independently reviewed, rolled out, or reverted.
- **Use a fully mocked service test:** rejected because it bypasses the constructor that caused the production failure.
- **Require uncontrolled internet in the default suite:** rejected because external DNS and page availability make the regression nondeterministic.

### Consequences

- The worker repair can land after its deterministic gates pass.
- Completing this change does not authorize unrestricted public crawler exposure.
- Egress hardening remains independently visible and cannot be implied by a passing worker test.

## ADR Review Verdict

- [x] Every durable decision in design.md is captured here, or no durable decision exists
- [x] Every superseded decision names its prior change and ADR id
- [x] No repository-level ADR file was created
