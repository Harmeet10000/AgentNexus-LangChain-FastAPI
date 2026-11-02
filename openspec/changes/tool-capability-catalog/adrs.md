# ADRs — tool-capability-catalog

## ADR-1: The policy file is the only author of grants

**Decision:** role → capability grants live in one static, versioned declaration; runtime context tuples are populated from it and nowhere else.
**Alternatives:** assembling grants at context-build sites (rejected: recreates the ad-hoc authority this change removes), per-call dynamic policies (rejected: unauditable until a consumer demands them).
**Consequence:** "why did role R have capability C?" is always a one-file grep; unknown capability identifiers fail at startup.

## ADR-2: Tags become derived views, never the contract

**Decision:** structured metadata is authoritative; authority tags are derived and must agree.
**Alternatives:** keep tags as the source (rejected: strings a selector or refactor can silently ignore — the exact weakness D4 exists to close).
**Consequence:** disagreement is a registration error; tag-based selection keeps working unchanged.

## ADR-3: Selector scoping is an intersection computed before ranking

**Decision:** the candidate set is policy ∩ candidates before relevance ranking applies.
**Alternatives:** post-ranking removal (rejected: an unauthorized tool would still influence ordering and telemetry of what remains).
**Consequence:** the projection is a subset of authorization by construction; the executor re-check remains the final gate.

## ADR-4: Egress is a classification, not a special case

**Decision:** tools whose model-chosen content leaves the trust boundary carry an egress classification and are gated by the standard executor contract.
**Alternatives:** a bespoke web-search guard (rejected: the next egress tool would re-open the hole).
**Consequence:** web search gains its grant in D4 as planned; every future egress tool inherits the gate by classification.
