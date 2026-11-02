# ADRs — agent-portfolio-traversal-tools

## ADR-1: Author the contract before the gate opens

**Decision:** spec the traversal tools while the edges they need do not exist, with the gate as a first-class requirement.
**Alternatives:** wait for edges (rejected: contracts invented under delivery pressure; the cap discipline arrives too late to be respected).
**Consequence:** the unblocking change lands against an agreed, testable boundary.

## ADR-2: The gate is binding-time and loud

**Decision:** binding a portfolio tool without relationship edges fails the graph build naming the gate.
**Alternatives:** runtime empty-successes (rejected: tools that answer from nothing — the vacuous-success class), a feature flag (rejected: flags get flipped; a build failure gets fixed).
**Consequence:** exposure is impossible by accident; the flag is the graph's state, not a config toggle.

## ADR-3: Traversal finds where; the evidence seam serves what

**Decision:** traversal returns identifiers, provenance, and handles; content materialization stays exclusively with the scoped evidence read.
**Alternatives:** returning clause text in traversal envelopes (rejected: forks the scoping and handle story, re-grows envelopes).
**Consequence:** one read path, one scoping contract; envelopes stay bounded.

## ADR-4: Named tools over a generic graph-query tool

**Decision:** three typed traversal tools instead of one generic query surface.
**Alternatives:** a generic tool (rejected: pushes Cypher-shaped thinking to the model, makes result contracts unenforceable).
**Consequence:** every result contract is checkable; the projection stays legible.
