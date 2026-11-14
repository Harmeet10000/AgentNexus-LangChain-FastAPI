# ADRs: add-semantic-answer-cache

## ADR-001: Ask-path semantic caching goes through a backend port; RedisVL ships first, LangCache stays behind the same port

- **Status:** accepted
- **Date:** 2026-11-15
- **Supersedes:** none

### Context

The user selected "both, abstracted": semantic caching ships now on RedisVL while leaving a
path open to Redis Cloud's managed LangCache service. The two differ in more than transport —
LangCache embeds server-side with its own model, which would create a second embedding path
and conflict with the `unified-embedding` one-path contract. Whatever is built now must not
leak backend-specific types or assumptions into the graph seam.

### Decision

All semantic-cache behavior in the ask path flows through one thin async port (lookup, store,
close) owned by the shared cache layer; the RedisVL adapter is the only implementation in this
change, selected by configuration. No code above the adapter boundary names the backend, the
vectorizer implementation, or the index technology. A future LangCache adapter implements the
same port and is selected by the same setting, in its own change.

### Alternatives

- Adopt LangCache directly now — rejected: second embedding path (violates unified-embedding),
  extra SaaS auth surface, and the answered backend decision.
- In-house vector similarity over Redis primitives — rejected: reimplements what redisvl
  provides (index lifecycle, HNSW query, tag filtering) for free.

### Consequences

- The port is a permanent seam: any backend change is adapter-local.
- redisvl-specific concepts (index schema, custom vectorizer, tag filters) are confined to the
  adapter; the partition digest is computed *above* the port so a LangCache adapter can reuse
  it as its matching predicate.
- Cost: one indirection layer for a two-method interface — accepted for the swap-out guarantee.

## ADR-002: The authorization partition is a single equality digest derived from the exact-match key derivation minus the query, and similarity is only evaluated inside it

- **Status:** accepted
- **Date:** 2026-11-15
- **Supersedes:** none

### Context

This repository has a documented history of cache-authorization defects (cache keys omitting
`user_id`, `docs/performance-agent-prompts-2026-09-09.md`), and the answered scoping decision
is per-user matching. Vector similarity that ranks across authorization boundaries and filters
afterwards is exactly that defect class. The exact-match answer-cache key derivation is the
reviewed, parity-pinned encoding of user, tenant, authorized scope, resolved document versions,
and evidence policy.

### Decision

The semantic cache's matching predicate is a single partition digest — SHA-256 over the same
sorted-key bytes the exact-match derivation uses, minus the query component. Similarity is
evaluated only among entries whose stored partition digest equals the request's; similarity
score never substitutes for, widens, or ranks across partitions. Any future cache layer
(LangCache adapter, tenant-shared mode) reuses this same derivation; there is exactly one
definition of the authorization partition in the codebase.

### Alternatives

- Individual tag filters per partition member — same guarantee, but a parallel encoding that
  can drift from the parity-pinned key derivation.
- Post-filtering after vector ranking — rejected: ranking across partitions already happened;
  this is the documented defect pattern.

### Consequences

- The modified `legal-corpus-retrieval` version boundary holds structurally, not by
  convention.
- The derivation function must be split cleanly (query-in vs query-free bytes) so the parity
  suite for the exact-match key keeps passing unchanged.
- Tenant-shared matching later = loosening one digest component in one place, behind config.

## ADR-003: Vector-space identity is an explicit match-predicate component, separate from the authorization partition

- **Status:** accepted
- **Date:** 2026-11-15
- **Supersedes:** none

### Context

An embedding model or output-dimension change shifts the vector space; distance measured
across spaces is meaningless noise in both directions — false matches and false misses. The
authorization digest (ADR-002) is deliberately scoped to authorization members and must not
absorb a non-authorization concern. The repository's own embedding-cache discipline is
already task-type/model/dimension-aware. Semantic caching is the first component here where
a comparison operation has pre-conditions, and vector-space identity belongs to that
contract.

### Decision

The semantic-cache match predicate is partition-digest equality AND embedding-identity
equality, where embedding identity is the configured embedding model id plus output
dimension, stored as its own tag attribute on every entry. The vector index is named per
configured dimension. Both identity components are computed above the backend port so any
adapter (including a future LangCache adapter) reuses the same predicate. Similarity is
evaluated only after both equality constraints are applied, never across them.

### Alternatives

- Fold the model id into the partition digest — rejected: conflates the authorization
  boundary with match validity, and buries a model swap in a hash that cannot be inspected
  or flushed by model.
- Ignore identity (rely on TTL expiry) — rejected: a real noise window on every model swap,
  however bounded.

### Consequences

- A model swap produces clean misses — not noisy matching — until old entries expire.
- A dimension change creates a fresh index naturally instead of failing on width mismatch.
- One additional tag filter per lookup and one additional attribute per entry.
- The served-answer content is unaffected by any of this: identity guards the *comparison*,
  not the answer.

## ADR Review Verdict

- [x] Every durable decision in design.md is captured here (D1 → ADR-001, D2 → ADR-002,
  D8 → ADR-003; D3–D7, D9 are tunable implementation choices, not architectural commitments)
- [x] No repository-level adr/ directory created
- [x] Supersession chains name prior change + ADR id (none superseded)
