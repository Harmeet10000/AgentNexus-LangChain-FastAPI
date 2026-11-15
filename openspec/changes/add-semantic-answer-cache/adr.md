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
change. `SEMANTIC_CACHE_BACKEND` accepts only `redisvl`; any other value fails settings
validation at startup rather than silently falling back. No code above the adapter boundary
names the backend, the vectorizer implementation, or the index technology. A future LangCache
adapter implements the same port and extends the accepted setting in its own change.

### Alternatives

- Adopt LangCache directly now — rejected: second embedding path (violates unified-embedding),
  extra SaaS auth surface, and the answered backend decision.
- In-house vector similarity over Redis primitives — rejected: reimplements what redisvl
  provides (index lifecycle, stock FLAT KNN cache schema, tag filtering) for free.

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

## ADR-003: Vector-space identity includes the embedding-contract revision, separate from the authorization partition

- **Status:** accepted
- **Date:** 2026-11-15
- **Supersedes:** none

### Context

An embedding model, output-dimension, or query-side instruction change shifts the vector
space; distance measured across spaces is meaningless noise in both directions — false matches
and false misses. Model and dimension are insufficient because the unified embedding path
applies query/document instructions and already versions its own embedding cache namespace
(`_CACHE_NAMESPACE = "embedding:v2"`): an instruction-template change produces a new contract
at the same model and width. The authorization digest (ADR-002) is deliberately scoped to
authorization members and must not absorb a non-authorization concern.

### Decision

The semantic-cache match predicate is partition-digest equality AND embedding-identity
equality over `(model id, output dimension, embedding-contract revision)`, stored as separate
filter attributes. The embedding path exports the public contract revision; the cache does not
define a parallel constant. The vector index is named per configured dimension. Identity is
computed above the backend port so any adapter (including a future LangCache adapter) reuses
the same predicate. Similarity is evaluated only after all equality constraints are applied,
never across them.

### Alternatives

- Model id + dimension only — rejected: silently compares vectors produced under different
  query-side instructions at the same width.
- Fold identity into the partition digest — rejected: conflates the authorization boundary
  with match validity, and buries a model/instruction swap in a hash that cannot be inspected
  or flushed by model.
- Ignore identity (rely on TTL expiry) — rejected: a real noise window on every model or
  instruction swap, however bounded.

### Consequences

- A model or instruction-contract swap produces clean misses — not noisy matching — until old
  entries expire; old-revision entries stay unreachable even at identical model and width.
- A dimension change creates a fresh index naturally instead of failing on width mismatch.
- One additional filter component and one additional attribute per entry.
- The served-answer content is unaffected by any of this: identity guards the *comparison*,
  not the answer.

## ADR-004: Semantic entries have a non-sliding maximum lifetime

- **Status:** accepted
- **Date:** 2026-11-15
- **Supersedes:** none

### Context

The semantic layer stores recoverable answer content and a bounded query preview. A sliding
TTL would let a frequently hit entry live indefinitely, extending both the privacy exposure
and the window in which a stale answer can be served. The graph re-executes nodes on
interrupt/replay, so a replay-time store could also refresh expiry if the adapter is not
explicit about preserving the original store timestamp.

### Decision

A generation carries a stable identity and absolute expiry established before publication,
no later than generation completion plus the configured TTL. Checkpoint resume/retry reuses
that deadline. Reads and replacement writes cannot extend an existing entry's expiry.
Use stock RedisVL entry serialization and deterministic key derivation, but publish the
hash plus `PEXPIREAT` in one bounded single-key Lua operation on the existing client; do
not call `astore` and assign expiry in a second request. The script checks server time,
rejects expired generations and older-over-newer replacements, and preserves the earlier
of existing finite expiry and incoming expiry. Validate inputs before mutation and handle
command failures: Redis scripts prevent interleaving, not rollback. Verify the required
permissions at startup and test publication failures against real Redis.
Hit-based/per-call refresh is disabled. A missing or expired absolute deadline or nonpositive
TTL is a miss. No deferred deletion or later read is needed to establish retention. A replay
after key expiry cannot resurrect an old generation; a genuinely new generation can create
a new entry. Query representation and answer envelope share that lifetime.

### Alternatives

- Sliding TTL — rejected: popularity must not make sensitive text or a stale answer immortal.
- Refresh on replay — rejected: replays are not fresh evidence and would undermine the bound.
- Separate hash write then expiry — rejected: a client interruption can leave a persistent
  value; a grace-period delete cannot close that publication window.

### Consequences

- Hot entries expire on the same fixed horizon as cold ones.
- Replayed graph work cannot extend retention.
- The exact-match cache's 900s hygiene TTL remains independent.

## ADR-005: The semantic index uses the stock FLAT schema and is created non-destructively

- **Status:** accepted
- **Date:** 2026-11-15
- **Supersedes:** none

### Context

The selected RedisVL cache API defines a stock schema with a FLAT cosine vector over
`prompt_vector`; its entry ID is derived from `prompt` plus filters. Inventing an HNSW or
`query_vector` schema would step outside the supported API and its schema validation. A
missing index is the normal first-boot state, but blindly recreating on boot would destroy
entries when the schema drifts or workers race.

### Decision

Use the stock semantic-cache schema (FLAT cosine FLOAT32) with custom filter fields for
partition and embedding identity, and explicit vectors from the unified embedder. At startup:
inspect; if missing, create with `overwrite=False`; if an already-exists race occurs,
re-inspect; validate the full schema; only then wire the adapter. An incompatible existing
schema is never dropped or overwritten; it disables semantic lookup/store for the process
with an explicit log. A missing index is distinguished from absent Search capability, ACL
denial, and connectivity failure.

### Alternatives

- Create-overwrite at boot — rejected: destructive on schema drift and races.
- HNSW/custom schema — rejected: unsupported by the stock cache API and would require
  reimplementing entry/TTL semantics.
- Lazy index creation on first lookup — rejected: moves a startup-shaped failure onto the ask
  path and complicates the disable flag.

### Consequences

- First boot works without an operator pre-creating the index.
- Schema mismatches fail closed and visibly, without data loss.
- Dedicated integration coverage is required for actual schema, filters, and expiry.

## ADR Review Verdict

- [x] Every durable decision in design.md is captured here (D1 → ADR-001, D2 → ADR-002,
  D8 → ADR-003, D3/D11 TTL → ADR-004, D4/D5 → ADR-005; D6, D7, D9, D10, D12 are tunable
  implementation or rollout choices, not architectural commitments)
- [x] No repository-level adr/ directory created
- [x] Supersession chains name prior change + ADR id (none superseded)
