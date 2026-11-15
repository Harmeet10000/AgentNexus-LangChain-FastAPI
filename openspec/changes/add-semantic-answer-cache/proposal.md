# Change: Add semantic answer cache to the ask path

**Class:** L

## Why

The ask path's answer cache is exact-match only: a paraphrased repeat of an already-answered
question ("how do I get my money back" vs "what is your refund policy") re-runs the whole
plan → retrieve → grade → generate pipeline and pays for the generation LLM call again. A
similarity-matched cache layer after planning and before retrieval converts near-duplicate questions
into cache hits — lower generation cost and lower tail latency for the ask API and for Agent
Saul's qna engine, which share the retrieval graph. The work is motivated by the Redis
semantic-caching pattern (RedisVL / LangCache); RedisVL is the selected backend because it
runs on the Redis Cloud database already in the stack.

## What Changes

- Add a similarity-matched (semantic) answer cache in the retrieval graph's ask flow: on an
  exact-match miss, look up a semantically similar previously stored answer **within the same
  authorization partition and embedding identity** before downstream retrieval/generation; store a
  qualifying fresh, non-degraded answer when semantic mode is not `off`.
- The authorization partition is non-negotiable metadata scoping — matches are considered
  only among entries stored under the same user, tenant, authorized document scope, resolved
  document versions, and evidence policy (per-user matching, per the answered scoping
  decision; tenant-level sharing is out of scope).
- Reuse the single embedding path (`unified-embedding` contract) for cache-side query and
  entry embeddings — no second embedding provider, no embedding model literals. The path
  exports a public embedding-contract revision (the module already versions its internal
  namespace) so model/instruction changes cannot compare incompatible spaces.
- Graceful degradation as a standing discipline: when the cache backend is unavailable, when
  startup cannot establish a compatible index, when lookup or store exceeds its deadline, or
  when the similarity threshold yields no eligible match, the ask path behaves exactly as it
  does today (uncached pipeline run). Missing indexes are created non-destructively at
  startup; incompatible schemas are never overwritten.
- Semantic entries have a non-sliding maximum lifetime fixed before publication; hits, observe
  checks, and replays cannot extend it.
- Backend abstraction: a small cache interface with a RedisVL `SemanticCache` adapter shipped
  now and a LangCache adapter expected later (answered backend decision: "both, abstracted").
  RedisVL is the only selectable backend until the LangCache adapter exists; configuration
  fails closed rather than silently selecting an unimplemented backend.
- Serve mode requires a labeled safe-substitution evaluation: a one-sided 95% false-hit upper
  bound ≤ 1% on unsafe substitutions among held-out would-serve decisions (zero errors
  requires ≥299 independent cases sampled from the intended workload). Known-negative
  false positives are reported separately; they are not the serving-accuracy denominator. The default threshold is a policy starting point, not a
  proven value. The 1% acceptance bound is an initial risk-policy choice, not a safety proof.
- **BREAKING**: none. No breaking endpoint contract change; the exact-match cache key
  derivation is unchanged; an optional, additive cache-provenance field on the ask response
  distinguishes `exact` from `semantic` hits (design decides the exact field name).

## Capabilities

### New Capabilities

- `semantic-answer-cache` — similarity-matched serving of ask answers under strict
  authorization partitioning, with degradation and provenance behavior.

### Modified Capabilities

- `legal-corpus-retrieval` — the "Answer cache keys are version-aware" requirement is
  extended to bind the version-aware serving boundary to similarity-matched caching as well:
  a similarity match SHALL NOT serve an answer stored under a different authorized scope,
  different resolved document versions, or a different pipeline version. The exact-match key
  derivation itself is unchanged.

## Impact

- **Code**: `src/app/utils/cache/` (new semantic cache adapter beside the existing
  Result-native Redis ops), `src/app/features/documents/retrieval/graph/nodes.py` (ask-plan
  lookup on miss; dual write on fresh answer), `src/app/ai/embeddings.py` (bridged as the
  cache vectorizer and exports the public embedding-contract revision — no embedding behavior
  change), `src/app/config/settings.py` (mode, RedisVL-only backend, threshold, TTL, preview and
  query bounds, lookup and store deadlines), `src/app/lifecycle/lifespan.py` (inspect/create/validate index;
  backend construction and disable flag).
- **APIs**: no breaking endpoint change; one additive optional provenance field on the ask
  response (`exact` / `semantic` / absent-on-miss).
- **Dependencies**: `redisvl` is already declared; this change deliberately raises its floor
  to `>=0.32.0,<0.33.0` (from the currently locked `0.28.0`) for non-sliding TTL controls,
  then refreshes the lock — it does not add a new package. Requires the Redis Cloud database's
  Search & Query capability — confirmed enabled (Redis Cloud Pro) — plus permissions for
  atomic scripted publication and absolute expiry (production Terraform pins 7.2).
- **Evaluation**: a labeled safe-substitution golden set and calibration/acceptance report
  under the evals inventory, plus a gated real-Redis integration test. These are required for
  serve mode, not optional research.
- **Async flows**: none — no new events, topics, or timers; expiry is Redis key TTL with a
  non-sliding maximum lifetime.
