# Change: Add semantic answer cache to the ask path

**Class:** L

## Why

The ask path's answer cache is exact-match only: a paraphrased repeat of an already-answered
question ("how do I get my money back" vs "what is your refund policy") re-runs the whole
plan → retrieve → grade → generate pipeline and pays for the generation LLM call again. A
similarity-matched cache layer in front of that pipeline converts near-duplicate questions
into cache hits — lower generation cost and lower tail latency for the ask API and for Agent
Saul's qna engine, which share the retrieval graph. The work is motivated by the Redis
semantic-caching pattern (RedisVL / LangCache); RedisVL is the selected backend because it
runs on the Redis Cloud database already in the stack.

## What Changes

- Add a similarity-matched (semantic) answer cache in the retrieval graph's ask flow: on an
  exact-match miss, look up a semantically similar previously stored answer **within the same
  authorization partition** before running the pipeline; store every freshly generated answer
  for future similarity lookup.
- The authorization partition is non-negotiable metadata scoping — matches are considered
  only among entries stored under the same user, tenant, authorized document scope, resolved
  document versions, and evidence policy (per-user matching, per the answered scoping
  decision; tenant-level sharing is out of scope).
- Reuse the single embedding path (`unified-embedding` contract) for cache-side query and
  entry embeddings — no second embedding provider, no embedding model literals.
- Graceful degradation as a standing discipline: when the cache backend is unavailable, when
  the search capability probe fails, or when the similarity threshold yields no eligible
  match, the ask path behaves exactly as it does today (uncached pipeline run).
- Backend abstraction: a small cache interface with a RedisVL `SemanticCache` adapter shipped
  now and a LangCache adapter expected later (answered backend decision: "both, abstracted");
  selection by configuration.
- **BREAKING**: none. No endpoint contract changes; the exact-match cache key derivation is
  unchanged; an optional, additive cache-provenance field on the ask response distinguishes
  `exact` from `semantic` hits (design decides the exact field name).

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
  cache vectorizer — no behavior change), `src/app/config/settings.py` (similarity threshold,
  TTL, backend selection settings), `src/app/lifecycle/lifespan.py` (backend construction +
  search-capability probe).
- **APIs**: no endpoint contract change; at most an additive optional provenance field on the
  ask response (`exact` / `semantic` / absent-on-miss).
- **Dependencies**: adds `redisvl` (uv). Requires the Redis Cloud database's Search & Query
  capability — confirmed enabled (Redis Cloud Pro).
- **Async flows**: none — no new events, topics, or timers; the only expiry is the existing
  TTL discipline on cached entries.
