# Capability: cache-deduplication

## Purpose
Prevent cache stampede (thundering herd) on hybrid search and embedding call hot paths using `stampede-cache` with multi-tier deduplication.

## Requirements

### R1: stampede-cache Integration
- Use `stampede-cache[redis]` library (v0.1.0+, MIT, multi-tier)
- `@coalesce(ttl=60)` decorator for in-flight request deduplication
- `distributed_coalesce()` for cross-instance dedup (K8s multi-worker)
- Optional: pgvector semantic cache for LLM query deduplication
- Redis backend for distributed thundering herd prevention via Lua scripts

### R2: Protected Paths
- **Hybrid search** (`DocumentQueryService.search`): dedupe `bm25_search + vector_search + trigram_search` fan-out
- **Embedding calls** (`_cached_embedding`): dedupe concurrent identical query embeddings
- **Leaderboard top-N**: dedupe leaderboard reads (already in tier-2 plan)
- **User profile**: dedupe profile reads (already in tier-2 plan)
- **Subscription status**: dedupe subscription reads (already in tier-2 plan)

### R3: Deduplication Key
- Cache key = SHA256 of (normalized_query + filter_params + user_id)
- Same key within TTL window → single execution, others await result
- Different keys → independent execution

### R4: Error Handling
- Errors are NOT cached (only successful results)
- If the single executing request fails, awaiting requests get the error (not a stale result)
- Fallback: if stampede-cache unavailable, fall through to uncached path

### R5: Configuration
- `CACHE_DEDUP_ENABLED: bool = Field(default=True)`
- `CACHE_DEDUP_TTL_SECONDS: int = Field(default=60)`
- `CACHE_DEDUP_MAX_ENTRIES: int = Field(default=1000)`
- `CACHE_DEDUP_DISTRIBUTED: bool = Field(default=True)` — use Redis for cross-worker dedup

### R6: Integration Pattern
```python
from stampede import coalesce

@coalesce(ttl=settings.CACHE_DEDUP_TTL_SECONDS)
async def _deduplicated_search(query: str, filters: dict) -> UnifiedSearchResponse:
    # Only one concurrent call executes this
    ...
```

### R7: Metrics
- Log dedup hits: `logger.debug("cache_dedup_hit", key=..., waiters=N)`
- Log dedup misses: `logger.debug("cache_dedup_miss", key=...)`
- Optional: Prometheus counter for dedup hits/misses

## Acceptance Criteria
- [ ] 10 concurrent identical search requests execute search exactly once
- [ ] Failed search doesn't return stale cache to other waiters
- [ ] `CACHE_DEDUP_ENABLED=false` disables dedup (bypass path)
- [ ] Import error on `stampede-cache` doesn't crash app (graceful fallback)
- [ ] Dedup hit/miss logged at DEBUG level
- [ ] Distributed coalescing works across 2+ workers via Redis

## Non-Goals
- Cache invalidation beyond TTL
- Event-driven cache invalidation
- Cache warming on startup
