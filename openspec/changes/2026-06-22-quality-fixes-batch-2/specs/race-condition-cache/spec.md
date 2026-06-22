# Race-Condition Cache Deduplication

## Scope

`documents/service.py` — `DocumentQueryService.search()` method

## Problem

```python
# Current (lines 229-283)
cache_key = _build_cache_key("documents:search", payload)
if not payload.bypass_cache and self.redis is not None:
    cached = await self.redis.get(cache_key)       # (A)
    if cached is not None:
        return response                             #

embedding = await embedding_client.aembed_query(...)  # (B) expensive
results = await asyncio.gather(...)                    # (C) expensive

if not payload.bypass_cache and self.redis is not None:
    await self.redis.setex(cache_key, ...)             # (D)
```

Two concurrent requests pass check (A), both execute (B) and (C), then both write to (D). Second request pays full cost for no benefit.

## Solution

Insert a Redis SETNX lock between cache miss and computation:

```python
cache_key = _build_cache_key("documents:search", payload)
if not payload.bypass_cache and self.redis is not None:
    cached = await self.redis.get(cache_key)
    if cached is not None:
        response = UnifiedSearchResponse.model_validate_json(cached)
        return response.model_copy(update={"cache_hit": True})

    lock_key = f"{cache_key}:lock"
    lock_acquired = await self.redis.setnx(lock_key, "1")
    if not lock_acquired:
        # paginating: another request is computing this query
        # wait up to 1.5s for it to finish
        for _ in range(30):
            await asyncio.sleep(0.05)
            cached = await self.redis.get(cache_key)
            if cached is not None:
                return UnifiedSearchResponse.model_validate_json(cached).model_copy(
                    update={"cache_hit": True}
                )
    else:
        await self.redis.expire(lock_key, 15)

# ... embed + search (unchanged) ...

if not payload.bypass_cache and self.redis is not None:
    await self.redis.setex(cache_key, DEFAULT_SEARCH_CACHE_TTL_SECONDS, response.model_dump_json())
    if lock_acquired:
        await self.redis.delete(lock_key)  # ponytail: lock released when cache set, TTL is safety net
```

## Edge Cases

| Case | Behaviour |
|------|-----------|
| Lock holder crashes before embedding | 15s TTL auto-releases lock; next request acquires lock and computes fresh |
| Lock holder crashes after setting cache | Lock is released (DELETE at end), or auto-expires at 15s. Cache entry persists normally |
| Redis is None | Lock skipped; existing behaviour preserved (no guard) |
| `bypass_cache=True` | Lock skipped; existing behaviour preserved |
| Polling times out (1.5s exhausted, no cache) | Request falls through to compute anyway (degraded but correct) |

## Verification

1. Unit test: mock Redis, send two concurrent `search()` calls with same query, assert `aembed_query` called exactly once
2. Integration: `pytest tests/ -k "test_search_cache"`
