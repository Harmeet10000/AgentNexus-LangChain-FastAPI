## Context

This change bundles five independent defensive-programming fixes discovered during a targeted codebase audit. Each has been independently verified as a real issue with a clear reproduction path and a minimal fix. They are batched into one change to reduce ceremony — each is small enough that separate proposals would generate more overhead than the fixes themselves.

---

## Decision 1: Redis SETNX cache lock for hybrid search

**Problem:** `DocumentQueryService.search()` (line 227 in `documents/service.py`) does:
```
1. Check cache (GET)
2. If miss: embed query
3. Search BM25 + vector + trigram
4. Set cache (SETEX)
```

Two concurrent requests with the same query both see a cache miss at step 1, both call the embedding API at step 2 (double cost), and both run the full search at step 3. The second pays ~2-3s latency + embedding token cost for no benefit.

**Decision:** Add a "computing" lock using `redis.setnx()` with a short TTL before the embed step:

```python
async def search(self, ...):
    cache_key = _build_cache_key("documents:search", payload)
    if not payload.bypass_cache and self.redis is not None:
        cached = await self.redis.get(cache_key)
        if cached is not None:
            return response.model_copy(update={"cache_hit": True})

        # ponytail: setnx lock with 15s TTL prevents concurrent duplicate computation
        lock_key = f"{cache_key}:lock"
        lock_acquired = await self.redis.setnx(lock_key, "1")
        if not lock_acquired:
            # Another request is computing — wait briefly then read its result
            for _ in range(30):
                await asyncio.sleep(0.05)
                cached = await self.redis.get(cache_key)
                if cached is not None:
                    return UnifiedSearchResponse.model_validate_json(cached).model_copy(
                        update={"cache_hit": True}
                    )
        else:
            await self.redis.expire(lock_key, 15)

    # ... embed + search ...

    if not payload.bypass_cache and self.redis is not None:
        await self.redis.setex(...)
        await self.redis.delete(lock_key)
    return response
```

If lock acquisition fails, the request polls for the cached result (up to 1.5s total). If the lock holder crashes before setting cache, the 15s TTL auto-releases.

**Rationale:** This is the simplest correct fix with zero new dependencies. The `async-cache-dedupe` library (already planned in tier-2 observability) would replace this once adopted, but that's a separate cross-cutting change. This lock is additive and trivially removed later.

**Alternatives considered:**
- *Striped lock per-cache-key in async-cache-dedupe* — not yet integrated — rejected
- *Always compute (no lock)* — acceptable at low concurrency, wasteful under load — rejected
- *Probabilistic early expiration* — fixes wrong problem (stale cache, not duplicate compute) — rejected

---

## Decision 2: Graphiti initialization health check

**Problem:** lifespan.py (line 150-162) initialises Graphiti after the `TaskGroup` has completed. The TaskGroup itself runs Neo4j setup as a task (line 112). If Neo4j fails in the TaskGroup, `neo4j_driver` is set to `None` (line 138), but the Graphiti block still runs `setup_graphiti()` which opens its *own* connection. This gives two possible failure modes:

1. Neo4j driver fails → `app.state.neo4j_driver = None` → Graphiti setup independently creates a new connection (it doesn't use the stored driver) → if Graphiti also fails, `app.state.graphiti = None` → app continues in degraded state silently.
2. Neo4j driver succeeds → Graphiti setup fails → `app.state.graphiti = None` → app continues with Neo4j driver but no Graphiti, which is semantically confusing (driver present, graph features absent).

**Decision:** The current behaviour is actually *correct for graceful degradation* — the existing `try/except` blocks handle both failure modes. The gap is **observability**: the health endpoint (`/health`) already has a `check_graphiti` probe (in `health_check.py:84`) that checks `getattr(app.state, "graphiti", None)` and returns `DEGRADED` when not initialised. This is sufficient.

Fix: **No code change needed** for lifespan. The existing graceful degradation is correct. What the user reported as a "bug" is actually working as designed. The health check already reports Graphiti/Neo4j degradation.

**However**, there IS a real gap: if the Neo4j driver fails but Graphiti independently succeeds (using its own connection), `app.state.neo4j_driver` is `None` while `app.state.graphiti` is set. The Neo4j health probe reports "not initialised" while Graphiti works fine. This is an **observability inconsistency**, not a runtime bug.

Fix: Add a lifespan-logged warning when `neo4j_driver is None` but `graphiti is not None`, and vice versa. This informs operators that state is inconsistent without adding runtime complexity.

---

## Decision 3: Embedding dimension validation

**Problem:** `_normalize_embedding()` silently truncates or zero-pads vectors to match `settings.EMBEDDING_DIMENSION` (default 768). If the model is switched (e.g., from Gemini 768-dim to OpenAI 1536-dim), all existing embeddings become silently wrong:

- Truncation: loses the last 768 dimensions of signal
- Padding: adds 768 dimensions of zeros that dilute the real signal in cosine similarity

The function is duplicated in three places:
- `src/app/features/documents/service.py:804`
- `src/app/shared/langgraph_layer/retrieval_kb/nodes.py:364`
- `src/app/shared/langgraph_layer/ingestion_kb/nodes.py:738`

**Decision:**

1. **Consolidate** the three copies into a single function in `src/app/utils/embedding.py` (new module).
2. **Add a warning log** when the actual embedding dimension differs from `settings.EMBEDDING_DIMENSION` — at the call site where the embedding is first computed (the `_cached_embedding` function and the `_embed_chunks` function).
3. **No breaking change** — the consolidation is a pure refactor; the function behaviour is identical.

This makes it feasible to later add a settings validation that compares `EMBEDDING_DIMENSION` against the configured model's expected dimension (a future change, not in scope here).

**Rationale:** Consolidation is the prerequisite for any future dimension validation. The warning log surfaces model mismatches during development (local dev, CI) without requiring a full schema migration.

---

## Decision 4: Celery task typed dispatch

**Problem:** Currently:
- Task definitions use `@celery_app.task(name="tasks.documents_ingest")` decorated functions in `src/tasks/*.py`
- Outbox events reference tasks via string `event_type` (e.g. `"tasks.documents_ingest"`)
- No validation that the string key corresponds to a registered task

There is already a `CeleryTaskRegistry` in `src/app/connections/celery_registry.py` with `TypedCeleryTask` base class. The `auth_email_tasks_typed.py` file demonstrates the pattern. However, the *invocation* side (outbox relay + any `send_task` call) still uses bare strings.

**Decision:**

1. **Add `typed_send` method on `CeleryTaskRegistry`** that validates kwargs against the registered Pydantic model before calling `celery_app.send_task()`.
2. **Register** the remaining unregistered tasks (`tasks.documents_ingest`, `tasks.search_ingest`) with their payload models.
3. **Update outbox relay** to use `CeleryTaskRegistry.typed_send()` instead of bare `celery_app.send_task()` — this validates kwargs at outbox flush time, catching payload mismatches before they become silent failures in the Celery worker.

This is an incremental step toward full migration. Not all tasks are converted — only the two outbox-dispatched tasks and the two email tasks that already have typed counterparts.

**Rationale:** The typed registry pattern already exists and is proven. Adding it to the outbox dispatch path catches payload contract violations at the earliest possible moment (when the outbox event is flushed, not when the worker deserialises).

---

## Decision 5: Middleware CORS verification

**Problem:** `main.py` has two CORS-related mechanisms:
1. `SecurityMiddleware.configure_cors(app=app, config=guard_config)` — Guard's built-in CORS helper that adds `CORSMiddleware` internally
2. `SecurityMiddleware` itself (the middleware instance) — also handles CORS headers as part of its security processing

The comment block in `main.py:47-57` describes the intended execution order but the actual middleware registration is:
- `SecurityMiddleware.configure_cors()` at line 61 (adds CORSMiddleware as an internal sub-middleware)
- `GZipMiddleware` at line 64
- `ApiDeprecationMiddleware` at line 67
- `SecurityMiddleware` at line 77
- `RequestStateLoggingMiddleware` at line 83

Guard's `SecurityMiddleware` already deduplicates CORS headers internally — it checks `Access-Control-Allow-Origin` before adding. However, there's no test or runtime verification that this is working correctly.

**Decision:**

1. **No code change** — the existing Guard-based CORS handling is correct. Guard's `SecurityMiddleware` has internal dedup logic.
2. **Add a comment** in `main.py` clarifying that `SecurityMiddleware.configure_cors()` is a helper that injects CORSMiddleware internally, and the dedup happens inside Guard's response handler.
3. **Add a verification command** to the tasks section: run `curl -H "Origin: http://example.com" -v http://localhost:8000/health 2>&1 | grep -c "access-control-allow-origin"` to verify a single CORS header.

**Rationale:** Changing the middleware order or removing Guard's CORS could introduce security regressions (preflight handling, origin validation). The safest fix is to document and verify the current behaviour, then fix only if the curl command shows duplicates.

---

## Goals / Non-Goals

**Goals:**
- Fix the TOCTOU cache race for hybrid search
- Improve observability of Graphiti/Neo4j state inconsistency
- Consolidate and instrument `_normalize_embedding()`
- Extend typed Celery dispatch to outbox-dispatched tasks
- Document and verify single-source CORS handling

**Non-Goals:**
- Replace search caching with `async-cache-dedupe` (separate change in tier-2)
- Full Celery task migration (incremental; one task at a time per existing process)
- Schema migrations or data backfills for embedding dimension
- Adding new embedding models or changing the default dimension
- Refactoring Guard's CORS implementation

## Risks / Trade-offs

- **[Lock contention]** The SETNX lock in search adds ~50ms polling per concurrent request (worst case ~1.5s if the computing request is slow). **Mitigation:** 15s lock TTL bounds worst-case wait. Under normal concurrency (<10 simultaneous identical queries), the lock is almost never contested.
- **[False warning]** Embedding dimension warning may trigger during rolling deployments if old code writes 768-dim and new code writes 1536-dim. **Mitigation:** Warning only, no runtime impact. Silenced once deployment completes.
- **[Over-engineering]** Five small fixes batched into one OpenSpec change may feel like scope creep. **Mitigation:** Each is independently testable and reversible. The batch only reduces ceremony; each task can be cherry-picked.
