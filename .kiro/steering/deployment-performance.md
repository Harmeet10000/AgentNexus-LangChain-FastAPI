---
inclusion: always
---

# Deployment & Runtime Performance

## Gunicorn `--preload`

Treat as deployment optimization for multi-worker Linux containers:

```bash
gunicorn --preload --workers 4 src.app.main:app
```

**What it does:** Preload mostly immutable app state in master process before forking, so workers share memory through Copy-on-Write.

**What it's NOT for:**
- Mutable caches
- Per-worker state
- Startup code with side effects that should run independently in each worker

Each worker still needs its own connection pools, Redis clients, etc.

## jemalloc

Treat as infrastructure/runtime optimization for memory tuning:

```dockerfile
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

**When to mention:** Multi-worker API containers with high memory pressure

**What it's NOT:** A Python code pattern; don't let it drive application design

## Connection Pooling

- Initialize all shared clients in FastAPI lifespan
- Store in `app.state`
- Reuse across requests
- Never create new connections per request

Reference: `src/app/lifecycle/lifespan.py`

## Pydantic Performance

For large collections, use `TypeAdapter`:

```python
# Bad - creates validator per item
users = [User.model_validate(u) for u in raw_users]

# Good - single validator for collection
users = TypeAdapter(list[User]).validate_python(raw_users)
```

## Async Concurrency

- Use `asyncio.gather(...)` for bounded, request-scoped work
- Use `asyncio.Queue` for unbounded, bursty workloads
- Never block event loop with sync I/O
- Use async clients: `motor`, `asyncpg`, `redis.asyncio`

## Streaming for Large Payloads

When payload can become large or unbounded, use `StreamingResponse`:

```python
@router.get("/download", response_class=StreamingResponse)
async def download() -> AsyncIterable[bytes]:
    async for chunk in get_file_chunks():
        yield chunk
```

Don't materialize full response in memory first.
