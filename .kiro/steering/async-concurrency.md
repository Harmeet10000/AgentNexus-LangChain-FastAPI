---
inclusion: always
---

# Async & Concurrency Patterns

## Core Rule: All I/O Must Be Async

- All I/O code must be async
- Use async clients: `motor`, `asyncpg`, `redis.asyncio`, `neo4j`, async LangChain integrations
- Never block the event loop with sync I/O

## Concurrency Patterns

### Bounded Fan-Out (Small, Known Task Set)
Use `asyncio.gather(...)` with a semaphore:
```python
semaphore = asyncio.Semaphore(10)

async def bounded_task(item):
    async with semaphore:
        return await process(item)

results = await asyncio.gather(*[bounded_task(item) for item in items])
```

**When to use:** Request-scoped, short-lived, bounded work (< 100 tasks)

### Unbounded/Bursty Workloads
Use bounded `asyncio.Queue` with worker tasks:
```python
queue = asyncio.Queue(maxsize=100)
workers = [asyncio.create_task(worker(queue)) for _ in range(10)]
```

**When to use:** Producers can outpace consumers, large/unbounded input, high memory pressure

### Never Use
- `asyncio.gather(...)` + semaphore as default for high-load, bursty, unbounded work
- Sync blocking code in async flows (use `asyncer` only as bridge)

## Async Clients

Always use async variants:
- MongoDB: `motor` (not `pymongo`)
- PostgreSQL: `asyncpg` (not `psycopg2`)
- Redis: `redis.asyncio` (not `redis`)
- Neo4j: async driver
- LangChain: async integrations

## Properties and Methods

- Use `@property` only for cheap, synchronous, side-effect-free access
- Use methods for operations with cost or side effects (I/O, DB, cache, network)
- Never perform persistence in property setters; use explicit methods
- Avoid async properties (they hide async cost behind attribute access)
