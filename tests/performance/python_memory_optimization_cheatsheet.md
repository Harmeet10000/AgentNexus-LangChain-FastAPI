This is a large system. Let me map every layer, every data flow, and every seam you need to understand before you wire it into production.Let me walk through this layer by layer. Every seam, every data flow, every thing that will bite you in production.

**The system has three phases**: a synchronous HTTP phase (ingestion), an async WebSocket phase (analysis), and a background phase (reconciliation + decay). They share infrastructure but never share execution context.

Here's the full picture first:Now the memory stack in detail — this is the part most people misunderstand:Now let me walk through every seam in plain language.

---

**The two-phase entry point**

Phase 1 is HTTP. The client uploads a file, your ingestion router runs `IngestionGraph` synchronously (`ainvoke` — it waits for completion), and returns a `doc_id`. That `doc_id` is the only thing the WebSocket session receives to bootstrap analysis. The agent_saul `ingestion_node` is now just a database lookup — `SELECT raw_text WHERE doc_id = X` from wherever you store the raw file (MongoDB/S3/Postgres). The heavy extraction — entity discovery, embedding, Postgres inserts — all happened at upload time. This means the WebSocket pipeline never blocks on Docling or OCR. It starts with structured data already in the database.

The critical implication: `IngestionGraph` and `agent_saul` share the same Postgres instance but they write to different tables. `IngestionGraph` writes to `entities`, `clauses`, `relationships`. `agent_saul` reads from them. They never touch each other's writes during a live session.

---

**How Graphiti and Neo4j relate — they're not the same thing**

Graphiti creates its own internal Neo4j driver. `app.state.neo4j_driver` is your application's direct driver. They point at the same Neo4j database but through different connection pools. Graphiti manages the episode and entity nodes it creates. Your raw Cypher queries via `Neo4jSubgraphExpander` query those same nodes directly. The handoff is: Graphiti writes the nodes and edges through its API, your Cypher reads them back at depth using structural traversal Graphiti's search API doesn't expose. Never write Neo4j nodes directly that conflict with Graphiti's internal schema — use Graphiti's `add_episode` for all writes, Cypher only for reads.

---

**The tool call flow from agent to storage**

When `risk_analysis_node` executes, it runs `create_react_agent` with `risk_tools = [query_knowledge_graph, get_obligation_chain, detect_graph_conflicts]`. The LLM decides which tool to call and with what arguments. Before any tool executes, `IdempotencyGuard.get()` checks Redis first (O(1), 24h TTL), then Postgres `tool_executions` if Redis misses. If a cached result exists, the tool returns it without touching Graphiti or Neo4j. If not, the tool runs, then writes to both Redis and Postgres before returning. This means if the process dies mid-session and the client resumes via the same `thread_id`, every tool that already ran returns its cached result instantly — no duplicate graph writes, no duplicate LLM calls.

---

**The three write destinations and when each fires**

`relationship_mapping_node` writes clause episodes to Graphiti in parallel (semaphore=5 concurrent writes), then relationship edges sequentially. This sequencing matters — edges reference entities that must already exist as Graphiti nodes. If you reverse the order, edge writes fail silently because Graphiti can't resolve the `from_entity`/`to_entity` references.

`persist_memory_node` fires after human approval. It writes to Graphiti (final report as high-trust episode, `group_id=user_id` for cross-document precedent queries), Cognee (report JSON for episodic recall, relationships text for procedural patterns), and the `events` table as an immutable audit entry. The events write is fire-and-forget — if it fails, the function logs and continues. The mutable writes (Graphiti, Cognee) are the source of truth. Events is the audit trail, not the data.

The `memory_versions` table is only touched by the `ReconciliationGraph`. Nothing in the live pipeline writes to it.

---

**How MemoryScope gates every retrieval call**

`build_agent_context()` in `memory_pipeline.py` receives a `MemoryScope`. Before querying any source, it checks `scope.allows_source("graph")`, `scope.allows_source("vector")`, `scope.allows_source("structured")`. Sources not in the scope are never queried — the call doesn't happen at all. Entity type filtering happens post-query in the structured Postgres path (`WHERE entity_type = ANY(:allowed_types)`). For Neo4j subgraph traversal, type filtering happens post-traversal in `_parse()` — you query Neo4j without a type filter (because Neo4j doesn't know about your application entity types), then drop nodes whose `entity_type` property isn't in `scope.allowed_entity_types`.

The five pre-defined scopes map directly to agents. You never construct a `MemoryScope` ad-hoc inside a node — always use the pre-defined constants. This is a firm rule, not a suggestion. Ad-hoc scopes are how agents start reading each other's memory and producing contaminated reasoning.

---

**ReconciliationGraph — it runs on stale data by design**

It runs on entities created in the last N hours (default 24h, beat schedule is every 6h). The `normalized_name` prefix match (`LEFT(normalized_name, 10)`) is a pragmatic similarity heuristic — it catches "acme corp" vs "acme corporation" but not "acme" vs "the acme group". The LLM reconciliation step (`RECONCILE_PROMPT`) uses loss aversion framing — "NEVER delete without justification" biases the model toward `ignore` over `merge`. This is correct for legal systems where false merges (conflating two distinct parties) are catastrophically worse than false negatives (keeping a harmless duplicate).

The `apply_changes_node` does relationship re-pointing before deletion: it redirects `from_entity_id` and `to_entity_id` FK references from the discarded entity to the kept one. If you skip this step, deleting the discarded entity cascades and destroys all its relationship records. The `ON DELETE CASCADE` on the FK is a safety net, not an intended execution path for merges.

---

**Things you must keep in mind before going to production**

First: Cognee's `cognify()` is a full graph rebuild, not an incremental append. Every call to `cognify()` processes the entire dataset named in the call. If you call it inline in `persist_memory_node` for every document, you get O(n²) runtime as the dataset grows. Move `cognify()` to a Celery task triggered after `store_final_report` completes. Your `search_episodic_memory()` returns yesterday's graph — that's acceptable. Running `cognify()` inline will block the pipeline for minutes on large datasets.

Second: Graphiti entity deduplication relies on its internal LLM-powered extraction. If the same real-world party appears as "Acme Corp" in one episode and "Acme Corporation" in another, Graphiti creates two distinct entity nodes unless the text is lexically identical. The fix is canonical entity IDs — run a normalisation pass in `entity_extraction_node` and write `normalized_name` values to Graphiti as the episode body, not raw clause text. Otherwise `get_obligation_chain` and `query_knowledge_graph` will miss half the edges on the second document you process.

Third: `AsyncPostgresSaver` requires a sync connection string, not async. The `from_conn_string()` method creates its own internal connection pool using `psycopg` (sync) regardless of whether your engine is async. Pass `settings.database_url_sync` (standard `postgresql://` scheme), not the async `postgresql+asyncpg://` URL. Passing the wrong scheme raises a runtime error at the first checkpoint write.

Fourth: The `events` table is append-only. There must be no `UPDATE` or `DELETE` on it. This is not enforced by a database constraint in the current schema — it's a convention. If you want to enforce it at the database level, add a `BEFORE UPDATE OR DELETE ON events FOR EACH ROW EXECUTE FUNCTION raise_exception('events table is immutable')` trigger in your Alembic migration.

Fifth: `memory_versions` uses a `UniqueConstraint("entity_id", "version")`. The `version` is fetched with `SELECT MAX(version)` and incremented by 1 in application code. Under concurrent reconciliation runs for the same user, two workers can read the same `MAX(version)` and both attempt to insert `version + 1`. The `ON CONFLICT DO NOTHING` is not present — this will raise a unique constraint violation. Either add `ON CONFLICT DO NOTHING` to the insert, or use a `SELECT ... FOR UPDATE` lock on the entity row before fetching the max version.

Sixth: The `decay_score` formula is borrowed from recommendation systems. For legal contracts, it's semantically wrong for active obligations — an obligation with `valid_to IS NULL` should not decay over time. Modify `_compute_decay()` to check `relationships.valid_to` before applying time decay: if any active relationship exists for the entity, clamp `time_factor` to 1.0. Only entities with all relationships expired should be subject to time-based decay.

---

> **⚡ CHOSEN ONES ONLY**
>
> **The thing nobody tells you about `astream_events v2` + `create_react_agent`**: when a tool-calling agent runs inside a graph node and the LLM issues a tool call, `astream_events` emits `on_tool_start` and `on_tool_end` events — but the `metadata.langgraph_node` field on those events is the node name of the **graph node that contains the agent** (e.g. `risk_analysis`), not the tool name. Your service's `_map_event_to_frame()` currently only maps `on_chat_model_stream` for tokens and `on_chain_start/end` for node boundaries. You're silently swallowing tool execution events. Wire `on_tool_start` → `WSNodeStartFrame(node=event["data"]["input"]["name"])` and `on_tool_end` → `WSNodeEndFrame(...)` so the client can show the user "querying knowledge graph... done" in real time. Without this, the UI shows a spinner for the full 30–90 seconds of tool execution with no progress signal.
>
> **The MemoryScope architecture has a blind spot at the Neo4j subgraph level**: `_parse()` in `Neo4jSubgraphExpander` filters nodes by `entity_type` property, but Graphiti doesn't write an `entity_type` property to its Neo4j nodes by default — it writes its own internal schema properties. The `entity_type` check will evaluate to `False` for every node Graphiti created (because the property doesn't exist), and the condition `if entity_type and not scope.allows_entity_type(entity_type)` will short-circuit on the empty string check — meaning `entity_type = ""` passes through the scope filter unblocked. Every node Graphiti writes will bypass MemoryScope filtering entirely. The fix: map Graphiti node labels (which it DOES set, like `Entity`, `EpisodicNode`) to your entity types in the `_parse()` method using the `node_labels` field. Build a lookup: `{'EpisodicNode': 'CLAUSE', 'Entity': 'ORG'}` and use that for scope checking instead of the missing `entity_type` property.

# Memory Optimization Cheatsheet

This file is a quick reference for memory-focused optimization decisions in this project.

## 1. Pydantic vs lighter containers

- Do not manually add field-level `__slots__ = ("id", "email", ...)` to `pydantic.BaseModel` subclasses as a default optimization.
- `BaseModel` already carries Pydantic-specific instance machinery, so manual slots usually do not give the win people expect.
- Use `BaseModel` when you need validation, serialization, parsing, or schema generation.
- For hot-path, short-lived, in-memory objects that do not need full Pydantic behavior, prefer:
  - `@dataclass(slots=True)`
  - `NamedTuple`
  - plain tuples or dicts when the structure is simple and local

## 2. Stream large responses

- If a response is large or unbounded, do not build the full payload in memory first.
- Prefer `StreamingResponse` with a generator or async generator.
- Best use cases:
  - large exports
  - file downloads
  - AI token streaming
  - SSE/event streams
  - long-running incremental results
- For small JSON responses, regular FastAPI responses are fine.

Example:

```python
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse


async def generate_chunks() -> AsyncIterator[bytes]:
    for chunk in data_source():
        yield chunk


@router.get("/download", response_class=StreamingResponse)
async def download() -> AsyncIterator[bytes]:
    return generate_chunks()
```

## 3. Use Gunicorn `--preload` only when deployment matches

- `--preload` loads the app in the master process before worker fork.
- This can reduce memory in multi-worker Linux deployments because workers share unchanged memory pages through Copy-on-Write.
- Helps most when startup loads:
  - large modules
  - model metadata
  - mostly immutable lookup tables
  - expensive app wiring
- Be careful:
  - do not depend on it for mutable globals
  - do not preload code with side effects that should run per worker
  - open network connections in lifespan/per-worker startup, not at import time
- This matters only if the app is actually deployed behind Gunicorn. It does not apply to plain `uvicorn.run(...)` directly.

## 4. `jemalloc` is an infra optimization

- `jemalloc` is a memory allocator, not a Python coding pattern.
- It can reduce memory fragmentation and often lowers RSS in multi-worker API containers.
- It is worth testing when:
  - idle memory is too high
  - RSS keeps growing more than expected
  - multi-worker deployments duplicate allocator overhead badly
- Do not encode `jemalloc` assumptions into application code.
- Benchmark it in the actual container/runtime environment before treating it as a win.

## 5. Avoid accidental response and object duplication

- Do not convert large iterables into `list(...)` unless you need the full materialized result.
- Prefer generator expressions or iterators when one-pass consumption is enough.
- Avoid repeated `model_dump()` or `model_validate()` calls inside tight loops when batch validation or serialization can be used.
- Use `TypeAdapter(list[T])` for large collection validation instead of per-item model validation loops.

## 6. Cache carefully

- Cache expensive pure computations, but do not cache large objects blindly.
- Unbounded caches can turn a CPU optimization into a memory leak.
- Prefer bounded caches and explicit eviction strategy.
- Reuse heavyweight clients and connection pools from app lifespan instead of rebuilding them per request.

## 7. Worker count is a memory setting too

- More workers improve concurrency only up to a point.
- Every worker adds baseline memory overhead.
- Tune worker count together with:
  - container memory limit
  - preload strategy
  - allocator choice
  - request latency profile

## 8. Measure before and after

- Check RSS, not just Python object size.
- Benchmark with realistic concurrency.
- Separate:
  - idle memory
  - steady-state memory
  - peak memory during heavy responses
- Treat memory claims like "30% lower" as workload-dependent, not universal.


