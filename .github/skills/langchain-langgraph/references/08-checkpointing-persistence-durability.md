# Checkpointing Persistence Durability

## Why Persistence Exists

LangGraph persistence saves graph state as checkpoints organized into threads. It enables:

- human-in-the-loop
- conversation memory
- time travel
- fault-tolerant execution

## Thread And Checkpoint Rules

When invoking a graph with a checkpointer, pass:

```python
{"configurable": {"thread_id": "..."}}
```

21. `checkpoint_id` is your best friend after pause and resume workflows.

36. The latest state of a graph can be viewed with:

```python
graph.get_state(config)
```

37. Full history can be viewed with:

```python
graph.get_state_history(config)
```

38. You can edit state using `update_state`. This creates a new checkpoint, not an in-place modification.

31. Delete all checkpoints for a thread:

```python
thread_id = "1"
checkpointer.delete_thread(thread_id)
```

## Checkpoint Namespace

Each checkpoint has a `checkpoint_ns` field.

- `""` means root graph
- `"node_name:uuid"` means subgraph checkpoint
- nested subgraphs join namespaces with `|`

Example:

```python
def my_node(state: State, config: RunnableConfig):
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
```

## Serializers

39. Checkpointers serialize state channel values using serializer objects.

`JsonPlusSerializer` is the default and handles many built-in LangChain and LangGraph types.

If needed, enable pickle fallback for unsupported types.

## Durable Execution Guidance

40. Durable execution exists to avoid repeating work and to replay non-deterministic behavior consistently.

Guidelines:

- wrap multiple side-effecting operations into separate tasks when replay matters
- isolate non-deterministic operations
- make side effects idempotent when possible

## Durability Modes

41. Durability modes from least to most durable:

- `exit`
- `async`
- `sync`

`exit` is best for performance but loses mid-execution recovery.

`async` is a good middle ground.

`sync` gives the strongest guarantees and the highest overhead.

## Deep Research Addendum (2026-10-25)

This addendum reconciles the concise rules above with the current LangGraph persistence model and the operational boundaries that matter in an async FastAPI service. It reflects research completed through 2026-10-24 for the requested 2026-10-25 cutoff; verify moving documentation and pinned package versions before release. Primary sources: [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [checkpoint reference](https://reference.langchain.com/python/langgraph/checkpoints), [stores](https://docs.langchain.com/oss/python/langgraph/stores), [durable execution](https://docs.langchain.com/oss/python/langgraph/durable-execution), and the [checkpoint-loading security advisory](https://github.com/advisories/GHSA-g48c-2wqr-h844).

### 1. Persistence is four separate planes

Checkpointing is thread-scoped workflow state. It records graph state and execution metadata at super-step boundaries: channel values, next nodes, configuration, metadata, parent configuration, and task information. It is not a Python process snapshot and does not restore an open socket, database connection, credential, coroutine stack, or arbitrary live object. See the [persistence model](https://docs.langchain.com/oss/python/langgraph/persistence).

| Plane | Owns | Lifetime | Do not use it as |
|---|---|---|---|
| Checkpointer/checkpoints | Thread state, progress, interrupts, task writes, history | One workflow/thread across invocations | A business ledger or credential vault |
| Store | Application-defined namespaced memory shared across threads | User/tenant/application scope | An implicit authorization layer |
| Runtime context | Trusted run inputs and live dependencies (`context=` / `Runtime.context`) | One invocation/resumption | Persisted state or a client-supplied tenant field |
| Business database/outbox | Orders, balances, approvals, payment outcomes, immutable facts | Domain-defined | A graph-history substitute |

The distinction is architectural, not cosmetic. A checkpoint may explain that a payment was requested; the payment database must prove whether it committed. Store memory may remember a preference; it should not be the source of truth for a balance. Rebuild runtime context from the authenticated principal and application services on every invocation. Keep secrets, clients, locks, file handles, and connection pools out of graph state.

### 2. Declaration and configuration patterns

The normal Python graph pattern is `builder.compile(checkpointer=saver, store=store)`. `compile()` also accepts `checkpointer=None`, `checkpointer=True`/`False` for subgraph inheritance controls, cache, interrupt configuration, debug, and a graph name; use the keyword form for readability. The functional API uses `@entrypoint(checkpointer=saver)`, and the current LangChain agent factory accepts `checkpointer=` and `store=`. See [StateGraph compile](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/graph/state.py), the [functional API](https://docs.langchain.com/oss/python/langgraph/functional-api), and the [LangChain v1 migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1).

```python
from langgraph.checkpoint.memory import InMemorySaver

saver = InMemorySaver()  # tests and local experiments only
graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "expense-123"}}
result = await graph.ainvoke(
    {"operation_id": "op-123"}, config, durability="sync"
)
```

The saver is a construction-time dependency; invocation configuration selects the addressed thread and optional historical checkpoint. Do not instantiate a saver per request, pass an unentered context manager, or expect `thread_id` to create persistence without a compiled checkpointer. `create_agent` in LangChain v1 supersedes legacy `create_react_agent`; migrate `prompt` to `system_prompt`, and keep agent state as the supported `TypedDict` shape. Direct `StateGraph` supports Pydantic state, but the v1 agent factory does not accept Pydantic/dataclass agent state. Keep Pydantic v2 models at the HTTP boundary and a small serializable state schema inside the graph.

### 3. Addressing: `thread_id`, `checkpoint_id`, `checkpoint_ns`

| Field | Meaning | Rule |
|---|---|---|
| `thread_id` | Stable workflow/conversation identity | Generate and authorize server-side; reuse for continuity |
| `checkpoint_id` | One saved checkpoint in that thread/namespace | Omit for latest; use the complete config returned by a snapshot/history operation for replay or fork |
| `checkpoint_ns` | Graph/subgraph address | Root is `""`; child namespaces are framework-generated (for example `node:uuid` and nested `a|b`) |

```python
config = {"configurable": {"thread_id": thread_id}}
latest = await graph.aget_state(config)
historical = await graph.aget_state(selected.config)
```

Never treat `checkpoint_ns` as a tenant identifier, and never let a browser submit arbitrary checkpoint configuration. The PostgreSQL schema keys checkpoints by `(thread_id, checkpoint_ns, checkpoint_id)`; all three are addressing data, not an authorization policy. Authorize ownership before `get_state`, history, stream, resume, replay, update, fork, and deletion. Keep IDs short opaque values even when the backend column is `TEXT`; do not rely on a troubleshooting page's stated length as a universal storage limit.

### 4. Checkpointers, stores, and sync/async APIs

The standard backend families are `InMemorySaver` (fast, non-durable), `SqliteSaver`/`AsyncSqliteSaver` (local file and small deployments), `PostgresSaver`/`AsyncPostgresSaver` (the usual production choice), and custom `BaseCheckpointSaver` implementations. A PostgreSQL shallow saver may be available in the installed package; verify its history/time-travel semantics before selecting it. See the [checkpoint API](https://reference.langchain.com/python/langgraph/checkpoints) and [PostgreSQL async source](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py).

| Intent | Synchronous | Asynchronous |
|---|---|---|
| Execute | `invoke` | `await ainvoke` |
| Stream | `stream` | `async for item in astream(...)` |
| Latest state | `get_state` | `await aget_state` |
| History | `get_state_history` | `async for snap in aget_state_history(...)` |
| State edit/fork | `update_state` | `await aupdate_state` |
| Raw saver read/list | `get`/`get_tuple`, `list` | `aget`/`aget_tuple`, `alist` |
| Delete thread | `delete_thread` | `await adelete_thread` |

Use graph-level APIs in application code; raw `put`/`put_writes` are saver internals and are not a substitute for a reducer-aware state edit. Never call blocking saver methods from an event-loop thread. `AsyncPostgresSaver` supports a long-lived async connection or pool; its connection-string helper is a context manager, so its context must span all graph use, not merely setup.

Store namespaces are tuples such as `(tenant_id, user_id, "preferences")`, with a key and dictionary value per item. `put`, `get`, `delete`, `search`, and namespace listing have async counterparts (`aput`, `aget`, `adelete`, `asearch`, `alist_namespaces`). Namespace prefixes match nested namespaces; semantic search requires an embedding model and configured indexed fields. Store data is cross-thread memory, not checkpoint history. Derive namespaces from an authenticated principal, never from an arbitrary request prefix. See [Store documentation](https://docs.langchain.com/oss/python/langgraph/stores).

### 5. Durability, pending writes, and exactly-once boundaries

Durability is a persistence timing choice, not a distributed transaction protocol. `sync` persists before the next step begins; `async` persists while the next step executes; `exit` persists when execution exits successfully, with an error, or on an interrupt. Documentation and implementation defaults can move between versions, so set `durability` explicitly where the failure window matters. See [durable execution](https://docs.langchain.com/oss/python/langgraph/durable-execution) and the [execution source](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/main.py).

| Mode | Best property | Failure window | Fit |
|---|---|---|---|
| `exit` | Lowest intermediate persistence overhead | Process crash can lose intermediate progress | Recomputable work |
| `async` | Overlaps persistence with execution | Crash can race checkpoint write | Latency-sensitive, repeat-safe work |
| `sync` | Strongest graph progress boundary | More I/O latency; external effects still separate | Approvals and critical transitions |

Pending writes are successful task/node writes from an unfinished super-step. If one parallel branch fails, persisted successful sibling writes can prevent those siblings from rerunning on recovery. A pending write is not a completed checkpoint, and neither proves that an external API effect was absent. A provider can accept a charge immediately before the worker dies; a retry can then repeat it.

Design every externally visible effect with a stable business idempotency key, such as `tenant:operation:capture:v1`, and persist/reconcile the provider receipt in a domain transaction. Use `sync` to improve workflow recovery, but never describe checkpoints, retries, tasks, or `sync` as end-to-end exactly-once execution.

### 6. Resume, interrupt, replay, time travel, and fork

Treat these as different commands with different authorization and side-effect policies:

| Operation | Typical action | Key behavior |
|---|---|---|
| Failure recovery | `await graph.ainvoke(None, config)` | Continues from saved progress; same thread |
| Dynamic interrupt | `await graph.ainvoke(Command(resume=value), config)` | Interrupting node restarts at its beginning; code before `interrupt()` runs again |
| New turn | `ainvoke(new_input, thread_config)` | Applies input to latest thread state |
| Inspect history | `aget_state_history(config)` | Read snapshots without executing |
| Historical replay | Invoke with a selected snapshot's full config | Skips earlier nodes, re-executes downstream nodes and effects |
| Corrective fork | `await graph.aupdate_state(selected.config, values, as_node=...)` then invoke returned config | Creates a new checkpoint branch; original history remains |

`update_state` is not an in-place SQL update. Reducers determine whether a channel is appended, merged, or replaced; `as_node` disambiguates the writer and controls successor selection. Always continue using the configuration returned by `update_state`. Historical replay is not a cached response: LLM calls, API calls, and interrupts after the selected checkpoint can run again. Expose replay/fork only behind operator authorization and an effect policy (often sandbox or read-only adapters). See [time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel) and [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts).

The functional API's `@entrypoint` shares the same checkpoint machinery; it accepts one positional input and can use `@task` for replay-safe side effects. Completed task results can be restored on resume, but interrupt ordering is part of the durable contract: changing the order of interrupt calls can mismatch resume values in old runs. Put non-deterministic work and side effects inside small tasks, keep task call order stable, and make each task idempotent.

### 7. Subgraph persistence and namespaces

Subgraph `checkpointer=None` (default) means per-invocation persistence inherited from the parent; `True` means per-thread child persistence; `False` means stateless execution. These are execution/isolation modes, not three root backend types.

| Mode | State across calls | Good fit | Trap |
|---|---|---|---|
| `None` | Fresh child invocation, with inherited persistence | Independent parallel subagents | No accumulated child conversation |
| `True` | Same child thread accumulates state | Continuing specialist assistant | Same namespace cannot safely handle concurrent calls |
| `False` | None | Cheap pure/recomputable helper | No child interrupts or crash recovery |

Subgraph checkpoints use framework-generated namespaces. A parent may expose a child as one super-step unless the child has its own checkpointing; do not assume parent history contains every internal child step. Calls to the same stateful subgraph namespace must be serialized; use per-invocation subgraphs for independent concurrent work. Namespace isolation prevents checkpoint collisions, but it is not tenant authorization. See [subgraph persistence modes](https://docs.langchain.com/oss/python/langgraph/use-subgraphs).

### 8. Serialization, encryption, and trust

`JsonPlusSerializer` is the default and handles many LangChain/LangGraph values, dates, enums, and Pydantic v2 values. Persist simple dictionaries, IDs, numbers, strings, and supported message types; do not persist clients, closures, credentials, or objects whose import path is unstable. The serializer source documents strict msgpack behavior, allowlists, and `pickle_fallback`; inspect your pinned version at [jsonplus.py](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py).

```python
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

serde = JsonPlusSerializer(
    pickle_fallback=False,
    allowed_msgpack_modules=None,  # safe built-in allowlist
)
saver = AsyncPostgresSaver(pool, serde=serde)
```

`pickle_fallback=True` increases compatibility at the cost of deserialization risk and upgrade burden; do not enable it for production state without a reviewed trust model. The documented encrypted serializer can wrap checkpoint payloads, for example `EncryptedSerializer.from_pycryptodome_aes()` using `LANGGRAPH_AES_KEY`; key management, rotation, and old-checkpoint reads remain application responsibilities. Encryption of serialized values does not automatically encrypt database identifiers, metadata, Store rows, logs, traces, or backups. Configure TLS, database-at-rest encryption, secret distribution, backup access, and log redaction separately.

The 2026 checkpoint-loading advisory describes unsafe msgpack deserialization when an attacker can modify trusted checkpoint bytes; its patched boundary is `langgraph` 1.0.10. Upgrade to a fixed release, restrict database writers, enable strict serializer policy, and treat imported checkpoint databases/backups as code-adjacent trusted material. Version fixes do not replace least privilege.

### 9. Production FastAPI lifecycle (async + Pydantic v2)

Create the async pool, saver, and compiled graph once in FastAPI `lifespan`; close them only after application-owned workers drain. Run `AsyncPostgresSaver.setup()` as a controlled deployment migration before traffic, not on every request. For manual psycopg construction, the documented PostgreSQL saver expects `autocommit=True` and `row_factory=dict_row`; use the package helper or match its contract. See [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events) and [AsyncPostgresSaver](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py).

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(
        CHECKPOINT_DSN
    ) as saver:
        await saver.setup()  # preferably a deploy step, not request code
        app.state.graph = build_graph(saver)
        yield
        # Drain application workers before this context exits.

app = FastAPI(lifespan=lifespan)
```

For higher concurrency, own an `AsyncConnectionPool` in lifespan and pass it to `AsyncPostgresSaver`; size pool limits against the total number of API workers, background workers, and other database clients. A process-local `asyncio.Lock` inside a saver is not a cross-process same-thread lock. Keep request DTOs (`BaseModel`, `extra="forbid"`, strict fields) separate from a small `TypedDict` graph state and a frozen runtime-context object containing the authorized tenant, user, and live adapters.

An HTTP request should authenticate first, resolve the authoritative thread owner/workflow version, acquire distributed same-thread admission, invoke with `context=...`, then render a Pydantic response from `aget_state`. For long-running work, enqueue a managed worker and return an operation status; do not rely on an untracked background coroutine that outlives a request and may race shutdown.

### 10. Concurrency and database transaction boundaries

Sharing one compiled graph is appropriate; mutating graph-global state per request is not. Concurrent requests against independent threads are safe only within pool and back-pressure limits. Concurrent writes against one thread need an explicit policy: reject, queue, or serialize using a distributed lease/fencing token. A larger pool does not solve same-thread ordering, and a stateful subgraph cannot be called concurrently in one namespace.

Do not hold an order/payment transaction open across an LLM call, graph wait, interrupt, or human approval. The checkpointer has its own persistence operations; pending writes may commit before the final checkpoint. A business row and a checkpoint therefore are not atomically committed merely because both use PostgreSQL. Use short domain transactions with unique operation constraints and a transactional outbox for business-change-plus-event publication. See [Psycopg transactions](https://www.psycopg.org/psycopg3/docs/basic/transactions.html) and the [transactional outbox pattern](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html).

### 11. Tenant isolation and operational security

The saver key is not an access-control policy. Enforce ownership before latest-state reads, history, streaming, resume, replay, fork, edits, deletion, and Store search. Derive `tenant_id` from verified authentication, carry it in runtime context, and validate state ownership at node boundaries. Use opaque IDs and an authoritative `(tenant, thread, workflow_version)` registry. Restrict database roles and checkpoint writers; if using PostgreSQL row-level security, test how tenant identity is set on every pooled connection/transaction. See [PostgreSQL row security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html).

Audit operator replay/fork/update/delete actions, checkpoint selection, workflow/schema version, actor, and effect policy. Redact message content, tokens, payment data, and secrets from traces and logs. Coordinate retention across checkpoints, Store, traces, backups, and encryption keys; deleting one visible checkpoint row is not a complete erasure policy.

### 12. Testing and observability matrix

| Test | Required assertion |
|---|---|
| Continuity | Same thread accumulates state; new thread does not |
| Restart | Kill/recreate worker and recover from intended backend, not RAM |
| Pending writes | Successful parallel sibling is not needlessly rerun after another fails |
| Effect/checkpoint gap | Crash after provider success does not duplicate an idempotent effect |
| Interrupt | Resume uses the same thread; duplicate/stale approvals are rejected |
| Replay/fork | Historical execution uses sandbox/read-only effects; original branch remains |
| Multi-worker race | Distributed admission/fencing wins over process-local locks |
| Tenant substitution | Other tenant cannot read/update/replay/delete state or Store memory |
| Upgrade | Old checkpoint fixtures load under new serializer, graph, and node names |
| Retention/key change | Cleanup, backup restore, key rotation, and old-state reads are understood |

Use `InMemorySaver` for fast unit tests, but use the production backend for restart, concurrency, migration, and retention tests. Record operation ID, authorized tenant, thread, selected checkpoint, graph/schema version, resume count, saver latency/errors, pool saturation, and idempotency conflicts; redact payloads. Observability explains runs; it is not the recovery ledger. Studio can inspect trajectories and rewind/edit/rerun, but those controls must be governed as production operations.

### 13. Version and migration traps

There are three migrations: saver database schema (`setup()`), application graph/state schema, and serializer/code compatibility. A saver migration does not migrate old state fields or import paths. Pin the complete package set and retain interrupted-run fixtures. Keep node names, interrupt ordering, reducer semantics, and serialized type paths stable or implement explicit compatibility/defaulting. Prefer a `state_version` field and a migration adapter at the graph boundary.

The v1 agent migration (`create_react_agent` to `create_agent`, `prompt` to `system_prompt`, and the agent node label change from `agent` to `model`) can break stream consumers and tests. `config_schema` is deprecated in favor of `context_schema`/`context=` in current source. `checkpoint_during` is deprecated; do not combine it with `durability`. Package release numbers and moving `main` source are observations, not a compatibility matrix; verify them against the lockfile and run a canary recovery test before rollout.

### 14. JavaScript parity (concise)

JavaScript uses the same concepts with different names and package boundaries. The official example uses `MemorySaver`, `InMemoryStore`, `compile({ checkpointer, store })`, and `thread_id` under `configurable`; invocation options carry durability:

```javascript
import { MemorySaver, InMemoryStore } from "@langchain/langgraph";

const checkpointer = new MemorySaver();
const store = new InMemoryStore();
const graph = builder.compile({ checkpointer, store });
const config = {
  configurable: { thread_id: "expense-123" },
  durability: "sync",
};
const result = await graph.invoke(input, config);
```

Use the separate [`@langchain/langgraph-checkpoint-postgres`](https://reference.langchain.com/javascript/langchain-langgraph-checkpoint-postgres) package for PostgreSQL. Do not copy Python class names, psycopg settings, serializer allowlists, or pool shutdown assumptions into JavaScript without checking that package's own reference. The shared rules remain: authorize thread IDs, keep runtime dependencies out of state, make external effects idempotent, and treat replay/fork as effectful operations.

### Decision summary

| Decision | Default for async FastAPI |
|---|---|
| Root persistence | Long-lived `AsyncPostgresSaver` in application lifespan |
| Critical transitions | Explicit `durability="sync"` plus idempotent business effects |
| State | Small serializable `TypedDict`; Pydantic v2 only at API boundary |
| Runtime dependencies | Reconstructed `context=` on each invocation/resume |
| Subagents | Per-invocation subgraphs unless multi-turn memory is required |
| Serialization | Strict JSON/msgpack policy; no casual pickle fallback; encryption where required |
| Same-thread concurrency | Distributed admission/queue/fencing, not a local lock |
| Transactions | Short business transactions + outbox; never assume checkpoint atomicity |
| Operations | Separate endpoints/policies for resume, replay, fork, update, and delete |

The veteran rule is to reason from crash windows: for every side effect, ask what happens after the provider commits but before the checkpoint, after the checkpoint but before the response, and after a worker lease expires. Checkpoint durability narrows workflow-loss windows; only domain idempotency, unique constraints, and explicit reconciliation close business-effect windows.

### Research bibliography

- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) · [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers) · [Checkpoint API](https://reference.langchain.com/python/langgraph/checkpoints)
- [Stores](https://docs.langchain.com/oss/python/langgraph/stores) · [Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api) · [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) · [Time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel) · [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) · [LangChain short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) · [LangChain v1 migration](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [StateGraph source](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/graph/state.py) · [execution source](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/langgraph/pregel/main.py) · [async PostgreSQL saver](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py) · [PostgreSQL schema/source](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py) · [JsonPlusSerializer](https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py)
- [LangGraph package](https://pypi.org/project/langgraph/) · [checkpoint package](https://pypi.org/project/langgraph-checkpoint/) · [PostgreSQL package](https://pypi.org/project/langgraph-checkpoint-postgres/) · [security advisory](https://github.com/advisories/GHSA-g48c-2wqr-h844)
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events) · [Psycopg transactions](https://www.psycopg.org/psycopg3/docs/basic/transactions.html) · [PostgreSQL row security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html) · [Transactional outbox](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html)
