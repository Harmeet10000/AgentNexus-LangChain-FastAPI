# Memory

## Quick Reference

Memory type | Scope | Stored in | Source notes
--- | --- | --- | ---
Short-term memory | One thread or conversation | Graph or agent state via checkpointer | 55, 57
Long-term memory | Across conversations or threads | LangGraph store namespace/key JSON docs | 30, 55, 57, 66
Semantic memory | Facts and concepts | Profile doc or collection | 66
Episodic memory | Past actions and experiences | Few-shot examples or retrieved examples | 66
Procedural memory | Rules and instructions | Prompts, code, weights, or editable instruction memory | 66

## Long-Term Memory API

Long-term memory is built on LangGraph stores, stored as JSON documents by namespace and key. Pass `store=...` to `create_agent`, then access it through `runtime.store` inside tools.

`InMemoryStore` is for development and testing. Use DB-backed stores for production.

Configure store indexing with embeddings and dimensions to support semantic memory search.

Namespace guidance:

- `(user_id, "memories")`
- `(org_id, "preferences")`
- `(user_id, app_context)`

## Preserved Source Notes

30. The recommended way to access the store is through the Runtime object.

55. Tools can access the current conversation state using `runtime.state`.

```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage

@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    messages = runtime.state["messages"]
```

Update state with `Command`:

```python
from langgraph.types import Command
from langchain.tools import tool

@tool
def set_user_name(new_name: str) -> Command:
    return Command(update={"user_name": new_name})
```

Access context through `runtime.context`.

Access store through `runtime.store`.

57. By default, agents use `AgentState` for short-term memory, especially conversation history through `messages`.

```python
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel

class CustomState(AgentState):
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(runtime: ToolRuntime[CustomContext, CustomState]) -> Command:
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    })
```

## File Memory Conventions

If memory has to be stored on disk, source notes suggest two file patterns:

1. `memory-dd-mm-yyyy.md`
2. `Memory.md`

The first is a daily running memory log.

The second is curated long-term facts, decisions, and preferences.

## Semantic Episodic Procedural Memory

66. Semantic memory stores facts.

Episodic memory stores experiences and past actions.

Procedural memory stores instructions and task rules.

Profile vs collection trade-off:

- profile: unified JSON document, easier centralization, harder updates at scale
- collection: higher recall and smaller atomic memories, but more search and deduplication complexity

Procedural memory can be refined by reflection or meta-prompting, where an agent updates its own instructions based on conversation history and feedback.

## Deep Research Addendum (2026-10-25)

This addendum expands the compact reference above into a production-oriented memory model. It reflects primary documentation and source material reviewed through 2026-10-24 for the requested 2026-10-25 cutoff; the date boundary is explicit because package APIs and moving documentation can change. Pin and test the exact versions deployed. Primary entry points are [LangGraph memory](https://docs.langchain.com/oss/python/langgraph/memory), [Stores](https://docs.langchain.com/oss/python/langgraph/stores), [short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory), and the [Store API](https://reference.langchain.com/python/langgraph/store/base/BaseStore).

### 1. Memory is not one persistence mechanism

Short-term memory is thread-scoped graph/agent state, typically message history plus task fields, persisted by a checkpointer. Long-term memory is application-defined data in a Store and can be read across threads. Runtime context is trusted, run-scoped identity and dependency input. None of these is automatically an authorization policy or a business system of record.

| Plane | Typical contents | Scope | Correct responsibility |
|---|---|---|---|
| Checkpoint / short-term state | Messages, current task state, pending work, interrupt progress | One `thread_id` | Resume and inspect a workflow |
| Store / long-term memory | Preferences, facts, evidence, examples, profiles | Namespace/key across threads | Reusable application memory |
| Runtime context | Authenticated tenant/user, feature flags, live clients | One invocation/resume | Reconstruct trusted dependencies |
| Procedural policy | Safety rules, tool permissions, workflow instructions | Deployment/application | Reviewed code/prompt/policy, not user-writable memory |
| Business DB/vector service | Orders, consent, ledger facts, externally indexed documents | Domain-defined | Authoritative invariants and retrieval infrastructure |

The framework supplies state, checkpointer, Store, runtime, and adapters. The application still owns identity, consent, provenance, conflict resolution, retention, deletion, and which memories may enter a prompt. A Store record saying “payment succeeded” is not evidence that the payment ledger committed; a recalled user instruction is data, not a system/developer instruction.

### 2. Semantic, episodic, and procedural memory

These labels describe what should be remembered and how it should be used; they are not three mandatory LangGraph backends.

| Kind | Meaning | Durable representation | Main hazard |
|---|---|---|---|
| Semantic | Stable facts, relationships, preferences | Profile fields or structured Store records | Stale/conflicting facts |
| Episodic | Past experiences, examples, successful trajectories | Dated summaries or retrieved few-shot examples | Irrelevant or unsafe imitation |
| Procedural | Rules for performing a task | Reviewed prompt, middleware, code, model weights, or controlled instruction memory | Prompt injection or self-modification |

Treat procedural policy as application-controlled. An agent may propose a procedural improvement from feedback, but promotion should be reviewed, versioned, tested, and auditable. Store episodic examples with source, timestamp, task type, outcome, and trust level; retrieve them as examples, never as executable instructions.

### 3. Profile versus collection versus hybrid

A profile is one continually updated JSON document. It is simple to load and present as current state, but every update must reconcile the whole document and can erase provenance. A collection stores smaller atomic records, improving targeted retrieval and deletion, but introduces duplicate detection, precedence, and context assembly.

| Shape | Pros | Cons | Use when |
|---|---|---|---|
| Profile | One read; easy current view | Large reconciliation; weak evidence history | Small, authoritative preference set |
| Collection | Granular writes/search/deletion; better evidence | Duplicates, conflicts, more retrieval work | Many facts/events or auditable memory |
| Hybrid | Curated current profile plus evidence collection | Requires precedence/consolidation policy | Product needs both low-latency reads and provenance |

Choose the shape per memory type, not by fashion. For a preference, deterministic key `language` can make retries idempotent. For episodic evidence, use an immutable event/evidence key and derive a curated profile asynchronously. Include `source`, `observed_at`, `confidence`, `schema_version`, and `supersedes` metadata where stale or disputed facts matter.

### 4. Store identity: namespaces and keys

Store namespaces are tuples of strings; a key addresses one item in that namespace. Prefix searches can span nested namespaces, but namespace structure is an organizational convention, not an ACL. Build it from verified server-side identity, for example:

```python
def memory_namespace(tenant_id: str, user_id: str) -> tuple[str, ...]:
    return ("tenant", tenant_id, "user", user_id, "memories")
```

Never accept `tenant_id`, `user_id`, namespace, or key ownership directly from a browser or model tool argument. Resolve the authenticated principal first, check thread/workspace ownership, then derive the namespace. For cross-agent sharing, give agents a deliberately shared namespace such as `(tenant, tenant_id, "workspace", workspace_id, "shared")` and expose only the operations each role requires.

Use deterministic keys for retry-safe upserts and stable evidence IDs for collection records. Random keys are appropriate only when every event is intentionally distinct. Keep namespace and key lengths bounded, normalize Unicode and case where the product requires it, and include a state/schema version in values rather than encoding version semantics solely into keys.

### 5. Store APIs and vector indexing

The Python `BaseStore` contract provides synchronous and asynchronous methods: `get`/`aget`, `put`/`aput`, `delete`/`adelete`, `search`/`asearch`, and `batch`/`abatch`; `list_namespaces`/`alist_namespaces` may be available in the installed implementation. Batch results correspond to input order. Use async methods in async graph nodes and FastAPI handlers; do not block the event loop with sync Store operations.

`search` can combine namespace prefix, semantic query, metadata filter, limit, offset, and TTL refresh options. A semantic query does not magically enable vectors: configure an index with an embedding function, fixed dimensions, and the fields to embed. Keep embedding dimensions consistent with the model and migration plan. `index=False` can preserve an item for exact retrieval while excluding it from vector search. Do not embed secrets, access tokens, raw unfiltered chat logs, or data unrelated to the retrieval purpose.

| Retrieval mode | Needed configuration | Strength | Failure mode |
|---|---|---|---|
| Key lookup | Namespace + key | Deterministic, low latency | Requires known identity |
| Prefix/filter search | Namespace prefix and metadata filter | Structured browsing | Can over-return or leak scope |
| Vector search | Index, embedding function, dimensions, indexed fields | Meaning-based recall | Drift, irrelevant matches, embedding cost |
| External vector DB | Application-owned ingestion/query adapter | Specialized scale/indexing | Separate auth, consistency, deletion, and tenancy |

Postgres Store/pgvector is infrastructure; it is not a memory extraction or authorization policy. Validate representative queries, tenant filters, stale records, duplicate records, and deletion propagation before treating semantic recall as production-ready.

### 6. Runtime access: `runtime.store`, `ToolRuntime`, and `InjectedStore`

Graph nodes can receive `Runtime` and use `runtime.store` plus `runtime.context`. Modern tools receive an injected `ToolRuntime`, which includes state, context, Store, and tool-call metadata while remaining hidden from the model-facing tool schema. `InjectedStore` is a lower-level annotation for injecting a persistent Store into tool arguments; retain it only where the pinned API requires it, and do not expose a raw Store parameter to the model.

```python
from langchain.tools import ToolRuntime, tool

@tool
async def save_preference(name: str, value: str, runtime: ToolRuntime) -> str:
    """Save a preference for the authenticated user."""
    ctx = runtime.context
    namespace = ("tenant", ctx.tenant_id, "user", ctx.user_id, "preferences")
    await runtime.store.aput(
        namespace,
        key=name,
        value={"name": name, "value": value, "schema_version": 1},
    )
    return "saved"
```

The model supplies `name` and `value`; the server supplies `runtime.context`, the Store, and namespace authority. Narrow tools (`save_preference`, `forget_preference`, `search_preferences`) are easier to authorize and audit than a generic “write anything to memory” tool. See [Python tools](https://docs.langchain.com/oss/python/langchain/tools), [Runtime](https://reference.langchain.com/python/langgraph/runtime/Runtime), and [InjectedStore](https://reference.langchain.com/python/langgraph.prebuilt/tool_node/InjectedStore).

### 7. Checkpointer versus Store

| Question | Checkpointer | Store |
|---|---|---|
| Scope | One graph thread | Cross-thread namespace/key |
| Main operation | Resume/inspect workflow | Read/write reusable records |
| Typical payload | State channels, messages, tasks | JSON values and metadata |
| Retrieval | Latest/history/checkpoint config | Key, prefix/filter, optional vector search |
| Deletion | Thread checkpoint/write history | Item/namespace deletion, TTL where supported |
| Security | Optional serializer encryption | Backend/database policy must be configured separately |

Use both when an assistant needs conversation continuity plus durable preferences. Do not put every long-term fact into message history, and do not use Store memory to reconstruct a half-completed graph run. A trimmed prompt is not checkpoint deletion; a deleted Store item is not removal from traces, caches, replicas, backups, or a vector index.

### 8. Async FastAPI and Pydantic v2 composition

Own the checkpointer, Store, connection pools, and compiled agent in FastAPI `lifespan`. Run backend `setup()` as a controlled deployment step before serving traffic, or make startup setup explicitly idempotent and serialized. Close pools after application-owned workers drain. Pydantic v2 validates the HTTP contract; keep agent state and Store values small, typed, serializable, and versioned.

```python
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

class ChatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str

@dataclass(frozen=True)
class AgentContext:
    tenant_id: str
    user_id: str

def ns(ctx: AgentContext) -> tuple[str, ...]:
    return ("tenant", ctx.tenant_id, "user", ctx.user_id, "preferences")

@tool
async def remember_preference(
    name: str, value: str, runtime: ToolRuntime[AgentContext]
) -> str:
    await runtime.store.aput(
        ns(runtime.context), name,
        {"name": name, "value": value, "schema_version": 1},
    )
    return "Preference saved."

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        saver = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(CHECKPOINT_DSN)
        )
        store = await stack.enter_async_context(
            AsyncPostgresStore.from_conn_string(STORE_DSN)
        )
        await saver.setup()
        await store.setup()
        app.state.agent = create_agent(
            model=MODEL,
            tools=[remember_preference],
            checkpointer=saver,
            store=store,
            context_schema=AgentContext,
        )
        yield

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(body: ChatIn, request: Request):
    principal = request.state.principal  # verified authentication dependency
    thread_id = await request.app.state.threads.require_owned(
        principal, request.headers["x-thread-id"]
    )
    context = AgentContext(principal.tenant_id, principal.user_id)
    return await request.app.state.agent.ainvoke(
        {"messages": [{"role": "user", "content": body.message}]},
        config={"configurable": {"thread_id": thread_id}},
        context=context,
    )
```

The example intentionally leaves authentication, thread ownership, rate limits, response shaping, and distributed same-thread admission as explicit application dependencies. Do not let a request body or model tool choose another user's `thread_id`. For high concurrency, use a pool-backed Store/checkpointer and size total connections across all FastAPI and worker processes. Use `runtime.store.aput`/`aget`/`asearch`/`adelete` in async paths; sync calls can block the event loop.

### 9. Hot-path writes, background extraction, and reflection

Hot-path memory writes are explicit user-visible actions. They are useful for “remember this” or a direct correction, but add latency and still require validation. Background extraction can inspect selected conversations, consolidate candidate facts, and update a profile without delaying the response. It must run as a durable job with retries, idempotency, observability, and authorization; an untracked `asyncio.create_task()` is not a delivery guarantee.

| Strategy | Advantages | Risks | Default use |
|---|---|---|---|
| Explicit tool/hot path | Immediate and explainable | Latency; model may over-save | User-confirmed preference/correction |
| Background manager | Lower response latency; consolidation | Delayed visibility; queue/retry complexity | Candidate facts from selected interactions |
| Human-reviewed | Strong provenance and safety | Slow and operationally expensive | Sensitive or consequential memory |

Store candidate records with source thread/checkpoint, author, timestamp, confidence, schema version, and deletion status. Consolidation should be deterministic or revision-aware; repeated jobs must upsert by a stable evidence ID rather than create duplicates. Reflection can improve a behavior prompt, but never permit it to weaken security policy, authorization, or data retention without reviewed deployment changes. LangMem's [background quickstart](https://langchain-ai.github.io/langmem/background_quickstart/) and [reference](https://langchain-ai.github.io/langmem/reference/) provide framework helpers; job ownership and policy remain application work.

### 10. Summarization, trimming, TTL, retention, and deletion

Separate context management from data erasure:

1. **Trim/summarize prompt context** to fit model limits; this changes what the next model call sees.
2. **Edit checkpoint state** if the application intentionally changes the stored thread history.
3. **Delete Store/checkpoint data** for privacy, retention, or account closure.

`trim_messages` and summarization do not erase old checkpoints. `delete_thread`/`adelete_thread` remove thread checkpoint/write data supported by the checkpointer, not copies in logs or business systems. Store TTL is backend-dependent; the documented API expresses minutes, and a PostgreSQL Store may require an explicitly started TTL sweeper. Deployment-managed TTL settings can separately govern Store records and checkpoint history: checkpoint TTL's `strategy` supports `delete` or `keep_latest`, while Store TTL uses `default_ttl`, `refresh_on_read`, and sweep behavior. Verify exact options in the pinned package; do not assume hosted deployment settings apply to self-hosted OSS.

#### In-loop context middleware

LangChain's current agent middleware can compact context during the agent loop, before persistence retention is applied. `SummarizationMiddleware` calls a configured model when a token/message/fraction trigger is reached, then keeps the configured recent context; `ContextEditingMiddleware` applies edits such as `ClearToolUsesEdit` to prune older tool results. A custom `@before_model` hook (imported from `langchain.agents.middleware`, with `state` and `Runtime` parameters) can implement deterministic trimming or message removal when the application needs a bespoke policy; related lifecycle hooks include `before_agent`, `after_model`, `after_agent`, `wrap_model_call`, and `wrap_tool_call`. The trade-off is extra model latency and summary drift for summarization versus information loss or lower auditability when messages/tool outputs are edited. Register middleware explicitly on `create_agent` and test tool-call/AI-message pairing, streaming, retries, and the persisted state produced by the policy:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    SummarizationMiddleware,
)

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        SummarizationMiddleware(
            model=summary_model,
            trigger=("tokens", 4_000),
            keep=("messages", 20),
        ),
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=100_000, keep=3)],
        ),
    ],
    checkpointer=checkpointer,
    store=store,
)
```

These are in-loop context transformations, not erasure guarantees: they affect what later model calls receive and what state updates are persisted, but do not by themselves remove older checkpoint rows, Store items, traces, logs, replicas, or backups. Checkpoint TTL's `strategy` is the separate deployment setting that supports `delete` or `keep_latest`; Store TTL uses `default_ttl`, `refresh_on_read`, and sweep behavior, without a `keep_latest` strategy. See the official [built-in middleware guide](https://docs.langchain.com/oss/python/langchain/middleware/built-in), [`SummarizationMiddleware` API](https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware), and [`ContextEditingMiddleware` API](https://reference.langchain.com/python/langchain/agents/middleware/context_editing/ContextEditingMiddleware).

Deletion is a workflow, not one method call. Authorize the request, delete/invalidate checkpoint thread state, delete Store records and derived vector entries, invalidate caches and queued extraction jobs, account for replicas/backups, and emit a deletion audit event without retaining the deleted secret. Keep legal/retention policy outside the graph state so a model cannot silently override it. See [LangSmith TTL configuration](https://docs.langchain.com/langsmith/configure-ttl).

### 11. Conflicts, concurrency, and idempotency

Store `put` is a write at a namespace/key; the base API does not promise compare-and-swap or revision conflict resolution. A read-modify-write profile update can lose a concurrent update. Use one or more of:

- deterministic keys and idempotent upserts for retry-safe writes;
- `revision`, `updated_at`, and `source` fields with application-side optimistic checks;
- database transactions or per-user/workspace serialization where updates must not overlap;
- immutable evidence records plus a deterministic consolidation job for collections.

Do not assume a checkpointer's internal lock coordinates multiple FastAPI workers, agents, or processes. Use distributed admission/leases for same-thread graph runs and explicit ownership for shared Store namespaces. If a memory write accompanies a business transaction, either make the memory eventually consistent through an outbox or define the business transaction boundary explicitly; a Store write and a domain DB commit are not atomically coupled by sharing PostgreSQL.

### 12. Prompt injection, poisoning, privacy, and encryption

Persisted memory is an attack surface. Retrieved text may contain instructions, an extraction model may promote an untrusted claim to durable memory, and an old user preference may conflict with current policy.

| Threat | Control |
|---|---|
| Cross-tenant read/write | Derive namespace from authenticated identity; authorize every operation |
| Memory poisoning | Store provenance/trust/confidence; require confirmation for consequential facts |
| Retrieved prompt injection | Treat memories as data; delimit and label them; never elevate them to policy |
| Secret retention | Minimize fields; denylist credentials/tokens; redact before Store/indexing/tracing |
| Stale/conflicting facts | Version, timestamp, source, supersession, and explicit precedence |
| Deletion failure | Delete Store/checkpoint/vector/cache copies and test backups/replicas |
| Deserialization exposure | Restrict database writers; configure serializers/backends securely |

Checkpoint encryption via an encrypted serializer protects serialized checkpoint values when configured, but it does not automatically encrypt Store rows, vector indexes, logs, traces, queues, or backups. Configure database/storage encryption, TLS, key rotation, least privilege, and trace redaction independently. Framework primitives do not infer consent or privacy classification.

### 13. Evaluation and observability

Evaluate memory over multi-turn and multi-tenant scenarios, not just whether one answer sounds good. A useful fixture set includes: explicit “remember” requests, corrections, stale facts, duplicate candidates, deleted facts, irrelevant memories, cross-thread sharing, cross-tenant substitution, prompt-injection payloads, concurrent writes, and embedding/index failures.

Measure at least:

- write precision: fraction of persisted memories that were intended and useful;
- retrieval recall and false-recall rate at a fixed namespace/tenant boundary;
- stale/conflict resolution correctness and deletion completeness;
- token, latency, and embedding cost added by memory retrieval;
- background extraction queue age, retry rate, idempotency conflicts, and consolidation failures;
- authorization denials, namespace mismatches, and attempted cross-tenant access.

Trace operation type, tenant class (not raw tenant secrets), thread, namespace class, item IDs, schema/version, retrieval filters, hit counts, and tool outcome. Mask raw memory content and sensitive inputs/outputs by default. Observability explains why an agent recalled something; it is not a substitute for the Store or checkpoint as the recovery record.

### 14. Version and deprecation traps

Pin the complete `langchain`, `langchain-core`, `langgraph`, Store/checkpoint integrations, LangMem, model adapters, FastAPI, and Pydantic dependency set. Current migration guidance replaces `langgraph.prebuilt.create_react_agent` with `langchain.agents.create_agent`, renames `prompt` to `system_prompt`, and moves static run data toward `context_schema` and `context=`. The v1 agent factory uses `TypedDict` agent state rather than Pydantic/dataclass state; keep Pydantic v2 at HTTP and Store-value boundaries. See the [LangChain v1 guide](https://docs.langchain.com/oss/python/migrate/langchain-v1).

Test upgrades against saved thread checkpoints, Store values, serializer fixtures, old state versions, and in-flight background extraction jobs. A backend `setup()` migration does not migrate application state fields, profile schemas, embedding dimensions, or serialized import paths. Add `schema_version` and compatibility defaults; migrate embedding indexes deliberately when changing models or dimensions. Do not infer a deployable compatibility matrix from an unpinned `main` branch or a blog example.

### 15. Concise JavaScript parity

LangGraph.js has the same checkpointer-versus-Store split. A schematic in-memory setup is:

```ts
import { createAgent } from "langchain";
import { MemorySaver, InMemoryStore } from "@langchain/langgraph";

const checkpointer = new MemorySaver();
const store = new InMemoryStore();
const agent = createAgent({ model, tools, checkpointer, store });

const result = await agent.invoke(
  { messages: [{ role: "user", content: "I prefer concise replies." }] },
  { configurable: { thread_id: authorizedThreadId } },
);
```

Use durable JavaScript checkpointer/Store packages for production, and verify exact constructors, async lifecycle, Store search/index options, and tool-runtime injection against the installed reference. The shared rules remain: server-owned identity, narrow tools, explicit namespaces, durable background jobs, idempotent writes, and independent deletion/retention policy. Do not translate Python serializer, psycopg, or Pydantic assumptions literally.

### Decision summary: production defaults

| Decision | Recommended baseline |
|---|---|
| Short-term memory | Async PostgreSQL checkpointer, explicit authorized `thread_id` |
| Long-term memory | Small structured async Store schema; add vectors only for measured recall needs |
| Identity | Derive namespace from authenticated tenant/user/workspace, never model text |
| Tool access | `ToolRuntime` with narrow read/write/delete tools; keep Store hidden from model schema |
| Data shape | Pydantic v2 at HTTP boundary; versioned JSON/TypedDict state and Store values |
| Writes | Hot path for explicit intent; durable background extraction for candidates/consolidation |
| Retrieval | Key/filter first; semantic index only with deliberate fields, dimensions, and evaluation |
| Retention | Separate prompt trimming, checkpoint retention, Store TTL, vector deletion, and legal erasure |
| Concurrency | Deterministic keys + revisions/idempotency; distributed serialization for shared threads |
| Security | Provenance/trust labels, prompt-data separation, least privilege, encryption/redaction |

The veteran rule is to distinguish “the model saw it,” “the application stored it,” and “the application is authorized to act on it.” Those are three different facts. A memory system becomes trustworthy when every recalled value has an owner, scope, provenance, retention rule, conflict policy, and deletion path—not merely when retrieval returns a relevant sentence.

### Research bibliography

- [LangGraph memory overview](https://docs.langchain.com/oss/python/langgraph/memory) · [Stores](https://docs.langchain.com/oss/python/langgraph/stores) · [Store API](https://reference.langchain.com/python/langgraph/store/base/BaseStore) · [Runtime](https://reference.langchain.com/python/langgraph/runtime/Runtime)
- [LangChain short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) · [Python tools and ToolRuntime](https://docs.langchain.com/oss/python/langchain/tools) · [InjectedStore](https://reference.langchain.com/python/langgraph.prebuilt/tool_node/InjectedStore)
- [AsyncPostgresStore](https://reference.langchain.com/python/langgraph.store.postgres/aio/AsyncPostgresStore) · [AsyncPostgresSaver](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver) · [PostgreSQL checkpoint source](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/README.md)
- [TTL configuration](https://docs.langchain.com/langsmith/configure-ttl) · [LangMem background quickstart](https://langchain-ai.github.io/langmem/background_quickstart/) · [LangMem reference](https://langchain-ai.github.io/langmem/reference/) · [LangMem SDK](https://www.langchain.com/blog/langmem-sdk-launch)
- [LangChain v1 migration](https://docs.langchain.com/oss/python/migrate/langchain-v1) · [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in) · [JavaScript memory](https://docs.langchain.com/oss/javascript/langgraph/add-memory) · [JavaScript short-term memory](https://docs.langchain.com/oss/javascript/langchain/short-term-memory)
