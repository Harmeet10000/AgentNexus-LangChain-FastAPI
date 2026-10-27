# Subgraphs And Streaming

## Subgraph Patterns

32. Two main subgraph patterns:

Call a subgraph inside a node:

- use when parent and subgraph have different schemas
- write a wrapper function that invokes the subgraph and maps state in and out

Add a compiled subgraph as a node:

- use when parent and subgraph share state keys
- add the compiled subgraph directly with `add_node`

34. Subgraph persistence modes:

- per-thread persistence when a subagent needs multi-turn memory
- stateless mode when the subgraph should behave like a plain function call

Use `checkpointer=True` for per-thread behavior.

Use `checkpointer=False` for stateless behavior.

## Subgraphs And Interrupts

48. When invoking a subgraph inside a node, the parent graph resumes from the beginning of the parent node where the subgraph was invoked, and the subgraph resumes from the beginning of the interrupted subgraph node.

```python
def node_in_parent_graph(state: State):
    some_code()
    subgraph_result = subgraph.invoke(some_input)

def node_in_subgraph(state: State):
    some_other_code()
    result = interrupt("What's your name?")
```

## Stream Version 2

35. With `version="v2"`, subgraph events use the same `StreamPart` format.

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,
    stream_mode=["values", "updates", "messages", "custom", "checkpoints", "tasks", "debug"],
    version="v2",
):
    ...
```

Every chunk is shaped like:

```python
{
    "type": "values" | "updates" | "messages" | "custom" | "checkpoints" | "tasks" | "debug",
    "ns": (),
    "data": ...,
}
```

`invoke(..., version="v2")` returns a `GraphOutput` object with `.value` and `.interrupts`.

---

## Production Research Addendum (2026-10-25)

This addendum was researched against official LangGraph/LangChain, FastAPI,
Starlette, and `sse-starlette` material available through 2026-10-24. The
requested cutoff is one day later than the available corpus, so changes
published on 2026-10-25 are not independently verified. Pin the installed
`langgraph`, `langchain-core`, checkpointer, model-provider, and transport
versions and run the compatibility tests below before shipping. The raw
research and metadata are retained in:

- `.research/parallel/langgraph-subgraphs-streaming-20261025-trun_44035068bd2f4cb3a4d1992a5a4e0ac6.md`
- `.research/parallel/langgraph-subgraphs-streaming-20261025-trun_44035068bd2f4cb3a4d1992a5a4e0ac6.json`

### 1. The architectural question: what is a subgraph boundary?

A subgraph is a compiled graph mounted inside another graph. The boundary is
not merely a code organization choice: it determines state ownership, the
checkpoint namespace, inspectability, replay behavior, and what a client can
observe while streaming.

| Composition | Use when | State contract | What the parent can inspect |
|---|---|---|---|
| `parent.add_node("worker", child)` | Parent and child intentionally share keys | Shared/compatible state keys and reducers | Child is statically discoverable; nested state can be inspected with `subgraphs=True` where persistence is enabled |
| Wrapper node calls `child.invoke`/`child.ainvoke` | Schemas differ, or only selected fields cross the boundary | Explicit input/output mapping | Direct calls in a graph node are generally discoverable; the wrapper is the right validation and redaction seam |
| Function/task helper calls child | Existing functional orchestration or a deliberate application-level abstraction | Mapping is owned by the helper | Do not assume parent-level static subgraph inspection; verify on the pinned release |
| Tool calls child | Tool routing is the product abstraction (for example, a specialist tool) | Tool input/output schema | Indirect tool calls are not statically visible to parent `get_state(..., subgraphs=True)` |
| Separate deployment/API call | Isolation, independent scaling, or an organizational boundary | Serialized API contract | Parent only sees the API result/events, not child graph state |

The safe default for different schemas is an adapter node:

```python
from langchain_core.runnables import RunnableConfig

async def research_node(state: ParentState, config: RunnableConfig) -> dict[str, str]:
    child_input = {"query": state["question"]}
    child_output = await research_graph.ainvoke(child_input, config=config)
    return {"answer": child_output["summary"]}

parent.add_node("research", research_node)
```

This makes ownership explicit. Validate the parent input before invoking the
child, validate the child's returned shape before writing parent state, and do
not copy the entire child state into the parent simply because it is
available. A shared-schema child is convenient, but it also expands the
surface on which reducers, retries, and public streams can leak implementation
details.

Do not rely on call-order-derived namespaces for durable identity. If a parent
node invokes the same child more than once, reordering those calls can change
the runtime path. Use separate named parent nodes or separate child instances
when stable identity matters. A graph mounted directly as a node has a clearer
topology than an arbitrary function or tool indirection.

### 2. Checkpoint lifetime and propagation

There are three child compile modes, and their names are easy to confuse:

| Child compile option | Lifetime | Suitable for | Important constraint |
|---|---|---|---|
| `checkpointer=None` (default) | Per invocation; the parent checkpointer is inherited for execution durability and interrupts | Independent specialist calls, fan-out workers, ordinary function-like subgraphs | No child memory between calls |
| `checkpointer=True` | Per thread; state is retained for the same `thread_id` | A multi-turn specialist or sub-agent with its own memory | The same stateful child/thread cannot safely be called concurrently |
| `checkpointer=False` | No child checkpoint | Stateless helper work where pause/recovery is not required | No child checkpoint-backed interrupt or crash recovery |

The parent graph must have a checkpointer for durable execution. A child with
`True` does not magically create a durable store if the parent run has no
thread identity or checkpointer. In production, use an async saver compatible
with the deployment database and initialize/close it in FastAPI lifespan. An
in-memory saver is a test tool, not restart durability.

The high-value distinction is between a checkpoint and a Store. A checkpoint
is the execution state of a particular thread/run; it is for pause/resume and
recovery. A Store is the right abstraction for durable cross-thread or
cross-user application memory. Do not turn a thread checkpoint into an
unbounded global profile database.

Per-thread subgraphs share a checkpoint namespace. Two concurrent calls to the
same stateful child can collide even if the calls originate in different
parent branches. Serialize those calls, use per-invocation children for
parallel work, or allocate genuinely isolated child instances/identities.
This is a correctness issue, not just a performance issue.

Durability mode is another independent knob. The documented choices are:

- `exit`: persist when the run exits;
- `async`: persist asynchronously while the next step proceeds; and
- `sync`: persist before advancing to the next step.

Choose `sync` where losing the most recent step is unacceptable, `async` when
the latency/recovery trade-off is acceptable, and `exit` only when intermediate
recovery is not a requirement. Measure the actual saver and database latency.

Two namespace shapes must not be conflated:

- stream `ns` is a tuple identifying the runtime execution path (`()` at the
  root in the v2 envelope); and
- checkpoint `checkpoint_ns` identifies persisted graph state, with nested
  components represented by a runtime/node-derived identifier.

Neither namespace is a user identity or an authorization boundary. Authorize a
`thread_id` against the authenticated principal before reading state or
starting a stream.

### 3. Interrupts, resume, recursion, and parent routing

`interrupt(payload)` pauses a node and returns a JSON-serializable request to
the caller. A checkpointer and `configurable.thread_id` are required to resume:

```python
from langgraph.types import Command

paused = await graph.ainvoke(
    inputs,
    config={"configurable": {"thread_id": thread_id}},
)
# Later, after validating the human's response:
resumed = await graph.ainvoke(
    Command(resume=validated_answer),
    config={"configurable": {"thread_id": thread_id}},
)
```

Resume is replay, not a continuation at the exact Python source line. The node
containing the interrupt runs again from its beginning. If a parent node called
an interrupted child, the parent node also restarts from its beginning. Put
irreversible effects after approval, split them into a later node, or make
them idempotent with an operation key. Otherwise a payment, ticket, email, or
tool call performed before `interrupt()` can be duplicated.

For multiple pending interrupts, resume by interrupt ID/value mapping rather
than assuming list order. Validate resume payloads as strictly as ordinary
inputs. Static breakpoints can pause at a graph boundary, but `interrupt()` is
the more explicit human-in-the-loop primitive.

`Command.PARENT` lets a child route to a node in its nearest parent graph. If
the child writes a key shared with the parent, the parent schema needs the
appropriate reducer. `Command` does not erase static edges: declaring both a
static outgoing edge and a dynamic `Command(goto=...)` can execute both. Pick
one routing mechanism per node unless the double transition is intentional.
`Send` is the dynamic fan-out/map-reduce primitive; each worker should receive
its own input and aggregate through a reducer.

Recursion limits cap super-steps in one execution. The current documentation
states a default of 1000 from LangGraph 1.0.6; set a deliberate
`recursion_limit` in runtime config for loops and nested agent topologies.
This is a safety ceiling, not a stopping condition. Test the combined parent
and child graph because exact nested accounting is version-sensitive.

### 4. Multi-agent topology choices

| Topology | Strength | Failure mode to design for |
|---|---|---|
| Supervisor + specialist subgraphs | Central routing, durable specialist boundaries, clear observability | Shared schemas and per-thread child calls can collide |
| Handoff via `Command.PARENT` | Specialist can return control with a state update | Missing reducer or mixed static/dynamic routing can produce unintended paths |
| `Send` map/reduce workers | Dynamic parallelism and worker-specific state | Concurrent writes need reducers; stateful child namespaces must not collide |
| Tool-based specialists | Natural LLM tool selection and compact public contract | Child is opaque to parent inspection and tool inputs are untrusted |
| Separate service/deployment | Independent scaling, tenant isolation, release cadence | Extra serialization, network failure, and distributed cancellation |

Use a graph boundary when you need durable execution, nested inspection,
interrupts, or explicit agent routing. Keep ordinary deterministic transforms
as ordinary functions. The veteran-level rule is to model state ownership
before modeling agent roles: most “multi-agent” bugs are reducer,
checkpoint-identity, or replay bugs wearing an LLM costume.

### 5. Streaming mode matrix

`stream()`/`astream()` can emit one mode or a list of modes. The seven modes
are different operational views, not interchangeable payload formats:

| Mode | Emission | Good client/use | Main risk |
|---|---|---|---|
| `updates` | Node-level state deltas after steps | Progress and incremental UI state | Deltas may contain sensitive values |
| `values` | Full graph state after each step | Internal replay/debug dashboards | Large payloads and maximum disclosure |
| `messages` | LLM message/token chunks plus metadata | Chat token UX and model provenance | Tool arguments, hidden content, or metadata may leak |
| `custom` | Application events sent by `get_stream_writer()` | Progress, domain status, business events | You own and must version the event schema |
| `checkpoints` | Checkpoint events/envelopes | Protected persistence diagnostics | Internal IDs and state are not a public contract |
| `tasks` | Task/Pregel lifecycle events | Node-level tracing and diagnosis | High volume; keep off ordinary client paths |
| `debug` | Broad execution diagnostics | Local or protected observability | Can expose internal state and implementation details |

For a single mode, legacy v1 returns the raw data. For multiple modes, v1
returns `(mode, data)` tuples. With `subgraphs=True`, v1 adds namespace
information and returns `(namespace, data)` tuples. The v2 contract unifies
the shape even with multiple modes and nested graphs:

```python
async for part in graph.astream(
    inputs,
    config=config,
    stream_mode=["messages", "custom", "updates"],
    subgraphs=True,
    version="v2",
):
    if part["type"] == "messages":
        message, metadata = part["data"]
        # part["ns"] identifies root or nested execution path.
    elif part["type"] == "custom":
        event = part["data"]
```

The v2 `StreamPart` has `type`, `ns`, and `data`. Filter on `type` and treat
`data` as mode-specific. Do not serialize every mode directly as a public API
because checkpoint/task/debug shapes are diagnostic and version-sensitive.

Token streaming uses `messages`; inspect metadata/tags to select the intended
model/node. Application progress uses `get_stream_writer()` inside a node and
the `custom` mode. If a model integration cannot stream, configure the model's
documented `disable_streaming=True` behavior instead of pretending that a
non-streaming call is token streaming. `subgraphs=True` includes child events,
but only events actually emitted by the selected modes appear.

The event-streaming API documented for LangGraph 1.2 is a separate, newer
generation: typed projections such as `stream.messages`, `stream.values`,
`stream.subgraphs`, `stream.output`, `stream.interrupts`, and
`stream.extensions` can be consumed independently. The low-level v2
`astream()` contract and the newer v3 event projections are not the same API.
Feature-gate v3 and normalize both generations at the application boundary.

### 6. Sync/async, backpressure, cancellation, and transport

Use `graph.stream()` for synchronous callers and `graph.astream()` for async
FastAPI paths. Do not run blocking model/database clients inside an async node.
For multiple modes, select only the modes needed by the consumer. A public
chat stream normally needs `messages` plus a small allowlisted `custom`
progress event; `values`, `tasks`, `checkpoints`, and `debug` belong on a
protected diagnostic path.

SSE is one-way and reconnect-friendly. WebSockets are appropriate when the
client must send cancellation, steering, or resume commands over the same
connection. Neither transport is a workflow durability mechanism. Decide
whether disconnect cancels request-scoped work or merely detaches from a
durable run that can later be reattached by authorized clients.

Bounded flow control matters: await socket sends, use a bounded queue/channel
when a producer can outrun a consumer, cap event size, and enforce idle/run
timeouts. On disconnect, cancel request-scoped tasks and close the stream in a
`finally` block. Cancellation of the HTTP task does not guarantee that every
already-running remote tool or external side effect has been canceled; use
idempotency and an explicit run-cancellation protocol for durable work.

### 7. FastAPI SSE adapter (allowlisted v2 events)

The following is a production-shaped adapter. Authentication, ACL lookup, and
input validation are deliberately explicit application seams:

```python
import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

def public_event(part: dict[str, object]) -> dict[str, object] | None:
    kind = part["type"]
    namespace = list(part["ns"])  # never use this as an ACL
    data = part["data"]
    if kind == "messages":
        message, metadata = data
        text = getattr(message, "content", None)
        if isinstance(text, str) and text:
            return {
                "type": "token",
                "namespace": namespace,
                "text": text,
                "node": metadata.get("langgraph_node"),
            }
    if kind == "custom" and isinstance(data, dict) and data.get("kind") == "progress":
        return {"type": "progress", "namespace": namespace,
                "label": str(data.get("label", ""))}
    if kind == "updates" and isinstance(data, dict):
        return {"type": "step", "namespace": namespace,
                "nodes": list(data.keys())}
    return None

@app.get("/threads/{thread_id}/stream")
async def stream_thread(thread_id: str, request: Request) -> EventSourceResponse:
    principal = await authenticated_principal(request)
    await authorize_thread(principal, thread_id)
    inputs = await validated_graph_input(request, principal)
    config = {"configurable": {"thread_id": thread_id}}

    async def events() -> AsyncIterator[dict[str, str]]:
        stream = graph.astream(
            inputs,
            config=config,
            stream_mode=["messages", "custom", "updates"],
            subgraphs=True,
            version="v2",
        )
        try:
            async for part in stream:
                if await request.is_disconnected():
                    return
                payload = public_event(part)
                if payload is not None:
                    yield {"event": str(payload["type"]),
                           "data": json.dumps(payload, default=str)}
        except asyncio.CancelledError:
            raise
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

    return EventSourceResponse(
        events(), ping=15, send_timeout=30,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

`Request.is_disconnected()` detects dropped clients. `sse-starlette` documents
heartbeat pings, send timeouts, generator cancellation, and bounded AnyIO
channels; reverse proxies may buffer SSE, so set an appropriate buffering
policy and keep proxy idle timeouts above the heartbeat interval. The numeric
values above are examples, not LangGraph requirements.

### 8. FastAPI WebSocket pump with explicit cancellation

```python
import asyncio
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/threads/{thread_id}")
async def graph_socket(websocket: WebSocket, thread_id: str) -> None:
    principal = await authenticate_websocket(websocket)
    await authorize_thread(principal, thread_id)
    await validate_allowed_origin(websocket)
    await websocket.accept()

    run = asyncio.create_task(
        pump_graph(websocket, thread_id, public_event)
    )
    try:
        while not run.done():
            command = await websocket.receive_json()
            if command.get("type") == "cancel":
                run.cancel()
                break
            # Whitelist validated resume/steer commands; never eval client data.
    except WebSocketDisconnect:
        if not run.done():
            run.cancel()
    finally:
        if not run.done():
            run.cancel()
            try:
                await run
            except asyncio.CancelledError:
                pass
```

For an interrupt, persist the pause and send a small interrupt event. Accept a
separately authenticated resume command containing a validated
`Command(resume=...)`; do not keep an expensive socket open waiting for human
approval.

### 9. API and deployment patterns

There are three useful API shapes:

1. **Request-scoped stream:** invoke and stream in one SSE/WebSocket request;
   simplest, but a disconnect usually ends the work.
2. **Durable run + attach:** create a run with an authorized thread ID, return
   a run identifier, and let clients attach/reconnect to a stream; best for
   long tools and human approval.
3. **Service-to-service subgraph:** call a separately deployed child over an
   authenticated API; best for independent scaling or release ownership, but
   requires an explicit serialized state/event contract and distributed
   tracing.

For all three, derive or authorize thread identity from the principal, bound
input/output sizes, allowlist tools, propagate request/run correlation IDs,
and keep retries compatible with child replay. A transport URL is not proof
that the caller owns the thread.

### 10. Performance and reliability guidance

| Optimize first | Why |
|---|---|
| Reduce stream modes and payload fields | `values` and diagnostics can dominate bandwidth and serialization cost |
| Use a compact application event schema | Avoid coupling clients to internal checkpoint/task formats |
| Choose per-invocation children for parallel calls | Avoid stateful-child namespace collisions and unnecessary checkpoint writes |
| Choose `async` durability where acceptable | Overlaps persistence with next-step work; measure recovery loss window |
| Batch/coalesce UI progress | Prevent token/progress floods from starving the consumer |
| Bound queues, run duration, and output size | Converts unbounded producer pressure into explicit backpressure/failure |
| Keep model/tool clients async in async graphs | Prevents event-loop starvation and misleading stream latency |

Do not “optimize” by disabling checkpointing on a workflow that must resume,
or by streaming full state to compensate for missing observability. Capture
protected task/debug events for operators and publish a deliberately smaller
projection to users.

### 11. Testing, observability, and security checklist

Test the following matrix against the exact locked versions:

- shared-schema child updates and different-schema adapters;
- per-invocation isolation, same-thread continuation, and concurrent child
  calls;
- `subgraphs=True` stream and `get_state(..., subgraphs=True)` namespace shape;
- interrupt, replay, multiple-interrupt resume, and duplicate-side-effect
  prevention;
- `Command.PARENT`, reducers, `Send`, and recursion-limit failure;
- every public stream mode in success and exception paths;
- v1/v2 normalization (and v3 feature-gating if installed);
- slow consumers, disconnects, cancellation, reconnect, and run ownership;
- tenant/thread ACL isolation and redaction of state, tool arguments, and
  diagnostic payloads.

Trace request ID, run ID, thread ID, graph node, child namespace, checkpoint
version, and event type. Do not return raw internal IDs or state to an
unauthorized client. Treat graph input, resume values, tool arguments, custom
events, and streamed text as untrusted. Validate schemas, constrain tools,
apply timeouts, redact secrets, and define checkpoint retention. The reviewed
documentation describes serialized checkpoint state but does not prescribe a
complete serializer-hardening policy; assess the selected saver/serializer,
encryption, retention, and migration behavior yourself.

### 12. Python/JavaScript parity and version traps

Python and LangGraph.js share the concepts—compiled nested graphs, explicit
state boundaries, persistence, and streamed events—but method signatures,
option spelling, event types, and release timing differ. Use the language-
specific docs and a language-specific adapter; do not paste Python
`async for`/`Command` examples into TypeScript and assume parity.

Guard these version-sensitive edges:

- v1 remains a legacy/default low-level shape in the cited guide; v2 requires
  `version="v2"` and is documented for newer LangGraph releases.
- Typed event projections (`stream_events`/`astream_events`, `version="v3"`)
  are a distinct newer API documented for LangGraph 1.2; feature-gate them.
- `GraphOutput.value` and `.interrupts` are the forward-looking accessors in
  the v2 streaming documentation; avoid relying on deprecated dict-style
  compatibility aliases.
- Recursion default and stream payload details are release-sensitive; set an
  explicit recursion limit and test golden event fixtures.
- LangChain agent streaming has its own interface/version lifecycle; do not
  assume it is identical to low-level LangGraph streaming.

### 13. Do/don't summary

| Do | Don't |
|---|---|
| Use an adapter for different schemas | Pass an entire parent state into every child |
| Pin versions and normalize stream events | Make client code depend on raw debug/checkpoint shapes |
| Make pre-interrupt effects idempotent | Assume resume continues at the interrupted source line |
| Serialize same-child/per-thread calls | Fan out concurrent calls into one stateful child namespace |
| Use allowlisted public events | Expose `values`, `debug`, tool args, or secrets by default |
| Distinguish SSE/WebSocket lifetime from run durability | Treat a socket disconnect as a durable-work cancellation guarantee |
| Use a Store for cross-thread memory | Stretch thread checkpoints into a global profile store |
| Set reducers for shared/fan-in writes | Rely on last-writer-wins for concurrent graph updates |

The veteran distinction is this: a graph stream is an observation of a
runtime, not the runtime's source of truth. Checkpoints define recoverability;
reducers define state convergence; namespaces locate nested execution; and
the transport merely delivers a selected, redacted projection. Designing those
four contracts separately prevents the most expensive production failures.

### Direct sources and bibliography

- [LangGraph subgraphs (Python)](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [LangGraph streaming (Python)](https://docs.langchain.com/oss/python/langgraph/streaming)
- [LangGraph event streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming)
- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph durable execution](https://docs.langchain.com/oss/python/langgraph/durable-execution)
- [LangGraph functional API](https://docs.langchain.com/oss/python/langgraph/functional-api)
- [LangGraph.js subgraphs](https://docs.langchain.com/oss/javascript/langgraph/use-subgraphs)
- [LangGraph.js streaming](https://docs.langchain.com/oss/javascript/langgraph/streaming)
- [LangGraph v1 migration](https://docs.langchain.com/oss/python/migrate/langgraph-v1)
- [LangGraph releases](https://github.com/langchain-ai/langgraph/releases)
- [LangChain streaming](https://docs.langchain.com/oss/python/langchain/streaming)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [Starlette request disconnect detection](https://www.starlette.io/requests/)
- [sse-starlette](https://github.com/sysid/sse-starlette)

#### Claim-to-source map

For fast review, these are the primary pages behind the version-sensitive
claims in this addendum: [subgraph composition and state inspection](https://docs.langchain.com/oss/python/langgraph/use-subgraphs),
[stream modes, `subgraphs=True`, and v1/v2 envelopes](https://docs.langchain.com/oss/python/langgraph/streaming),
[typed event projections](https://docs.langchain.com/oss/python/langgraph/event-streaming),
[interrupt replay and resume](https://docs.langchain.com/oss/python/langgraph/interrupts),
[parent routing, reducers, `Send`, and recursion](https://docs.langchain.com/oss/python/langgraph/graph-api),
[checkpoint scope and durability](https://docs.langchain.com/oss/python/langgraph/checkpointers),
[durable execution](https://docs.langchain.com/oss/python/langgraph/durable-execution),
[JavaScript parity](https://docs.langchain.com/oss/javascript/langgraph/streaming),
[WebSocket lifecycle](https://fastapi.tiangolo.com/advanced/websockets/), and
[SSE heartbeat/backpressure behavior](https://github.com/sysid/sse-starlette).
