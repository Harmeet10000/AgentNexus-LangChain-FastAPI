# Interrupts HITL Resume

## Core Resume Rules

43. Key points about resuming:

- use the same thread ID that was used when the interrupt occurred
- the value passed to `Command(resume=...)` becomes the return value of the interrupt call
- the node restarts from the beginning of the node where the interrupt was called
- only pass JSON-serializable resume values

42. Starting points for resuming workflows:

- in a `StateGraph`, the starting point is the beginning of the node where execution stopped
- for a subgraph call inside a node, both the parent node and the interrupted subgraph node may restart from their beginnings

44. You can place interrupts directly inside tool functions for approval or editing before execution.

## Do Not Catch Interrupts Broadly

45. Do not wrap interrupt calls in bare `try/except`.

Good:

```python
def node_a(state: State):
    interrupt("What's your name?")
    try:
        fetch_data()
    except Exception as e:
        print(e)
    return state
```

Also good when catching specific exceptions:

```python
def node_a(state: State):
    try:
        name = interrupt("What's your name?")
        fetch_data()
    except NetworkException as e:
        print(e)
    return state
```

Bad:

```python
def node_a(state: State):
    try:
        interrupt("What's your name?")
    except Exception as e:
        print(e)
    return state
```

## Do Not Reorder Interrupt Calls

46. Keep interrupt ordering stable across executions.

Do not:

- conditionally skip interrupts in a way that changes call order
- loop interrupts over non-deterministic lists
- pass complex unserializable values such as functions

Bad examples from source:

```python
def node_a(state: State):
    name = interrupt("What's your name?")
    if state.get("needs_age"):
        age = interrupt("What's your age?")
    city = interrupt("What's your city?")
    return {"name": name, "city": city}
```

```python
def node_a(state: State):
    results = []
    for item in state.get("dynamic_list", []):
        result = interrupt(f"Approve {item}?")
        results.append(result)
    return {"results": results}
```

## Idempotency Around Interrupts

47. Use idempotent operations before interrupts, or move side effects after the interrupt or into separate nodes.

Good:

```python
def node_a(state: State):
    db.upsert_user(user_id=state["user_id"], status="pending_approval")
    approved = interrupt("Approve this change?")
    return {"approved": approved}
```

```python
def node_a(state: State):
    approved = interrupt("Approve this change?")
    if approved:
        db.create_audit_log(user_id=state["user_id"], action="approved")
    return {"approved": approved}
```

Bad:

```python
def node_a(state: State):
    audit_id = db.create_audit_log({
        "user_id": state["user_id"],
        "action": "pending_approval",
        "timestamp": datetime.now()
    })
    approved = interrupt("Approve this change?")
    return {"approved": approved, "audit_id": audit_id}
```

---

## Maintainer Addendum (researched 2026-10-25)

This addendum is a version-sensitive reference for LangGraph 1.x-style interrupt APIs and LangChain agents. The canonical conceptual guide is [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts); the implementation source is [the `interrupt`/`Command` definition](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py). Check the installed package versions before copying an example: rolling documentation and a repository `main` branch are not a substitute for the lockfile.

### 1. What is actually paused

There are two related but different mechanisms:

| Mechanism | Where configured | Purpose | How it resumes |
| --- | --- | --- | --- |
| Dynamic `interrupt(value)` | Inside node/tool code | Ask for data or a decision at a data-dependent point | `Command(resume=value)` with the same `thread_id` |
| Static breakpoint | `interrupt_before=[...]` / `interrupt_after=[...]` at invocation | Debugging, inspection, or an operator stop at a known node boundary | `invoke(None, config)` / `invoke(null, config)` |

Dynamic interrupts save the checkpoint and surface the JSON-serializable payload under `__interrupt__` for ordinary `invoke()`, or through `stream.interrupts`/`stream.interrupted` with event streaming. They wait indefinitely until resumed. A new `thread_id` is not “the same run with a typo”; it addresses a new persistent cursor and therefore starts a new thread. The [graph API guide](https://docs.langchain.com/oss/python/langgraph/graph-api) documents the distinction and the rule that `resume` is the input command intended for continuing an interrupted run.

The runtime implements the pause with an internal graph-bubble-up signal. It is deliberately not a normal application exception. A broad handler around the call catches the signal and prevents the graph runtime from checkpointing the pause. Catch only exceptions from the code after the interrupt, or catch a specific library exception at an adapter boundary. `NodeInterrupt` is a deprecated compatibility class; migrate to `from langgraph.types import interrupt` (see [the deprecation in `errors.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/errors.py)).

### 2. Minimal, typed Python contract

The smallest production-shaped graph has all three prerequisites: a checkpointer, a stable thread identifier, and a serializable interrupt payload.

```python
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict


class Approval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    reason: str = ""


class State(TypedDict, total=False):
    change_id: str
    approval: Approval
    outcome: str


def request_approval(state: State) -> dict[str, Approval]:
    decision = interrupt(
        {
            "kind": "change_approval",
            "change_id": state["change_id"],
            "question": "Approve this change?",
        },
        response_schema=Approval,
    )
    return {"approval": decision}


def finish(state: State) -> dict[str, str]:
    return {"outcome": "approved" if state["approval"].approved else "rejected"}


graph = (
    StateGraph(State)
    .add_node("request_approval", request_approval)
    .add_node("finish", finish)
    .add_edge(START, "request_approval")
    .add_edge("request_approval", "finish")
    .add_edge("finish", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "tenant-7:change-123"}}
pending = graph.invoke({"change_id": "change-123"}, config)
# pending["__interrupt__"][0].value is the review payload.

completed = graph.invoke(
    Command(resume={"approved": True, "reason": "Reviewed in ticket SEC-42"}),
    config,
)
assert completed["approval"] == Approval(
    approved=True, reason="Reviewed in ticket SEC-42"
)
```

`response_schema` is a validation contract for the value supplied on resume. Python Pydantic models, `TypedDict`s, and dataclasses are returned in their corresponding validated shape. A raw JSON Schema dictionary is exposed to clients but is not itself a server-side validator; validate at the application boundary as well. Invalid typed input raises a validation error and can be corrected with another `Command(resume=...)` on the same thread. See the [typed response-schema section](https://docs.langchain.com/oss/python/langgraph/interrupts#interrupt-response-schemas).

`Command(update=...)`, `Command(goto=...)`, and `Command(graph=...)` have a different job: they are node-return control primitives for updating state and routing. They are not a replacement for `Command(resume=...)` as the input to continue a pending interrupt. A plain dictionary is the appropriate input for a fresh user turn; using `Command(update=...)` as that input can re-enter the latest checkpoint and appear stuck.

### 3. Multiple, concurrent, and nested interrupts

Interrupt answers are correlated by runtime identity, not by a UI row number invented by the client.

* Multiple calls in one task are matched in call order. Keep the call sequence stable across every re-execution; do not let a set, unordered database result, or changing conditional determine which interrupt is “first.”
* Fan-out/parallel branches can produce several pending interrupts. Resume them with a mapping from each opaque interrupt ID to its answer when the API exposes IDs. Do not derive an ID from the payload or persist assumptions about its format.
* In LangChain middleware’s default batched mode, the resume value is `{"decisions": [...]}` and list order must match the action order. Per-call mode emits one interrupt per gated call and uses an interrupt-ID mapping instead. These are different protocols even though both eventually use `Command(resume=...)`.
* In a parent node that calls a subgraph, an interrupt in the child can replay the parent node and the interrupted child node from their beginnings. Any parent-side network call, mutation, or generated random identifier before the subgraph call is therefore replay-sensitive. Put the child call in a small replay-safe node or make the operation idempotent.
* If a node requires a sequence of user corrections, prefer one `interrupt()` per invocation and a conditional edge back to the node. A `while True` loop containing `interrupt()` replays earlier loop iterations on every resume and can multiply side effects. Store the current prompt/validation error in state and return from the node after one question.

This is why an interrupt payload should carry a stable business key and a display-safe snapshot, while the server stores the complete pending request and expected state version separately. The payload is presentation data, not an authorization token.

### 4. Tool approval with `HumanInTheLoopMiddleware`

For a LangChain `create_agent`, use the middleware when the policy is “pause selected tool calls before execution.” The policy is explicit and per tool:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ToolCallRequest,
)
from langgraph.checkpoint.memory import InMemorySaver


def is_destructive_sql(request: ToolCallRequest) -> bool:
    query = str(request.tool_call["args"].get("query", ""))
    return not query.lstrip().upper().startswith("SELECT")


agent = create_agent(
    model="provider:model-you-have-pinned",
    tools=[read_data, execute_sql, write_file],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_data": False,
                "execute_sql": {
                    "allowed_decisions": ["approve", "reject"],
                    "when": is_destructive_sql,
                    "description": "DBA approval required for writes",
                },
                "write_file": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
            },
            interrupt_mode="batched",  # default; use per_call when needed
        )
    ],
    checkpointer=InMemorySaver(),
)
```

The documented decision types are `approve` (run unchanged), `edit` (run a modified action), `reject` (deny with feedback), and `respond` (the human supplies the result for an ask-user-style tool). Do not use `respond` to deny a side-effecting tool: it is treated as a successful tool result. For `edit`, retain the original tool name and submit arguments that pass that tool’s schema; server-side authorization still runs after the edit. The [official HITL guide](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) documents these contracts and its source examples.

| Middleware mode | Interrupt shape | Resume shape | Best fit |
| --- | --- | --- | --- |
| `batched` (default) | One request containing all gated calls for a model turn | `{"decisions": [{"type": ...}, ...]}` in action order | One reviewer screen for a turn |
| `per_call` | One interrupt per gated tool call, each with its response schema | Map each interrupt ID to one decision | Independent reviewers, queue workers, or parallel approvals |

`when` predicates are evaluated again after resume. They must be deterministic for the same tool call. If a predicate changes from true to false, the reviewer’s answer may be ignored; in batched mode a partial change can be an error. Base a predicate on immutable call arguments and policy, not on a clock, random value, or mutable external flag. This is a subtle source of “I rejected it, but it ran” incidents.

The current source also moves low-level `HumanInterrupt`, `ActionRequest`, and related types toward `langchain.agents.interrupt`; imports from older `langgraph.prebuilt.interrupt` examples may emit deprecation warnings. Treat the [prebuilt compatibility source](https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/interrupt.py) as a migration clue, not a reason to pin an obsolete import forever.

### 5. Durable execution and side-effect boundaries

Checkpointing is persistence of graph execution state; it is not an authorization system and is not the same thing as a cross-thread application store. In development, `InMemorySaver` makes tests easy. In production, select a durable saver such as an async PostgreSQL or MongoDB-backed checkpointer supported by the installed release. The deployment data-plane guide explains that checkpoints, threads, and long-lived resources have distinct storage responsibilities.

Durability mode is a latency/recovery trade-off. Current releases expose modes commonly described as:

| Mode | Checkpoint timing | Trade-off |
| --- | --- | --- |
| `sync` | Wait for each checkpoint write | Strongest boundary guarantee, higher latency |
| `async` | Write in the background | Lower latency, a process crash can lose the newest not-yet-flushed boundary |
| `exit` | Persist at completion | Lowest overhead, weakest mid-run recovery |

Pin the mode and verify its exact spelling against the installed `langgraph` version. For human approval, a checkpoint must be durable before the API returns “pending”; otherwise a worker crash can leave the reviewer with an approval request that has no resumable state. A practical pattern is `sync` (or an equivalent transactional boundary) for the pause-producing run, plus an application outbox for the review record.

Think in terms of a replay contract:

1. Before `interrupt()`: reads, deterministic computation, or idempotent upserts only. Use an idempotency key such as `(tenant_id, thread_id, change_id, node_name)`.
2. At `interrupt()`: persist a pending review row containing the opaque interrupt ID, state/checkpoint version, allowed decisions, and an expiry. Do not place secrets in the display payload.
3. After approval: execute the side effect in a separate node or service with an idempotency key; record the result before returning success.
4. On rejection/expiry: transition the review row exactly once and route to a terminal/recovery path.

“After the interrupt” does not mean “exactly once”: the node can still be retried or the client can repeat a resume request. Exactly-once external effects require the external system’s idempotency or a transaction/outbox protocol; LangGraph’s checkpoint is not a distributed transaction with your payment, email, or database provider.

### 6. Async FastAPI approval endpoint

The web layer should authenticate the reviewer and derive tenant/thread ownership from the session. A client-provided `thread_id` is a lookup key, not proof of permission.

```python
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, ConfigDict, Field
from langgraph.types import Command

router = APIRouter(prefix="/v1/reviews")


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(pattern="^(approve|reject)$")
    reason: str = Field(default="", max_length=2000)


class Principal(BaseModel):
    tenant_id: str
    user_id: str
    can_approve: bool


@router.post("/{review_id}/decision")
async def decide(
    review_id: Annotated[str, Path(min_length=1)],
    body: ReviewDecision,
    principal: Annotated[Principal, Depends(current_principal)],
) -> dict[str, str]:
    review = await review_repository.get_for_update(review_id)
    if review is None or review.tenant_id != principal.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if not principal.can_approve:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    if review.status != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT)

    # The transaction atomically claims the one-shot decision. It also checks
    # the checkpoint/revision recorded when the interrupt was created.
    await review_repository.claim(review_id, principal.user_id, review.revision)
    config = {"configurable": {"thread_id": review.thread_id}}
    result = await graph.ainvoke(
        Command(resume={"approved": body.type == "approve", "reason": body.reason}),
        config,
    )
    await review_repository.complete(review_id, result_revision=review.revision + 1)
    return {"status": "completed"}
```

In a real service, replace the illustrative `HTTPException`s with the project’s typed policy exceptions and response envelope. The important invariants are: `review_id` is an application-owned capability; the repository checks tenant and expected revision; the decision is single-use; the graph is resumed with the original thread; and a stale/missing pending checkpoint becomes a controlled `409`, not a second execution.

For long-running workers, do not hold an HTTP request open while waiting for a human. Return a pending review resource, publish the interrupt to the UI/queue, and invoke `ainvoke(Command(resume=...))` from a bounded worker after authorization. If using stream events, treat disconnects as transport events, not approval or cancellation; reconnect by reading the same thread’s pending state.

### 7. Streaming, deployment, and observability

Use event streaming when the client needs progressive output and a clear pause signal. The current guide recommends `stream_events(..., version="v3")`; inspect `stream.interrupted` and `stream.interrupts`, render the payload, then issue another stream with `Command(resume=...)` and the same config. Ordinary `invoke()` remains useful for request/response APIs and exposes `result["__interrupt__"]`.

For a deployed graph, the shape is usually:

```text
start run -> stream/invoke -> interrupt + durable checkpoint
          -> 202 review resource / websocket event
reviewer decision (authorized, single-use)
          -> resume same thread -> stream final updates -> terminal state
```

Log a correlation tuple such as `(tenant_id, thread_id, run_id, node, interrupt_id, review_id)`, but redact payload content and secrets. Record decision type, reviewer identity, policy version, state revision, and latency. LangSmith tracing is useful for graph-level diagnostics; it does not replace application audit logs or tenant authorization. The [LangGraph interrupt guide](https://docs.langchain.com/oss/python/langgraph/interrupts) points to LangSmith for debugging.

### 8. Error taxonomy and migration traps

| Situation | Correct handling |
| --- | --- |
| `interrupt()` pauses | Let the graph runtime handle its internal signal; surface payload to the caller |
| Resume fails response-schema validation | Return a safe validation error; retry with corrected `Command(resume=...)` on the same thread |
| Tool argument is invalid | Let the agent/tool validation path handle it; do not mislabel it as a reviewer decision |
| Network or provider failure | Apply a bounded retry/typed adapter policy; ensure replay does not duplicate effects |
| Unexpected bug/invariant failure | Let it bubble to monitoring; do not turn it into an approval or rejection |
| `NodeInterrupt` in an old example | Replace with `langgraph.types.interrupt`; update payload/resume handling |

The internal source distinguishes invalid resume validation from other tool errors so a bad answer can be corrected without losing the pending interrupt. Do not catch `Exception` around the entire node merely to convert every failure into “please approve again”; that hides bugs and can swallow the interrupt signal itself.

### 9. Security checklist

* Authorize the reviewer against the tenant, resource, and policy version; a valid thread ID alone is not sufficient.
* Treat interrupt payloads and edited arguments as untrusted input. Validate schema, size, content type, and allowed fields at the API boundary.
* Never let `edit` change a tool into an arbitrary tool or bypass service authorization. Enforce the same authorization path for approved and edited calls.
* Do not expose bearer tokens, raw credentials, or sensitive retrieved documents in interrupt payloads, logs, browser storage, or traces. Use a short-lived review ID and server-side lookup.
* Claim decisions atomically and reject replayed, expired, superseded, or already completed decisions.
* Bind the checkpoint to a tenant and graph/version identity. A user must not resume a thread belonging to another tenant or a different workflow deployment.
* Rate-limit decision endpoints and audit who approved, rejected, or edited what. Approval is an authorization event, not merely UI state.

### 10. Test matrix and anti-patterns

| Test | Assertion |
| --- | --- |
| First run | A checkpointer is required; pending payload and opaque ID are returned |
| Same-thread approval | `interrupt()` receives the resume value and the graph reaches the terminal state |
| Wrong thread | No state is resumed; request is denied or starts no unintended workflow |
| Invalid typed answer | Validation error; corrected answer on the same thread succeeds |
| Replay | Pre-interrupt idempotent operation is not duplicated; post-approval operation uses an idempotency key |
| Reject/edit | Rejection avoids the side effect; edit stays within the tool schema and policy |
| Parallel approvals | Every interrupt ID is mapped to the intended answer; no positional guesswork across branches |
| Conditional policy | `when` returns the same result before and after resume |
| Subgraph | Parent and child replay-safe behavior is asserted |
| Security | Cross-tenant, stale revision, expired, and duplicate decisions fail closed |

Avoid these veteran-level failure modes:

* treating a checkpoint as a transaction across an external API;
* using a mutable global list or unordered result to determine interrupt order;
* putting a non-idempotent `create_*` call before the interrupt;
* resuming with a plain dictionary or `Command(update=...)` when a pending interrupt requires `Command(resume=...)`;
* using `respond` as a rejection in `HumanInTheLoopMiddleware`;
* trusting a reviewer’s edited tool arguments without re-running authorization and validation;
* assuming a streaming disconnect cancels the graph or resolves the review;
* returning the entire checkpoint or secret-bearing payload to the browser;
* catching all exceptions around `interrupt()` and silently continuing.

### 11. Concise JavaScript parity

The JavaScript API uses the same model with camel-case option names in middleware and `new Command({ resume: ... })`:

```typescript
import { Annotation, Command, MemorySaver, StateGraph, START, END, interrupt } from "@langchain/langgraph";

const State = Annotation.Root({
  changeId: Annotation<string>(),
  approved: Annotation<boolean | null>(),
});

const graph = new StateGraph(State)
  .addNode("approve", (state) => ({
    approved: interrupt({ changeId: state.changeId, question: "Approve?" }),
  }))
  .addEdge(START, "approve")
  .addEdge("approve", END)
  .compile({ checkpointer: new MemorySaver() });

const config = { configurable: { thread_id: "tenant-7:change-123" } };
const pending = await graph.invoke({ changeId: "change-123" }, config);
const completed = await graph.invoke(new Command({ resume: true }), config);
```

Use Zod `responseSchema` for typed resume validation in JavaScript. The same replay, ordering, durable-checkpoint, authorization, and idempotency rules apply; only naming and the async iteration surface differ. The [LangGraph JS HITL guide](https://github.com/langchain-ai/langgraphjs/blob/main/docs/docs/concepts/human_in_the_loop.md) is the parity reference.

### Decision guide

| Need | Prefer | Why |
| --- | --- | --- |
| Human supplies arbitrary structured data | Dynamic `interrupt()` in a narrow node | Payload and response contract are explicit |
| Approval of selected agent tools | `HumanInTheLoopMiddleware` | Policy, decision schema, and tool-call lifecycle are standardized |
| Debug pause at a known graph boundary | Static breakpoint | No human payload protocol is needed |
| Independent review queues for parallel calls | Middleware `per_call` or explicit ID mapping | Avoids positional ambiguity |
| High-value external mutation | Approval node + idempotent action node + audit/outbox | Separates authorization from effect and makes replay safe |
| Cross-thread user preferences | Application store | A checkpointer is a thread cursor, not a long-term domain store |

### Source bibliography (retrieved 2026-10-25)

1. [LangGraph interrupts conceptual guide](https://docs.langchain.com/oss/python/langgraph/interrupts) — dynamic/static behavior, prerequisites, resume rules, schemas, multiple interrupts, streaming, and debugging.
2. [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) — `Command` input/output semantics and resume distinction.
3. [LangChain human-in-the-loop middleware](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) — `HumanInTheLoopMiddleware`, `interrupt_on`, decisions, batched/per-call modes, conditional predicates, and tool edits.
4. [LangGraph `types.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py) — current `Command` and `interrupt` signatures and implementation documentation.
5. [LangGraph `errors.py`](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/errors.py) — graph bubble-up and invalid-resume distinctions; deprecated `NodeInterrupt`.
6. [LangGraph prebuilt interrupt compatibility source](https://github.com/langchain-ai/langgraph/blob/main/libs/prebuilt/langgraph/prebuilt/interrupt.py) — legacy HITL types and migration notices.
7. [LangGraph JS human-in-the-loop source guide](https://github.com/langchain-ai/langgraphjs/blob/main/docs/docs/concepts/human_in_the_loop.md) — JavaScript `interrupt`/`Command` parity.
8. [LangSmith data plane](https://docs.langchain.com/langsmith/data-plane) — deployment persistence responsibilities and checkpoint backends.
9. [Managed deep-agent resolve-interrupt API](https://docs.langchain.com/langsmith/managed-deep-agents-api/runs/resolve-interrupt) — deployed-thread resolution endpoint shape and authorization surface.

> **Chosen-ones insight:** An interrupt is a deterministic replay boundary disguised as a question. The UI sees “waiting for a human,” but the runtime sees “re-enter this node with a cached answer.” The production design that survives retries is therefore not the prettiest approval dialog; it is the one whose pre-interrupt code can be replayed, whose post-approval effect has an idempotency key, and whose reviewer decision is an authenticated, single-use state transition.

### Version-sensitive identity and deployment notes

The `Interrupt` object’s identity fields have changed across LangGraph releases. Older examples may mention `interrupt_id`, `ns`, `when`, or `resumable`; newer references use `id` and expose a different task/stream envelope. Treat IDs as opaque, normalize them at your adapter boundary, and pin the framework version used by the worker and API. Never build a durable approval key by parsing a namespace string. For managed Agent Server deployments, use the authenticated SDK/API for the deployed thread rather than serializing an in-process Python `Command` object directly over the wire; the [managed resolve-interrupt API](https://docs.langchain.com/langsmith/managed-deep-agents-api/runs/resolve-interrupt) is an example of a server-side operation that resolves a thread by authenticated path parameter.

The LangGraph repository has also published security advisories affecting resource/action authorization in some SDK surfaces. Review the [GHSA-fvww-7h3r-vfhp advisory](https://github.com/langchain-ai/langgraph/security/advisories/GHSA-fvww-7h3r-vfhp) against the exact SDK version in use, and test denied reads and resumes against the deployed service. A decorator or UI role label is not evidence that the underlying resource authorization is enforced.
