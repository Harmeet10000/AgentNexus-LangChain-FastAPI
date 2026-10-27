# Multi Agent Patterns

## Why Multi-Agent

65. Multi-agent systems coordinate specialized components to tackle complex workflows, but not every complex task requires this approach. A single agent with the right tools and prompt can sometimes achieve the same goal.

Reasons people usually want multi-agent systems:

- context management
- distributed development
- parallelization

## Main Patterns

26. Four architectural patterns:

- Subagents (Supervisor Pattern)
- Handoffs Pattern
- Skills Pattern
- Router Architecture

27 and 65. Additional patterns table:

Pattern | How it works
--- | ---
Subagents | Main agent coordinates subagents as tools
Handoffs | Tool calls update state and routing changes behavior dynamically
Skills | Specialized prompts or knowledge are loaded on demand
Router | Input is classified and directed to one or more specialized agents
Custom workflow | Bespoke LangGraph execution mixing deterministic and agentic behavior

## Communication Patterns

27. Agents usually exchange information through:

- Shared State
- Tool-Based Communication

19. Best practice for multi-agent systems is to use `AIMessage` objects with `tool_calls` to hand off tasks.

18. Circular delegation is possible. Loop safety is limited and may depend on the surrounding orchestration system.

## Pattern Trade-Offs

Subagents:

- strong for distributed development and parallelization
- weak for direct user-to-subagent interaction

Handoffs:

- strong for multihop conversations and direct interaction
- weaker for independent distributed development

Skills:

- strong for progressive disclosure and bounded context loading

Router:

- strong for parallelization
- weak for multihop sequences

Custom workflow:

- recommended when production control flow must be explicitly designed for the domain

---

## Maintainer Addendum (researched 2026-10-25)

This reference separates documented LangChain/LangGraph primitives from higher-level architecture labels. The primary conceptual guide is [LangChain’s multi-agent overview](https://docs.langchain.com/oss/python/langchain/multi-agent); the low-level control-flow contract is the [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api). APIs and package boundaries are version-sensitive, so pin `langchain`, `langchain-core`, `langgraph`, `langgraph-prebuilt`, provider packages, and the SDK together.

### 1. First decide whether another agent is justified

Multi-agent is an information-flow and authority decision, not a maturity badge. A second model call is worthwhile when it creates a real boundary: a different tool permission set, a different context window, an independently testable specialist, or bounded parallel work. If the task is one coherent objective and one agent can use a few well-described tools, another agent usually adds latency, tokens, coordination state, and failure modes without adding capability.

| Choose | When it fits | Cost / risk |
| --- | --- | --- |
| One agent + tools | One objective, shared context, modest tool set, simple policy | Long context and tool-selection drift can grow |
| Subagents as tools | A main conversation needs specialist work but should retain one user-facing voice | Supervisor must summarize and select context; each call adds latency/tokens |
| Handoff | The active specialist should own the next conversational turns | Context and authorization must cross an explicit boundary; loops need limits |
| Router | Clear domains, stateless or short-lived classification, parallel retrieval | Router misclassification and result synthesis become new failure points |
| Skills / progressive disclosure | One agent needs bounded, on-demand instructions or knowledge | Loading the wrong skill can distort behavior; not a substitute for permission checks |
| Custom `StateGraph` | Typed state, deterministic transitions, retries, interrupts, compensation, or strict auditability | More code and explicit reducers/state contracts |
| Peer/swarm/network | Specialists negotiate or route among peers with no permanent supervisor | More difficult termination, authorization, history, and observability; this is an architecture, not one built-in LangGraph mode |

The official router guide describes the core distinction well: a router is a dedicated classification/dispatch step, while subagents are dynamically called by a conversation-aware supervisor. Use a router for clear categories and parallel sources; use a supervisor when decisions depend on evolving context. A one-agent baseline should be an acceptance test: compare quality, latency, token cost, and failure rate before committing to orchestration.

### 2. Pattern matrix and ownership boundaries

| Pattern | Coordinator | Communication | State visibility | Best for | Failure to guard |
| --- | --- | --- | --- | --- | --- |
| Subagents-as-tools | Main agent | Tool calls + returned summaries | Workers receive an explicit projection | Specialist research/calculation while preserving one voice | Full-history leakage and token blow-up |
| Supervisor / tool-calling | Main agent or graph node | Named tools, structured results | Supervisor owns conversation; workers are bounded | Multi-step delegation with policy at one gate | Tool descriptions that invite delegation loops |
| Handoff | Active agent | Handoff tool returns `Command` to parent | New agent owns a selected state projection | Customer support or stage transitions | Ambiguous ownership and circular transfer |
| Router / dispatcher | Classifier node or LLM | `Command(goto=...)` or `Send(...)` | Usually stateless per request | Distinct verticals and map/reduce | Wrong route, reducer conflicts, lost constraints |
| Skills | One agent | Prompt/context loaded on demand | Same agent state, narrower instructions | Progressive disclosure and context reduction | Treating a prompt as an authorization boundary |
| Hierarchical teams | Supervisor of supervisors | Nested tools/subgraphs | Each team has local state plus a contract | Large domains with bounded team responsibilities | Deep recursive delegation and expensive history |
| Peer/network/swarm | Multiple agents | Explicit messages or handoff edges | Shared or projected state by policy | Dynamic negotiation with no fixed leader | Non-termination, race conditions, unclear audit owner |
| Custom workflow | Graph designer | Typed state, edges, `Command`, `Send` | Explicit channels and reducers | Production workflows with deterministic controls | Overbuilding agentic behavior where a function suffices |

“Swarm,” “network,” and “peer-to-peer” are useful architectural descriptions, but they do not imply a single official `SwarmGraph` API. Build them from ordinary graph nodes, tools, `Command`, and `Send`, or adopt a package whose lifecycle/version policy you have evaluated.

### 3. Subagents as tools: isolate context without losing authority

The tool-calling pattern gives the parent control over when to delegate, what to send, and how much of the result to expose to the user. A worker should receive a focused task object, not an implicit copy of every message and credential.

```python
from pydantic import BaseModel, ConfigDict, Field
from langchain.tools import tool


class ResearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    constraints: list[str] = Field(default_factory=list)
    output_format: str = "brief_with_sources"


@tool(args_schema=ResearchTask)
async def research_specialist(
    question: str,
    constraints: list[str],
    output_format: str,
) -> str:
    """Research one bounded question and return a source-backed brief."""
    result = await research_graph.ainvoke(
        {
            "question": question,
            "constraints": constraints,
            "output_format": output_format,
        }
    )
    return result["brief"]


supervisor = create_agent(
    model=supervisor_model,
    tools=[research_specialist, billing_specialist],
    system_prompt=(
        "Delegate only bounded tasks. Preserve user constraints in every task. "
        "Synthesize returned evidence; do not claim a worker's action happened "
        "unless its structured result says it completed."
    ),
)
```

The tool is an API boundary: its name, description, argument schema, timeout, and result shape teach the supervisor what delegation means. Do not expose a generic `run_agent(task: str)` when the worker has materially different permissions. Name tools with stable verbs (`research_specialist`, `lookup_invoice`) and state what they do, what they do not do, expected latency, and whether they mutate anything. A tool schema constrains shape; the worker service must still enforce authorization and resource scope.

The official subagent pattern supports either returning the final worker answer or returning a structured result for the supervisor to synthesize. Prefer a small typed result such as `{status, findings, citations, unresolved, cost}` over dumping the worker’s full transcript into the parent. Keep an escape hatch for failure details and prior constraints; an aggressive “last message only” projection can silently drop negation, safety rules, or unresolved work.

### 4. Supervisor, hierarchy, and handoff

A supervisor is a policy-bearing coordinator. It may call workers as tools (recommended when the user should continue speaking to one agent) or route to worker subgraphs. A handoff changes who owns the next conversational turn; it should be explicit and observable.

For a subgraph-to-parent handoff, `Command.PARENT` is the documented primitive:

```python
from typing import Literal, TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.types import Command


class TeamState(TypedDict, total=False):
    messages: list
    active_agent: str
    task_description: str


@tool
def transfer_to_billing(
    task_description: str,
    runtime: ToolRuntime,
) -> Command[Literal["billing"]]:
    """Transfer the current request to the billing specialist."""
    state: TeamState = runtime.state
    tool_call_id = runtime.tool_call_id
    if tool_call_id is None:
        raise ValueError("transfer_to_billing requires a tool call ID")
    tool_message = ToolMessage(
        content="Transferred to billing specialist.",
        name="transfer_to_billing",
        tool_call_id=tool_call_id,
    )
    return Command(
        graph=Command.PARENT,
        goto="billing",
        update={
            "messages": state["messages"] + [tool_message],
            "active_agent": "billing",
            "task_description": task_description,
        },
    )
```

When a subgraph updates a state key shared with its parent, define a reducer on the parent key. A handoff payload should include the stable task/resource ID, relevant constraints, authority scope, current state revision, and a concise user-visible reason. Avoid forwarding credentials or the entire history by default. If a team should not be able to modify parent state, route through a parent dispatcher that validates the handoff and applies a narrow update.

Use one routing mechanism per node unless you intentionally want both paths: static edges still execute when a node also returns a dynamic `Command(goto=...)`. Python nodes returning `Command` should annotate destination literals so the graph renderer and type checker know possible targets; JavaScript builders use the corresponding `ends` declaration where required. The graph API documents these constraints.

The separately maintained [`langgraph-supervisor-py`](https://github.com/langchain-ai/langgraph-supervisor-py) package now recommends direct tool-based supervisors for most use cases because they provide more control over context engineering. Use the package only when its compatibility and message-history semantics match your pinned LangChain version; do not assume an external `create_supervisor` wrapper is the same as the core `create_agent` or `StateGraph` APIs.

### 5. Router, `Send`, map/reduce, and conflict semantics

The official router pattern classifies input, dispatches zero or more specialists, then synthesizes results. For one destination, return `Command(goto=...)`; for parallel fan-out, return `Send` objects carrying per-branch inputs.

```python
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class State(TypedDict):
    query: str
    findings: Annotated[list[dict[str, str]], add]
    answer: str


def route(state: State) -> list[Send]:
    destinations = classify_query(state["query"])
    return [
        Send(
            destination,
            {"query": focused_query, "source": destination},
        )
        for destination, focused_query in destinations
    ]


def synthesize(state: State) -> dict[str, str]:
    # Sort by source or stable request index before prompting the synthesizer.
    findings = sorted(state["findings"], key=lambda item: item["source"])
    return {"answer": render_answer(state["query"], findings)}


builder = StateGraph(State)
builder.add_node("route", route)
builder.add_node("github", github_worker)
builder.add_node("notion", notion_worker)
builder.add_node("slack", slack_worker)
builder.add_node("synthesize", synthesize)
builder.add_edge(START, "route")
builder.add_edge("github", "synthesize")
builder.add_edge("notion", "synthesize")
builder.add_edge("slack", "synthesize")
builder.add_edge("synthesize", END)
graph = builder.compile()
```

The reducer is part of the correctness contract: concurrent workers cannot safely write the same scalar channel without a merge policy. List append is only safe if ordering does not matter or the reducer carries a stable key. For production, prefer a keyed reducer or return `{request_id, source, result}` and sort/deduplicate in a synthesis node. Bound fan-out by policy, reject duplicate destinations, and set per-worker deadlines. A `Send` branch is not a free thread; it consumes model/provider capacity and can amplify failures.

Parallel fan-out has two different conflict classes:

1. State conflicts: two branches write the same key. Use reducers or isolate branch state and merge explicitly.
2. Business conflicts: two specialists propose mutually incompatible actions. A reducer can concatenate proposals but cannot decide authority; add a deterministic adjudication node or human review.

Use `asyncio.gather` only for bounded request-scoped work under the application’s concurrency budget. For unbounded or bursty workloads, use a bounded queue/worker pool outside the graph and feed results back through an explicit state transition.

### 6. Context engineering and history projection

Context is a contract, not an accidental transcript. Before delegating, classify each field:

| Field | Usually send? | Reason |
| --- | --- | --- |
| Stable resource/task ID | Yes | Correlation and authorization |
| User’s exact request and constraints | Yes | Prevents loss of negation and scope |
| Relevant structured state | Yes | Allows typed worker behavior |
| Full message history | Only when necessary | Continuity versus cost/leakage |
| Other worker’s hidden reasoning | No | Not a stable or safe interface |
| Credentials/session tokens | No | Inject trusted dependencies server-side |
| Prior failures and unresolved questions | Usually yes | Prevents repeated mistakes |
| Raw retrieved documents | Only relevant excerpts/IDs | Controls prompt size and disclosure |

Three useful projection modes are:

- `full_history`: best continuity, highest token and privacy cost;
- `last_message`/final-result: cheapest, but can omit constraints and evidence;
- `focused_task`: a typed envelope containing request, scope, prior decisions, and required output schema.

The supervisor library exposes full-history versus last-message options, but those are integration choices rather than universal LangGraph state modes. A custom graph should make the projection a named function and test it independently. Preserve provenance (`source_agent`, `request_id`, `revision`) in structured results so synthesis can cite and reconcile rather than blindly trust the last string.

Skills are the lightest boundary: load specialized instructions or knowledge only when needed, keeping one agent and one authority context. If a “skill” changes which resources the agent may access, enforce that in tool/service authorization too; prompt text is not least privilege.

### 7. Handoffs versus a swarm/network

Handoff is a directed, explicit ownership transfer. A swarm/network is a broader decentralized design in which peers can propose the next peer. Build a swarm from explicit graph edges and typed messages, then add controls that a central supervisor would otherwise provide:

- a visited-agent set and maximum hop/step budget;
- an allowlist of legal destinations per agent;
- a correlation ID and immutable task revision;
- a terminal owner responsible for the user-visible answer;
- cycle detection and a deterministic fallback/escalation path;
- per-agent tool/resource permissions;
- a trace span for every transfer and result.

Never let a model emit an arbitrary node name and pass it directly to `goto`. Map a validated symbolic destination (`"billing"`) through a server-owned routing table. A peer’s ability to hand off is a capability; it should be narrower than “can invoke any graph node.”

### 8. Persistence, streaming, interrupts, and human oversight

Compile stateful multi-agent graphs with a checkpointer and pass the same `thread_id` across invocations. The checkpointer stores thread-scoped execution state; a long-term store is a separate cross-thread memory mechanism. Use durable savers in production and in-memory savers only for tests/prototypes.

Subagents and handoff nodes can interrupt for approval or clarification. The interrupted parent/child node may replay from its beginning, so all code before an interrupt or subgraph call must be replay-safe. Place irreversible work after approval in an idempotent action node. For streaming, normalize the installed API’s pending representation (`__interrupt__`, graph output, or stream interrupt projection) into an application event such as `{kind: "review_required", thread_id, interrupt_id, payload}`; do not infer completion merely because token output ended.

An HTTP service should return a pending resource rather than hold a request open:

```python
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from langgraph.types import Command


class DelegateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=4000)
    request_id: str = Field(min_length=1, max_length=100)


class ReviewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    revision: int = Field(ge=0)


router = APIRouter(prefix="/v1/agent-runs")


@router.post("")
async def start_run(
    body: DelegateRequest,
    request: Request,
    principal: Annotated[Principal, Depends(current_principal)],
) -> dict[str, str]:
    thread_id = await run_repository.create_thread(
        principal.tenant_id, body.request_id
    )
    config = {"configurable": {"thread_id": thread_id}}
    result = await request.app.state.graph.ainvoke(
        {"task": body.task, "tenant_id": principal.tenant_id}, config
    )
    return await present_run(thread_id, result)


@router.post("/{thread_id}/handoff-review")
async def review_handoff(
    thread_id: str,
    body: ReviewBody,
    request: Request,
    principal: Annotated[Principal, Depends(current_principal)],
) -> dict[str, str]:
    review = await review_repository.claim_for_update(
        thread_id, principal.tenant_id, body.revision, principal.user_id
    )
    if review is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT)
    config = {"configurable": {"thread_id": thread_id}}
    result = await request.app.state.graph.ainvoke(
        Command(resume={"approved": body.decision == "approve"}), config
    )
    await review_repository.complete(review.id, result)
    return await present_run(thread_id, result)
```

This skeleton intentionally leaves `Principal`, repository locking, graph construction, and `present_run` application-owned. The endpoint must check tenant/resource ownership and graph revision before reading or resuming state, claim a decision once, and use an idempotency key. For a middleware-managed tool review, the body must match the middleware’s decision schema (`approve`, `edit`, `reject`, or `respond`), not this custom `approved` shape.

### 9. Error propagation, limits, and failure design

Keep orchestration failures distinct:

| Failure | Owner | Response |
| --- | --- | --- |
| Invalid route/destination | Router boundary | Reject or route to safe fallback; never invoke arbitrary node |
| Worker validation error | Worker contract | Typed failure returned to supervisor; no silent success |
| Transient provider/network error | Adapter/worker | Bounded retry with backoff and idempotency |
| Worker domain rejection | Worker | Structured `status="rejected"` result; supervisor decides next path |
| Unhandled bug/invariant failure | Runtime/observability | Bubble with correlation; do not ask a human to “approve” a bug |
| Human interrupt | Graph runtime | Persist and resume on the same thread |
| Recursion/delegation budget exhausted | Orchestrator | Terminal escalation/failure with trace evidence |
| Partial fan-out | Aggregator | Explicit quorum/partial-result policy; never assume all branches completed |

Use graph retries for transient node failures only when the node is replay-safe. Bound recursion, tool calls, subagent depth, wall-clock duration, token spend, and parallel fan-out. A circular handoff is not prevented by a prompt; persist a hop counter or visited set in state and enforce it in the router.

### 10. Security and least authority

Every agent boundary should answer four questions: which tenant/resource is in scope, which tools are callable, who may approve an action, and which state fields can be updated. Model-generated task text, destinations, tool arguments, summaries, and citations are untrusted input.

* Give each worker only the tools and data needed for its role; use server-injected identity/context rather than model arguments for authority.
* Validate destinations against a static registry. Do not expose `goto` or arbitrary tool names directly to a browser.
* Re-authorize after handoff and after human edits. A supervisor’s approval is not a substitute for the worker service’s resource check.
* Keep secrets out of messages and shared state; use runtime context or dependency injection for clients and credentials.
* Use tenant-scoped thread IDs, ownership checks, and single-use decision records. A valid thread ID is a cursor, not permission.
* Redact or hash sensitive payloads in traces; retain enough provenance to audit which worker proposed and executed an action.
* Treat long-term memory as a data store with retention and deletion policy, not as an unbounded transcript shared by all agents.

### 11. Evaluation and observability

Measure the system as a workflow, not only as a final answer:

| Dimension | Example metric |
| --- | --- |
| Routing | Correct specialist rate, fallback rate, invalid-destination rate |
| Delegation | Calls per request, depth, cycles, handoff latency |
| Quality | Evidence coverage, synthesis correctness, unresolved-constraint rate |
| Cost | Input/output tokens per worker, cache hit rate, cost per successful request |
| Reliability | Retry count, partial fan-out, timeout, checkpoint recovery, duplicate effect |
| Safety | Unauthorized tool attempts, policy denials, cross-tenant access tests |
| Human oversight | Approval latency, rejection/edit rate, stale/replayed decisions |

Give every run, task, worker invocation, handoff, and external effect a correlation ID. Include the projection version and worker/graph revision in traces so a change in context engineering is distinguishable from a model change. Store structured worker results and provenance; avoid using raw model transcripts as the only audit record.

### 12. Python and concise JavaScript parity

The JavaScript graph primitives mirror the Python concepts: `new Command({ goto: ... })`, `new Send(destination, payload)`, `graph: Command.PARENT`, reducers in the state schema, and `createAgent`/middleware camel-case options. Keep `thread_id` in the configurable runtime object where the current examples require it, and do not mechanically rename every field in a wire contract.

```typescript
import {
  Annotation,
  Command,
  Send,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";

const State = Annotation.Root({
  query: Annotation<string>(),
  task: Annotation<string>(),
  activeAgent: Annotation<string>(),
  taskDescription: Annotation<string>(),
});

type RouteTarget = { agent: "billing"; query: string };

const classify = (query: string): RouteTarget[] => [{ agent: "billing", query }];

function route(state: typeof State.State) {
  return classify(state.query).map(
    (target) => new Send(target.agent, { query: target.query }),
  );
}

const handoff = (state: typeof State.State) =>
  new Command({
    update: { activeAgent: "billing", taskDescription: state.task },
    goto: "billing",
    graph: Command.PARENT,
  });

const billing = (_state: typeof State.State) => ({ activeAgent: "billing" });
const synthesize = (state: typeof State.State) => ({
  taskDescription: `Handled: ${state.query}`,
});

const graph = new StateGraph(State)
  // Send values are returned by a conditional edge, not a normal node update.
  .addNode("route", (state) => state)
  .addNode("billing", billing)
  .addNode("synthesize", synthesize)
  .addEdge(START, "route")
  .addConditionalEdges("route", route)
  .addEdge("billing", "synthesize")
  .addEdge("synthesize", END)
  .compile();
```

JavaScript’s Zod schemas are useful for validating classification and handoff envelopes. The same authority, reducer, replay, and termination rules apply. Test Python and JS adapters independently because tool-call, message, stream, and middleware field names can differ by package/version.

### 13. Version and deprecation traps

* The old `langgraph.prebuilt.create_react_agent` surface is deprecated in LangGraph v1 in favor of `langchain.agents.create_agent`; do not mix old examples with current middleware options without checking the migration guide.
* `langgraph-supervisor-py` is a separate package and its own README recommends direct tool-based supervisors for most use cases. It is not a requirement for supervisor architecture.
* `Command` routing requires declared destinations/type annotations for graph rendering and may interact unexpectedly with static edges; test the compiled topology.
* Shared parent/subgraph state updates require reducers. A message reducer that blindly appends can duplicate messages on replay; use an ID-aware reducer such as `add_messages` when replacing/updating messages is possible.
* Parallel tool-call handoffs and `Command(graph=Command.PARENT, goto=[Send(...)])` are compatibility-sensitive. If preserving parent updates is critical, route through a parent dispatcher and test multiple sibling calls under the exact pinned versions.
* Graph migrations for interrupted threads are stricter than migrations for completed threads: renaming/removing a node that an interrupted thread may enter can break resumption. Version graph definitions and drain or migrate pending threads deliberately.

### Decision checklist

1. Can one agent with typed tools meet the quality and latency target? If yes, start there.
2. If adding agents, what boundary is real: context, authority, specialization, or parallel work?
3. Is the pattern a subagent tool, handoff, router, skill, or custom graph? Name it explicitly.
4. What exact state projection crosses the boundary, and which reducer merges concurrent updates?
5. How are destination, tenant, resource, revision, and reviewer authority validated?
6. What stops recursion, fan-out explosion, duplicate effects, and stale resumes?
7. Which failures are retried, returned as structured worker errors, escalated to a human, or terminal?
8. Which graph/checkpointer/SDK versions and stream/message envelopes are tested?
9. Can traces explain every delegation and synthesis without retaining sensitive transcripts?

### Source bibliography (retrieved 2026-10-25)

1. [LangChain multi-agent overview](https://docs.langchain.com/oss/python/langchain/multi-agent) — trade-offs and supported subagents, handoffs, skills, and router patterns.
2. [Subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents) — subagent-as-tool context and orchestration pattern.
3. [Handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs) — active-agent transfer and state/routing semantics.
4. [Router](https://docs.langchain.com/oss/python/langchain/multi-agent/router) — `Command` routing, `Send` parallel fan-out, stateful/stateless trade-offs, and synthesis.
5. [Skills](https://docs.langchain.com/oss/python/langchain/multi-agent/skills) — progressive disclosure/context loading.
6. [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) — `Command`, `Send`, reducers, subgraph parent navigation, and migration constraints.
7. [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) — checkpoint, thread, and store boundaries.
8. [LangGraph streaming](https://docs.langchain.com/oss/python/langgraph/streaming) — stream modes and deployment-facing progress.
9. [`langgraph-supervisor-py`](https://github.com/langchain-ai/langgraph-supervisor-py) — separate supervisor package and its current recommendation to prefer direct tool-based supervisors for most cases.
10. [LangGraph `Command` source](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py) — current primitive implementation and type contract.
11. [LangGraph Swarm API reference](https://reference.langchain.com/python/langgraph-swarm) — optional swarm-oriented package/reference; evaluate its version and lifecycle separately from core `StateGraph`.
12. [Migration from `langgraph-supervisor`](https://docs.langchain.com/oss/python/migrate/langgraph-supervisor) — current recommendation and compatibility boundary for direct tool-based supervisors.

> **Chosen-ones insight:** The hidden cost of multi-agent is not the extra model call; it is the extra *authority boundary*. Every boundary needs a context projection, a state merge rule, a termination budget, and an audit owner. If you cannot name those four things, you have not designed a multi-agent system—you have made the model call itself recursively and called the recursion architecture.
