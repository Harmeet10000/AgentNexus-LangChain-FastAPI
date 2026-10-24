# Middleware And Guardrails

## Supported Middleware Types

60. The following middleware work with any LLM provider:

- Summarization
- Human-in-the-loop
- Model call limit
- Tool call limit
- Model fallback
- PII detection
- To-do list
- LLM tool selector
- Tool retry
- Model retry
- LLM tool emulator
- Context editing
- Shell tool
- File search
- Filesystem
- Subagent

## Composing Guardrails

61. Combine multiple guardrails by stacking them in the middleware array. They execute in order.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, send_email_tool],
    middleware=[
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),
        SafetyGuardrailMiddleware(),
    ],
)
```

## Runtime Access In Middleware

62. LangChain `create_agent` runs on LangGraph's runtime under the hood.

The runtime exposes:

- Context
- Store
- Stream writer

Use the Runtime parameter for node-style hooks. For wrap-style hooks, access runtime through `ModelRequest`.

## Custom Middleware Hooks

63. Middleware provides two styles of hooks:

- Node-style hooks: `before_agent`, `before_model`, `after_model`, `after_agent`
- Wrap-style hooks: `wrap_model_call`, `wrap_tool_call`

Node-style example:

```python
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 50:
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
    return None

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None
```

Wrap-style example:

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")
```

## State Updates From Middleware

Node-style hooks return a dict directly.

```python
from langchain.agents.middleware import after_model, AgentState
from langgraph.runtime import Runtime
from typing import Any
from typing_extensions import NotRequired

class TrackingState(AgentState):
    model_call_count: NotRequired[int]

@after_model(state_schema=TrackingState)
def increment_after_model(state: TrackingState, runtime: Runtime) -> dict[str, Any] | None:
    return {"model_call_count": state.get("model_call_count", 0) + 1}
```

Wrap-style model hook returning `ExtendedModelResponse`:

```python
from typing import Callable
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    AgentState,
    ExtendedModelResponse,
)
from langgraph.types import Command
from typing_extensions import NotRequired

class UsageTrackingState(AgentState):
    last_model_call_tokens: NotRequired[int]

@wrap_model_call(state_schema=UsageTrackingState)
def track_usage(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ExtendedModelResponse:
    response = handler(request)
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={"last_model_call_tokens": 150}),
    )
```

68. Quick reminder:

- `wrap_model_call`
- `wrap_tool_call`
- `before_model`
- `after_model`
- `before_agent`
- `after_agent`



# Deep research current as of 2026-10-23: LangChain Python and LangGraph middleware for production agents. Use primarily official LangChain/LangGraph documentation, API reference, source code, release notes, and official examples; use high-quality secondary sources only to clarify gaps. First build a complete taxonomy and inventory of all currently supported prebuilt agent middleware in LangChain Python and any LangGraph middleware-like primitives, clearly distinguishing true AgentMiddleware from graph nodes, runtime hooks, callbacks, interceptors, checkpointing, stores, and prebuilt agent components. For every prebuilt middleware give exact current package/import path, supported version, purpose, lifecycle hooks used, configuration knobs, state/context schema changes, sync/async behavior, ordering/composition rules, persistence or checkpoint implications, streaming implications, failure semantics, security/privacy consequences, overhead, incompatibilities, and a concise use-or-avoid recommendation. Cover custom middleware authoring in depth: AgentMiddleware base class, decorator APIs, before_agent, before_model, wrap_model_call and async form, after_model, wrap_tool_call and async form, after_agent, short-circuiting, Command or state update behavior, tool/model request mutation, structured state, context, runtime, thread safety, retry/idempotency, error propagation, observability, testing, and middleware ordering. Verify exact APIs rather than relying on memory.\n\nThen recommend a concrete minimal production middleware stack for AgentNexus, a FastAPI application using LangGraph for a streaming Agent Saul, durable PostgreSQL checkpoints and a separately projected transcript, Redis ownership-token thread locks, WebSocket streaming and revocation checks, dependency-injected model/tool providers, typed boundaries, and strict crash/replay/idempotency requirements. Separate: adopt now, adopt conditionally, implement custom, and avoid. Explain exact order and why order changes behavior. Avoid duplicating responsibilities already owned by transport security, repository transactions, transcript projection, or ThreadLock. Address summarization and context editing versus durable history, model/tool retry versus streaming and non-idempotent tools, model fallback, model/tool call limits, human-in-the-loop, PII/privacy, tool selection, todo/planning, prompt caching, context engineering, observability, cost/latency budgets, tenant policy, and production rollout.\n\nDeliver a detailed cited report with: executive recommendation; comprehensive middleware matrix; lifecycle and ordering model; recommended Agent Saul stack with sequence diagram or ordered flow; custom middleware designs with small accurate Python code skeletons; decision table with pros/cons; migration plan in phases; tests and failure-injection plan; metrics/SLOs; version-compatibility notes; and primary-source links. Explicitly identify uncertain or version-sensitive claims. Do not assume JavaScript APIs apply to Python.

## Content

# Production Middleware for LangChain and LangGraph Agents

## Executive recommendation

- **Keep Control Planes Separate**: `AgentMiddleware` runs around a LangChain agent's model/tool lifecycle, while graph nodes, checkpointers, stores, and transport controls have distinct responsibilities -> use middleware for agent-loop policy, not as a substitute for AgentNexus authentication, Redis ownership locks, repository transactions, transcript projection, or durable graph state. [4] [19] [34]
- **Pin the Exact Runtime**: the repository now locks `langchain==1.4.7`, `langchain-core==1.7.0`, and `langgraph==1.2.16`; keep the middleware reference and import tests aligned with those exact versions rather than treating documentation as a universal compatibility matrix. [9] [10]
- **Bound Work Explicitly**: use `ModelCallLimitMiddleware` and `ToolCallLimitMiddleware` to bound runaway agent loops; they support run-level and thread-level limits, but thread-wide enforcement depends on durable state/checkpoint configuration -> choose limits from measured product budgets and test restart behavior. [17] [17] [20] [7]
- **Do Not Retry Side Effects Blindly**: tool retry middleware can repeat calls, and durable graph execution may replay work after interruption or failure -> put idempotency keys and authorization inside tool/repository boundaries; only retry operations that are safe to repeat or whose effects are deduplicated. [23] [12] [24]
- **Keep the Transcript Canonical**: summarization and context editing can change the working context, whereas checkpointing saves graph state -> preserve the separately projected transcript as the application record, and treat summaries or cleared tool results as reconstructible context optimizations. [22] [18] [34] [35]
- **Use Interrupts for Consequential Actions**: LangChain's human-in-the-loop middleware pauses tool execution, and LangGraph interrupts resume through persisted graph state -> gate only designated high-impact tools, then recheck tenant authorization and lock ownership when execution resumes. [15] [2]
- **Treat Provider Tools as Privileged**: shell, memory, file search, prompt caching, and provider moderation integrations have materially different security and data behavior -> add only the capabilities the product needs, constrain workspace paths and provider data exposure, and keep shell out of a general-purpose production chat agent by default. [7] [13] [11]

## 1. Taxonomy: middleware is only one layer

**Verified framework distinction.** `AgentMiddleware` is a LangChain agent-loop extension point: it can run hooks around agent/model/tool steps and wrap model or tool calls. It is not interchangeable with a LangGraph node or a generic runtime callback. The base API exposes state/context parameters and lifecycle methods, while `StateGraph` nodes and reducers express graph orchestration and state transitions. [4] [5] [19]

| Primitive | What it owns | Persistence / ordering | AgentNexus decision |
|---|---|---|---|
| `AgentMiddleware` | Agent lifecycle hooks and wrappers around model/tool work | Ordered composition; state updates can be returned from hooks; middleware itself is not a durable store [5] [5] | Use narrowly for model/tool-loop policy if Saul is built with `create_agent` |
| LangGraph node / edge | Explicit workflow steps, branching, retries, state reducers, side-effect boundaries | Graph state can be checkpointed at super-step boundaries; graph topology, not middleware order, determines node sequence [19] [34] | Use for application workflows, authorization-sensitive transitions, and durable tool execution |
| Runtime/context | Per-invocation dependencies and contextual values supplied to graph/agent execution | Context is runtime input, not a substitute for persisted conversation state [19] | Pass tenant, user, provider, policy, and request-scoped services from the application boundary |
| Checkpointer / store | Checkpointer persists thread-scoped graph state; store supports longer-lived application memory | A checkpointer supports resume/recovery; it is not the independently projected transcript [34] | Keep PostgreSQL checkpointing and transcript projection separate |
| Callback / tracing handler | Observes execution events and emits telemetry | Observability hooks do not define authorization or graph control flow | Instrument through a single tracing/metrics policy; avoid duplicate transcript/audit writers |
| Interrupt / `Command` | Pauses and resumes graph execution or changes graph state | Resume requires persisted execution context for durable human approval [2] | Use for human approval, not as a replacement for WebSocket revocation or tenant auth |
| Prebuilt agent component | Agent tools, prompts, provider-specific behavior, and middleware | May change tool inventory or prompt; not all components persist state or expose streaming guarantees | Inventory and approve individually; do not equate a convenience tool with a control-plane primitive |

**Decision insight.** For Agent Saul, use the direct graph for durable business workflow and application-owned boundaries. If the graph contains a LangChain prebuilt agent loop, attach `AgentMiddleware` to that loop; do not try to make middleware own the WebSocket, lock lease, transcript, or transaction lifecycle.

## 2. Built-in LangChain middleware inventory

The documented catalog includes generic controls and provider-specific middleware. Imports below use the documented top-level export where available, with the class module shown when the API reference identifies it. A package export list and current reference pages are not perfectly aligned, so validate imports against the pinned environment; documentation does not provide one universal compatibility matrix for every provider/model. [8] [27]

| Middleware and import | Purpose, hooks, and knobs | State, operational caveats, recommendation |
|---|---|---|
| `ToolErrorMiddleware` - `from langchain.agents.middleware import ToolErrorMiddleware` | Converts tool exceptions into error tool messages, or can let them propagate. Supports `on_error`, optional `aon_error`, and tool selection. [30] [30] | It does not catch argument-validation errors occurring before tool execution. Avoid returning raw exception text to users. Adopt only if the agent needs a controlled tool-error message; keep detailed diagnostics in protected telemetry. |
| `ToolRetryMiddleware` - `langchain.agents.middleware.tool_retry.ToolRetryMiddleware` | Wraps tool calls synchronously/asynchronously. Options include `max_retries`, `tools`, `retry_on`, `on_failure`, exponential backoff settings, and jitter; documented defaults include two retries, factor 2, initial delay 1 second, maximum delay 60 seconds, and jitter. [23] [23] [23] | A retry can duplicate a side effect. Conditional: restrict to known idempotent reads or operations with application idempotency keys; test interaction with error conversion and streaming. |
| `ModelRetryMiddleware` - `langchain.agents.middleware.model_retry.ModelRetryMiddleware` | Wraps model calls with retry policy, retry predicate, backoff, jitter, and failure behavior; async form is available. [26] [26] | It can convert final failure to an error response or raise, depending on configuration. Conditional: use a small bounded retry for transient provider errors; verify streaming and provider adapter behavior. |
| `ModelFallbackMiddleware` - `langchain.agents.middleware.model_fallback.ModelFallbackMiddleware` | Tries a primary model and then configured fallback models; accepts `first_model` and `additional_models`, and wraps synchronous/asynchronous model calls. [31] [31] [31] | Compatibility of tool calling, structured output, safety policy, and streaming must be verified for each model pair. Conditional: use only where fallback semantics are equivalent and no user-visible partial answer has already streamed. |
| `ModelCallLimitMiddleware` - `langchain.agents.middleware.model_call_limit.ModelCallLimitMiddleware` | Limits model calls per run and/or thread; `exit_behavior` is `end` or `error`; implements before/after model hooks in sync and async forms. [17] [17] [17] | Thread limit needs reliable persisted state to survive independent runs/restarts. Adopt a run limit; add thread limit only after verifying checkpoint behavior. |
| `ToolCallLimitMiddleware` - `langchain.agents.middleware.tool_call_limit.ToolCallLimitMiddleware` | Limits tool calls per run/thread and can target `tool_name`; supports `continue`, `error`, or `end` behavior. [20] [20] | Adds control state and may alter how the agent sees blocked calls. Adopt a run-level cap; use per-tool caps for expensive or high-impact tools. |
| `HumanInTheLoopMiddleware` - `langchain.agents.middleware.human_in_the_loop.HumanInTheLoopMiddleware` | `interrupt_on` configures tools and approve/edit/reject/respond actions; `description_prefix`, `edit_notice`, and `interrupt_mode` affect approval UX. It wraps tool calls and requires checkpointed execution to resume. [15] [15] [15] [15] | Missing tool configuration can mean auto-approval, so explicitly enumerate sensitive tools. `per_call` mode is version-sensitive; verify it against the locked 1.4.7 environment. Use only for deliberate human gates. |
| `SummarizationMiddleware` - `langchain.agents.middleware.summarization.SummarizationMiddleware` | Summarizes older conversation messages based on token/message thresholds and retains recent context; supports model and threshold-related configuration. [22] [7] [7] | Changes the agent's working message context and incurs a model call. Do not treat the summary as the transcript or legal record. Conditional: enable when context measurements show a need; test factual retention and recovery. |
| `ContextEditingMiddleware` - `langchain.agents.middleware.context_editing.ContextEditingMiddleware` | Applies configured context edits, including clearing older tool results; supports edit list and token counting configuration. [18] | Context reduction is not transcript deletion. Conditional: use for stale tool output, with replay and streaming tests; do not use as the sole history-retention policy. |
| `PIIMiddleware` - `langchain.agents.middleware.pii.PIIMiddleware` | Detects configured PII types such as email, credit card, IP, MAC, and URL; strategies include block, redact, mask, and hash, with input/output/tool-result switches and configurable detector. [25] [25] [25] | Generic detection is not comprehensive DLP. Output streaming redaction is version-sensitive and requires supported package versions. Adopt only with a clearly defined data policy and tests against every wire/log surface. |
| `TodoListMiddleware` - `langchain.agents.middleware.todo.TodoListMiddleware` | Adds a `write_todos` tool and planning instructions; prompt and tool description can be configured. [7] | Introduces agent-visible planning state and tool overhead. Conditional for long multi-step tasks; avoid for short conversational turns or where workflow is already explicit in LangGraph. |
| `LLMToolSelectorMiddleware` - `langchain.agents.middleware.tool_selection.LLMToolSelectorMiddleware` | Uses an LLM structured-output step to choose relevant tools; options include model, prompt, max tools, always-include tools, parsing retries, and failure behavior. [21] | Adds model latency/cost and another selection decision. Conditional for large tool inventories; retain deterministic allowlists for security-sensitive tools. |
| `ProviderToolSearchMiddleware` - documented top-level middleware export | Defers provider-supported tools and searches/selects them through provider tool-search behavior; takes `searchable_tools`; unsupported deferred-tool providers can raise `ValueError`. [28] [28] | Provider/model availability is not uniform. Conditional only after testing the exact provider/model and ensuring tool policy remains enforced server-side. |
| `FilesystemFileSearchMiddleware` - `langchain.agents.middleware.file_search.FilesystemFileSearchMiddleware` | Adds glob/grep-style search over a configured `root_path`; can use ripgrep or Python fallback and has a maximum file size setting. [16] [16] | Exposes files under the configured root to agent queries. Avoid for broad application hosts; use an isolated, read-only, tenant-scoped workspace if required. |
| `ShellToolMiddleware` - `langchain.agents.middleware.shell_tool.ShellToolMiddleware` | Provides persistent shell session, workspace root, startup/shutdown commands, and execution policy (including host/container options). [36] [7] | Host execution can expose the host; redaction after execution does not prevent exfiltration. Interrupt/persistent-session interaction is a documented caveat. Avoid in Agent Saul's general production tool set. |
| Anthropic middleware - `langchain_anthropic.middleware` | Includes prompt caching, Claude bash, text editing, memory, and file search components. Prompt caching has TTL/cache configuration; bash supports workspace and execution policy; memory/file features add tools or state. [13] [13] [13] | Provider-specific behavior and model support must be pinned. Use prompt caching only for stable, non-sensitive prefixes; use memory/file tools only in explicit scoped workspaces. |
| OpenAI moderation - `langchain_openai.middleware.OpenAIModerationMiddleware` | Moderates configurable input, output, and tool-result surfaces; supports moderation model and actions such as end, error, or replace. [11] [11] | Provider-specific and adds a moderation call. Conditional on policy, latency budget, and data-processing approval; do not assume it replaces application authorization. |
| AWS Bedrock prompt caching - `langchain_aws.middleware.BedrockPromptCachingMiddleware` | Provider-specific prompt-cache behavior listed in the official middleware material. [7] [14] | Confirm exact Bedrock model, TTL, and current import in the locked package. Cache is not conversational memory. Conditional when Bedrock and stable prefixes are already in use. |
| `LLMToolEmulator` - `langchain.agents.middleware.tool_emulator.LLMToolEmulator` | Simulates selected tools for testing; defaults and model configuration are documented in the API reference. [33] [33] | Test utility, not a production tool executor. Avoid in production. |

**Takeaway.** Adopt simple, measurable controls first: run-level model/tool limits, narrow PII handling if required, and explicit approval only for risky tools. Every built-in that changes provider calls, context, or tool availability needs a pinned-version contract test because documented streaming and state details are not uniform.

## 3. Lifecycle, composition, and custom middleware

The documented hook family includes `before_agent`, `before_model`, `wrap_model_call`, `after_model`, `wrap_tool_call`, and `after_agent`, with async counterparts where supported. A pre/post hook can return a state update or no update; wrappers receive a request and handler, and may call the handler zero, one, or multiple times. This enables short-circuiting and retry, but repeated calls make idempotency and duplicate streaming output application concerns. [4] [5] [5] [5] [5]

Ordering is consequential: before hooks run in declared order, after hooks unwind in reverse, and wrappers compose as nested calls. The first wrapper can therefore observe or transform the request/response around later wrappers; test the exact order for fallback, retry, redaction, and error conversion instead of assuming ordering from a JavaScript API or an intuitive list. [5]

| Hook | Appropriate use | Avoid |
|---|---|---|
| `before_agent` | Validate invocation context, install per-run policy, fail closed before work | Network lock acquisition or authorization based on client-supplied tenant IDs |
| `before_model` | Apply input policy, count or annotate model attempt | Mutating shared middleware instance state across concurrent requests |
| `wrap_model_call` / `awrap_model_call` | Model routing, bounded retry/fallback, request/response instrumentation | Retry after a partial streamed answer without a client-visible recovery protocol |
| `after_model` | Inspect response, update typed state, enforce response policy | Treating output filtering as a substitute for input/tool policy |
| `wrap_tool_call` / async form | Enforce tool-call policy, add observation, wrap safe retry | Owning repository transactions or retrying non-idempotent writes without dedupe |
| `after_agent` | Finalize per-run metadata or attach safe state updates | Canonical transcript writes duplicated outside the transcript transaction/outbox |

A custom class can declare structured state/context through `AgentMiddleware` generics and schema properties; hooks receive state and runtime, while wrappers receive the request/handler. State updates are reducer-applied. Model wrapper responses can carry an `ExtendedModelResponse` with a `Command` update, and tool wrappers can return a `Command`; `Command` changes graph state rather than silently rewriting the separately owned transcript. [4] [5] [5] [5]

```python
from typing import Any
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, Runtime

class InvocationPolicy(AgentMiddleware):
    """Stateless policy example; real policy comes from trusted runtime context."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        context = runtime.context
        if not context.tenant_id or not context.user_id:
            raise PermissionError("Missing trusted invocation identity")
        return None
```

The type names above follow the documented API family; verify them against the installed patch version because reference/export layouts can change. A wrapper skeleton should be kept equally conservative:

```python
class ToolObservation(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        # Authorization remains in the tool/repository boundary too.
        started = self.metrics.start_tool(request)
        try:
            result = handler(request)
            self.metrics.finish_tool(started, "ok")
            return result
        except Exception:
            self.metrics.finish_tool(started, "error")
            raise
```

Use only request attributes confirmed by the installed API, keep middleware instances stateless or concurrency-safe, and put tenant-specific services in invocation runtime/context rather than mutable class fields. Async hooks and wrappers exist, but sync/async execution behavior is adapter-dependent; do not block an event loop with synchronous I/O. The docs recommend graceful error handling and independent middleware testing; application policy should define whether an exception propagates, becomes an error result, or stops the run. [5] [5]

**Takeaway.** Middleware is a policy seam, not a transaction manager. Keep authorization and deduplication close to the side effect even if an outer middleware provides defense-in-depth.

## 4. AgentNexus recommended stack and execution order

**AgentNexus case study and recommendation.** The application already has strong ownership boundaries: FastAPI/WebSocket controls authentication and revocation, Redis has a thread lock with ownership token, PostgreSQL stores durable checkpoints, and a separate projection stores the transcript. The design should preserve that separation. A graph invocation should receive a server-validated `thread_id` and trusted tenant/user context; the graph should not trust IDs or approval messages sent by the browser.

```text
WebSocket request
  -> authenticate + authorize + revocation check (FastAPI)
  -> acquire/validate Redis ownership token
  -> construct trusted runtime context and request-scoped providers
  -> LangGraph invoke(thread_id=server-owned thread, checkpointer=Postgres)
       -> before_agent: tenant/run policy validation
       -> before_model: input policy / optional PII transform
       -> model limit
       -> fallback (outer) -> retry (inner; only if retry failure propagates)
       -> tool wrapper: explicit approval for sensitive tool calls
       -> tool authorization + idempotency key + repository transaction
       -> emit stream events to WebSocket; project transcript once via app outbox
  -> verify lock token before release; persist final projection/status
```

The ordering shown for model fallback and retry is a recommendation, not a guarantee for every provider adapter: configure retry to propagate final failure if the outer fallback is expected to catch it, then test fallback exhaustion and partial-stream behavior. For tools, avoid stacking retry plus error conversion until an integration test proves which wrapper sees the raw exception; otherwise a converted error may look like success to an outer retry layer. Middleware composition nesting is documented, but each combination's exact failure/stream semantics are not. [5] [26] [31]

| Category | Adopt now | Adopt conditionally | Implement custom / keep outside middleware | Avoid |
|---|---|---|---|---|
| Identity and tenant policy | Trusted identity in runtime context; fail-closed entry check | Custom input/output PII policy if required | FastAPI auth, revocation, Redis token validation, repository authorization | Trusting prompt instructions or model-selected tool names as authorization |
| Work bounds | `ModelCallLimitMiddleware` and `ToolCallLimitMiddleware` run limits | Thread limit after restart/replay verification | End-to-end budget accounting if cost must include provider usage and tools | Unbounded tool loops |
| Reliability | Narrow model retry for transient failures | Model fallback for equivalent tool/structured-output behavior | Idempotency keys, transactional outbox, repository-level retry | Retrying non-idempotent tool calls blindly |
| Human approval | None for ordinary reads | HITL for explicitly enumerated consequential writes | Reauthorize on resume; keep lock owner and approval identity checks application-owned | Broad approvals or relying on default behavior for unlisted tools |
| Context | Separate durable checkpoint and transcript | Summarization/context editing when measured context pressure justifies it | Transcript projection and retention policy | Treating a summary as canonical history or assuming cache equals memory |
| Tool selection/planning | Deterministic allowlist and explicit graph plan | LLM tool selector for a large, low-risk tool catalog; Todo middleware for long tasks | Tenant-specific tool policy and business workflow | Dynamic unrestricted tool discovery for privileged operations |
| Provider features | Stable, non-sensitive prompt prefix for cache if provider supports it | Provider moderation/caching after latency/privacy review | Dependency-injected provider selection and key management | Shell on host, unscoped filesystem, emulator in production |

**Takeaway.** The minimal initial stack is tenant/run validation, bounded model/tool calls, and tool-boundary authorization/idempotency. Add PII, human approval, summaries, retries, fallback, or tool selection only when a defined product requirement and tests support them.

## 5. Context, streaming, replay, and security decisions

**Durability versus context management.** LangGraph checkpointers save graph snapshots for resumption; they do not make an application transcript projection unnecessary. Summarization may replace or condense the agent's current message context, while context editing clears older tool results. The client/application should retain its full transcript record independently; Anthropic's context-management documentation similarly distinguishes server-side editing from the client's unmodified history and describes compaction as the primary long-conversation strategy. [34] [22] [18] [35] [35] [35]

**Streaming and retries.** A model fallback decision after partial tokens creates a user-visible discontinuity; prefer choosing the model before streaming or constrain fallback to pre-output failures. Tool retries can be safe for reads, but writes require an idempotency key tied to a stable action/run identifier and a repository transaction or outbox. Durable replay reinforces this: side effects must be idempotent or recorded as completed work at a durable boundary. Treat these as engineering requirements and test them in the actual deployment topology, not as middleware guarantees. [23] [12] [24] [3]

**PII, prompt caching, and privacy.** PIIMiddleware offers regex/function detection and configurable input/output/tool-result application; it is a useful policy component but does not establish complete data-loss prevention. Prompt caching can reduce repeated-prefix work, but cached prompt content is not memory and should not include tenant-specific secrets unless the provider's cache isolation and policy are explicitly acceptable. Keep logs, trace payloads, checkpoint state, transcript storage, and WebSocket events in the privacy review. Just-in-time retrieval of only needed context is a sound design pattern, but is still a recommendation rather than a LangChain middleware guarantee. [25] [13] [37] [37]

**Human-in-the-loop and lock safety.** Interrupts pause a graph and resume it with persisted state. Since approval may outlive the original WebSocket or Redis lease, on resume revalidate tenant authorization, revocation, action parameters, and ownership token; approval should bind to a digest of the proposed tool/action. This revalidation belongs in AgentNexus, not in a prompt or assumed middleware feature. [2] [15]

## 6. Rollout, tests, and operating budgets

| Phase | Work | Exit criteria |
|---|---|---|
| 0. Pin and isolate | Lock LangChain/LangGraph and provider packages; run import smoke tests; capture the agent graph and middleware order | Reproducible install and exact API/type checks on the deployment image |
| 1. Guardrails | Add run-level call limits; trusted runtime context; tool-level authorization and idempotency | Unauthorized tenant/tool tests fail closed; loop cap is observable |
| 2. Persistence and replay | Exercise Postgres checkpoints, transcript projection/outbox, Redis token lifecycle, resume after process loss | No duplicate transcript item or side effect on retry/replay; stale owner cannot continue or release another owner's lock |
| 3. Streaming and failures | Test cancellation, disconnect, provider timeout, model retry/fallback, tool failure, partial output | Defined client behavior for every terminal and retry outcome; no duplicate final answer |
| 4. Optional controls | Add PII, HITL, summarization/context editing, moderation, tool selector, or provider caching one at a time | Per-feature privacy review, feature flag, rollback path, and latency/cost comparison |
| 5. Canary and expand | Canary by tenant/cohort, compare baseline and treated runs, review traces and checkpoint correctness | SLOs and rollback thresholds pass under load and chaos tests |

**Failure-injection suite.** Kill the worker before and after a tool side effect, after checkpoint write, during transcript projection, during an approval pause, and after Redis lease expiry. Inject provider timeout, tool schema error, retry exhaustion, fallback incompatibility, malformed PII detector output, WebSocket disconnect, cancellation, duplicate invocation, stale lock token, and Postgres unavailability. Verify that errors are classified, secrets are not exposed, the transcript projection is idempotent, and retry/replay cannot apply the same side effect twice.

**Metrics and SLO design.** Establish measured baselines before setting numeric targets. Track end-to-end time to first token and completion latency; model/tool calls per run; retry and fallback rates; token and provider cost per completed task; context size and summary rate; tool error/approval/rejection rates; checkpoint save/resume/replay count; duplicate-idempotency suppression; WebSocket disconnects; lock contention/expiry; and transcript projection lag. Segment by tenant, model, tool, and release. Keep content out of metric labels and redact trace payloads. Define budgets per task class and stop or degrade predictably when exhausted.

**Takeaway.** A release is not production-ready until replay, disconnect, lock-expiry, and side-effect duplication tests pass, not merely because the agent returns a correct answer in a happy-path evaluation.

## 7. Compatibility notes and uncertainties

The reviewed environment locks LangChain `1.4.7`, LangChain Core `1.7.0`, and LangGraph `1.2.16`. Pin exact patch versions and re-check the package index at release time. The middleware reference and a source-export snapshot differ in the visible set of exported symbols; use documented API references plus import tests against the lockfile rather than assuming every class is stable at the package root. [9] [10] [27]

Provider tool search, moderation, prompt caching, shell, and memory components are provider- or model-dependent. The reviewed references do not give a complete cross-provider compatibility, thread-safety, or streaming matrix. Likewise, exact persistence semantics for every built-in's internal counters/state and every middleware combination are not uniformly documented. Mark these as integration-test obligations, not facts. Do not import JavaScript middleware names or semantics unless the Python reference explicitly documents the same API.

## Synthesis

LangChain `AgentMiddleware` is best understood as a local policy/composition layer around an agent's model and tool loop; LangGraph nodes and checkpointing are the workflow and durability layer; FastAPI, Redis, repository transactions, and transcript projection are AgentNexus application ownership boundaries. They differ in scope, persistence, and failure domain, and combining them should reinforce rather than duplicate responsibilities. [4] [19] [34]

The main tradeoff is convenience versus control. Built-ins reduce implementation effort for limits, PII transformations, retries, approval, and context reduction, but their provider compatibility, stream behavior, and state details vary. Explicit graph nodes and application-owned tool boundaries require more code, yet they make replay and transaction semantics visible. AgentNexus should use middleware for bounded, testable policy; keep effects idempotent and authorization in business boundaries; and adopt provider conveniences only when exact-version tests show they preserve those guarantees.

## References

1. *Persistence*. https://docs.langchain.com/oss/python/langgraph/persistence
2. *Interrupts*. https://docs.langchain.com/oss/python/langgraph/interrupts
3. *Streaming - Docs by LangChain*. https://docs.langchain.com/oss/python/langgraph/streaming
4. *AgentMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/types/AgentMiddleware
5. *Custom middleware*. https://docs.langchain.com/oss/python/langchain/middleware/custom
6. *Agents*. https://docs.langchain.com/oss/python/langchain/agents
7. *Prebuilt middleware*. https://docs.langchain.com/oss/python/langchain/middleware/built-in
8. *middleware | langchain | LangChain Reference*. https://reference.langchain.com/python/langchain/middleware
9. *Release langchain==1.4.7 · langchain-ai/langchain · GitHub*. https://github.com/langchain-ai/langchain/releases/tag/langchain%3D%3D1.4.7
10. *langgraph · PyPI*. https://pypi.org/project/langgraph/1.2.16/
11. *OpenAI middleware integration*. https://docs.langchain.com/oss/python/integrations/middleware/openai
12. *Persistence - Docs by LangChain*. https://docs.langchain.com/oss/python/langgraph/durable-execution
13. *Anthropic middleware integration*. https://docs.langchain.com/oss/python/integrations/middleware/anthropic
14. *AWS middleware integration*. https://docs.langchain.com/oss/python/integrations/middleware/aws
15. *HumanInTheLoopMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/human_in_the_loop/HumanInTheLoopMiddleware
16. *FilesystemFileSearchMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/file_search/FilesystemFileSearchMiddleware
17. *ModelCallLimitMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/model_call_limit/ModelCallLimitMiddleware
18. *ContextEditingMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/context_editing/ContextEditingMiddleware
19. *Graph API overview*. https://docs.langchain.com/oss/python/langgraph/graph-api
20. *ToolCallLimitMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/tool_call_limit/ToolCallLimitMiddleware
21. *LLMToolSelectorMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/tool_selection/LLMToolSelectorMiddleware
22. *SummarizationMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware
23. *ToolRetryMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/tool_retry/ToolRetryMiddleware
24. *Fault tolerance*. https://docs.langchain.com/oss/python/langgraph/fault-tolerance
25. *PIIMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/pii/PIIMiddleware
26. *ModelRetryMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/model_retry/ModelRetryMiddleware
27. *langchain/libs/langchain_v1/langchain/agents/middleware/__init__.py at 307cdcac9e4e3d7cdcc86eeb3a4688b486f8e1d8 · langchain-ai/langchain · GitHub*. https://github.com/langchain-ai/langchain/blob/307cdcac9e4e3d7cdcc86eeb3a4688b486f8e1d8/libs/langchain_v1/langchain/agents/middleware/__init__.py
28. *ProviderToolSearchMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/provider_tool_search/ProviderToolSearchMiddleware
29. *PostgresSaver*. https://reference.langchain.com/python/langgraph.checkpoint.postgres/PostgresSaver
30. *ToolErrorMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/tool_error/ToolErrorMiddleware
31. *ModelFallbackMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/model_fallback/ModelFallbackMiddleware
32. *langgraph/libs/checkpoint-postgres/README.md at main · langchain-ai/langgraph · GitHub*. https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/README.md
33. *LLMToolEmulator*. https://reference.langchain.com/python/langchain/agents/middleware/tool_emulator/LLMToolEmulator
34. *Checkpointers*. https://docs.langchain.com/oss/python/langgraph/checkpointers
35. *Context editing*. https://platform.claude.com/docs/en/build-with-claude/context-editing
36. *ShellToolMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/shell_tool/ShellToolMiddleware
37. *Effective context engineering for AI agents \ Anthropic*. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
