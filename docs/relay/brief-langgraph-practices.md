# Brief: LangChain/LangGraph practices from this repo's own docs

Scout leg. Terrain only — no prescriptions beyond what the cited docs state.
Generated 2026-08-17.

## Ground truth: installed versions

From `.venv/lib/python3.12/site-packages/*.dist-info` and `uv.lock`:

| Package | Version | uv.lock line |
|---|---|---|
| `langgraph` | **1.1.2** | `uv.lock:4136` |
| `langgraph-checkpoint` | **4.0.1** | `uv.lock:4153` |
| `langgraph-checkpoint-postgres` | **3.0.4** | `uv.lock:4166` |
| `langgraph-prebuilt` | **1.0.8** | `uv.lock:4181` |
| `langgraph-sdk` | 0.3.6 | `uv.lock:4194` |
| `langchain` | **1.2.12** | `uv.lock:3519` |
| `langchain-core` | **1.2.28** | `uv.lock:3615` |
| `langchain-google-genai` | **4.2.1** | `uv.lock:4001` |
| `tenacity` | **9.1.4** | `uv.lock:8445` |
| `langchain-classic` | 1.0.3 | venv only |

Also installed: `langchain-community` 0.4.1, `langchain-openai` 1.1.12,
`langchain-anthropic` 1.3.5, `langchain-text-splitters` 1.1.1,
`langchain-mcp-adapters` 0.2.2, `langchain-tavily` 0.2.17, `google-genai` 1.67.0.

**Consequence for the planner:** this is a langchain 1.x / langgraph 1.1 stack.
Any doc passage written against langgraph 0.2-era API (`MessageGraph`,
`langchain.chains`, `langchain_community` tool imports, `HumanInterrupt`) is
misleading and flagged below.

## Source inventory

| Source | Verdict | Size |
|---|---|---|
| `.github/LangChain-LangGraph_organized_reference.md` | **Substantive**, not a stub | 2189 lines |
| `.github/skills/langchain-langgraph/references/` | 13 topic files + `index.md` | 48 KB |
| `.opencode/skills/langchain-langgraph/references/` | **Byte-identical duplicate** except missing `index.md` | 44 KB |
| `.github/skills/langchain-langgraph/SKILL.md` vs `.opencode/.../SKILL.md` | **Differ** — two divergent copies | — |
| `docs/superpowers/plans/2026-04-13-reconciliation-langgraph-package-split.md` | 85 lines | see below |
| `docs/superpowers/specs/2026-05-28-langchain-langgraph-skill-redesign.md` | 69 lines | see below |
| `.kiro/skills/langgraph-*` (14 dirs) | **ALL BROKEN SYMLINKS** — dead end | 0 |

`diff -rq` of the two reference trees: the only difference is
`Only in .github/skills/langchain-langgraph/references/: index.md`. The 13
numbered reference files are identical. The `SKILL.md` files are NOT identical.

## Nature of the organized reference

`.github/LangChain-LangGraph_organized_reference.md:3-5` states it is a
reorganized copy of `.github/LangChain-LangGrpah_thingies.md` (typo in original
filename preserved), cross-checked against "Docs by LangChain MCP on
2026-04-16", refreshed 2026-04-17. Structure: a group index
(`:9-15`), per-group Quick Reference tables, `Added / Cross-Checked Notes`
tables (the only vetted material), then `Preserved Source Notes` — raw numbered
notes, unedited. Note 73 is explicitly **unmapped/needs manual review**
(`:15`).

Read the `Added / Cross-Checked Notes` tables as higher-confidence than the
`Preserved Source Notes`, which are personal shorthand.

---

## (i) `MessagesState` for Agent A → Agent B

**Prescription (as documented):** the repo docs never name `MessagesState`.
They prescribe a **custom `TypedDict` with an explicitly annotated reducer**:

```python
class MyCustomState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    step_count: int
```

**Citation:** `.github/skills/langchain-langgraph/references/07-langgraph-state-nodes-edges.md:26-36`
("Use `TypedDict` for LangGraph state schemas") and `:5` ("Use reducers such as
`add_messages` or `operator.add` for mergeable state").
Reinforced at `.github/LangChain-LangGraph_organized_reference.md:100`:
"custom schema for create_agent can be a TypedDict only, can be defined via
state_schema or middleware" — i.e. **not** a Pydantic BaseModel for agent state.
`.github/LangChain-LangGraph_organized_reference.md:79-80` shows the same shape
with `operator.add`.

**Structured non-message payloads:** the docs answer this indirectly and
consistently — **they belong in separate state channels, not in `messages`.**
The `MyCustomState` example puts `user_id` and `step_count` as sibling keys
alongside the reduced `messages` channel
(`07-langgraph-state-nodes-edges.md:32-36`). The data-source table at
`.github/LangChain-LangGraph_organized_reference.md:143-146` separates Runtime
Context (static config, conversation-scoped) / State (short-term memory,
conversation-scoped) / Store (long-term, cross-conversation) — payloads are
State keys, not message content.
`.github/LangChain-LangGraph_organized_reference.md:130-139` adds the
transient-vs-persistent split: Model Context is transient (what the LLM sees for
one call), state writes are persistent.

**Handoff pattern:** `11-multi-agent-patterns.md:39` —
"Best practice for multi-agent systems is to use `AIMessage` objects with
`tool_calls` to hand off tasks." Communication is via **Shared State** or
**Tool-Based Communication** (`:34-37`). Pattern menu and trade-offs at
`:24-30` and `:43-66`; "Custom workflow — recommended when production control
flow must be explicitly designed for the domain" (`:64-66`).

**Version caveat:** `MessagesState`, `add_messages` ID-collision/overwrite
semantics, `RemoveMessage`, and `REMOVE_ALL_MESSAGES` are **entirely absent**
from every repo doc. The repo docs also show `add_messages` used *manually*
(`07:14-18`: `state["messages"] = add_messages(state["messages"], [new_msg])`),
which is a reducer-internals call, not the normal node-return idiom of
`return {"messages": [msg]}` shown at
`.github/LangChain-LangGraph_organized_reference.md:85`. Partial-update
semantics are undocumented here.

**Anti-pattern the docs warn against:**
`.github/LangChain-LangGraph_organized_reference.md:105` — "always normalise
agent state after fetching from checkpointer so that there is no version
mismatch". And `:109` (note 22) — preserve `extras["signature"]` on `AIMessage`
across turns or the model re-reasons from scratch; this is a concrete
message-fidelity failure mode for A→B handoff.

**Internal contradiction (flagged):**
`07-langgraph-state-nodes-edges.md:60-88` shows `create_agent(...)` constructed
*inside* `research_node`/`review_node` with a fresh `MemorySaver()` per node.
`.github/LangChain-LangGraph_organized_reference.md:64` and `:96-98` call
exactly this an anti-pattern ("forces the agent to be rebuilt on every single
step... prevents effective caching"). `07:90` admits the conflict:
"this source example is preserved, but elsewhere in the reference there is a
stronger recommendation to keep full agent loops out of nodes."

### (i) addendum — the decisive passage

`.github/LangChain-LangGraph_organized_reference.md:1341-1345` (note 67) is the
single most load-bearing line for this todo:

> "As of langchain 1.0, custom state schemas must be TypedDict types. Pydantic
> models and dataclasses are no longer supported. Defining custom state via
> middleware is preferred over defining it via `state_schema` on `create_agent`
> ... `state_schema` is still supported for backwards compatibility.
> Use TypedDict: Required for LangGraph state schemas.
> Use Pydantic models: These are generally used for data configuration, tool
> arguments, or schema validation in standard LangChain components."

So: **state = TypedDict; Pydantic = tool args / DTO validation.** This resolves
the "subclass vs custom TypedDict vs Pydantic" question in the docs' own voice.
The subclass form the docs show is `class CustomState(AgentState)` — subclassing
`langchain.agents.AgentState`, attached via `AgentMiddleware.state_schema`
(`:1346-1365`), **not** `MessagesState`.

`MessagesState` is named exactly once in the whole corpus, descriptively:
`.github/LangChain-LangGraph_organized_reference.md:1479` — "Agent B receives
the full MessagesState, allowing it to see the history." The surrounding
handoff sketch (`:1473-1479`): Agent A returns an `AIMessage` with a
`transfer_to_*` tool call; a Router node reads that specific tool call and
routes the edge to Agent B; Agent B sees full history.

**`add_messages` semantics, as documented:**
`.github/LangChain-LangGraph_organized_reference.md:1462` —
"# Messages are merged by ID (deduplication)". That is the repo's only statement
of ID semantics. Message-ID *collision* consequences (overwrite-in-place),
`RemoveMessage`, and `REMOVE_ALL_MESSAGES` are **not covered anywhere** — see
Gaps.

**Loop-safety caveat, verbatim** (`:1471`, note 18): "Circular delegation is
possible... There's no loop detection beyond `completed_agents` in
`SupervisorState`, and that only works in the supervisor graph — not in the
tool-based `MultiAgentSystem`." Note this sentence describes *this repo's own
prior code*, not the library.

Guard rail for any handoff graph: `:1492` (note 20) — "Always set a
`recursion_limit` (max steps) in your LangGraph and a timeout on your LLM calls."
Also `07-langgraph-state-nodes-edges.md:49`.

**Architecture menu** (`:1496-1521`): Network of Agents is called
"too loose, unreliable, and costly for production"; Supervisor gives "better
control"; **"Custom Architectures: The most recommended approach for production,
where the control flow is specifically designed for the domain rather than
relying on off-the-shelf patterns"** (`:1521`).

---

## Tools & structured output

**Target form for tools (documented):** `@tool` decorator + an explicit
**Pydantic `args_schema`**, with `Field(description=...)` on every argument.

`.github/LangChain-LangGraph_organized_reference.md:1481-1490`:
> "To ensure the LLM doesn't just guess what your database needs, you should
> define a Pydantic model and pass it to the `@tool` decorator. This populates
> the `parameters` field in the JSON schema sent to the model, effectively
> 'forcing' the LLM to adhere to your structure."

with the example `@tool(args_schema=DatabaseQuery)` where `DatabaseQuery` is a
`BaseModel` whose fields all carry `Field(description=...)`.
Reinforced by note 1 at `:43`: "add Field description for tool instead of simple
docstrings."

`:1480` — `InjectedToolArg`: "tells LangGraph: 'The LLM should not provide this
argument; instead, the system should inject it from the state or config at
runtime' (e.g., a `user_id`)." This is the documented way to pass user scope
into a tool without exposing it to the model.

**`StructuredTool` is never mentioned in the corpus.** The docs' target form is
`@tool(args_schema=PydanticModel)`. Absence noted, not interpreted.

**Version caveat:** `:1465` (note 15) — "`@tool` functions can be `def` or
`async def`. If your tool calls an API, make it `async def`." Combined with
note 7 (`:56`) "use async functions, methods and packages in langchain and
langGraph" and `07:6`.

**Import anti-pattern, explicit** (`:60-63`, note 9):
```
CORRECT — from langchain_tavily import TavilySearch
WRONG   — from langchain_community.tools.tavily_search import TavilySearchResults
```
i.e. dedicated provider packages, never `langchain_community` paths.

### Structured output — target path

**Prescription:** two documented paths, chosen by layer.

1. **Agent-level:** `create_agent(..., response_format=...)`, result lands in
   `structured_response`. Signature at
   `.github/skills/langchain-langgraph/references/04-model-selection-and-structured-output.md:53-72`:
   `response_format: Union[ToolStrategy[T], ProviderStrategy[T], type[T], None]`.
   Passing a bare schema type lets LangChain pick provider-native structured
   output when supported, else tool-calling
   (`.github/LangChain-LangGraph_organized_reference.md:36`).
2. **Model-level:** `model.with_structured_output(Schema, include_raw=True)`
   when you need token counts / metadata — `04-...:79` (note 49).

**Version caveat (matches installed stack):** `04-...:77` —
"`ProviderStrategy.strict` requires `langchain>=1.2`." Installed `langchain` is
**1.2.12** (`uv.lock:3520`), so `strict` is available. Also
`.github/LangChain-LangGraph_organized_reference.md:37`: native
structured-output support is read from **model profile** data on
`langchain>=1.1`; if profile data is unavailable, pass a custom profile or pick
a strategy manually. Profile shape at `04-...:81-93` (`model.profile` →
`max_input_tokens`, `image_inputs`, `reasoning_output`, `tool_calling`;
`init_chat_model("...", profile=custom_profile)`).

**Anti-pattern:** note 2 at
`.github/LangChain-LangGraph_organized_reference.md:45` — "use structured output
everywhere for llm output, tool output, MCP output". Free-text parsing is the
thing being ruled out. And `:942-943` / `04-...:79`: discarding the raw
`AIMessage` loses usage metadata.

**Dynamic `response_format`** per state/store/role via `@wrap_model_call` +
`request.override(response_format=...)` — `04-...:97-127`,
`.github/LangChain-LangGraph_organized_reference.md:660-763`.

---

## Prompts

**What the docs actually say** (`02-prompts-and-messages.md:5-9`, mirrored at
`.github/LangChain-LangGraph_organized_reference.md:1416`):

- `PromptTemplate` — single string inputs.
- `ChatPromptTemplate` — structured chat (system / human / AI).
- `MessagesPlaceholder` — a slot where a list of messages (history) is injected.

**Text vs message prompts** (`02-...:17-27`): text prompts for a single
standalone request with no history and minimal complexity; message prompts for
multi-turn, multimodal (image/audio/file), or when including system
instructions. Raw-message form shown at `02-...:29-36`
(`[SystemMessage(...), HumanMessage(...), AIMessage(...)]` → `model.invoke`).

**Brace escaping: NOT COVERED.** `string.Template`, `partial_variables`, and
`{{`/`}}` escaping appear **nowhere** in this repo's corpus (grepped
`string\.Template|partial_variables|escap` across
`.github/LangChain-LangGraph_organized_reference.md`, all 14 reference files,
and both `SKILL.md` copies — zero hits). This is a hard **Gap**; the planner must
fall back to `langchain-core` 1.2.28 library docs.

**Observable convention in every doc example, though:** all dynamic prompts are
built with plain Python f-strings inside `@dynamic_prompt` / `@wrap_model_call`
and returned as `str` or appended as a raw message dict — the templating engine
is bypassed entirely. See `02-...:90-103`, `:118-136`, `:151-165`, `:176-201`,
`:217-241`. Notably `02-...:190-193` and `:228-233` interpolate multi-line
payloads via f-string, not a template. Reported as terrain, not advice.

**Interaction with TOON:** notes 3-6 at
`.github/LangChain-LangGraph_organized_reference.md:47-54` mandate TOON for
serialization to/from both LLMs and tools, and quote the target format
verbatim: `"Communicate data using TOON format. Declaring uniform arrays as
key[N]{field1, field2}: val1, val2. Minimal punctuation. No braces."` — note
`key[N]{...}` itself contains braces, which is exactly the collision the todo
names. Note 6 also asks an **open question the docs never answer**: "should i use
chains for repeatable action for toon conversion". Note 1-after-24 (`:114`)
extends TOON to "agents, chats, RAG, web search results, after tool LLM invoke
and everywhere else". Note 8 (`:58`): "trim/remove tool output in a multi step
agent conversation".

**Message-fidelity anti-pattern (repeat, because it bites prompt rebuilding):**
`02-...:13` — preserve `extras["signature"]` on `AIMessage` across turns.

---

## (j) Retries — `tenacity`

**`tenacity` is named ZERO times in this repo's entire LangChain/LangGraph doc
corpus.** So is `RetryPolicy`, and so is `.with_retry()`. Verified by grep over
`.github/LangChain-LangGraph_organized_reference.md`, all 14 files in
`.github/skills/langchain-langgraph/references/`, and both `SKILL.md` copies.
`tenacity` **is** an installed dependency at **9.1.4** (`uv.lock:8445`).
The three-layer model in the todo is therefore **NOT** sourced from repo docs —
it must come from library docs.

**What the docs DO prescribe about retries:**

| Layer | Documented form | Citation |
|---|---|---|
| Intent | "have proper retry mechanism for tools with idempotent execution as mention in langchain docs" (note 12, typo `idenpotent` in source) | `.github/LangChain-LangGraph_organized_reference.md:107`, `01-langchain-overview.md:84` |
| Model call | manual retry loop inside `@wrap_model_call` middleware | `05-middleware-and-guardrails.md:93-105`, mirror at `.github/LangChain-LangGraph_organized_reference.md:1248-1258` |
| Tool call | `ToolNode(..., handle_tool_errors=...)` — catch all, custom message, handler fn, or selected exception types only | `.github/LangChain-LangGraph_organized_reference.md:38` (cross-checked note) |
| Middleware capability list | "Tool retry — automatically retry failed tool calls with exponential backoff. Model retry — ... failed model calls with exponential backoff." | `.github/LangChain-LangGraph_organized_reference.md:1117-1118`, `05-...:15-16` |

The `@wrap_model_call` mechanism is described as the retry seam:
`.github/LangChain-LangGraph_organized_reference.md:1235-1236` — "Intercept
execution and control when the handler is called. Use for retries, caching, and
transformation. You decide if the handler is called zero times (short-circuit),
once (normal flow), or multiple times (retry logic)."

### The replay trap — the docs DO cover this, concretely

This is the strongest documented finding for todo (j). Three passages compose
into the trap:

1. **`.github/LangChain-LangGraph_organized_reference.md:1628`** (note 43):
   "The node restarts **from the beginning of the node** where the interrupt was
   called when resumed, **so any code before the interrupt runs again**."
   Mirror: `09-interrupts-hitl-resume.md`. Also `:1622-1623` (note 42): for a
   `StateGraph`, "the starting point is the beginning of the node where
   execution stopped"; for a subgraph call, the starting point is the *parent
   node* that called the halted subgraph.
2. **`:1612`** (note 40): "If a node contains multiple operations with side
   effects (e.g., logging, file writes, or network calls), **wrap each operation
   in a separate task**. This ensures that when the workflow is resumed, the
   operations are not repeated, and their results are retrieved from the
   persistence layer." Plus `:1613`: encapsulate non-deterministic operations in
   tasks/nodes so resumption "follows the exact recorded sequence of steps with
   the same outcomes."
3. **`:1614`**: "Use Idempotent Operations... if a task starts but fails to
   complete successfully, the workflow's resumption will **re-run the task**,
   relying on recorded outcomes... Use idempotency keys or verify existing
   results to avoid unintended duplication."

Compressed statement of the trap in the docs' own terms: **the checkpointer's
recovery unit is the node, not the statement.** Retry state held in local
variables inside a node (a `tenacity` attempt counter, a partial accumulator) is
not a checkpointed channel, so on replay the node re-enters at its first line
with that counter reset — the retry budget is silently multiplied, and every
non-idempotent side effect before the failure point re-fires. The docs' remedy is
task decomposition (`:1612`) plus idempotency keys (`:1614`), not a bigger retry
decorator.

**Anti-pattern the docs explicitly warn against** — `:1633-1662` (note 45):
"**Do not wrap interrupt calls in try/except.** The way that `interrupt` pauses
execution at the point of the call is by throwing a special exception. If you
wrap the interrupt call in a try/except block, you will catch this exception and
the interrupt will not be passed back to the graph." Prescribed instead:
separate interrupt calls from error-prone code, and catch **specific** exception
types (worked good/good/bad triple at `:1637-1662`). This generalizes directly
to any bare-`Exception` retry wrapper — including `tenacity`'s default
`retry_if_exception_type(Exception)` — placed around code that may `interrupt`.
Reinforced: `:1664` (note 46) "Do not reorder interrupt calls within a node";
`:1710` / `09-...:90` "Use idempotent operations before interrupt", `:1752`
"Do not perform non-idempotent operations before interrupt";
`:1401` "Keep interrupt ordering stable and avoid non-idempotent side effects
before pause points."

**Version caveat:** all retry guidance here is middleware-era (`langchain` 1.x
`create_agent` + `@wrap_model_call`), consistent with installed 1.2.12. But the
langgraph-native `RetryPolicy` / `CachePolicy` args to `add_node`, and
`durability=` on `invoke`, are only partially covered (durability modes are —
see below; `RetryPolicy` is not).

---

## Checkpointing

**Prescription:** a real checkpointer requires (a) an **async** backend class,
(b) a `thread_id` in config on every invocation, (c) a serializer that can encode
every state channel value.

**Citations:**

- **Async backend, mandatory in production:**
  `.github/LangChain-LangGraph_organized_reference.md:1465` (note 15) —
  "Checkpointers: `SqliteSaver` (Sync) vs. `AsyncSqliteSaver` (Async). In
  production, **always use the async version to avoid blocking your DB
  connection pool**." Mirror: `07-langgraph-state-nodes-edges.md:40`.
  Installed backend: `langgraph-checkpoint-postgres` **3.0.4** (`uv.lock:4166`)
  — i.e. `AsyncPostgresSaver` is available; the docs' Sqlite examples are
  illustrative only.
- **`thread_id` is not optional:** `08-checkpointing-persistence-durability.md:14-18`
  and `.github/LangChain-LangGraph_organized_reference.md:1409` — "When invoking
  a graph with a checkpointer, pass `{"configurable": {"thread_id": "..."}}`; the
  checkpointer uses `thread_id` to save/load checkpoints and resume after
  interrupts." Resume requires the **same** `thread_id` used at interrupt time
  (`:1626`, note 43).
- **`checkpoint_id`** is the handle for post-pause resume — `:1494` (note 21),
  `08-...:20`. Formerly named `thread_ts` (`:1494`).
- **Read APIs:** `graph.get_state(config)` → latest `StateSnapshot`;
  `graph.get_state_history(config)` → history, **newest first**
  (`:1410`, `:1600`, `08-...:22-32`).
- **Writes create checkpoints, never mutate:** `:1602` (note 38) —
  "`update_state` ... creates a new checkpoint with the updated values — it does
  not modify the original checkpoint. The update is treated the same as a node
  update: **values are passed through reducer functions when defined, so channels
  with reducers accumulate values rather than overwrite them.**" This is
  load-bearing for any `messages` channel: `update_state` on an
  `add_messages`-reduced channel appends, it does not replace.
- **Namespaces:** `checkpoint_ns` is `""` for the root graph, `node_name:uuid`
  for a subgraph, joined with `|` when nested (`:1597-1598`, `08-...:43-49`).
  Read inside a node via `config["configurable"]["checkpoint_ns"]`
  (`:1595-1596`).
- **Serialization:** `JsonPlusSerializer` is the default, uses **ormsgpack + JSON**
  under the hood, handles LangChain/LangGraph primitives, datetimes, enums
  (`:1604-1609`, `08-...:58-64`). For unsupported objects (the doc names Pandas
  DataFrames) use the `pickle_fallback` argument. **Implication for the refactor:
  any non-primitive object placed in a state channel must be
  `JsonPlusSerializer`-encodable or the checkpoint write fails.**
- **Durability modes**, least → most: `exit` / `async` / `sync`
  (`:1616-1619`, `08-...:76-88`). `exit` persists only on exit (success, error,
  or interrupt) — "best performance for long-running graphs but ... you cannot
  recover from system failures (like process crashes) that occur mid-execution."
  `async` persists while the next step executes, "small risk that LangGraph does
  not write checkpoints if the process crashes." `sync` persists before the next
  step starts, "high durability at the cost of some performance overhead."
- **Thread deletion:** `checkpointer.delete_thread(thread_id)` (`08-...:36-41`).
- **Checkpointer vs Store:** `:1412` — "Checkpointers persist per-thread state.
  Stores persist data across threads and are the correct place for long-term
  memory." Practical separation rules at `06-runtime-state-store-context.md:47-53`:
  State = active working memory (conversation-scoped); Store = cross-thread
  durable memory; Runtime Context = **immutable invocation-scoped configuration**;
  Model Context = the exact prompt-time view for one call.

**What breaks when the checkpointer is `None`** — the docs do **not** address
this case directly. Inferable only: every capability listed at
`08-...:5-10` (HITL, conversation memory, time travel, fault-tolerant execution)
depends on persistence, and `interrupt`/`Command(resume=...)`/`get_state`/
`get_state_history`/`update_state` all key off `thread_id`. Recorded as **Fog**,
not asserted.

**Anti-pattern the docs warn against:** `.github/LangChain-LangGraph_organized_reference.md:105` (note 11) — "always normalise
agent state after fetching from checkpointer so that there is no version
mismatch". A checkpoint written under
one state schema, read back after the schema changed, is a live hazard for this
refactor.

---

## (f) Graph in `app.state` via lifespan

**`FastAPI`, `app.state`, and `lifespan` appear ZERO times in this repo's
LangChain/LangGraph doc corpus.** (Fixed-string grep for `FastAPI`, `app.state`,
`lifespan`, `startup` across the 2189-line reference, all 14 reference files, and
both `SKILL.md`s.) The web-framework integration boundary is entirely
undocumented here.

**What the docs DO establish — build-once, reuse-forever:**

- **`.github/skills/langchain-langgraph/references/01-langchain-overview.md:10`**
  (Production Reminders): "**Do not rebuild model instances or agent instances on
  every call.**"
- **`01-...:86`** (note 24): "Model instances are rebuilt on every call.
  `build_chat_model()` constructs a new model every time it is called. **The model
  object should be a module-level singleton or per-spec singleton since it is
  stateless.**" Mirror at `.github/LangChain-LangGraph_organized_reference.md:111-112`,
  which names the concrete class: `ChatGoogleGenerativeAI`. This is a critique of
  **this repo's own `build_chat_model()`** — pre-existing terrain, not hypothetical.
- **`01-...:41`**: "Creating an agent instance (`create_agent`) inside a node is
  an anti-pattern because it forces the agent to be rebuilt on every single step,
  which is inefficient, **prevents effective caching**, and complicates testing."
  Mirror `.github/LangChain-LangGraph_organized_reference.md:64`.
- **Worked build-once shape:** `01-...:43-70` — module-level `init_chat_model`
  instances, a single shared agent, `StateGraph(...)` → `add_node`/`add_edge` →
  `graph = workflow.compile()` **at module scope**, and nodes that only invoke
  the pre-built object. Mirror `.github/LangChain-LangGraph_organized_reference.md:65-94`
  (comment at `:87`: "# Build graph once"; `:94`: "Usage:
  `graph.invoke({...})` # Reuses agent/model").
- **`01-...:72`**: "Nodes should focus on **how to process the state**, not how to
  configure an agent. By keeping the agent logic outside the node, your nodes
  become cleaner and easier to unit test."
- **`.github/skills/langchain-langgraph/SKILL.md:22`** (High-Value Repo Rules):
  "**Reuse model and agent instances instead of rebuilding them inside each call
  path.**" And `SKILL.md:21`: "Prefer explicit LangGraph orchestration for complex
  workflows instead of burying a full agent loop inside a graph node."
- **`.opencode/skills/langchain-langgraph/SKILL.md:110`** (Quick Reminders):
  "Do not create agent instances inside LangGraph nodes unless there is a narrow,
  well-justified reason."

**Per-process vs per-request — what the docs support:**

| Concern | Docs' scope | Citation |
|---|---|---|
| Model instance | stateless → **singleton** | `01-...:86` |
| Compiled graph / agent | build once, reuse | `01-...:43-70`, `SKILL.md:22` |
| Checkpointer | async, holds a **DB connection pool** | `.github/LangChain-LangGraph_organized_reference.md:1465` |
| Store | long-term, cross-thread | `:1412`, `06-...:50` |
| `thread_id` / `checkpoint_id` | **per invocation**, in `config["configurable"]` | `08-...:14-18`, `:1409` |
| Runtime Context (user_id, API keys, permissions) | "Static configuration", **conversation-scoped**, "immutable invocation-scoped configuration" | `06-...:21`, `06-...:51` |

The clean line the docs draw: **`config`/Runtime Context is the per-request
carrier; the graph, model, checkpointer, and store are not.** `06-...:47-53`
("Practical Separation Rules") is the load-bearing citation. `:1417` adds:
"`agent.ainvoke(input, config)`: The input matches your `AgentState`. The config
... contains configurable parameters like `thread_id` or `user_id`."

**What breaks if you compile per request:** the docs assert inefficiency, lost
caching, and harder testing (`01-...:41`) plus connection-pool blocking for sync
checkpointers (`:1465`). They do **not** discuss per-request compile against a
shared checkpointer, event-loop affinity of an async connection pool, or
`app.state` typing. **Fog.**

---

## `Send` / fan-out and reducers

**`Send` is named ZERO times in this repo's entire LangChain/LangGraph doc
corpus.** So is `langgraph.types.Send` (grep found only
`from langgraph.types import Command` — `:883`, `:1291`, `:1865`, `:1929`,
`03-tools-and-toolruntime.md:159`, `05-...:136`, `12-memory.md:45`, `:64`).
Neither "fan-out", "fanout", nor "map-reduce" appears. **Hard Gap** — the
planner must source `Send` semantics from langgraph 1.1.2 library docs.

**Nearest documented material — reducers, which is the half the docs do cover:**

- `07-langgraph-state-nodes-edges.md:5`: "Use reducers such as `add_messages` or
  `operator.add` for mergeable state."
- `.opencode/skills/langchain-langgraph/SKILL.md:112`: "**Use reducers for
  append-style state fields or they will be overwritten.**" — the clearest
  statement of the accumulate-vs-clobber rule in the corpus.
- Signature by example only: `Annotated[list, add_messages]`
  (`07-...:33`, `:1371`) and `Annotated[list, operator.add]`
  (`01-...:59`, `:80`). No explicit `Callable[[T, T], T]` reducer contract is
  written down anywhere.
- `:1602` (note 38): `update_state` values "are passed through reducer functions
  when defined, so channels with reducers accumulate values rather than
  overwrite." Confirms reducers apply to external writes too, not just node
  returns.
- **`Command(update={...})`** is the documented dynamic-routing/state-write
  primitive the corpus does cover — from tools (`03-...:159-165`), from
  middleware (`05-...:136-150`, `:1291-1308`), from memory writes
  (`12-memory.md:45-50`, `:64-77`), and from guardrails (`:883-898`).
- **Concurrency warning, indirect:** `:1500` and `:1514` name parallelization as
  the strength of the Subagents and Router patterns respectively, but the corpus
  never states what happens when two parallel branches write the same channel
  without a reducer. `SKILL.md:112` is the only hint.

---

## Embeddings

**`init_embeddings` — ZERO mentions. `GoogleGenerativeAIEmbeddings` — ZERO
mentions. `task_type` — ZERO mentions. No batch-size ceiling is documented
anywhere.** The provider-agnostic-vs-direct question is **not answered** by this
repo's docs. Hard Gap; `langchain-google-genai` is installed at **4.2.1**
(`uv.lock:4002`) and its own docs must be the source.

**What the docs DO say:**

- **Caching is the headline finding.** `.github/LangChain-LangGraph_organized_reference.md:2049`
  (note 23): "Embeddings aren't cached. **`aembed_batch` calls the API every
  time.** Embeddings for the same text are deterministic — a simple LRU cache
  keyed on **SHA256(text)** would eliminate redundant API calls entirely."
  Softened restatement at `13-retrieval-rag.md:28`. This is a critique of this
  repo's own embedding path and names `aembed_batch` — the only batching API
  mentioned in the corpus.
- **Documented cache mechanism:** `CacheBackedEmbeddings.from_bytes_store(
  underlying_embeddings, store, namespace=underlying_embeddings.model)` with
  `LocalFileStore("./cache/")` — `13-retrieval-rag.md:30-54`, mirror `:2116-2135`.
  **Version caveat:** the import is `from langchain_classic.embeddings import
  CacheBackedEmbeddings` and `from langchain_classic.storage import
  LocalFileStore`. `langchain-classic` **is** installed (1.0.3), so this works —
  but `langchain_classic` is the v0-compat shim, and the docs' own note 9
  (`:60-63`) rules against legacy import paths. Also note `namespace=
  underlying_embeddings.model` requires the embeddings object to expose `.model`.
- **Dimensions** appear only for Store semantic search: `12-memory.md:19` /
  `:1847` — "Configure store indexing with embeddings and dimensions to support
  semantic memory search with `store.search(..., query=..., limit=...)`."
- **Pipeline shape:** `13-...:5-14` — sources → loaders → documents → chunks →
  embeddings → vector store → retriever → LLM answer. Mirror `:2042`.
- **Architecture choice worth flagging:** `13-...:22` — "If you already have a
  strong existing knowledge base such as SQL, CRM, or internal docs, **you do not
  necessarily need to rebuild it as a vector store.** You can expose it as a tool
  and pass retrieved context to the LLM." And `13-...:24`: "Add retrieval
  evaluation before production: retrieved documents, relevance, groundedness, and
  answer correctness."
- Splitters: `RecursiveCharacterTextSplitter` and
  `CharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", ...)`
  — `13-...:72-95`. Vector-store interface (`add_documents` / `delete` /
  `similarity_search`, metadata `filter=`): `13-...:106-145`.

---

## Status of the two `docs/superpowers/` documents

### `docs/superpowers/plans/2026-04-13-reconciliation-langgraph-package-split.md`

**SUPERSEDED — twice over.** State it plainly:

1. **Already implemented.** The plan's goal (`:5`) was to rename
   `reconsiliation` → `reconciliation` and split the monolith into
   `state.py` / `prompt.py` / `graph.py` / `pipeline_node.py` (`:16-22`). On disk
   today: `src/app/shared/langgraph_layer/reconciliation/` exists with
   `__init__.py`, `state.py`, `prompts.py`, `graph.py`, `nodes.py`. The misspelled
   `reconsiliation/` directory is gone. The split landed (with `nodes.py` and
   `prompts.py` rather than the plan's `pipeline_node.py`/`prompt.py`), so even
   the file names in the plan no longer match reality. Every checkbox in the file
   is still unchecked (`- [ ]` at `:24, :29, :39, :48, :58, :69, :77, :82`) —
   the plan was never marked complete, which is why it still reads as live.
2. **The target is being deleted.** The current refactor removes the
   reconciliation package; a plan to reorganize it has no remaining subject.

**One artifact from it is still relevant as a contradiction**, not as guidance —
`:31-37` prescribes:
```python
class ReconciliationState(BaseModel):
    model_config = ConfigDict(extra="forbid")
```
A **Pydantic** state schema. See Contradictions below.

Also note the sibling packages the plan treated as the reference shape
(`ingestion_kb`) — and the actual on-disk peers of the deleted package:
`agent_saul/`, `ingestion_kb/`, `open_deep_search/`, `retrieval_kb/`, plus
top-level `checkpointer.py` and `kb_retry.py` in
`src/app/shared/langgraph_layer/`. Those last two file names are the repo's own
prior art for the checkpointing and retry todos.

### `docs/superpowers/specs/2026-05-28-langchain-langgraph-skill-redesign.md`

**Implemented, and it explains the duplicate-tree finding.** Marked "Approved for
implementation on 2026-05-28" (`:3`). Its File Plan (`:27-56`) targets
`.opencode/skills/langchain-langgraph/` exclusively — `SKILL.md`,
`references/index.md`, and all 13 numbered files. All 13 exist in both trees with
the May 28 timestamps. Editorial rules at `:58-63` explain why the reference
files read as raw notes: "Preserve almost all original notes and examples. Keep
important opinionated guidance **even when the source wording is rough**. Fix only
structural issues." `:17` mandates `.github/LangChain-LangGraph_organized_reference.md`
stay untouched — so the reference doc is the canonical source and the skill files
are a derived access layer. Confirmed by
`.github/skills/langchain-langgraph/SKILL.md:8` and `:48`.

This spec is **not** superseded; it is done. It contains no refactor guidance.

**No `openspec/` coverage:** there is no `openspec/specs/` entry and no
`openspec/changes/` entry for LangGraph state, checkpointing, retries, or the
reconciliation deletion. (Checked for a spec covering this area by domain
concept — graph state, persistence, agent orchestration — not by the request's
wording.)

---

## Contradictions

1. **Pydantic state schema: plan vs reference doc.**
   `docs/superpowers/plans/2026-04-13-reconciliation-langgraph-package-split.md:32`
   prescribes `class ReconciliationState(BaseModel)`, while
   `.github/LangChain-LangGraph_organized_reference.md:1341` and
   `06-runtime-state-store-context.md:59` both say "As of LangChain 1.0, custom
   state schemas must be `TypedDict` types. **Pydantic models and dataclasses are
   no longer supported.**" **Unresolved nuance:** note 67 is written about
   `create_agent`'s `state_schema`; raw `langgraph` `StateGraph` has its own
   position on Pydantic state schemas that this corpus never states. The repo's
   existing `state.py` files across five packages may follow either. Flagged as
   Fog below — do not treat note 67 as settled for bare `StateGraph`.
2. **Agent-inside-node: reference file vs reference file.**
   `07-langgraph-state-nodes-edges.md:60-88` and
   `.github/LangChain-LangGraph_organized_reference.md:1418-1457` both show
   `create_agent(...)` built **inside** a node with a fresh `MemorySaver()`
   per node. `01-langchain-overview.md:41`/`:74-78`,
   `.github/LangChain-LangGraph_organized_reference.md:64`/`:96-98`,
   `.github/skills/langchain-langgraph/SKILL.md:21`, and
   `.opencode/skills/langchain-langgraph/SKILL.md:110` all call that an
   anti-pattern. `07-...:90` acknowledges the conflict but preserves the example
   anyway (per the spec's "preserve almost all original notes" rule,
   `spec:60`). **The anti-pattern side has four citations to the example's two;
   the example is preserved source, the prohibition is editorial + cross-checked.**
3. **Dataclass for context vs "dataclasses no longer supported."**
   `02-prompts-and-messages.md:114`, `:146`, `:213` and `06-...` all use
   `@dataclass class Context` for `context_schema=`. Note 67 forbids dataclasses
   for **state**, not for **context**. Two different schema slots; easy to
   conflate. Not a real contradiction, but it reads as one.
4. **`langchain_classic` imports vs the dedicated-import rule.**
   `13-retrieval-rag.md:34-35` imports `CacheBackedEmbeddings` and
   `LocalFileStore` from `langchain_classic`, while note 9
   (`.github/LangChain-LangGraph_organized_reference.md:60-63`) forbids legacy
   import paths. `langchain-classic` 1.0.3 is installed, so the code runs; the
   tension is stylistic.
5. **`create_react_agent` vs `create_agent`.**
   `01-langchain-overview.md:45` imports `create_react_agent` from
   `langgraph.prebuilt` with the inline comment "# or
   `langchain.agents.create_agent`", while every other passage in the corpus uses
   `create_agent`. `langgraph-prebuilt` 1.0.8 is installed so both exist. The
   corpus never states which is current for langchain 1.2.
6. **Doc vs installed version — HITL names already migrated.**
   `.github/LangChain-LangGraph_organized_reference.md:102-104` lists the
   deprecations as a table (`HumanInterruptConfig` → `InterruptOnConfig`,
   `ActionRequest` → `InterruptOnConfig`, `HumanInterrupt` → `HITLRequest`).
   Presented as "deprecated vs new things" — on `langchain` 1.2.12 the left column
   is gone, so any repo code still on the left column is broken, not merely dated.
7. **Two divergent `SKILL.md` copies.** See Repo hygiene.

---

## Gaps — repo docs do NOT cover these todos at all

The planner must fall back to library docs (`langgraph` 1.1.2,
`langchain-core` 1.2.28, `langchain-google-genai` 4.2.1, `tenacity` 9.1.4) for:

| Topic | Grep evidence |
|---|---|
| `Send` / map-style fan-out | zero hits for `Send(`, `fan-out`, `fanout` |
| Reducer function **signature** (only shown by example) | no `Callable[[T,T],T]` contract stated |
| `MessagesState` as a class | one descriptive mention (`:1479`); never imported or subclassed |
| `RemoveMessage`, `REMOVE_ALL_MESSAGES` | zero hits |
| `add_messages` ID-collision / overwrite semantics | only "merged by ID (deduplication)" (`:1462`) |
| `tenacity` | zero hits (installed 9.1.4) |
| `RetryPolicy` on `add_node` | zero hits |
| `.with_retry()` on Runnables | zero hits |
| `CachePolicy` | zero hits |
| FastAPI / `app.state` / `lifespan` integration | zero hits |
| Brace escaping (`{{` / `}}`) in `ChatPromptTemplate` | zero hits — **and this collides directly with the mandated TOON format's `key[N]{...}` braces (`:54`)** |
| `partial_variables` | zero hits |
| `string.Template` | zero hits |
| `init_embeddings` | zero hits |
| `GoogleGenerativeAIEmbeddings`, `task_type` | zero hits |
| Embedding batch-size ceiling | zero hits; only `aembed_batch` named (`:2049`) |
| `StructuredTool` | zero hits — docs' target is `@tool(args_schema=Model)` |
| Checkpointer `None` behaviour | never discussed |

---

## Repo hygiene

1. **`.kiro/skills/` is a dead end — 14 broken symlinks.** Every
   `langchain-*` / `langgraph-*` / `deep-agents-*` / `langsmith-*` /
   `framework-selection` entry under `.kiro/skills/` is a dangling symlink into a
   `.agents/skills/` tree that contains only the `openspec-*` skills. Confirmed
   broken; contents unreadable. The dangling names —
   `deep-agents-core`, `deep-agents-memory`, `deep-agents-orchestration`,
   `framework-selection`, `langchain-dependencies`, `langchain-fundamentals`,
   `langchain-middleware`, `langchain-rag`, `langgraph-fundamentals`,
   `langgraph-human-in-the-loop`, `langgraph-persistence`, `langsmith-dataset`,
   `langsmith-evaluator`, `langsmith-trace` — indicate what guidance *was* meant
   to exist. The `openspec-*` entries in that directory are real directories, not symlinks, and resolve fine — the breakage is confined to the LangChain/LangGraph/LangSmith/deep-agents set (`readlink` shows e.g. `.kiro/skills/langgraph-fundamentals -> ../../.agents/skills/langgraph-fundamentals`, target absent).
2. **Duplicate skill tree, partially divergent.**
   `.github/skills/langchain-langgraph/references/` and
   `.opencode/skills/langchain-langgraph/references/` hold **byte-identical**
   copies of all 13 numbered files (`diff -rq`: no content differences). But:
   - `index.md` exists **only** in `.github/...` (the `.opencode` copy is missing
     the file its own `SKILL.md` workflow step 1 tells you to open first —
     `.github/skills/langchain-langgraph/SKILL.md:12` "Open `references/index.md`
     first").
   - The two `SKILL.md` files **differ structurally**. `.github/...SKILL.md` is a
     48-line dispatcher (workflow, High-Value Repo Rules, file list).
     `.opencode/...SKILL.md` is 113 lines titled "LangChain LangGraph Reference
     Index" — it has absorbed the index content, carries a per-file topic map
     (`:29-107`) and a "Quick Reminders" block (`:108-113`) that the `.github`
     copy lacks. **Neither is a superset**: the `.github` copy's "High-Value Repo
     Rules" (`:19-27`) and the `.opencode` copy's "Quick Reminders" are different
     rule lists. Any refactor touching skill guidance must edit both, and the
     `.opencode` spec (`spec:27-29`) only ever named the `.opencode` paths.
3. **Filename typo in the canonical source chain.**
   `.github/LangChain-LangGraph_organized_reference.md:3` points at
   `.github/LangChain-LangGrpah_thingies.md` — "LangGrpah" transposed. That
   misspelled file is declared the untouchable original (`spec:17`).
4. **Note 73 is unmapped.** `.github/LangChain-LangGraph_organized_reference.md:15`
   and `13-retrieval-rag.md:149-151`: "Present in source but not mapped by the
   organizer. Preserved in the source document for future review." Whatever note
   73 says lives only in the misspelled `_thingies.md` file, never in the
   organized copy.
5. **Stale plan left unchecked.** The reconciliation split plan is fully
   implemented on disk but every checkbox is still `- [ ]`, making a completed
   plan indistinguishable from a pending one.

---

## Fog

Things I could not establish from this repo's docs, and what it would take:

1. **Whether bare `StateGraph` still accepts Pydantic state schemas on langgraph
   1.1.2.** Note 67 (`:1341`) is scoped to langchain's `create_agent`. The five
   existing `state.py` files under `src/app/shared/langgraph_layer/*/` would
   settle what the repo actually does — I did not read source (docs-only remit).
   Resolve by: reading those five files, or `langgraph` 1.1.2 `StateGraph`
   docstring.
2. **Whether the doc corpus's `create_agent`/middleware material even applies to
   the graphs being refactored.** The corpus is heavily `create_agent`-centric,
   yet `SKILL.md:21` and `:1521` both push toward hand-built `StateGraph`
   orchestration. Which mode the target packages use is a source question.
3. **`add_messages` behaviour on ID collision.** "Merged by ID (deduplication)"
   (`:1462`) does not say whether the later message **replaces** the earlier one
   in place or is dropped. This matters for Agent A → Agent B handoff. Library
   source (`langgraph.graph.message.add_messages`) is the only authority here.
4. **What actually breaks with a `None` checkpointer.** Docs list what
   persistence *enables* (`08-...:5-10`) but never the `None` failure mode.
   Resolve by: reading `src/app/shared/langgraph_layer/checkpointer.py` — its
   existence suggests the repo already made this decision somewhere.
5. **Whether `kb_retry.py` already implements the retry layer being asked for.**
   `src/app/shared/langgraph_layer/kb_retry.py` exists and is named for exactly
   this concern, but the docs never reference it and I did not read source. This
   is the single highest-value unknown for todo (j) — prior art may already exist
   under a name the request never uses.
6. **Whether the `Send` fan-out pipeline being promoted to production is
   `ingestion_kb` or `open_deep_search`.** Both have the file shape for it;
   `open_deep_search/utils.py` and `config.py` are unique to it. The docs name
   neither.
7. **Where TOON serialization is implemented.** Notes 3-6 and 1-after-24 mandate
   it pervasively (`:47-54`, `:114`) and note 6 leaves the "should i use chains
   for repeatable action for toon conversion" question **open in the source**. No
   reference file covers TOON mechanics.
8. **Whether note 73 contains anything relevant.** It exists only in the
   unread `.github/LangChain-LangGrpah_thingies.md`. Resolve by reading that file
   — I stayed within the organized copy per the spec's canonical-source framing.
