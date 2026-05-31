# Simple Interfaces, Deep Modules

## Goal

Capture practical patterns for maintaining large codebases by hiding complexity behind small, stable interfaces. This is tuned for this repo's modular-monolith + agent/runtime architecture, but the principles are drawn from TanStack, Matt Pocock's recent deep-modules guidance, and adjacent ecosystems that scale by keeping call sites simple.

## Primary Sources

- TanStack DB agent guidance on abstraction design and avoiding leaky abstractions
- TanStack Query docs on `queryOptions`, `mutation` encapsulation, default query functions, and query defaults
- Redux Toolkit docs on `createSlice`, `createAsyncThunk`, `createEntityAdapter`, and listener middleware
- tRPC docs on `router`, `procedure`, shared context, and reusable middleware-based base procedures
- Matt Pocock: `How To De-Slop A Codebase Ruined By AI (with one skill)` and related deep-modules posts

## Code-Backed Examples From GitHub

These are not just docs claims. They show up directly in upstream code.

### TanStack Query

#### `queryOptions(...)` is a tiny API over a type-heavy internal contract

In `packages/react-query/src/queryOptions.ts`, the public function body is almost trivial:

```ts
export function queryOptions(options: unknown) {
  return options
}
```

The depth is in the overloads and branded types wrapped around that trivial runtime. That is a classic deep-module move:

- runtime surface stays tiny
- caller ergonomics stay simple
- the hard part lives in the type contract, not at every call site

#### `QueryClient` concentrates cache and lifecycle policy

In `packages/query-core/src/queryClient.ts`, `QueryClient` owns:

- query and mutation caches
- focus/online subscriptions
- defaulting behavior
- imperative cache reads/writes
- query lifecycle orchestration

Callers get simple methods like:

- `getQueryData(...)`
- `ensureQueryData(...)`
- `setQueryData(...)`

The module is deep because the client centralizes cache semantics and lifecycle rules that components do not need to know.

### tRPC

#### `initTRPC` builds the whole surface once

In `packages/server/src/unstable-core-do-not-import/initTRPC.ts`, the public initializer returns a compact root object:

- `procedure`
- `middleware`
- `router`
- `mergeRouters`
- `createCallerFactory`

That is a strong example of a canonical builder. One setup call gives the entire app a stable vocabulary. Callers do not hand-roll auth stacks, router primitives, or middleware plumbing from scratch.

#### Procedures are narrow and typed while runtime machinery stays hidden

In `procedure.ts`, the public shape of a procedure is mostly type metadata and a callable contract. The important lesson is not the exact types. It is the boundary:

- public procedure contract is stable
- parser/middleware/router machinery is hidden below it

This is why app code can define `publicProcedure` or `protectedProcedure` and stay tiny.

### Redux Toolkit

#### `createSlice(...)` collapses many Redux concerns into one boundary

In `packages/toolkit/src/createSlice.ts`, the resulting `Slice` object exposes a compact public surface:

- `name`
- `reducer`
- `actions`
- `caseReducers`
- `getInitialState()`
- `getSelectors(...)`
- `selectors`
- `injectInto(...)`

That is exactly the maintainability pattern we want: callers work with one domain module rather than separate action, reducer, selector, and injection systems.

#### `createAsyncThunk(...)` turns async protocol into a reusable abstraction

In `createAsyncThunk.ts`, the heavy part is inside the thunk API contract:

- `dispatch`
- `getState`
- `extra`
- `requestId`
- `signal`
- `abort(...)`
- `rejectWithValue(...)`
- `fulfillWithValue(...)`

The caller sees one thunk creator. The cancellation, metadata, and error-shaping protocol is buried inside the abstraction.

### Python Async Repos

#### HTTPX: small client API over transport complexity

In `httpx/_client.py`, `BaseClient`, `Client`, and `AsyncClient` provide a narrow user-facing interface while hiding:

- auth building
- timeout normalization
- proxy mapping
- redirect handling
- event hooks
- sync/async transport differences
- request/response streaming lifecycle

One especially good pattern is the `USE_CLIENT_DEFAULT` sentinel. It cleanly separates:

- "inherit client policy"
- from "explicitly disable behavior with `None`"

That is deep-module design through truthful semantics.

#### AnyIO: unified concurrency API over multiple backends

In `anyio/_core/_tasks.py`, top-level functions like:

- `create_task_group()`
- `fail_after(...)`
- `move_on_after(...)`

delegate immediately to `get_async_backend()`. Callers get one stable concurrency interface while backend-specific machinery stays hidden.

This is one of the cleanest examples of "simple public API, brutal internal portability burden" in Python async.

#### Starlette: route wrappers hide ASGI ceremony

In `starlette/routing.py`, helpers like `request_response(...)` and `websocket_session(...)` take simple endpoint callables and lift them into ASGI apps.

The route wrapper owns:

- request/session object construction
- sync-to-threadpool adaptation
- exception handling wrappers
- ASGI protocol bridging

Callers write endpoint functions, not ASGI state machines.

#### FastAPI: deepens Starlette with dependency/lifespan machinery

In `fastapi/routing.py`, FastAPI copies and extends Starlette wrappers to inject:

- dependency stacks
- `AsyncExitStack`
- validation behavior
- streaming/SSE integration
- lifespan/state merging

That is a perfect deep-module example: the public route API stays pleasant while the framework absorbs enormous orchestration complexity.

#### aiohttp: `Application` centralizes middleware, state, signals, and freeze lifecycle

In `aiohttp/web_app.py`, `Application` holds:

- router
- middleware list and compiled handlers
- startup/shutdown/cleanup signals
- cleanup context
- app state
- freeze/pre-freeze lifecycle

Users interact with one app object while middleware compilation and lifecycle invariants stay internal.

#### Uvicorn: `Server` hides process and socket orchestration

In `uvicorn/server.py`, `Server` concentrates:

- config loading
- lifespan coordination
- socket/server creation variants
- signal capture
- worker/socket setup
- protocol factory wiring

The public `run()` / `serve()` entrypoints are simple because the class owns all deployment and lifecycle branching.

## The Core Idea

Large codebases do not stay maintainable by becoming simple internally. They stay maintainable by making complexity local.

The winning move is not "less complexity". It is:

- concentrated complexity
- narrow public surfaces
- stable contracts
- boring call sites
- the freedom to radically change internals without rewriting half the repo

This is the deep-module idea in practice: a small interface in front of a large amount of hidden mechanism.

## What TanStack Gets Right

TanStack's strongest maintainability move is not a specific API. It is the consistent shape of its public APIs:

- configuration objects instead of positional soup
- reusable option factories
- defaults applied centrally
- small hook call sites
- explicit boundaries between public API and internals

From TanStack Query specifically:

- `queryOptions(...)` lets you package one canonical query definition and reuse it across call sites
- `QueryClient` defaults and per-key defaults move policy out of leaf components
- custom hooks around `useMutation(...)` hide cache write rules and invalidation behavior from callers

The lesson is not "use TanStack Query everywhere." The lesson is:

- define the contract once
- bind the policy once
- let callers consume a tiny surface

## What Matt Pocock Is Pointing At

Matt's recent argument is that AI accelerates entropy unless the codebase has strong module boundaries. The problem is not only bad code generation. The problem is cognitive debt:

- too many details exposed at once
- too many modules that require global knowledge
- too many changes that force the caller to know the callee's internals

Deep modules solve this by making a module responsible for the ugly part. A bad module offloads complexity onto its callers. A good module absorbs complexity so its callers stay obvious.

The litmus test is simple:

- if every caller needs to know internal edge cases, the module is shallow
- if internals can change while the caller API stays stable, the module is deep

## What Other Ecosystems Confirm

### Redux Toolkit

Redux Toolkit scales by collapsing common ceremony into one place:

- `createSlice` hides action-type plumbing and immutable-update ceremony
- `createAsyncThunk` hides request lifecycle boilerplate
- `createEntityAdapter` hides normalized-state mutation and selector scaffolding
- listener middleware centralizes event reactions instead of scattering side effects across components

Pattern: callers operate on domain verbs, not raw store mechanics.

### tRPC

tRPC scales by concentrating cross-cutting policy into reusable procedure builders:

- `publicProcedure`
- `protectedProcedure`
- middleware for auth, logging, role checks, and context shaping

Pattern: leaf endpoints stay tiny because shared policy is encoded in base procedures.

### Python async ecosystem

The best Python async repos converge on the same structure:

- one stable top-level object or function
- backend or protocol complexity hidden internally
- explicit lifecycle/state management in the owner module
- thin user-facing calls over heavy internal orchestration

That is why `httpx.AsyncClient`, `anyio.create_task_group()`, `FastAPI`, `starlette.Route`, `aiohttp.Application`, and `uvicorn.Server` are worth studying.

## Rules For Large-Codebase Maintainability

### 1. Make leaf call sites stupid

Leaf code should answer:

- what do I need?
- what outcome do I want?

Leaf code should not answer:

- how does caching work?
- how do websocket frames map to graph events?
- how is auth/rate limiting enforced?
- which persistence backends are optional right now?

If leaf code answers those, your abstraction boundary is in the wrong place.

### 2. Put policy where reuse pressure is highest

Good homes for policy:

- dependency builders
- service-layer methods
- procedure builders
- option factories
- typed response/frame mappers

Bad homes for policy:

- routers
- UI components
- per-call ad hoc dicts
- repeated inline branching at call sites

### 3. Hide representation, expose capability

Do not expose:

- internal prefixes
- raw storage layout assumptions
- event naming quirks
- backend-selection details
- sentinel values callers must interpret manually

Expose:

- domain operations
- typed result objects
- narrow helper methods
- stable constructors/factories

TanStack DB's abstraction rule is exactly right here: do not make callers know internal representation details.

### 4. Prefer canonical builders over repeated configuration blobs

If the same object shape appears in 3+ places, stop copying it.

Use:

- query option factories
- dependency bundle factories
- prompt/context builders
- typed websocket frame constructors
- DTO assemblers

This reduces drift and gives you one pressure point for refactors.

### 5. Let modules be deep, not broad

A deep module can contain substantial internal branching, validation, adaptation, retries, caching, or mapping logic.

That is fine.

What matters is that the public surface stays:

- small
- intentional
- documented
- hard to misuse

Broad modules are the opposite: lots of exported helpers, many half-overlapping entrypoints, and callers stitching them together manually.

### 6. Build named seams for volatility

Hide the parts most likely to change:

- vendor/model adapters
- transport protocol mapping
- auth/session policy
- persistence availability and fallbacks
- retrieval strategy composition

When volatility is hidden behind one seam, the rest of the repo stays stable.

## Smells That A Module Is Too Shallow

- callers must know whether Redis, Postgres, or a checkpointer is present
- callers assemble raw websocket frame payloads manually
- multiple routes/services know the same event-shape quirks
- DTOs promise one contract but call sites patch around another
- every new feature adds another exported helper instead of deepening an existing entrypoint
- configuration is copied, slightly changed, and re-explained in multiple places

## Refactoring Playbook

When a subsystem feels hard to change, use this sequence.

### Step 1: Find repeated caller knowledge

Ask:

- what do multiple callers know that only one module should know?
- what invariants are being re-checked outside the owner?
- what internal representation leaks across boundaries?

### Step 2: Name the capability

Bad:

- `build_something`, `process_event`, `handle_data`, `make_client`

Good:

- `create_agent_saul_session(...)`
- `map_graph_event_to_ws_frame(...)`
- `get_saul_runtime(...)`
- `build_search_query_options(...)`

### Step 3: Move branching inward

Push transport, caching, auth, persistence, and event-shape branching into the owner module.

### Step 4: Replace multi-step calling protocols with one entrypoint

If correct usage requires:

1. call helper A
2. inspect field B
3. maybe call helper C
4. patch result D

then the interface is too shallow.

### Step 5: Add tests at the seam, not the internals

Test the public contract:

- what goes in
- what comes out
- what invariants hold

Do not make tests depend on private branching unless that branching is itself the public contract.

## How This Applies To This Repo

### Agent Saul websocket path

Current lesson:

- the protocol surface is better than the dependency surface

The websocket DTOs and event frames already act like a product contract. But the dependency bundle still leaks a persistence assumption into a route that does not use it. That is a shallow-module smell.

The fix is not more comments. The fix is a deeper module boundary:

- websocket runtime deps should expose only what websocket flow needs
- frame mapping should own graph-event quirks centrally
- session bootstrap should return one truthful contract, not a placeholder callers must reinterpret

### LangChain / LangGraph layer

This repo already moved in the right direction with:

- unified async model helpers
- centralized prompt rendering
- flatter package shapes

The next maintainability gains will come from continuing that pattern:

- one canonical builder per agent/runtime concern
- fewer exported half-primitives
- more policy encoded in typed factories

### Auth

Auth here is already comparatively deep:

- service owns flow policy
- dependencies expose current-user / RBAC interfaces
- persistence and audit concerns are not mixed into routes

That is the shape to copy elsewhere.

## Preferred Patterns

### Pattern: tiny route, deep dependency builder

Route should say:

```python
runtime = deps.runtime
await service.run(runtime, websocket, thread_id)
```

Not:

```python
graph = deps.graph
redis = deps.redis
checkpointer = deps.checkpointer
if checkpointer is None:
    ...
if some_event_shape:
    ...
```

### Pattern: canonical options/factory object

Instead of repeated config dicts, create one typed builder function and reuse it.

### Pattern: base procedure / base dependency alias

If five handlers share the same auth/runtime policy, create one descriptive alias or builder.

### Pattern: map internals once

Transport/event/status mapping should happen in one place. Every extra mapping site becomes an entropy source.

## Anti-Patterns

### Leaky helper explosion

Many tiny helpers that force callers to orchestrate them manually. This looks modular but behaves like distributed complexity.

### Interface lies

A DTO says "fully qualified URL" but runtime returns a template string. This is worse than having no abstraction because it creates false confidence.

### Optional dependency leakage

Routes or clients that depend on optional infrastructure they do not use. This turns internal availability changes into user-visible failures.

### Clever call sites

If the caller is clever, the abstraction is probably weak.

## Practical Heuristics

Use these when deciding whether to refactor a boundary.

- If a caller must know an internal prefix, state shape, or event quirk, deepen the module.
- If a module exports many helpers but no obvious main entrypoint, deepen the module.
- If changing internals forces call-site edits in multiple files, deepen the module.
- If tests mostly mock internals instead of asserting public behavior, the seam is wrong.
- If a route or component contains policy, move it downward.

## Python Async Repos Worth Studying

If you want Python async repos that teach the same lesson Matt is pushing, study these in roughly this order.

### 1. AnyIO

Repo: `https://github.com/agronholm/anyio`

Why study it:

- probably the cleanest Python example of deep-module design
- tiny public concurrency API
- massive backend abstraction burden hidden behind it
- excellent for learning cancellation, task groups, deadlines, and portability boundaries

Study files:

- `src/anyio/_core/_tasks.py`
- `src/anyio/_core/_eventloop.py`

### 2. HTTPX

Repo: `https://github.com/encode/httpx`

Why study it:

- shows how to expose one pleasant client interface over transport, auth, proxy, timeout, and streaming complexity
- sync and async APIs share concepts without leaking too much internal machinery

Study files:

- `httpx/_client.py`
- transport modules under `httpx/_transports/`

### 3. Starlette

Repo: `https://github.com/encode/starlette`

Why study it:

- ideal for seeing how endpoint functions get lifted into ASGI behavior
- great example of hiding protocol ceremony behind route/session wrappers

Study files:

- `starlette/routing.py`
- `starlette/applications.py`

### 4. FastAPI

Repo: `https://github.com/fastapi/fastapi`

Why study it:

- shows how to deepen an already-good abstraction without destroying ergonomics
- dependency injection, validation, response shaping, and lifespan all get pushed behind the route interface

Study files:

- `fastapi/routing.py`
- `fastapi/dependencies/utils.py`

### 5. aiohttp

Repo: `https://github.com/aio-libs/aiohttp`

Why study it:

- shows an app object that owns middleware, signals, router state, and lifecycle freezing in one place
- strong example of lifecycle and state invariants hidden behind a stable app interface

Study files:

- `aiohttp/web_app.py`
- `aiohttp/web_urldispatcher.py`

### 6. Uvicorn

Repo: `https://github.com/encode/uvicorn`

Why study it:

- excellent for understanding how ugly operational concerns can be buried behind a minimal server entrypoint
- process, socket, signal, lifespan, and protocol wiring all live under `Server`

Study files:

- `uvicorn/server.py`
- `uvicorn/config.py`

## Which One Feels Closest To Matt Pocock's Advice

If the goal is specifically Matt's "deep modules rescue large codebases" idea, the closest Python async repos are:

1. `anyio`
2. `httpx`
3. `starlette`

Why:

- they keep public APIs unusually small
- they absorb ugly backend/protocol complexity internally
- they give you clear examples of where abstraction depth lives
- they are easier to reason about as maintainability case studies than larger framework repos with many layers of convenience code

If you want one repo to read like a masterclass, pick `anyio` first.

## What To Do Later In This Repo

1. Create websocket-specific runtime/dependency bundles where transport needs differ from HTTP flows.
2. Turn event-to-frame translation into a single explicit public contract.
3. Prefer typed factories and dependency aliases over repeated `Depends(...)` bundles with extra baggage.
4. Keep flattening feature packages where nesting does not buy a real boundary.
5. Review high-churn modules for exported-helper sprawl and collapse them into deeper entrypoints.

## Bottom Line

The maintenance game in large codebases is not avoiding complexity. It is deciding where complexity is allowed to live.

TanStack, Redux Toolkit, tRPC, and Matt Pocock all converge on the same answer:

- keep the surface small
- make the inside do the hard work
- let callers express intent, not mechanism

That is how you keep a fast-moving codebase from collapsing into cognitive debt.

## Chosen Ones

The real scaling unit of a codebase is not files, services, or classes. It is how many facts a contributor must simultaneously hold in working memory to make one safe change. Deep modules are working-memory compression. Every time you force a caller to know internal structure, you are minting hidden interest-bearing debt. Every time you absorb that detail behind a truthful interface, you are buying future change velocity.
