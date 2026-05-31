# Agent Saul Websocket Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove websocket-path brittleness in Agent Saul, fix incorrect websocket protocol behavior, and add targeted regression coverage for the session bootstrap and frame-mapping flow.

**Architecture:** Keep the existing router/service/websocket-security split. Narrow the websocket dependency bundle to only the infrastructure the route actually uses, fix the frame emission contract in `AgentSaulService`, and align the HTTP session bootstrap response with its documented API contract. Add small focused tests around the new seams instead of broad end-to-end scaffolding.

**Tech Stack:** FastAPI, Pydantic v2, LangGraph, Redis, pytest, Ruff, ty

---

### Task 1: Decouple websocket startup from unused checkpointer state

**Files:**
- Modify: `src/app/features/agent_saul/dependencies.py`
- Modify: `src/app/features/agent_saul/router.py`
- Modify: `src/app/features/agent_saul/dto.py` only if dependency-related contract names need clarification
- Test: `tests/unit/agent_saul/test_dependencies.py`

- [ ] Split the current `AgentSaulDeps` dependency bundle into a websocket-specific dependency bundle that only requires `CompiledStateGraph` and `Redis`.
- [ ] Keep any HTTP or future persistence-aware paths free to depend on `AsyncPostgresSaver`, but stop injecting it into `saul_ws_endpoint(...)` unless the route or service actually uses it.
- [ ] Preserve the existing `get_saul_checkpointer(...)` helper for persistence-aware flows instead of deleting it outright.
- [ ] Add a unit test proving the websocket dependency bundle can be resolved without `app.state.langgraph_checkpointer`.
- [ ] Add a unit test proving `get_saul_checkpointer(...)` still raises `ServiceUnavailableException("Persistence layer is unavailable")` when explicitly requested and unavailable.

### Task 2: Fix unreachable websocket state-update emission

**Files:**
- Modify: `src/app/features/agent_saul/service.py`
- Modify: `src/app/features/agent_saul/dto.py` only if the emitted frame contract changes
- Test: `tests/unit/agent_saul/test_service_frames.py`

- [ ] Refactor `AgentSaulService._map_event_to_frame(...)` so top-level `on_chain_end` handling does not return before evaluating whether a `WSStateUpdateFrame` should be emitted.
- [ ] Choose one explicit contract and encode it clearly:
- [ ] Either emit `WSNodeEndFrame` and `WSStateUpdateFrame` through separate code paths from the caller, or fold state/status information into a single intentional frame strategy.
- [ ] Keep the final behavior deterministic for the frontend: no duplicate ambiguous `on_chain_end` handling branches.
- [ ] Add a unit test for a top-level `on_chain_end` event without `status`, asserting the expected `WSNodeEndFrame` behavior.
- [ ] Add a unit test for a top-level `on_chain_end` event with `output["status"]`, asserting the intended state-update behavior after the refactor.

### Task 3: Fix websocket rate-limit exception construction

**Files:**
- Modify: `src/app/features/auth/websocket_security.py`
- Test: `tests/unit/agent_saul/test_websocket_security.py`

- [ ] Change `_raise_websocket_rate_limit(...)` to raise an exception instance with `raise WebSocketRateLimitExceeded()`.
- [ ] Add a focused unit test that asserts the helper raises `WebSocketRateLimitExceeded` with the expected `error_code`, `message`, and `retryable` semantics.
- [ ] Keep the rest of the limiter wiring unchanged.

### Task 4: Fix the session bootstrap websocket URL contract

**Files:**
- Modify: `src/app/features/agent_saul/router.py`
- Modify: `src/app/features/agent_saul/dto.py`
- Test: `tests/unit/agent_saul/test_router_session_creation.py`

- [ ] Resolve the mismatch between `CreateSessionResponse.ws_url` documentation and the actual value returned by `create_session(...)`.
- [ ] Pick one contract and make code plus docs match exactly:
- [ ] Preferred: derive the actual websocket URL from the incoming request host/scheme and return a real fully qualified URL.
- [ ] Acceptable fallback: rename/document the field as a relative websocket path if the backend intentionally does not know the public host.
- [ ] Remove the literal `"{host}"` placeholder from the runtime response.
- [ ] Add a unit test covering the selected contract so the route cannot regress back to a placeholder string.

### Task 5: Add focused regression coverage for websocket bootstrap flow

**Files:**
- Create: `tests/unit/agent_saul/__init__.py`
- Create: `tests/unit/agent_saul/test_dependencies.py`
- Create: `tests/unit/agent_saul/test_service_frames.py`
- Create: `tests/unit/agent_saul/test_websocket_security.py`
- Create: `tests/unit/agent_saul/test_router_session_creation.py`

- [ ] Create a new `tests/unit/agent_saul/` test package so websocket-path tests live near each other.
- [ ] Keep tests narrow and seam-oriented; do not introduce full websocket integration harnesses unless unit seams prove insufficient.
- [ ] Cover the four regressions this plan is fixing:
- [ ] websocket dependency bundle no longer requires unused checkpointer state
- [ ] frame mapping no longer shadows state-update behavior
- [ ] rate-limit helper raises a concrete exception instance
- [ ] session bootstrap response no longer returns a fake `{host}` URL

### Task 6: Verify the websocket hardening pass

**Files:**
- No source changes required unless checks fail

- [ ] Run `uv run pytest tests/unit/agent_saul -q`
- [ ] Run `uv run ruff check src/ tests/`
- [ ] Run `uv run ty check src/`
- [ ] Fix any issues introduced by the hardening changes.

## Research Notes

This plan should be executed with one architectural bias: make the websocket path a deeper module with a smaller public surface.

### Maintainability heuristics to apply during implementation

- Keep leaf call sites stupid. The router should not know persistence availability details it does not use.
- Hide representation, expose capability. Graph-event quirks should be normalized in one mapper instead of leaking into multiple layers.
- Prefer a canonical runtime/dependency builder over a broad dependency bundle with optional baggage.
- A DTO contract must be truthful. A placeholder URL is an interface lie and should be treated as a maintainability bug, not just a formatting issue.
- Test the seam, not the internals. The important regression tests are around dependency resolution, public frame behavior, and session bootstrap contract.

### Why these fixes matter beyond the immediate bugs

Research across TanStack, Redux Toolkit, tRPC, and Matt Pocock's recent deep-module guidance converges on the same lesson: large codebases stay changeable when complexity is concentrated behind small stable interfaces.

Applied here:

- The websocket dependency bug is a shallow-module smell: callers are forced to carry an internal persistence assumption.
- The duplicated `on_chain_end` logic is a weak-seam smell: event-shape interpretation is not owned clearly enough in one place.
- The `ws_url` placeholder is an interface-truthfulness bug: public contracts cannot require callers to infer hidden deployment details.
- The rate-limit exception bug is small, but it shows why cross-cutting policy should stay encoded in one precise helper instead of being treated casually.

### Desired end state

After this plan is implemented, the websocket path should move closer to this shape:

- router depends on a websocket-specific runtime contract
- runtime owns transport-facing mapping behavior
- optional persistence concerns stay behind persistence-specific seams
- session bootstrap returns one precise public contract
- tests describe the public behavior callers rely on

## Review Findings

### High

1. `src/app/features/agent_saul/dependencies.py:84-89` and `src/app/lifecycle/lifespan.py:176-187`
Problem: the websocket route depends on `AgentSaulDepsAnnotated`, which always requires `get_saul_checkpointer(...)`, but the lifespan checkpointer setup is currently commented out. `src/app/features/agent_saul/router.py:138-144` only uses `deps.graph` and `deps.redis`, so websocket startup is currently coupled to unavailable infrastructure it does not use.
Fix: create a websocket-specific dependency bundle that injects only `CompiledStateGraph` and `Redis`, and use that bundle in `saul_ws_endpoint(...)`.

2. `src/app/features/agent_saul/service.py:237-259`
Problem: `_map_event_to_frame(...)` has two `on_chain_end` branches for the same top-level node condition. The first returns `WSNodeEndFrame` immediately, so the later `WSStateUpdateFrame` path is likely unreachable.
Fix: collapse the duplicated branch structure into one intentional `on_chain_end` handler or move multi-frame emission out of `_map_event_to_frame(...)` so status updates are not shadowed.

### Medium

3. `src/app/features/agent_saul/router.py:55` and `src/app/features/agent_saul/dto.py:152-156`
Problem: `CreateSessionResponse.ws_url` is documented as a fully qualified websocket endpoint, but `create_session(...)` returns `f"ws://{{host}}/..."`, which is a literal template string, not a usable URL.
Fix: either compute the real websocket URL from request context or explicitly change the response contract to a relative path and document it that way.

### Low

4. `src/app/features/auth/websocket_security.py:73-74`
Problem: `_raise_websocket_rate_limit(...)` raises `WebSocketRateLimitExceeded` as a class rather than constructing `WebSocketRateLimitExceeded()`.
Fix: raise the exception instance so the custom initializer and message semantics are applied predictably.

## Chosen Ones

The subtle smell is not “websockets are too complex.” It is that the protocol boundary is cleaner than the dependency boundary. Your DTOs and HITL frames are already acting like a product contract, but the dependency graph still reflects an older persistence-centric design. When the transport contract is sharper than the wiring contract, production bugs tend to show up as mysterious availability failures rather than obvious type errors. Fix the wiring first; then the websocket stack becomes appropriately sophisticated instead of accidentally fragile.
