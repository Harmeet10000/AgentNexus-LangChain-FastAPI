> Change class: **L** — adds a new public protocol surface, a new frontend runtime and toolchain, a new authorization/transport path for the same graph, and new HTTP read endpoints over the transcript store.

## Why

Agent Saul is a resumable, interrupt-driven legal review graph, but its only client contract is a bespoke WebSocket frame protocol (`api/websocket/endpoint.py`, `api/dto.py`). That means no maintained chat UI can consume it, every client re-implements node progress, plan editing and clause overrides, and a pending human approval dies with the socket that carried it. Adopting AG-UI 1.0 as the agent↔UI contract — and CopilotKit as the first client — replaces a one-off protocol with a versioned specification, makes interrupts durable state instead of a live connection, and gives us a working review console without inventing components.

## What Changes

- Add an **AG-UI 1.0 projection** of Agent Saul runs over **HTTP + SSE**, derived from the same `astream_events` v2 projection that already feeds the WebSocket frames. One event mapper, two transports.
- Introduce a **transport port** in the agent-saul application layer and split `AgentSaulService.run_session()` into a start phase (run to terminal or interrupt) and a resume phase (run from checkpoint), so one service implementation serves both a socket that awaits a mid-turn command and a request that cannot. The WebSocket adapter composes the two phases in a loop to preserve today's behaviour exactly.
- Model human review as the AG-UI **terminal-run interrupt**: `RunFinished { outcome: { type: "interrupt", interrupts: [...] } }`, resumed by a new run carrying `RunAgentInput.resume[]`. A paused approval becomes a durable checkpoint, not an open socket.
- Emit `STATE_SNAPSHOT` / `STATE_DELTA` (RFC 6902), `STEP_STARTED` / `STEP_FINISHED`, `ACTIVITY_SNAPSHOT` / `ACTIVITY_DELTA`, and `SUBAGENT_*` so node progress, plan state and delegated work are protocol-level facts rather than UI guesswork.
- Publish the interrupt payload as a **JSON Schema per interrupt type** (`PLAN_APPROVAL`, `HUMAN_REVIEW_REQUIRED`, `CLARIFICATION_NEEDED`, `OCR_REUPLOAD`) generated from the existing Pydantic domain models. This includes adding a typed override model for `human_review_required`, which today accepts `list[dict[str, Any]]`. The two gates whose payloads schema forms render badly get hand-built editors.
- Add **transcript read endpoints** — `GET /agent-saul/sessions` and a paginated `GET /agent-saul/sessions/{session_id}/transcript` returning JSON — over the existing transcript store, preserving redaction/erasure semantics already enforced by the NDJSON export.
- Ship a **CopilotKit React console** (`web/`, Vite + React + TypeScript) that performs session preflight, scopes the run to authorized documents, streams the turn, renders the plan/review gates, and reloads history from the transcript endpoints. No Node server: the CopilotKit runtime is mounted in FastAPI.
- Treat the console as a legal-review workspace: safe Markdown rendering, stable scroll while evidence is reviewed, keyboard-first composer and gates, attention-aware history, responsive panels, and locally persisted presentation preferences only.
- Keep `WS /agent-saul/ws/{thread_id}` working and behaviourally unchanged. It is documented as deprecated but supported; retiring it is a separate change.
- **BREAKING (internal only):** the WS frame mapper becomes a transport-neutral projection, and `AgentSaulService.run_session()` becomes a start/resume pair behind a transport port. WebSocket frame shapes, close codes and close ordering do not change; call sites and the service's constructor signature do.

## Scope / Non-Goals

In scope: the AG-UI event contract and SSE transport, interrupt/resume lifecycle, the transcript read API, the console application, authorization parity with the WebSocket path, observability for the new surface, contract tests proving both transports agree, and the deployment/static-hosting wiring.

Out of scope: removing or reshaping the WebSocket protocol; model or provider selection in the UI; a general-purpose chat product; document upload from the console; mobile/native clients; multi-agent or third-party agent support; replacing CopilotKit with a second UI framework; and any change to graph topology, budget enforcement, thread locking, or transcript retention policy.

**Planning assumptions (confirm before apply):**
1. One change delivers backend projection + console; WS stays.
2. AG-UI runs over SSE; no WebSocket custom binding for AG-UI.
3. No Node runtime in production — CopilotKit's Python runtime is mounted in FastAPI.
4. History comes from the transcript store, not from a UI-side or vendor-side store.
5. The console is served from the API origin, so the existing `access_token` cookie authenticates both HTTP and stream requests.
6. This change is additive and must not modify WS frame shapes, so it can land alongside `harden-agent-saul-evidence-pipeline`.

## Capabilities

### New Capabilities

- `agent-ui-streaming`: AG-UI 1.0 event contract for Agent Saul runs — run lifecycle, text, state, activity, subagent and interrupt events over HTTP+SSE, including resume semantics, authorization parity with the WebSocket path, and transport-equivalence guarantees.
- `agent-saul-transcript-query`: authorized listing of a caller's sessions and paginated JSON transcript reads over the existing transcript store, with redaction and erasure represented rather than reconstructed.
- `agent-console-frontend`: the CopilotKit review console — session preflight and document scoping, streaming turn rendering, plan and human-review gates, history, and degradation/error behaviour.

### Modified Capabilities

None. The WebSocket contract, graph topology, budget, thread locking and transcript retention requirements are unchanged; this change adds surfaces beside them.

## Impact

- New backend package: `src/app/features/agent_saul/api/agui/` (SSE endpoint, request/response adapters), `src/app/features/agent_saul/application/projection/` (transport-neutral event mapping extracted from `application/service.py`), and `application/ports/sink.py` (the transport port the service writes through).
- Touched backend files: `application/service.py` (mapper extraction and the `run_session` start/resume split — the largest single edit in this change), `api/websocket/endpoint.py` (composes the two phases behind the port, behaviour unchanged), `api/http/router.py` (two GET routes and the AG-UI POST), `api/v1/__init__.py` (already includes the feature router), `domain/state.py` (typed review-override model), `main.py` / `server.py` (static mount + SSE-safe middleware), `lifecycle/lifespan.py` (runtime wiring if a client object is needed).
- New frontend package: `web/` (Vite, React 19, TypeScript, Tailwind, CopilotKit React SDK) plus its own lockfile and CI job.
- Dependencies: AG-UI Python SDK (`ag-ui-protocol`) and the CopilotKit Python runtime (`copilotkit`). Exact pins verified in design; both are new third-party dependencies with their own release cadence.
- Infrastructure: SSE must pass through the reverse proxy unbuffered (`X-Accel-Buffering: no`, `proxy_buffering off`, idle timeout above the longest run); static assets served from the API origin.
- Tests: cross-transport contract tests that replay one recorded event stream through both projections, plus authorization, redaction and interrupt-resume suites.
- Docs: `docs/architecture.md`, the manual testing plan, and a new ADR for the protocol choice.

## Risks

| Risk | Mitigation |
| --- | --- |
| Two transports drift and behave differently | The mapper is transport-neutral and single-source; a recorded-stream contract test asserts both projections emit equivalent facts. |
| Terminal-run interrupts change reconnect behaviour | Resume is keyed on `thread_id` and served by the same checkpoint resume path; an explicit test covers disconnect-during-interrupt then resume. |
| CopilotKit's Python runtime lags the Node one | Spike the Python runtime before UI work starts; fallback is `@ag-ui/client`'s `HttpAgent` against the same AG-UI endpoint, with CopilotKit used as components only. |
| SSE buffering or proxy idle timeouts truncate long legal runs | Explicit proxy config in the migration plan, plus a heartbeat and a resume-on-disconnect test. |
| The service split regresses WebSocket behaviour (close codes, ordering, lock release, T3 abort) | The WS adapter composes start+resume in the same order the current loop uses; a golden-frame replay test from a recorded run asserts identical frames and close sequence before the AG-UI path is enabled. |
| Merge conflicts with `harden-agent-saul-evidence-pipeline` (same files, 86 open tasks) | Additive-only files where possible, the projection + service split lands first and alone, and HITL payload construction is untouched; rebase before apply if that change lands first. |
| A second chat store re-creates the erasure problem | The console keeps no durable history of its own; it renders from the transcript store and clears client caches on logout. |
| Paused approvals resumed by a user who lost authority | Every resume re-resolves live tenant/legal authority, identical to the WebSocket path, and is tested with a role revocation between interrupt and resume. |
| Scope creep into a general chat product | Non-goals are explicit; the console has no model picker, no upload, and no free-form tools. |
