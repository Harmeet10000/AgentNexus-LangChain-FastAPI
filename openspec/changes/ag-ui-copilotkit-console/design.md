> Change class: **L**.

## Change-Local ADRs

- `adr.md#ADR-001` — coherent with D1, D3 and D9.
- `adr.md#ADR-002` — coherent with D2 and D13.

## Context

Agent Saul already separates three things cleanly, and the adoption rides on that separation rather than fighting it:

- `application/service.py` drives the graph with `astream_events` v2, detects `interrupt()` pauses via `aget_state()`, and resumes with `Command(resume=...)`. `AgentSaulService` owns thread locking (`thread_lock.py`), budget (`domain/budget.py`), transcript recording (`transcript/recorder.py`) and resume-time authorization.
- **`AgentSaulService` is currently typed to the WebSocket transport and owns an interactive loop.** It is constructed with `ws_security: WebSocketSecurityService` and `ws_context: WebSocketSecurityContext`, and `run_session(self, ws: WebSocket, ...)` both drives the graph and awaits the next client command mid-turn (`application/service.py:246-266`, `:333-340`). SSE has no mid-turn command channel, so this is the real refactor boundary — not the event mapper.
- `_map_event_to_frame()` (`service.py:482`) is a **static, transport-shaped** mapper: LangGraph event → `WSOutbound` frame. It is the only place where graph events become wire facts.
- `PreparedTranscriptExport` (`transcript/export.py`) already streams rows in `(turn_seq, ordinal)` order with a keyset cursor `(turn_seq, ordinal, row_id)` and enforces ownership plus schema gating in `prepare()`.

Constraints that shape this design:

- The WebSocket path is in active use by an 86-task in-flight change (`harden-agent-saul-evidence-pipeline`) that edits the same files. This change must be additive and must not reshape WS frames.
- Authentication is a bearer access token from the `Authorization` header or the `access_token` cookie (`auth/http/dependencies.py:258`); tenant is selected by `x-tenant-id`, `?tenant_id=`, or a namespaced subprotocol, and is only a selector, never authority.
- Streaming responses already have a precedent that bypasses the shared envelope: the NDJSON export returns `StreamingResponse` with `response_model=None` and renders preflight failures through `render_result`.
- There is no Node toolchain in the repository; the deploy target is Python-only.

## Goals / Non-Goals

**Goals:**

- One graph-event projection, two transports, with a mechanical contract test proving they cannot drift.
- A run that survives client disconnects, page reloads and reverse proxies while paused at a human gate.
- Interrupt payloads whose accepted shape is generated from the Pydantic domain models rather than hand-maintained.
- A console that adds zero new identity, authority or transcript-of-record.
- Additive deployment: feature-flagged, no data migration, rollback by flag.

**Non-Goals:**

- Retiring or reshaping `WS /agent-saul/ws/{thread_id}`.
- A model/provider picker, document upload, or general chat features in the console.
- Running a Node service, or adopting CopilotKit Cloud / Assistant Cloud persistence.
- Changing graph topology, checkpoint retention, budget accounting or thread-lock semantics.
- Protocol-level support for non-Agent-Saul agents.

## Decisions

### D1 — AG-UI 1.0 over HTTP+SSE; no AG-UI custom binding over WebSocket

AG-UI 1.0 defines HTTP+SSE and HTTP+Protobuf as transports and permits custom bindings, so keeping the WebSocket would be legal. We choose SSE anyway:

- CopilotKit and `@ag-ui/client` speak SSE out of the box; a custom binding forces us to ship a client transport adapter, which is the thing we are trying to delete.
- The AG-UI interrupt model is *terminal run + resume as a new run*, which is naturally an HTTP request, not a socket. Holding a socket across a human pause is exactly the failure mode observed in the current design.
- SSE needs no `Upgrade` proxying, which removes the `X-Forwarded-*` trust subtleties already encoded in `get_websocket_url()`.

*Alternatives:* AG-UI over WebSocket (keeps `websocket_security` presence registration verbatim, but needs a custom client binding); bespoke JSON-over-SSE (no ecosystem, re-invents the spec).

**Consequence:** presence/audit currently piggybacks on `websocket_security.register_connection`. SSE runs have no socket to register; run start/finish is already recorded as audit facts, and any additional lease requirements are deferred (see Open Questions).

### D2 — Extract a transport-neutral projection, a transport port, and a start/resume phase split

Three related moves, landed together because each depends on the next:

1. **Projection.** Move `_map_event_to_frame` into `application/projection/` as a pure function `project(event, step) -> ProjectionEvent`, a **closed union** of graph facts (token delta, step started/finished, status change, state patch, interrupt opened, delegated subagent start/finish, run terminal). Both encoders end with `assert_never` over that union, so a new fact cannot be silently dropped by one transport.
2. **Port.** Introduce `application/ports/sink.py` with a `RunSink` Protocol (emit a projection event, close with a reason, and an optional presence hook). `AgentSaulService` depends on the Protocol; `WebSocketSecurityService` is no longer a constructor parameter at the service boundary — the WS adapter wraps it.
3. **Phases.** Split `run_session()` into `start_run()` (acquire lock, hydrate, drive the graph until terminal or interrupt) and `resume_run(command)` (re-authorize, resume from checkpoint, drive until terminal or interrupt). The WS adapter composes them in a loop, preserving today's await-resume behaviour, close codes and lock/T3 ordering. The SSE endpoint calls exactly one phase per HTTP request, which is what makes terminal-run interrupts natural.

*Alternatives:* duplicate the mapping in the AG-UI module (drifts immediately); have AG-UI consume `WSOutbound` frames and translate (couples the new protocol to the deprecated shape); keep `run_session` intact and run the graph inside the SSE generator's task while polling Redis for a resume command (re-creates the socket model with none of its guarantees).

**Consequence:** this is the largest edit in the change and the only one that touches the working WS path. It is protected by a golden-frame replay test: one recorded `astream_events` capture is pushed through the refactored path and compared to the frames the current code emits, including close ordering.

### D3 — Model interrupts as terminal runs, and key resume on `(thread_id, interrupt_id)`

`hitl_interrupt` becomes `RUN_FINISHED { outcome: { type: "interrupt", interrupts: [...] } }`. Each `Interrupt` carries:

| Field | Source |
| --- | --- |
| `id` | stable per interrupt occurrence, derived from `thread_id` + checkpoint step + interrupt index |
| `reason` | the `HITLInterruptType` value (`plan_approval`, `human_review_required`, `clarification_needed`, `ocr_reupload`) |
| `message` | the interrupt's human-readable prompt |
| `responseSchema` | JSON Schema generated from the payload model for that interrupt type |
| `metadata` | interrupt-specific non-authoritative hints (for example a UI hint for the plan editor) |

Resume is a new run whose `RunAgentInput.resume[]` addresses every open interrupt; the encoder validates the payload against the same schema before constructing `Command(resume=...)`. Authorization is re-resolved inside `AgentSaulService` exactly as the WebSocket path does, so no new authority path is introduced.

*Alternatives:* keep the socket open and await a frame (current behaviour); accept a partial resume set (leaves the graph in an ambiguous pause).

### D4 — Schema-driven contract, hand-built editors

`responseSchema` is generated from the existing Pydantic models (`PlanStep`, the human-review override shape) via `model_json_schema()`. The generic renderer is the fallback; the two gates that matter get hand-written components:

- plan approval: ordered step table with edit / remove / add / reorder, plus reject-with-feedback;
- human review: findings with original-vs-override comparison and per-finding disposition.

*Alternatives:* generate every form (degrades the clause-override UX that justifies the product); hand-write every payload contract (drifts from the domain models).

### D5 — CopilotKit's Python runtime, static SPA, no Node

The CopilotKit runtime is mounted in FastAPI and the console is a Vite-built static bundle served from the API origin at `/console`. This keeps one deploy, one auth plane, and one set of secrets.

*Alternatives:* Next.js + Node runtime (adds a second TLS/auth/log surface and a second lockfile ecosystem in a Python-only repo); `@ag-ui/client`'s `HttpAgent` with CopilotKit used only as components (the documented fallback if Python runtime parity blocks the spike).

**Go/no-go:** a one-day spike must prove the Python runtime can serve the AG-UI endpoint the console needs. If it cannot, fall back to `HttpAgent` + CopilotKit components without changing the protocol or the specs.

### D6 — Same-origin cookie authentication, with explicit CSRF posture

The console is served from the API origin, so the existing `access_token` cookie authenticates both page and stream requests, and `SameSite=Lax` blocks cross-site POSTs. Run creation is a `POST` and therefore state-changing, so the endpoint additionally requires an `Origin`-matching check and a custom `X-Agent-Console` header; a missing or mismatched pair is refused. Bearer-header auth remains supported for tooling.

*Alternatives:* separate origin + CORS + `SameSite=None` (widens cookie exposure and reintroduces the WS header problem); short-lived stream ticket exchange (an extra round trip and a new token type for no additional guarantee here).

### D7 — Transcript reads reuse the export pipeline

The JSON read endpoint wraps `PreparedTranscriptExport` with a different transport adapter: same `prepare()` preflight, same keyset cursor, same redaction representation, JSON instead of NDJSON. Pagination adds a `limit` and returns `nextCursor` / `hasMore`.

*Alternatives:* a separate query path (re-implements ordering and redaction, and they will disagree eventually).

### D8 — Streaming responses bypass the envelope; preflight failures do not

The AG-UI endpoint returns `EventSourceResponse` (add `sse-starlette` as a direct dependency; it is already resolved transitively) once the run is authorized and started. Anything that fails **before** the first event — auth, tenant, document scope, malformed input — is answered with the normal `APIResponse` envelope via `render_result`/`http_error`. After the first event, failures become `RUN_ERROR`. This mirrors the export endpoint's existing split and keeps the envelope contract test green.

### D9 — Keep the WebSocket endpoint, mark it deprecated

The WS endpoint stays, unchanged on the wire, documented as deprecated in `docs/architecture.md` and the manual testing plan. Its removal is a follow-up change after `harden-agent-saul-evidence-pipeline` lands.

*Alternatives:* cut over now (forces a rewrite of the in-flight change and removes the only working client).

### D10 — Cancellation is an SSE abort, and it cancels the run

AG-UI 1.0 defines no cancel event, so cancellation is expressed by aborting the HTTP request. The endpoint MUST treat an abort after `RUN_STARTED` as a cancel: stop the graph at the next safe boundary, release the thread lock, record a cancelled terminal state in the transcript, and leave the checkpoint resumable. An abort before the first event starts no run at all.

*Alternatives:* detach and let the graph finish in the background (burns budget on legal runs the user walked away from, and leaves a terminal state nobody observed); treat abort as a Transient error and let the checkpoint stay mid-node (leaves the lock to TTL expiry).

### D11 — Duplicate run submissions are rejected, not coalesced

AG-UI messages carry a `Message.id`. A run POST whose `(threadId, messageId)` matches an in-flight run is refused with a stable `RUN_ALREADY_ACTIVE` code instead of starting a second graph execution. A retry after a dropped stream is expressed as a resume for the same `threadId`, not as a repost.

*Alternatives:* coalesce onto the existing run (requires a fan-out registry and changes transcript semantics); rely on `ThreadLock` alone (the lock rejects the second writer but surfaces as an opaque infrastructure error).

### D12 — Typed review overrides, added permissively for wire compatibility

`WSResumeMessage.overrides` is `list[dict[str, Any]]` today (`api/dto.py:75`), which cannot produce a meaningful `responseSchema`. Add a `ReviewOverride` model (target reference, kind, original value, override value, rationale) in `domain/state.py` and use it for both transports. To avoid breaking existing WebSocket clients, the model validates permissively (optional fields, unknown keys preserved) while the AG-UI `responseSchema` advertises the strict shape.

*Alternatives:* keep the open dict and publish `{"type": "object"}` (the console then cannot validate, and the editor is hand-written against nothing); tighten the WS type immediately (breaks any client sending an ad-hoc dict).

### D13 — Engineering contracts that apply to every new backend module

- Service and repository methods return `Result[T, FeatureError]` per `RESULT-PATTERN.md`; new failure modes join `features/agent_saul/domain/errors.py` and every `match` over that union closes with `assert_never`.
- Pre-stream failures answer through `render_result` / `http_error`; failures after the first event become `RUN_ERROR`. No raising across the HTTP path.
- `ProjectionEvent` is a closed union; both encoders close with `assert_never`; a golden-frame test pins the WebSocket bytes before and after the refactor.
- `sse-starlette` becomes a direct dependency (already resolved transitively) and is used through one adapter module, so the SDK surface stays replaceable.
- No `except Exception` in the new packages except named degradation boundaries with `add_note` and a `# noqa: BLE001 — reason`.

### D14 — The console is a review workspace, not a chat transcript skin

The UI uses three explicit surfaces: session history, the transcript/decision workspace, and a collapsible progress rail. On narrow screens these become focused panels or drawers rather than compressed columns. The state model preserves selection, scroll anchor, draft, and an in-progress decision across layout changes, but never writes content or identifiers to browser storage.

Assistant output uses one sanitized Markdown adapter with raw HTML disabled and unsafe URL schemes rejected. Streaming scroll follows only when the reviewer is already near the end; otherwise a “jump to latest” control accumulates an unread count. The composer submits on `Ctrl+Enter` / `Meta+Enter`, keeps failed drafts in memory, and creates no duplicate optimistic turn.

Session summaries carry status, updated time and an attention-required reason so history can be filtered before pagination without reading transcript content. Theme, reduced-motion and panel-size preferences are the only locally persisted UI state.

*Alternatives:* a generic CopilotKit chat layout (does not expose long-running legal review state or gates clearly); persist the whole client store for convenient reload (creates a second transcript store and weakens erasure); render raw model HTML (turns generated content into an XSS boundary).

### D15 — SSE runs use lifecycle audit facts, not connection presence leases

The existing WebSocket presence lease answers “which sockets are connected,” while an SSE run is a finite request that can terminate at an interrupt and resume later. Registering it as socket presence would create false online state during human review. The AG-UI path therefore records structured run-start, interrupt, resume, cancel and terminal facts with correlation identifiers, and exposes active-run metrics; it does not create a WebSocket presence lease.

*Alternatives:* reuse the WebSocket presence sorted set (misstates a paused reviewer as connected and couples the new adapter to socket infrastructure); add a separate SSE lease (duplicates active-run state without an operational consumer).

## Risks / Trade-offs

| Risk | Mitigation |
| --- | --- |
| [Service split regresses WebSocket behaviour — close codes, frame order, lock release, T3 abort ordering] → golden-frame replay test from a recorded run, landed before the AG-UI endpoint is enabled; the WS adapter composes start+resume in the current loop's order. |
| [Interrupt id is not reproducible after a process restart if `aget_state()` does not expose a stable step/index] → the spike confirms the source; fallback is to persist the interrupt id on the checkpoint record and read it back at resume. |
| [Projection drift between transports] → one pure `project()` function plus a fixture-driven test that replays a recorded `astream_events` capture through both encoders and diffs node order, statuses, interrupt descriptors and terminal outcome. |
| [CopilotKit Python runtime parity] → one-day spike with a documented fallback to `@ag-ui/client` `HttpAgent` + CopilotKit components; the protocol and specs do not change either way. |
| [SSE buffering / idle cuts] → `X-Accel-Buffering: no` header from the endpoint, `proxy_buffering off` and a read timeout above the longest legal run in the proxy config, plus a heartbeat interval and a reconnect test. |
| [Interrupt id collisions across checkpoints] → id includes thread id, checkpoint step and interrupt index; stale ids are rejected against the live `aget_state()` interrupt set. |
| [Merge conflict with the in-flight hardening change] → additive files where possible, projection extraction lands first and alone, and no WS frame shape changes; rebase before apply if that change lands first. |
| [Cookie-authenticated POST is CSRF-reachable] → `SameSite=Lax` + `Origin` check + required custom header, tested with a cross-origin form post. |
| [Console becomes a second chat store] → no durable client persistence; transcript is always re-read from the API; caches cleared on logout/tenant switch; tested. |
| [Schema-generated forms regress the review UX] → schema is the contract, not the renderer; the two critical gates are hand-built and covered by interaction tests. |
| [New third-party dependencies churn] → exact pins for `ag-ui-protocol`, `copilotkit`, and the CopilotKit React packages; a single adapter module isolates the SDK surface. |
| [Generated legal content becomes an XSS vector] → one sanitized Markdown adapter, raw HTML disabled, unsafe schemes rejected, CSP enforced, and adversarial rendering tests. |
| [Slow SSE clients accumulate unbounded events] → ordered bounded buffering, a measurable overflow path, safe cancellation, and resumable checkpoints. |
| [Streaming auto-scroll makes evidence review unusable] → scroll anchoring follows only near the end; otherwise show unread count and “jump to latest”. |

## Migration Plan

No data migration: the change adds endpoints, a static bundle and one flag.

1. **Spike (1 day, throwaway branch).** Prove the Python CopilotKit runtime serves the AG-UI endpoint and that a React page can stream from it. Go/no-go for D5; record the pin versions.
2. **Projection + port + phase split.** Land `application/projection/`, `application/ports/sink.py`, and the `start_run`/`resume_run` split; rewire the WS adapter to compose the phases. The golden-frame replay test must pass and the WS integration suite must be unchanged before this merges. Deployable alone.
3. **AG-UI endpoint.** Add `POST /api/v2/agent-saul/agui/runs` behind `AGENT_CONSOLE_ENABLED` (default off). Auth, scope narrowing, interrupt/resume and the SSE framing tests land here.
4. **Transcript read endpoints.** Add the two GETs behind the same flag; parity test against the NDJSON export.
5. **Console.** Build `web/`, mount at `/console` behind the flag; serve only when built; keep `/api`, `/swagger.json`, `/metrics`, `/health` and `/api-docs` precedence intact.
6. **Deploy order:** backend with flag off → proxy SSE config → static bundle → flag on for internal users → widen. Rollback is flag-off; no schema, no data, no WS change.

**Rollback:** disable the flag and unmount `/console`. The WS path and every existing endpoint behave exactly as before, because the only touched existing code is the mapper's location.

## Open Questions

- Which telemetry naming the new endpoint uses (`saul.agui.run.*` versus reusing existing `saul_*` event names) — cosmetic, decided during implementation.
- Whether the console ever needs a locale switch. Not required by these specs; deferred.
