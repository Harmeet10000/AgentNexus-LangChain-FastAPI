## 1. Spike and dependency pins

- [ ] 1.1 Spike the CopilotKit Python runtime mounted in FastAPI: serve one AG-UI SSE run that emits text, a state delta and an interrupt, and resume it from a minimal React page — verification: throwaway branch demo described in design.md, decision recorded as go/no-go for D5.
- [ ] 1.2 Record exact pins for `ag-ui-protocol`, `copilotkit` and the CopilotKit React packages in `pyproject.toml` and `web/package.json` — verification: `uv lock` and the web lockfile contain exact versions, no ranges.
- [ ] 1.3 Confirm `aget_state()` exposes a stable step/index at interrupt time for interrupt-id derivation; if not, record the checkpoint-record fallback in design.md — verification: a written note with the observed state shape.

## 2. Transport port, projection and phase split (WebSocket parity first)

- [ ] 2.1 Add `application/projection/events.py` with the closed `ProjectionEvent` union (`token_delta`, `step_started`, `step_finished`, `status_changed`, `state_patch`, `interrupt_opened`, `subagent_started`, `subagent_finished`, `run_terminal`) — verification: `ty check` accepts the union and a unit test asserts each member round-trips.
- [ ] 2.2 Move the `_map_event_to_frame` logic into `project(event, step) -> ProjectionEvent | None`, preserving the untyped LangGraph event guards — verification: unit tests ported from the existing mapper cases pass.
- [ ] 2.3 Add the WebSocket encoder `ProjectionEvent -> WSOutbound` and delete the old mapper; close the union with `assert_never` — verification: `ty check` reports no missing case; no call site references the removed mapper.
- [ ] 2.4 Capture one real `astream_events` fixture and add a golden-frame test asserting the refactored WS path emits the same frames and close order as the pre-refactor recording — verification: test fails when a frame field is altered deliberately, passes otherwise.
- [ ] 2.5 Add `application/ports/sink.py` with the `RunSink` Protocol (emit event, close with reason, optional presence hook) — verification: the service imports the Protocol and no longer imports `WebSocket*` types.
- [ ] 2.6 Split `run_session()` into `start_run()` and `resume_run(command)`, preserving lock acquisition, owner gate, reconcile, hydrate, T3 abort ordering and lock release — verification: WS integration suite passes unmodified.
- [ ] 2.7 Rewire `api/websocket/endpoint.py` to build a sink adapter around `websocket_security` and compose start+resume in the current loop order — verification: golden-frame test and close-code tests pass.
- [ ] 2.8 Remove `ws_security` / `ws_context` from the service constructor in favour of the sink port — verification: constructor signature contains no WebSocket type; `ty check` clean.
- [ ] 2.9 Run `uv run ruff format src/ tests/`, `uv run ruff check src/ tests/`, `uv run ty check src/` and `uv run pytest` — verification: all green with no new suppressions.

## 3. AG-UI run endpoint

- [ ] 3.1 Add `api/agui/dto.py` with the accepted `RunAgentInput` subset (`threadId`, `messages`, `state`, `resume[]`) using `extra="forbid"` — verification: unknown-field and missing-scope requests produce validation errors.
- [ ] 3.2 Add `api/agui/router.py` with `POST /api/v2/agent-saul/agui/runs` behind `AGENT_CONSOLE_ENABLED` (default off), returning `EventSourceResponse` — verification: flag-off returns 404; flag-on streams.
- [ ] 3.3 Implement `api/agui/encoder.py` mapping `ProjectionEvent` to `ag_ui.core` events, closed with `assert_never` — verification: every union member has a mapping test.
- [ ] 3.4 Emit `RUN_STARTED` → content events → `RUN_FINISHED { outcome: { type: "success" } }` for a non-interrupt run — verification: an end-to-end test asserts exactly one terminal event and no events after it.
- [ ] 3.5 Emit `STATE_SNAPSHOT` at run start and `STATE_DELTA` (RFC 6902) for status, plan and step changes — verification: applying the deltas to the snapshot reconstructs the final state in a test.
- [ ] 3.6 Emit `ACTIVITY_*` and one `STEP_STARTED`/`STEP_FINISHED` pair per top-level node — verification: a run with N nodes yields N matched pairs in order.
- [ ] 3.7 Emit `SUBAGENT_STARTED` / `SUBAGENT_FINISHED` / `SUBAGENT_ERROR` for delegated work with distinct ids — verification: a delegation run attributes specialist output to the subagent id, not the parent.
- [ ] 3.8 Route pre-stream failures through `render_result` / `http_error` and post-first-event failures to `RUN_ERROR` — verification: an auth failure returns the standard envelope; a mid-stream failure returns only `RUN_ERROR`.
- [ ] 3.9 Set SSE headers (`text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`), emit `: ping` comment heartbeats and close after the terminal event — verification: a test with an injected silent node observes comment heartbeats and no non-protocol event types.
- [ ] 3.10 Authorize the POST with the existing cookie/header token extraction plus tenant selector, and require a matching `Origin` and an `X-Agent-Console` header — verification: a cross-origin form POST without the header is refused; same-origin console request succeeds without CORS preflight.
- [ ] 3.11 Narrow the document scope server-side through `DocumentAuthorizationDep` before the run starts — verification: a request naming an out-of-scope document id never reads it and the run scope excludes it.
- [ ] 3.12 Implement cancellation on client abort: stop at the next safe boundary, release the lock, record a cancelled terminal state, keep the checkpoint resumable — verification: abort-mid-run, resume-after-cancel and abort-before-start tests.
- [ ] 3.13 Reject a duplicate `(threadId, messageId)` run with `RUN_ALREADY_ACTIVE` and start no second execution — verification: a concurrent-duplicate test observes one graph run and one refusal.
- [ ] 3.14 Close any open assistant message with `TEXT_MESSAGE_END` before the interrupt terminal event and open a new message id on resume — verification: message-lifecycle test asserts no content follows a closed id.
- [ ] 3.15 Add the new failure modes to `features/agent_saul/domain/errors.py` and close every dispatch with `assert_never` — verification: `ty check` proves exhaustiveness; no `except Exception` added outside named degradation boundaries.
- [ ] 3.16 Bound SSE buffering and define the slow-consumer overflow path without reordering events — verification: a stalled-consumer test observes bounded memory, lock release, cancellation telemetry and a resumable checkpoint.

## 4. Interrupt and resume lifecycle

- [ ] 4.1 Derive a stable interrupt id from thread, checkpoint step and interrupt index (or the 1.3 fallback) — verification: the same paused checkpoint yields the same id across a process restart.
- [ ] 4.2 Emit `RUN_FINISHED { outcome: { type: "interrupt", interrupts: [...] } }` with `id`, `reason`, `message`, `responseSchema` and `metadata` — verification: one test per interrupt type asserts every field is present and `reason` matches the Agent Saul interrupt enum.
- [ ] 4.3 Add a permissively-validated `ReviewOverride` model to `domain/state.py` and use it in `WSResumeMessage.overrides` without breaking existing WS payloads — verification: a legacy ad-hoc override dict still validates; an AG-UI-shaped override validates strictly.
- [ ] 4.4 Generate each `responseSchema` from the same Pydantic model the validator uses and publish it on the interrupt — verification: a property test asserts schema-accepted payloads validate and schema-rejected payloads are refused.
- [ ] 4.5 Implement the resume run: require `resume[]` to address every open interrupt, reject stale or foreign ids, re-resolve live authority, then build `Command(resume=...)` — verification: incomplete, stale, foreign and cross-tenant resume tests.
- [ ] 4.6 Test authority revoked between interrupt and resume — verification: the resume is refused and the graph neither finalizes nor persists.
- [ ] 4.7 Test disconnect-during-interrupt then resume — verification: the resumed stream emits `STATE_SNAPSHOT` before any further event and the graph continues from the checkpoint.

## 5. Transcript read API

- [ ] 5.1 Add `GET /api/v2/agent-saul/sessions` returning tenant-scoped, newest-first, paginated summaries with no transcript content — verification: two-tenant test shows only authorized sessions.
- [ ] 5.2 Add `GET /api/v2/agent-saul/sessions/{session_id}/transcript` reusing `PreparedTranscriptExport` with a JSON adapter, `limit`, `nextCursor` and `hasMore` — verification: page-one and page-two tests use the returned cursor.
- [ ] 5.3 Return redacted rows as `content: null` and `redacted: true` — verification: an erased turn appears in position and offers no reconstructed body.
- [ ] 5.4 Enforce authorization and non-disclosure: cross-tenant returns not-found, same-tenant without authority is refused — verification: both cases assert no metadata leaks.
- [ ] 5.5 Gate on supported transcript schema version before returning any row — verification: an unsupported-version session fails as a whole.
- [ ] 5.6 Assert JSON read and NDJSON export return identical rows and ordering — verification: parity test over one populated session.
- [ ] 5.7 Assert keyset stability when rows are appended between pages — verification: no duplicates and no gaps across pages.
- [ ] 5.8 Add server-side session filters for status, attention-required state and authorized document id; include last-updated time and safe interrupt reason in summaries — verification: filters are applied before pagination and return no transcript or response payload.

## 6. Console application

- [ ] 6.1 Scaffold `web/` (Vite, React 19, TypeScript, Tailwind) with lockfile, build script and lint/typecheck scripts — verification: `npm ci && npm run build` produces `web/dist`.
- [ ] 6.2 Wire the CopilotKit provider to `POST /api/v2/agent-saul/agui/runs` and render streamed assistant text — verification: a manual run shows tokens arriving incrementally.
- [ ] 6.3 Implement session preflight and the document-scope picker, showing ingestion readiness and blocking non-ready documents — verification: `not_ready` and `failed` documents cannot start a run.
- [ ] 6.4 Render a progress rail from step and activity events (active node, completed steps, status) — verification: a long node keeps the rail live with no text output.
- [ ] 6.5 Build the plan approval editor: ordered steps, edit/remove/add/reorder, approve, reject-with-feedback, schema validation, single-flight submission — verification: interaction tests cover modify-then-approve, invalid plan blocked, double click sends once.
- [ ] 6.6 Build the human review gate: findings with evidence, per-finding overrides showing original and override together, approve/reject — verification: override payload matches the interrupt schema and the run stays pending until confirmed.
- [ ] 6.7 Implement clarification answers and the OCR re-upload handoff (link only, no upload) — verification: OCR interrupt renders guidance and starts no upload.
- [ ] 6.8 Implement history from the session and transcript endpoints with pagination and erased-row rendering — verification: reopened session renders in order and loads further pages.
- [ ] 6.9 Implement cancel, resume-after-drop and retryable-vs-terminal error handling — verification: a dropped stream offers resume keyed on `threadId`; a non-retryable code stops and explains.
- [ ] 6.10 Route unauthorized responses to sign-in without retry loops, and clear all local state on logout or tenant switch — verification: 401 mid-run redirects once; tenant switch shows an empty thread list.
- [ ] 6.11 Accessibility pass: keyboard reachability for every decision control, pending/error state exposed to assistive technology, no colour-only approve/reject — verification: keyboard-only walkthrough plus an automated accessibility scan.
- [ ] 6.12 Assert no transcript content is persisted in `localStorage`/`sessionStorage` — verification: a test inspects storage after a full session.
- [ ] 6.13 Add the single sanitized Markdown/citation renderer with raw HTML disabled, safe external-link attributes and plain-text copy — verification: adversarial HTML, URL-scheme, partial-stream and clipboard tests pass.
- [ ] 6.14 Add the multiline composer, `Ctrl+Enter`/`Meta+Enter` submission, recoverable in-memory draft, anchored streaming scroll, unread count and “jump to latest” — verification: interaction tests cover a failed send and reading earlier evidence during token arrival.
- [ ] 6.15 Add status/document history filters, last-updated labels and text-plus-icon attention badges while preserving selection across pagination — verification: filtering pending reviews and loading another page do not close the active transcript.
- [ ] 6.16 Add the responsive review workspace, focused narrow-screen drawers, distinct loading/empty/offline/reconnecting states, theme and reduced-motion preferences — verification: viewport-change tests preserve an unsent decision and storage contains only allowlisted presentation preferences.
- [ ] 6.17 Add browser-level acceptance coverage for start, stream, cancel/resume, plan edit, review override, reload-at-interrupt, history pagination, tenant switch and keyboard-only operation — verification: the built SPA passes the scripted suite against a deterministic backend fixture.

## 7. Serving, configuration and proxy

- [ ] 7.1 Mount `web/dist` at `/console` with SPA fallback, preserving precedence for `/api`, `/swagger.json`, `/metrics`, `/health` and `/api-docs` — verification: route-precedence test for each reserved path.
- [ ] 7.2 Wire `AGENT_CONSOLE_ENABLED` through settings and application setup, default off — verification: flag-off serves no console, AG-UI endpoint, session-list endpoint or JSON transcript endpoint.
- [ ] 7.3 Document and apply proxy settings for SSE: buffering off, read timeout above the longest run, forwarded-header trust unchanged — verification: a proxied run longer than the old idle timeout completes.
- [ ] 7.4 Set a CSP and static-asset cache policy for the console — verification: the built console loads under the CSP with no inline-script violations.
- [ ] 7.5 Add a CI job for the web build, lint, typecheck, component tests and deterministic browser acceptance suite alongside the existing Python jobs — verification: CI fails on a deliberate type error and on a broken critical interaction in `web/`.

## 8. Observability, documentation and rollout

- [ ] 8.1 Emit structured logs for run start, interrupt, resume, cancel and terminal outcome with thread, tenant and correlation ids — verification: one log line per transition in a scripted run.
- [ ] 8.2 Add metrics for run duration, interrupt wait time, resume refusals by reason and stream aborts — verification: metrics appear at `/metrics` after a scripted run.
- [ ] 8.3 Update `docs/architecture.md`, `docs/manual-testing-plan.md` and the README with the AG-UI surface, the console and the deprecation note for WS — verification: docs describe every new endpoint and the deprecation.
- [ ] 8.4 Add the cross-transport equivalence test to CI from a recorded event stream — verification: deliberately altering one encoder fails the test.
- [ ] 8.5 Run the internal rollout and a flag-off rollback drill — verification: console and AG-UI endpoint disappear, WS run still completes, no data change.
- [ ] 8.6 Run `openspec validate ag-ui-copilotkit-console --type change --strict`, validate `contracts/asyncapi.yaml`, and verify every discovery event maps to a spec scenario and contract message — verification: all checks exit zero and the traceability review has no orphan event or task.

## Task Traceability and Verification Matrix

The `Implements` and `verify` entries below are part of their corresponding checkbox task. `sendRunEvents` and `receiveRunCommand` refer to `contracts/asyncapi.yaml`; a dash means the task has no async channel.

| Task | Implements (requirement / scenario) | Channel | verify |
|---|---|---|---|
| 1.1 | Run stream endpoint / First run; Interrupt lifecycle / Plan approval pause | receiveRunCommand, sendRunEvents | `test -f openspec/changes/ag-ui-copilotkit-console/spike-report.md` — go/no-go and observed run recorded |
| 1.2 | Run stream endpoint / First run | receiveRunCommand, sendRunEvents | `uv lock --check && npm --prefix web ci` — exact locked versions resolve |
| 1.3 | Interrupt lifecycle / Stale or foreign interrupt id | sendRunEvents | `rg 'interrupt.*(step|checkpoint|index)' openspec/changes/ag-ui-copilotkit-console/design.md` — observed source or fallback recorded |
| 2.1 | Transport equivalence / Recorded run replayed | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/application/projection -q` — projection union tests pass |
| 2.2 | Transport equivalence / Recorded run replayed | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/application/projection -q` — mapper cases pass |
| 2.3 | Transport equivalence / Recorded run replayed | sendRunEvents | `uv run ty check src/app/features/agent_saul` — encoder union exhaustive |
| 2.4 | Transport equivalence / Recorded run replayed | sendRunEvents | `uv run pytest tests/contract/agent_saul/test_transport_equivalence.py -q` — golden frames match |
| 2.5 | Transport equivalence / Recorded run replayed | sendRunEvents | `uv run ty check src/app/features/agent_saul` — service has no transport types |
| 2.6 | Run lifecycle / Successful completion; Interrupt lifecycle / Plan approval pause | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/application -q` — phase lifecycle passes |
| 2.7 | Transport equivalence / Recorded run replayed | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/api/websocket -q` — frame and close parity passes |
| 2.8 | Transport equivalence / Recorded run replayed | sendRunEvents | `uv run ty check src/app/features/agent_saul` — constructor boundary clean |
| 2.9 | Transport equivalence / Recorded run replayed | sendRunEvents | `uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest` — all gates pass |
| 3.1 | Run stream endpoint / Empty run input refused | receiveRunCommand | `uv run pytest tests/unit/features/agent_saul/api/agui/test_dto.py -q` — validation cases pass |
| 3.2 | Run stream endpoint / First run | receiveRunCommand, sendRunEvents | `uv run pytest tests/unit/features/agent_saul/api/agui/test_router.py -q` — flag and stream cases pass |
| 3.3 | Run lifecycle / Successful completion | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/api/agui/test_encoder.py -q` — all projection members map |
| 3.4 | Run lifecycle / Successful completion | sendRunEvents | `uv run pytest tests/integration/features/agent_saul/test_agui_run.py -q` — one terminal event |
| 3.5 | State projection / Status transition | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/api/agui/test_state_projection.py -q` — patches reconstruct state |
| 3.6 | Activity and step projection / Node progress observable | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/api/agui/test_activity_projection.py -q` — pairs ordered |
| 3.7 | Subagent attribution / Specialist delegation | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/api/agui/test_subagents.py -q` — ids attributed |
| 3.8 | Run lifecycle / Terminal error | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/api/agui/test_errors.py -q` — pre/post stream errors differ correctly |
| 3.9 | Stream transport safety / Long silent node | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/api/agui/test_sse.py -q` — headers and comment heartbeat pass |
| 3.10 | Stream transport safety / Authenticated from console origin | receiveRunCommand | `uv run pytest tests/unit/features/agent_saul/api/agui/test_security.py -q` — CSRF and same-origin cases pass |
| 3.11 | Authorization parity / Client widens document scope | receiveRunCommand | `uv run pytest tests/unit/features/agent_saul/api/agui/test_scope.py -q` — unauthorized docs unread |
| 3.12 | Run cancellation / all scenarios | receiveRunCommand, sendRunEvents | `uv run pytest tests/integration/features/agent_saul/test_agui_cancel.py -q` — abort lifecycle passes |
| 3.13 | Duplicate run submission / Concurrent duplicate | receiveRunCommand | `uv run pytest tests/integration/features/agent_saul/test_agui_idempotency.py -q` — one graph run |
| 3.14 | Message identity / Resume opens a new message | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/api/agui/test_message_lifecycle.py -q` — ids close and rotate |
| 3.15 | Run lifecycle / Terminal error | receiveRunCommand, sendRunEvents | `uv run ty check src/app/features/agent_saul` — error dispatch exhaustive |
| 3.16 | Stream transport safety / Slow consumer exceeds stream bound | sendRunEvents | `uv run pytest tests/integration/features/agent_saul/test_agui_backpressure.py -q` — bounded cancellation passes |
| 4.1 | Interrupt lifecycle / Stale or foreign interrupt id | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/api/agui/test_interrupt_ids.py -q` — restart stability passes |
| 4.2 | Interrupt lifecycle / Plan approval pause | sendRunEvents | `uv run pytest tests/unit/features/agent_saul/api/agui/test_interrupts.py -q` — all descriptors complete |
| 4.3 | Interrupt response schema / Untyped payload not exposed | receiveRunCommand | `uv run pytest tests/unit/features/agent_saul/domain/test_review_override.py -q` — legacy and strict shapes pass |
| 4.4 | Interrupt response schema / Plan payload schema matches validation | receiveRunCommand, sendRunEvents | `uv run pytest tests/property/test_interrupt_schemas.py -q` — schema and validator agree |
| 4.5 | Interrupt lifecycle / Incomplete resume refused | receiveRunCommand | `uv run pytest tests/integration/features/agent_saul/test_agui_resume.py -q` — invalid resume set refused |
| 4.6 | Authorization parity / Role revoked before resume | receiveRunCommand | `uv run pytest tests/integration/features/agent_saul/test_agui_resume_authority.py -q` — revoked authority blocks transition |
| 4.7 | Interrupt lifecycle / Resume after disconnect | receiveRunCommand, sendRunEvents | `uv run pytest tests/integration/features/agent_saul/test_agui_reconnect.py -q` — snapshot precedes continuation |
| 5.1 | Session listing / Caller lists own sessions | — | `uv run pytest tests/unit/features/agent_saul/api/test_session_query.py -q` — tenant pagination passes |
| 5.2 | Paginated transcript read / Read first page | — | `uv run pytest tests/unit/features/agent_saul/api/test_transcript_query.py -q` — cursors page correctly |
| 5.3 | Redaction represented / Erased turn remains visible | — | `uv run pytest tests/unit/features/agent_saul/api/test_transcript_query.py -q` — redaction preserved |
| 5.4 | Transcript authorization / both scenarios | — | `uv run pytest tests/unit/features/agent_saul/api/test_transcript_query_security.py -q` — disclosure cases pass |
| 5.5 | Schema gating / Unsupported schema version | — | `uv run pytest tests/unit/features/agent_saul/api/test_transcript_query.py -q` — whole request fails |
| 5.6 | Paginated transcript read / Export and read agree | — | `uv run pytest tests/contract/agent_saul/test_transcript_parity.py -q` — rows identical |
| 5.7 | Pagination stability / Rows appended between pages | — | `uv run pytest tests/unit/features/agent_saul/api/test_transcript_pagination.py -q` — no gap or duplicate |
| 5.8 | Session listing / Filter sessions requiring attention | — | `uv run pytest tests/unit/features/agent_saul/api/test_session_query.py -q` — pre-page filters and safe summaries pass |
| 6.1 | Document-scoped session start / Start authorized document | — | `npm --prefix web ci && npm --prefix web run build` — static build succeeds |
| 6.2 | Streaming turn presentation / Progress visible | sendRunEvents | `npm --prefix web test -- --run stream` — incremental text renders |
| 6.3 | Document-scoped session start / Document not ready | — | `npm --prefix web test -- --run preflight` — non-ready docs blocked |
| 6.4 | Streaming turn presentation / Progress visible | sendRunEvents | `npm --prefix web test -- --run progress` — silent node remains visible |
| 6.5 | Plan approval gate / all scenarios | receiveRunCommand | `npm --prefix web test -- --run plan-approval` — editor interactions pass |
| 6.6 | Human review gate / both scenarios | receiveRunCommand | `npm --prefix web test -- --run human-review` — override contract passes |
| 6.7 | Clarification and re-upload gates / both scenarios | receiveRunCommand | `npm --prefix web test -- --run interrupt-forms` — bounded answer and handoff pass |
| 6.8 | History and transcript rendering / both scenarios | — | `npm --prefix web test -- --run history` — pages and erased rows render |
| 6.9 | Authentication, degradation and reconnect / Retryable interruption | receiveRunCommand, sendRunEvents | `npm --prefix web test -- --run recovery` — cancel/resume/error paths pass |
| 6.10 | Authentication, degradation and reconnect / Expired session | — | `npm --prefix web test -- --run auth` — one redirect and state clear |
| 6.11 | Accessible approval controls / Keyboard-only review | — | `npm --prefix web run test:a11y` — automated scan and keyboard assertions pass |
| 6.12 | Server-derived session state / Logout clears local state | — | `npm --prefix web test -- --run storage` — no content persists |
| 6.13 | Safe rich-content rendering / both scenarios | sendRunEvents | `npm --prefix web test -- --run rich-content` — adversarial render and copy pass |
| 6.14 | Composer and transcript navigation / both scenarios | receiveRunCommand, sendRunEvents | `npm --prefix web test -- --run composer-scroll` — draft and anchor behavior pass |
| 6.15 | Session findability / both scenarios | — | `npm --prefix web test -- --run session-list` — filtering and selection pass |
| 6.16 | Responsive workspace / both scenarios | — | `npm --prefix web test -- --run workspace` — layout state and storage allowlist pass |
| 6.17 | All console requirements / critical journeys | receiveRunCommand, sendRunEvents | `npm --prefix web run test:e2e` — deterministic browser suite passes |
| 7.1 | Document-scoped session start / Start authorized document | — | `uv run pytest tests/unit/test_console_static_mount.py -q` — reserved routes win |
| 7.2 | Run stream endpoint; Session listing; Paginated transcript read | receiveRunCommand, sendRunEvents | `uv run pytest tests/unit/test_agent_console_flag.py -q` — every surface hidden flag-off |
| 7.3 | Stream transport safety / Long silent node | sendRunEvents | `uv run pytest tests/integration/test_agent_console_proxy.py -q` — long proxied run completes |
| 7.4 | Safe rich-content rendering / Model output contains active content | sendRunEvents | `npm --prefix web run test:csp` — bundle loads without violations |
| 7.5 | All console requirements / critical journeys | receiveRunCommand, sendRunEvents | `npm --prefix web run lint && npm --prefix web run typecheck && npm --prefix web run test:e2e` — CI gates pass |
| 8.1 | Run lifecycle; Interrupt lifecycle; Run cancellation | receiveRunCommand, sendRunEvents | `uv run pytest tests/unit/features/agent_saul/test_agui_observability.py -q` — one log per transition |
| 8.2 | Run lifecycle; Interrupt lifecycle; Stream transport safety | receiveRunCommand, sendRunEvents | `uv run pytest tests/unit/features/agent_saul/test_agui_metrics.py -q` — metrics exported |
| 8.3 | All requirements / operator documentation | — | `rg 'AG-UI|/console|deprecated' README.md docs/architecture.md docs/manual-testing-plan.md` — surfaces documented |
| 8.4 | Transport equivalence / Recorded run replayed | sendRunEvents | `uv run pytest tests/contract/agent_saul/test_transport_equivalence.py -q` — contract fixture passes |
| 8.5 | Run stream endpoint / flag-off rollback | receiveRunCommand, sendRunEvents | `uv run pytest tests/integration/features/agent_saul/test_console_rollback.py -q` — console disappears and WS succeeds |
| 8.6 | All discovery/spec/contract requirements | receiveRunCommand, sendRunEvents | `openspec validate ag-ui-copilotkit-console --type change --strict` — exits zero; AsyncAPI and traceability checks pass |

## Task Dependency Graph

- **Wave 1 — feasibility and facts:** 1.1, 1.2, 1.3
- **Wave 2 — transport-neutral foundation:** 2.1, 2.2, 2.3, 2.4, 2.5
- **Wave 3 — service phase split and WS parity:** 2.6, 2.7, 2.8, 2.9
- **Wave 4 — AG-UI contract and safety:** 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16
- **Wave 5 — durable interrupts:** 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
- **Wave 6 — query surfaces:** 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
- **Wave 7 — console core:** 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12
- **Wave 8 — UI safety and quality of life:** 6.13, 6.14, 6.15, 6.16, 6.17
- **Wave 9 — serving and deployment:** 7.1, 7.2, 7.3, 7.4, 7.5
- **Wave 10 — operations and acceptance:** 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
