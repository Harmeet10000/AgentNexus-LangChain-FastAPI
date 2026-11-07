## Completeness

- [x] Every requirement in all three delta specs has at least one scenario.
- [x] Error paths are named for run input, resume mismatch, stale interrupt, unsupported schema, cross-tenant read, unauthorized resume and retryable stream failure.
- [ ] **Cancellation is only specified in the frontend.** `agent-console-frontend` says cancel stops presentation and leaves the session resumable, but no requirement defines what the *server* does when the SSE request is aborted mid-run. Budget, thread lock and transcript terminal state are unaddressed. → must fix.
- [ ] **No server-side idempotency requirement.** The frontend blocks double-click, but a network retry or a second tab can post the same run twice. `ThreadLock` exists but the AG-UI contract does not state the observable outcome. → must fix.
- [ ] **Assistant message identity across resume is unspecified.** On resume, it is undefined whether `TEXT_MESSAGE_CONTENT` continues the pre-interrupt `messageId` or opens a new one, so a client cannot render deterministically. → must fix.
- [x] Delta operations are correct: all three capabilities are new, so only `ADDED` is used, and each new capability has a `## Purpose`.

## Correctness

- [x] Requirements use SHALL/MUST and are observably testable; scenarios use `####` and WHEN/THEN.
- [x] Scenarios describe observable behaviour rather than restating requirements.
- [ ] **"Emit a heartbeat" is unsafe as written.** A heartbeat must be an SSE comment line (`: ping`), not a new AG-UI event type; AG-UI 1.0 defines 31 event types and consumers must accept unknown *well-formed* events, but inventing a 32nd type is a protocol violation. The requirement must say which. → must fix.
- [ ] **`responseSchema` for `human_review_required` has no model to generate from.** `WSResumeMessage.overrides` is `list[dict[str, Any]]` (`api/dto.py:75`). The design asserts "JSON Schema generated from the payload model for that interrupt type", which is untrue for overrides. Either introduce a typed override model or declare an open-object schema; the design must choose. → must fix.
- [x] `skip_specs` is not used; the change genuinely changes observable behaviour.

## Standards

- [x] Specs avoid class/function names and framework choices; those appear only in design.
- [ ] **Design does not state the Result/exception contract for the new endpoints.** Per `RESULT-PATTERN.md`, new service and repository methods must return `Result[T, FeatureError]`; pre-stream failures must go through `http_error()`/`render_result`, and the new error cases must join the feature's closed error union with `assert_never` coverage. → must fix.
- [ ] **`ProjectionEvent` is not declared as a closed union with `assert_never` at both encoders.** Without that, drifting encoders fail silently at runtime instead of at type-check time. → must fix.
- [ ] **No golden-frame test for the WebSocket adapter.** The proposal calls the mapper move an internal breaking change, but nothing pins the existing wire bytes. Extracting the mapper must be proven byte-identical for a recorded stream. → must fix.
- [x] Secrets handling is untouched; no new secret is introduced.
- [x] No `match`/`case` on `Success`/`Failure` is proposed.

## Risk

- [ ] **CRITICAL — the design understates the refactor.** `AgentSaulService` is constructed with `ws_security: WebSocketSecurityService` and `ws_context: WebSocketSecurityContext`, and `run_session(self, ws: WebSocket, ...)` (`application/service.py:246-266`, `:333-340`). The service is typed to the WebSocket transport *and* owns the interactive await-resume loop. Serving AG-UI therefore requires more than extracting `_map_event_to_frame`: the service needs a transport port, and `run_session` must split into a start phase (run to terminal or interrupt) and a resume phase (run from checkpoint), because SSE cannot await a mid-stream command. The design's claim that "the only change to existing code paths" is the mapper is false as written. → must fix; this is the largest task in the change and it is currently unsized.
- [x] Authorization parity is specified per run and per resume, including role revocation between interrupt and resume.
- [x] Redaction is represented, never reconstructed, and the JSON read reuses the export preflight.
- [x] CSRF is addressed with `SameSite=Lax`, an `Origin` check and a required custom header.
- [x] SSE buffering, idle timeouts and heartbeat are called out with concrete proxy mitigations.
- [ ] **Interrupt id staleness contract is asserted but not sourced.** The design derives the id from `thread_id` + checkpoint step + interrupt index; the review should confirm `aget_state()` exposes a stable step/index at resume time, otherwise the id is not reproducible after a process restart. → verify during the spike; record the answer.
- [x] The additive/flagged rollout and flag-off rollback are credible and require no data migration.
- [x] Durable decisions (protocol adoption, transport-neutral service port) are captured directly in the change-local `adr.md`.

## Verdict

**VERDICT:** `CHANGES-REQUESTED`

Must fix before `tasks.md`:

1. **Redesign D2 and the proposal's impact section** around a transport port plus a start/resume phase split of `AgentSaulService`; size it as the change's largest task, and state which WS behaviours must remain byte-identical.
2. Add a **run cancellation** requirement to `agent-ui-streaming`: SSE abort stops the graph at the next safe boundary, releases the thread lock, records a cancelled terminal state, and leaves the run resumable from checkpoint.
3. Add a **duplicate run / idempotency** requirement with a scenario for a retried POST on an active thread.
4. Add a **message identity across resume** requirement and scenario.
5. Correct the heartbeat requirement to an **SSE comment line**, not an AG-UI event.
6. Decide and record the **`human_review_required` resume schema** source — typed override model preferred — in design and in the streaming spec.
7. State the **Result/`http_error()` contract** for new endpoints and the **closed `ProjectionEvent` union with `assert_never`**, plus a **golden-frame WS test**.
8. Confirm during the spike that `aget_state()` yields a **reproducible interrupt id**; record the mechanism in design.

## Addendum — 2026-11-01, after fixes

1. **Addressed.** `proposal.md` now lists the transport port and the start/resume split in What Changes, Impact and Risks, and records it as the internal breaking change. `design.md` D2 was rewritten around projection + port + phase split, with the WS loop preserved by adapter composition.
2. **Addressed.** `agent-ui-streaming` gained `Requirement: Run cancellation` with three scenarios (mid-run abort, resumable after cancel, abort before start). `design.md` D10 records the decision and the rejected alternatives.
3. **Addressed.** `agent-ui-streaming` gained `Requirement: Duplicate run submission` with concurrent-duplicate and retry-after-drop scenarios. `design.md` D11 records `RUN_ALREADY_ACTIVE`.
4. **Addressed.** `agent-ui-streaming` gained `Requirement: Message identity across resume`; the frontend spec gained a matching resume scenario.
5. **Addressed.** `Requirement: Stream transport safety` now specifies a `: ping` SSE comment and explicitly forbids a new event type.
6. **Addressed.** `design.md` D12 adds a permissively-validated `ReviewOverride` model; `agent-ui-streaming` gained `Requirement: Interrupt response schema is authoritative`, including the rule that an untyped payload may not be published as an open object schema.
7. **Addressed.** `design.md` D13 states the Result/`http_error` contract, the closed `ProjectionEvent` union with `assert_never` at both encoders, the golden-frame WS test, and the single-adapter rule for `sse-starlette`.
8. **Recorded as a risk, to be verified in the spike.** `design.md` Risks now carries the interrupt-id reproducibility risk with a concrete fallback (persist the id on the checkpoint record). This is deferred rather than resolved because it cannot change the specs, the approach, or the task ordering — the fallback is already in the task list.

**VERDICT:** `APPROVED`

## Addendum — 2026-11-07, planning-integrity and UI quality audit

- [x] Added the missing event-storm discovery artifact with stable identities, policies, read models and unresolved hotspots.
- [x] Added the missing AsyncAPI 3.0 contract for start/resume input and the ordered AG-UI event stream, with scenario traceability.
- [x] Consolidated the two proposed decisions into the change-local `adr.md`; no repository-level ADR files are used.
- [x] Added bounded slow-consumer behavior so SSE cannot accumulate unbounded memory.
- [x] Closed the flag-off gap for session and JSON transcript endpoints.
- [x] Added safe generated-content rendering, anchored streaming scroll, keyboard composer behavior, attention-aware history, responsive workspace behavior, non-sensitive preference storage and browser-level acceptance coverage.
- [x] Added final strict OpenSpec, AsyncAPI and discovery-to-contract traceability gates.

`openspec validate ag-ui-copilotkit-console --type change --strict` passes after these changes.

**VERDICT:** `APPROVED`
