## Purpose

Defines the AG-UI 1.0 contract by which an authorized client starts, observes, interrupts and resumes one Agent Saul graph run over a single ordered event stream, independent of the transport that carries it.

## ADDED Requirements

### Requirement: Run stream endpoint

The system SHALL expose an AG-UI run endpoint at `POST /api/v2/agent-saul/agui/runs` that accepts an AG-UI `RunAgentInput` (`threadId`, `messages`, `state`, optional `resume[]`) and responds with an ordered `text/event-stream` of AG-UI events for exactly one run.

The endpoint MUST reject an input that omits both a user message and a `resume[]` entry, and MUST reject an unknown or malformed `threadId` rather than silently creating an unscoped run.

#### Scenario: First run for a new thread

- **WHEN** an authenticated client posts a `RunAgentInput` with a `threadId`, a user message, and an authorized document scope
- **THEN** the response is `text/event-stream` beginning with `RUN_STARTED` and ending with `RUN_FINISHED` or `RUN_ERROR`

#### Scenario: Empty run input is refused

- **WHEN** a client posts a `RunAgentInput` with neither a user message nor a `resume[]` entry
- **THEN** the request is refused with the project error envelope and no run is started

### Requirement: Run lifecycle events

Every run SHALL emit `RUN_STARTED` before any content event and SHALL terminate with exactly one terminal event: `RUN_FINISHED` carrying an `outcome`, or `RUN_ERROR` carrying a stable error code.

`RUN_FINISHED.outcome` MUST be `{ type: "success" }` for a completed run and `{ type: "interrupt", interrupts: [...] }` for a run paused for human input; it MUST NOT be omitted.

#### Scenario: Successful completion

- **WHEN** the graph reaches its terminal node without an interrupt
- **THEN** the stream ends with `RUN_FINISHED` whose outcome is `{ type: "success" }` and no further events follow

#### Scenario: Terminal error

- **WHEN** the run fails with an error that the session cannot recover from
- **THEN** the stream emits `RUN_ERROR` with a stable code and closes without emitting `RUN_FINISHED`

### Requirement: Text message streaming

Model output SHALL stream as `TEXT_MESSAGE_START`, one or more `TEXT_MESSAGE_CONTENT`, and `TEXT_MESSAGE_END` events for each assistant message, in generation order, and each content event MUST carry the `messageId` opened by its start event.

#### Scenario: Streamed assistant answer

- **WHEN** a node emits token deltas during a run
- **THEN** the client receives a start event, the deltas in order, and a matching end event for that `messageId`

#### Scenario: Interrupt before message end

- **WHEN** the graph pauses at an interrupt while an assistant message is open
- **THEN** a `TEXT_MESSAGE_END` for the open `messageId` is emitted before the terminal interrupt event

### Requirement: State projection

The stream SHALL establish run state with `STATE_SNAPSHOT` before the first `STATE_DELTA`, and SHALL communicate workflow status, plan, current step and document readiness through `STATE_DELTA` operations that are valid RFC 6902 JSON Patch against the last synchronized state.

A client MUST be able to reconstruct the same state by applying every delta in order; the system MUST emit a fresh `STATE_SNAPSHOT` after any resume and after any reconnect.

#### Scenario: Status transition

- **WHEN** the graph moves to a new `WorkflowStatus`
- **THEN** a `STATE_DELTA` replaces the status field and the plan field when it changed

#### Scenario: Resume resynchronizes state

- **WHEN** a paused run is resumed
- **THEN** the resumed run emits a `STATE_SNAPSHOT` before any further delta

### Requirement: Activity and step projection

The system SHALL project graph execution progress outside the transcript as `ACTIVITY_SNAPSHOT` / `ACTIVITY_DELTA` and `STEP_STARTED` / `STEP_FINISHED` pairs, so a client can render node-level progress without inferring it from message content.

Each `STEP_STARTED` MUST name the node and its 1-based position and MUST be followed by a matching `STEP_FINISHED` or by the terminal event that ended the run.

#### Scenario: Node progress is observable

- **WHEN** a run executes a top-level graph node
- **THEN** the client receives `STEP_STARTED` for that node and a later `STEP_FINISHED` for the same node

#### Scenario: Interrupted step closes

- **WHEN** a node pauses the run at an interrupt
- **THEN** the in-flight step is closed and the activity state reports the pause before the terminal event

### Requirement: Subagent attribution

Delegated work SHALL be attributed with `SUBAGENT_STARTED`, `SUBAGENT_FINISHED` and `SUBAGENT_ERROR` events that carry an id distinguishing the delegated run from its parent, and no delegated event may be reported as parent-run text.

#### Scenario: Specialist delegation

- **WHEN** the graph delegates to a specialist agent
- **THEN** the delegated work is bracketed by `SUBAGENT_STARTED` and `SUBAGENT_FINISHED` with the same id

### Requirement: Interrupt lifecycle and durable resume

A run paused for human input SHALL terminate with `RUN_FINISHED` whose outcome is `{ type: "interrupt", interrupts: [...] }`, where each interrupt carries a stable `id`, a `reason` derived from the Agent Saul interrupt type, a human-readable `message`, and a `responseSchema` describing the accepted resume payload.

The system MUST NOT keep the run open awaiting an answer, MUST represent a paused run as resumable state addressed by `threadId` plus interrupt `id`, and MUST resume through a new run whose `RunAgentInput.resume[]` addresses every open interrupt. A resume that omits any open interrupt MUST be refused.

#### Scenario: Plan approval pause

- **WHEN** the graph reaches a `PLAN_APPROVAL` interrupt
- **THEN** the run terminates with an interrupt outcome whose interrupt carries a stable id, reason `plan_approval`, and a schema for the approve/reject/modify payload

#### Scenario: Resume after disconnect

- **WHEN** a client disconnects while an interrupt is pending and later posts a resume run for the same `threadId`
- **THEN** the graph continues from the checkpoint and the stream emits a `STATE_SNAPSHOT` before further events

#### Scenario: Incomplete resume is refused

- **WHEN** a resume run addresses fewer interrupts than the pending set
- **THEN** the resume is refused with `RUN_ERROR` and the pending interrupts remain resumable

#### Scenario: Stale or foreign interrupt id

- **WHEN** a resume references an interrupt id that is not open for that thread and caller
- **THEN** the resume is refused with `RUN_ERROR` and no graph transition occurs

### Requirement: Authorization parity on every run and resume

Every run and every resume SHALL authorize the caller from live server-side state before graph execution: membership in the selected tenant, current legal roles, and the document allowlist narrowed by the documents capability. Client-supplied identity, roles, tenant, or document scope MUST NOT widen access.

Authority revoked between the interrupt and the resume MUST cause the resume to be refused.

#### Scenario: Role revoked before resume

- **WHEN** a reviewer's role is revoked after the interrupt and before the resume
- **THEN** the resume is refused and the graph does not finalize or persist

#### Scenario: Client widens document scope

- **WHEN** a run input lists a document id outside the caller's authorized scope
- **THEN** the scope is narrowed server-side and the out-of-scope id is never read

### Requirement: Interrupt response schema is authoritative

The `responseSchema` published on an interrupt and the server-side resume validation SHALL be derived from the same typed payload model, so a payload the schema accepts is not rejected by the server for shape reasons, and a payload the schema rejects cannot be accepted.

An interrupt type whose payload has no typed model MUST NOT be published with an unconstrained object schema.

#### Scenario: Plan payload schema matches validation

- **WHEN** the system publishes the `plan_approval` interrupt schema
- **THEN** a payload built from that schema passes resume validation and a payload missing every plan step is refused by both

#### Scenario: Untyped payload is not exposed

- **WHEN** an interrupt type's payload is only an open dictionary
- **THEN** the system either introduces a typed model for it or fails the run rather than publishing `{"type": "object"}`

### Requirement: Run cancellation

Aborting the stream request after the run has started SHALL cancel the run: the graph stops at the next safe boundary, the thread lock is released, the transcript records a cancelled terminal state, and the checkpoint remains resumable.

An abort before the first event MUST start no run and MUST NOT take the thread lock.

#### Scenario: Client aborts mid-run

- **WHEN** a client aborts the stream while a node is executing
- **THEN** the run stops, the lock is released, and the transcript shows a cancelled run rather than a completed one

#### Scenario: Cancelled run is resumable

- **WHEN** a client resumes a cancelled thread
- **THEN** the graph continues from the last checkpoint and the stream reports the state it actually finds

#### Scenario: Abort before start

- **WHEN** a client aborts the request before any event is emitted
- **THEN** no run is created and no lock is taken

### Requirement: Duplicate run submission

A run request whose semantic identity — `threadId` plus the input message id — matches a run already in flight SHALL be refused with a stable `RUN_ALREADY_ACTIVE` code and MUST NOT start a second graph execution.

Recovering from a dropped stream SHALL be expressed as a resume for the same `threadId`, not as a repost of the same message.

#### Scenario: Concurrent duplicate

- **WHEN** two requests for the same thread and message id arrive while the first is running
- **THEN** the second is refused with `RUN_ALREADY_ACTIVE` and only one graph execution occurs

#### Scenario: Retry after a drop

- **WHEN** a stream drops and the client retries
- **THEN** the client resumes the thread and does not repost the message

### Requirement: Message identity across resume

Any assistant message open when the run pauses SHALL be closed with `TEXT_MESSAGE_END` before the terminal event, and a resumed run SHALL open a new assistant message id rather than appending content to a closed id.

#### Scenario: Resume opens a new message

- **WHEN** a paused run is resumed and the assistant continues
- **THEN** the new content carries a message id that was not closed by the interrupt

### Requirement: Transport equivalence with the WebSocket contract

The event facts published through AG-UI and through the existing WebSocket frames SHALL be produced by one transport-neutral projection of graph events, and both transports MUST report the same node sequence, status transitions, interrupt type and terminal outcome for the same run.

#### Scenario: Recorded run replayed through both transports

- **WHEN** one recorded graph event stream is projected to AG-UI events and to WebSocket frames
- **THEN** the compared node order, status transitions, interrupt descriptors and terminal outcomes are identical

### Requirement: Stream transport safety

The SSE response SHALL declare `text/event-stream`, disable intermediary buffering, emit an SSE comment heartbeat (a `: ping` line, not an AG-UI event type) while a node is silent, and close after the terminal event. Authentication SHALL accept the same credential sources as the existing WebSocket endpoint, and the endpoint MUST be usable from the API origin without a CORS preflight.

The transport MUST preserve event order with bounded buffering. A slow or disconnected consumer MUST NOT cause unbounded memory growth; after the configured bound is exceeded the server SHALL cancel the request at a safe boundary, record the abort, release the lock, and preserve a resumable checkpoint.

#### Scenario: Long silent node

- **WHEN** a node produces no events for longer than the configured heartbeat interval
- **THEN** the stream emits a comment heartbeat and the connection is not closed by an idle timeout, without introducing an event type outside the protocol's set

#### Scenario: Authenticated from the console origin

- **WHEN** the console running at the API origin posts a run with the access-token cookie
- **THEN** the run is authorized without a CORS preflight and without a second login

#### Scenario: Slow consumer exceeds the stream bound

- **WHEN** a client consumes events more slowly than the bounded stream buffer permits
- **THEN** the server does not grow memory without bound, terminates the request safely, releases the lock, and leaves the thread resumable
