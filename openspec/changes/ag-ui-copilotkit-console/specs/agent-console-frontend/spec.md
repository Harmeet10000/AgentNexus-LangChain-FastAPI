## Purpose

Defines the behaviour of the CopilotKit-based review console: how a user selects an authorized document scope, starts and observes an Agent Saul run, acts on each human-review gate, and reloads history — without holding authority or durable state that belongs to the backend.

## ADDED Requirements

### Requirement: Document-scoped session start

The console SHALL start a run only after a successful session preflight that returns a `thread_id` and an authorized document scope, and SHALL present only documents the backend reports as authorized for the caller.

The console MUST NOT allow a run with an empty document scope, MUST display non-ready ingestion status for a selected document, and MUST NOT treat a client-side document list as authority.

#### Scenario: Start with an authorized document

- **WHEN** the user selects an authorized, ready document and submits a question
- **THEN** the console performs preflight, then starts the run with the returned `thread_id` and the returned scope

#### Scenario: Document not ready

- **WHEN** the selected document reports `not_ready` or `failed` ingestion
- **THEN** the console blocks the run and explains the ingestion state instead of starting it

#### Scenario: Scope cannot be widened in the client

- **WHEN** a crafted client adds a document id that was not returned in the preflight scope
- **THEN** the console does not include it and the run proceeds with the server-narrowed scope

### Requirement: Streaming turn presentation

The console SHALL render assistant text as it streams and SHALL show run progress derived from protocol activity and step events — current node, completed steps, and overall status — rather than inferring progress from message text.

Cancelling a run MUST stop presenting further output and MUST leave the session resumable by `thread_id`.

#### Scenario: Progress is visible during a long node

- **WHEN** the run is executing a node that has produced no text for several seconds
- **THEN** the console still shows the active node and completed step count

#### Scenario: Cancel mid-run

- **WHEN** the user cancels an in-progress run
- **THEN** streaming stops, no approval is submitted, and the session remains resumable

#### Scenario: Resume opens a new assistant message

- **WHEN** the user resumes a paused session
- **THEN** the console renders the resumed output as a new assistant message and never appends to the message closed by the interrupt

### Requirement: Plan approval gate

On a `plan_approval` interrupt the console SHALL render the plan as ordered steps with their action and description, and SHALL support approve, reject with feedback, and modify by editing, removing or adding steps before approval.

The submitted resume payload MUST validate against the interrupt's `responseSchema`; the console MUST refuse to submit an invalid payload and MUST disable submission while a resume is in flight.

#### Scenario: Approve unmodified plan

- **WHEN** the user approves the plan as presented
- **THEN** the console submits an approve resume addressing that interrupt id and waits for the server to confirm continuation

#### Scenario: Modify then approve

- **WHEN** the user edits one step description and removes another before approving
- **THEN** the submitted payload contains the complete modified plan and validates against the interrupt schema

#### Scenario: Invalid payload blocked

- **WHEN** an edited plan violates the interrupt's schema, such as an empty step list
- **THEN** the console blocks submission and explains the violation instead of sending the resume

#### Scenario: Double submission prevented

- **WHEN** the user activates the approve control twice before the server responds
- **THEN** exactly one resume is sent

### Requirement: Human review gate

On a `human_review_required` interrupt the console SHALL render the review findings and their evidence, and SHALL support per-finding or clause-level overrides before approve or reject, showing the original and overridden value together.

The console MUST NOT finalize, persist, or display a finalized result before the server confirms the run continued past the gate.

#### Scenario: Override a finding

- **WHEN** a reviewer changes the disposition of one finding and approves
- **THEN** the resume carries the override list for that interrupt and the console shows the pending state until the server responds

#### Scenario: No premature finalization

- **WHEN** the reviewer approves a human-review gate
- **THEN** the console does not mark the session complete until a terminal success event is received

### Requirement: Clarification and re-upload gates

On a `clarification_needed` interrupt the console SHALL offer a bounded free-text answer. On an `ocr_reupload` interrupt the console SHALL explain the required remediation and link to the existing document upload flow, and MUST NOT perform an upload itself.

#### Scenario: Answer a clarification

- **WHEN** the assistant asks a clarifying question
- **THEN** the console submits the typed answer as the resume payload for that interrupt

#### Scenario: OCR remediation is a handoff

- **WHEN** the interrupt requires a re-uploaded document
- **THEN** the console shows the reason and a link to the document flow without starting an upload

### Requirement: Server-derived session state

The console SHALL derive run state from the event stream and from the transcript endpoints. It MUST NOT maintain a durable local copy of transcript content, and MUST clear any transient cache on logout or tenant change.

#### Scenario: Reload the page mid-session

- **WHEN** the user reloads while a session is paused at an interrupt
- **THEN** the console restores the session from the transcript endpoint and offers the still-valid resume controls

#### Scenario: Logout clears local state

- **WHEN** the user logs out or switches tenant
- **THEN** cached transcript content and in-memory run state are discarded

### Requirement: History and transcript rendering

The console SHALL list the caller's sessions from the session endpoint and render a selected transcript page by page, preserving display order and representing redacted rows as erased rather than as missing.

#### Scenario: Older session opened

- **WHEN** the user opens a session from history
- **THEN** the console renders its transcript in display order and loads further pages on demand

#### Scenario: Erased turn

- **WHEN** a transcript page contains a redacted row
- **THEN** the console shows it as erased in its original position and offers no reconstruction

### Requirement: Authentication, degradation and reconnect

The console SHALL authenticate with the same credential as the API on the API origin, redirect to sign-in on an unauthorized response, and MUST NOT retry an unauthorized or forbidden resume automatically.

On a retryable stream failure the console SHALL offer a resume by `thread_id`; on a non-retryable error it SHALL surface the stable error code and stop retrying.

#### Scenario: Expired session mid-run

- **WHEN** a stream request is refused as unauthorized
- **THEN** the console stops, preserves the thread id, and sends the user to sign-in without retry loops

#### Scenario: Retryable interruption

- **WHEN** the stream drops with a retryable failure
- **THEN** the console offers to resume the same thread and, on resume, reflects server state rather than replaying optimistic messages

### Requirement: Accessible approval controls

Every approval, rejection and modification control SHALL be operable by keyboard, SHALL expose its pending and error state to assistive technology, and SHALL NOT rely on colour alone to distinguish approve from reject.

#### Scenario: Keyboard-only review

- **WHEN** a keyboard-only user reaches a plan approval gate
- **THEN** focus order reaches each step editor and each decision control, and the submitted state is announced

### Requirement: Safe rich-content rendering

The console SHALL render assistant output with a single sanitized Markdown pipeline. It MUST disable raw HTML, remove unsafe URL schemes, open external links with safe `rel` attributes, preserve code and citation readability, and provide a plain-text copy action that excludes hidden UI labels.

Streaming partial Markdown MUST remain readable while tokens arrive and MUST NOT execute markup that becomes dangerous only after later tokens complete it.

#### Scenario: Model output contains active content

- **WHEN** an assistant message contains a script tag, inline event handler, `javascript:` URL, or crafted partial HTML
- **THEN** the console displays inert text or safe Markdown and executes no active content

#### Scenario: Reviewer copies an answer

- **WHEN** the reviewer copies an assistant message containing citations and code
- **THEN** the clipboard receives readable plain text with citation labels and code content in display order

### Requirement: Composer and transcript navigation ergonomics

The message composer SHALL support multiline input, use `Ctrl+Enter` or `Meta+Enter` to submit, expose the shortcut in its accessible name or help text, and retain the in-memory draft when a recoverable request fails. Sensitive draft or transcript content MUST NOT be written to browser storage.

The transcript SHALL follow new output only while the reviewer is already near the end. If the reviewer scrolls upward, new output MUST preserve their position and display a keyboard-accessible “jump to latest” control with an unread count.

#### Scenario: Reviewer reads earlier evidence during streaming

- **WHEN** the reviewer scrolls away from the end while tokens continue to arrive
- **THEN** the viewport stays anchored, an unread count increases, and activating “jump to latest” returns to the newest content

#### Scenario: Recoverable send failure

- **WHEN** submission fails before the server accepts the message
- **THEN** the composer keeps the draft in memory, explains the retryable failure, and does not create a duplicate optimistic message

### Requirement: Session findability and attention state

The history view SHALL support filtering the loaded session list by status and document, SHALL show the last-updated time, and SHALL distinguish sessions waiting for reviewer input from running, failed, cancelled, and completed sessions using text plus iconography.

Opening, filtering, or paginating history MUST preserve the currently selected session until the reviewer explicitly chooses another session.

#### Scenario: Reviewer finds pending work

- **WHEN** the reviewer filters history to sessions requiring attention
- **THEN** only sessions with an open interrupt are shown and each identifies the interrupt reason without exposing transcript content

#### Scenario: More history is loaded

- **WHEN** the reviewer loads the next session page while a transcript is open
- **THEN** the selected transcript remains open and the added sessions keep newest-first order

### Requirement: Responsive workspace and non-sensitive preferences

The console SHALL provide a usable single-column layout on narrow screens and a resizable history/transcript/progress workspace on wider screens. Moving between layouts MUST NOT discard an in-memory draft, pending decision, selected session, or scroll position.

Theme, reduced-motion, and panel-size preferences MAY be persisted locally, but the stored values MUST contain no message, transcript, document, interrupt, tenant, or user content. Loading, empty, offline, reconnecting, and terminal states SHALL have distinct text and accessible status announcements.

#### Scenario: Review continues on a narrow viewport

- **WHEN** the viewport becomes narrow while a plan decision is being edited
- **THEN** the editor remains reachable in a focused panel or drawer and the unsent decision is preserved

#### Scenario: Preference storage is inspected

- **WHEN** a completed session has been viewed and browser storage is inspected
- **THEN** storage contains only allowlisted presentation preferences and no content or identifiers from the session
