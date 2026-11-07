# Discovery: Event Storm ag-ui-copilotkit-console

## Actors

| Actor | System role |
|---|---|
| Reviewer | Starts a document-scoped run, observes progress, and answers human gates. |
| Review console | Presents server-owned state and sends commands; it owns no durable transcript or authority. |
| AG-UI run adapter | Authorizes one HTTP run or resume request and projects it as an ordered SSE stream. |
| Agent Saul runtime | Executes the graph, checkpoints it, opens interrupts, and records terminal state. |
| Transcript query service | Exposes authorized session summaries and paginated persisted rows. |
| Identity and document authorization services | Resolve live tenant, role, and document authority for every start and resume. |

## Commands (blue commands)

| Command | Actor | Payload essentials | Produces events |
|---|---|---|---|
| Start run | Reviewer | `threadId`, input `messageId`, message, server-narrowed document scope | `RUN_STARTED`, content/state/activity events, then one terminal event |
| Resume run | Reviewer | `threadId`, every open `interruptId`, schema-valid response | Fresh state snapshot, progress events, then one terminal event |
| Cancel run | Reviewer | Abort signal for the active `runId` | Cancelled terminal state and resumable checkpoint |
| Read sessions | Review console | Tenant selector, cursor, optional status/document filters | Session-summary read model page |
| Read transcript | Review console | Authorized `sessionId`, keyset cursor, limit | Transcript-page read model |

## Domain Events (orange stickies — past tense)

| Event (past tense) | Payload | Stable identity | Producer |
|---|---|---|---|
| Run started (`RUN_STARTED`) | run/thread ids and start timestamp | `runId` within `threadId` | Agent Saul runtime |
| Assistant message streamed (`TEXT_MESSAGE_*`) | message id, ordered content deltas | `messageId` | Projection layer |
| Run state synchronized (`STATE_SNAPSHOT` / `STATE_DELTA`) | run state or RFC 6902 operations | `runId` plus monotonic event order | Projection layer |
| Activity changed (`ACTIVITY_*`) | active/completed activity facts | `runId` plus activity id | Projection layer |
| Step changed (`STEP_STARTED` / `STEP_FINISHED`) | node, position, status | `runId` plus step id | Projection layer |
| Subagent changed (`SUBAGENT_*`) | delegated run id, parent id, status/error | delegated `subagentId` | Projection layer |
| Interrupt opened (`RUN_FINISHED` interrupt outcome) | interrupt id, reason, message, response schema | `interruptId` within `threadId` | Agent Saul runtime |
| Run finished (`RUN_FINISHED`) | success or interrupt outcome | `runId` | Agent Saul runtime |
| Run failed (`RUN_ERROR`) | stable code, retryability, safe message | `runId` | AG-UI run adapter |
| Run cancelled | thread/run ids and last checkpoint | `runId` | Agent Saul runtime |

## Policies (lilac — WHEN/THEN)

| WHEN | THEN | Kind (reactive / command / timer+expiry) |
|---|---|---|
| A start or resume command is received | Re-resolve live tenant, role, and document authority before taking the thread lock | command |
| `RUN_STARTED` is emitted | Emit ordered protocol facts and exactly one terminal event | reactive |
| An interrupt opens | Close any open assistant message, checkpoint, release the lock, and finish the run with the interrupt descriptor | reactive |
| A resume omits, stales, or foreign-addresses an interrupt | Refuse it without transitioning the graph | command |
| The client aborts after start | Stop at the next safe boundary, release the lock, record cancellation, preserve resumability | reactive |
| A node is silent for the heartbeat interval | Emit an SSE `: ping` comment without adding a protocol event | timer+expiry |
| A slow consumer exceeds the bounded stream buffer | Cancel the request safely rather than accumulating unbounded memory | timer+expiry |
| Authentication or tenant context changes | Clear console run state and transcript views | reactive |

## Read Models

| Read model | Serves | Derives from | Staleness tolerance |
|---|---|---|---|
| Session summary page | History, status filters, and “needs attention” badges | Session metadata and open-interrupt state | Read committed; newest-first cursor page |
| Transcript page | Reopening history and reconstructing display order | Transcript rows using `(turn_seq, ordinal, row_id)` | Read committed; append-safe keyset page |
| Live run projection | Messages, progress rail, decisions, reconnect state | Ordered AG-UI stream plus fresh snapshots | No reordering; reconnect begins with a snapshot |

## Hotspots

| # | Question | Who decides | Resolved in |
|---|---|---|---|
| 1 | Does `aget_state()` expose a restart-stable interrupt index? | Backend spike owner | Task 1.3; checkpoint-record fallback in design D3 |
| 2 | Does CopilotKit’s Python runtime support the required interrupt/resume flow? | Spike owner | Task 1.1; design D5 fallback |
| 3 | What bounded-buffer size and slow-client timeout fit legal runs? | Runtime owner | Task 3.16 and deployment telemetry |
| 4 | Are run audit facts sufficient without WebSocket presence leases? | Security/operations | Design open question; must resolve before rollout |

