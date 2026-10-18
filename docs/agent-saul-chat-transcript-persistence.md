# Agent Saul — Chat Transcript Persistence: Design (v3)

**Status:** Draft v3 (v2 verdict: accept with three must-fixes — all applied, plus the three should-fixes and polish items). Text changed since v2 is tagged [v3]. No migration or code until final sign-off. **Audience:** the agent that will implement this, and a reviewer who wants the five Definition-of-Done answers in one read (see §12).

**Provenance.** This doc is built from the architecture brief, the review answers, and the v1→v2 review rounds; facts marked **[verify]** are assumptions the implementer must confirm against code before relying on them. They are collected in §14.

## Changes from v1

- **B1, fixed by deletion.** The one non-derived write (`turn_closed(failed)` from a loop exception at T3) is removed. An exception is never evidence; the checkpoint is the only witness (§1, §5.1, §6, §9, §10, §12).
- **B2, lock built.** A per-thread Redis lock is held for connection lifetime and renewed by heartbeat. A second connection is refused with `THREAD_BUSY` (§5.5, §9). D6 is resolved.
- **B3, aliveness is an input.** `project()` takes an injected `aliveness` argument fed by the lock heartbeat. The sweep passes `False` unless the heartbeat is fresh (§4.1, §5.3, §6).
- **Minors 1–7:** expiry from the fresh snapshot (§8); T0 awaited in T3 (§5.2); GC audit stamp (§8); orphan sweep moves cached status (§5.3); repository wording (§10); erasure stores NULL content (§3.2, §8, §11); paths and resolved [verify] items (§14).
- **D1–D6:** accepted. D4 gains a test assertion (§13).

## Changes from v2

- **Must-fix 1: `derive_status` is total again.** v2 narrowed `active` to require a fresh heartbeat but never re-closed the table: a snapshot with a running turn, dead heartbeat, and checkpoint ts *inside* the grace period matched no row. v3 adds an explicit NoOp row and states the totality invariant (§4.1, §5.3).
- **Must-fix 2: `close_turn()` removed from the repository (§10).** The v2 deletion of the non-derived write did not propagate to the API list. Every terminal row — including any `turn_closed` row — is written through `apply_projection` from a snapshot-derived projection.
- **Must-fix 3: heartbeat cadence specified (§5.5).** Renewal is time-driven (every TTL/3), independent of WS traffic; renewal failure reports `aliveness=false`; TTL expiry remains the crash fence.
- **Should-fix: Redis-outage semantics made consistent (§5.5, §6, §9).** No verdicts from a blind sweep: when Redis is unavailable, connections are refused (fail-closed, deliberate) and sweeps skip status derivation entirely — statuses freeze rather than decay.
- **Should-fix: `plan_decision` content rule defined (§3.4, §14).** The row carries a terse derived label, never the user's full text; whether the modify path persists richer text into state is a new verify item.
- **Should-fix: `derive_status` signature made explicit (§4.1)** — reply-row knowledge comes from `turn_rows`, never from `SnapshotView`.
- **Polish:** §12.3 typo fixed; §13 gains the totality/NoOp test, a Redis-outage test, and an assert that the sweep's closing row is a projection output; §14 resolves the `AgentError` shape against code and adds one verify (terminal-error classification).

---

## 1. Summary

1. The **checkpoint** (AsyncPostgresSaver, exists) is the only authority on what happened in a run and the only thing that can resume it.
2. The **transcript** (`chat_sessions` / `chat_messages`) is an **append-only projection** of the checkpoint, written only by a pure projector reading `aget_state` snapshots. It is never a second source of truth, so the two cannot fork. Transcript ⊆ checkpoint by construction.
3. **Long-term memory** (cognee) is untouched. It is still written only by `persist_memory_node` after approval, and the transcript layer never writes to it.
4. Status (`clarifying`, `completed`, …) is **derived** from the snapshot (`state.next`, `state.tasks` interrupts, `final_report`, checkpoint ts). The loop never sets it by hand. The session row caches the derived value for list views.
5. Idempotency is two-level. **Batch level** is `(thread_id, checkpoint_id)`. **Item level** is a stable `source_key` per row, so replaying from a *later* checkpoint cannot duplicate (see §4.3).
6. A turn that dies before the first super-step **never existed**. No ingress marker row is written.
7. Failure policy: **fail open** for anything that only degrades presentation or analytics (transcript lag, token totals, memory read/write). **Fail closed** for anything that would act on state the runtime doesn't understand, or for the wrong person (schema version, identity).

**[v2] Chain-of-custody rule.** The transcript admits only snapshot-derived facts. Wall-clock time, exceptions, and connection state are inputs to the projector, never evidence in themselves.

**[v3] Totality rule.** `derive_status` is a total function: every snapshot maps to exactly one outcome — a derived status or an explicit NoOp. If a new input dependency narrows any row's match set, the table must be re-closed over the full cross product of inputs; an unreachable input combination in review is a bug in review.

---

## 2. Layers and authority

| Layer | Store | Authoritative for | Written by |
| --- | --- | --- | --- |
| A. Conversation state | Checkpointer (`thread_id`) | "Resume the run", "what is the graph's state" | LangGraph, every super-step (exists) |
| B. Transcript | `chat_sessions`, `chat_messages` | "Show me my chats", export, audit, per-turn token totals | Projector via `ChatTranscriptRepository` (new) |
| C. Long-term memory | cognee (tenant-partitioned) | Approved report recall | `persist_memory_node` only (exists) |
| D. Document KG | Graphiti/Neo4j | Clause/relationship graph | Document pipeline (separate; not touched) |

Rules:

- Questions about *conversation continuity* read A. Questions about *history and listing* read B. B never answers "can this run resume".
- B is rebuildable from A while A exists. After checkpoint GC (§7), B is the only record. This is why GC always follows a verified projection catch-up.
- No raw SQL against checkpoint tables. Only public saver/graph APIs (`aget_state`, `aget_state_history`, `adelete_thread`).
- Never write the raw LangGraph event stream to client or DB. The projector sees snapshots, not events.

---

## 3. Data model

Two existing tables are extended. `message_role` (created in migration 0014) keeps its name and values.

### 3.1 `chat_sessions` (`session_id` == `thread_id`, stored as the UUID string)

Existing: `id`, `session_id` (unique), `user_id`, `title`, `extra_metadata`, `created_at`, `updated_at`.

Add:

- `schema_version smallint not null default 1`. This is the **only** place the transcript schema version lives. Unknown value ⇒ refuse at read (§9).
- `status` (new PG enum `chat_session_status`: `active, clarifying, resumed, completed, abandoned_expired, failed`). This is a cache of the *latest turn's* derived status.
- `turn_count int not null default 0`
- `last_checkpoint_id text null`, `last_checkpoint_ts timestamptz null`: the projection watermark. `last_checkpoint_ts` is also the base for pause expiry.
- `created_at` / `updated_at` become `server_default=func.now()` (project convention). The projector sets `updated_at = now()` on every applied projection.

### 3.2 `chat_messages`

Existing: `id`, `session_id`, `user_id`, `role` (`user|assistant|system`), `content`, `model`, `tokens_used`, `extra_metadata`, `created_at`.

Add:

- `turn_seq int not null`: 1-based turn ordinal within the session, assigned by the projector (§4.2).
- `ordinal int not null`: deterministic position within the turn (§4.2). Display order is `(turn_seq, ordinal)`, **not** insert order, because reconciliation can insert late.
- `kind varchar(32) not null` with a CHECK constraint (not a PG enum, since kinds will evolve): `query, clarification_prompt, clarification_reply, plan_prompt, plan_decision, final_summary, turn_closed`.
- `source_key text not null`: stable identity of the projected item (§4.3).
- `source_checkpoint_id text not null`: provenance, the checkpoint the row was projected from.
- `redacted_at timestamptz null`: erasure marker (§8).
- `created_at` becomes `server_default=func.now()`.

**[v2]** `content` becomes nullable if it is not already, so erasure can store NULL ("no content") rather than an empty string ("empty content"). **[verify current nullability]**

### 3.3 Constraints and indexes

- `UNIQUE (session_id, source_key)` on messages. This is the item-level idempotency guard.
- `(session_id, turn_seq, ordinal)` on messages for transcript reads. This replaces the briefed `(session_id, id)` because ordering is by turn and ordinal (decision D3).
- `(user_id, updated_at DESC)` on sessions for chat listing (briefed as `created_at`; a chat list is by last activity — decision D3). Drop the old single-column `user_id` index if it becomes a redundant prefix.

### 3.4 Row kinds

| kind | role | Derived from | `source_key` |
| --- | --- | --- | --- |
| `query` | user | `HumanMessage` tagged `saul_kind=query` in `messages` | `message.id` |
| `clarification_prompt` | assistant | interrupt payload in `state.tasks[].interrupts` | `interrupt:{interrupt.id}` |
| `clarification_reply` | user | untagged `HumanMessage` following a clarification prompt | `message.id` |
| `plan_prompt` | assistant | planner interrupt payload | `interrupt:{interrupt.id}` |
| `plan_decision` | user | post-resume state (`plan_approved`, `plan`) | `decision:{interrupt.id}` |
| `final_summary` | assistant | summary/title from `final_report` | `final:{query message.id}` |
| `turn_closed` | system | terminal non-completion: `abandoned_expired` or `failed` | `closed:{query message.id}` |

Notes:

- Assistant-side content mostly does **not** live in `messages`; prompts live in interrupt payloads and the result in `final_report`. This is why rows are *synthesized* from the snapshot **[verify]**.
- `final_summary` stores a summary/title, never the full report JSON. The report lives in memory (post-approval) and the checkpoint (pre-approval).
- **[v3] `plan_decision.content` is a terse derived label** — `approved` or `modified` — never the user's full text and never the plan itself. The plan stays in `state.plan` / the checkpoint (pre-approval) and in memory (post-approval). If the modify path does persist richer text into state ([verify 10](#user-content-section-14)), `extra_metadata` may carry a *pointer* (e.g. `{modified: true}`), not the text.
- `extra_metadata` carries structured, content-free facts only: `interrupt{id, checkpoint_id}`, `hitl{type, answers_prompt}`, `usage{input, output}`, `tool_calls[{name, node, ok, ms}]` (no arguments or outputs), `errors[{node, code}]` **[v3: shape from `AgentError` — `message` and `retryable` are not recorded; `message` is content, and terminal classification is a derivation rule, not a stored flag]**, `report_ref` (`legal::{user_id}::reports#{doc_id}`, approved runs only).

---

## 4. Projection

### 4.1 One pure function, one place that reads graph internals

```
snapshot_to_view(StateSnapshot) -> SnapshotView                  # the ONLY code that reads state.next / state.tasks
derive_status(view, turn_rows, now, aliveness) -> Status | NoOp  # total: every snapshot maps to exactly one outcome
project(view, existing_keys, now, aliveness) -> Projection       # pure: rows to insert, derived status
```

`SnapshotView` is a frozen Pydantic model: `user_id`, `checkpoint_id`, `checkpoint_ts`, `step`, `next`, `interrupts`, `messages`, `final_report_summary`, `errors`, `schema_version`. Interrupt detection must not fork between the service and the projector, so both go through `snapshot_to_view`. `hydrate_state` has already gate-checked the state version before any snapshot is projected.

**[v2]** `aliveness` (bool) is an injected input from the lock heartbeat (§5.5). It is never looked up inside `snapshot_to_view` or `project`. Aliveness, `now`, and exceptions are inputs; only the snapshot is evidence.

**[v3] Input discipline.** Knowledge that is snapshot-independent — such as whether the current turn already has a reply row — enters through `turn_rows` (the turn's already-projected rows), never by enriching `SnapshotView`. `SnapshotView` stays a pure snapshot fact.

**[v3] Totality invariant.** `derive_status` returns a status or the explicit NoOp for *every* input combination. See §5.3 for the closed table.

### 4.2 What a turn is

A **turn** is one logical run on a thread: a start frame through END, including every HITL exchange inside it. It is **not** a checkpoint, because checkpoints are written per super-step, including internal ones like hydration and gateway.

- **Boundaries.** The service tags the seeded user query at the edge (`HumanMessage(..., additional_kwargs={"saul_kind": "query"})` in `_build_initial_state`). A turn is the span from one tagged query to the next. Untagged `HumanMessage`s inside a span are HITL replies. No graph-state schema change is needed **[verify: additional_kwargs survives checkpoint serialization]**.
- **`turn_seq`.** Assigned by the projector when it first inserts a `query` row: `session.turn_count + 1`, inside the projection transaction under a row lock on the session. Later items in the turn resolve `turn_seq` from their query row's `source_key`. There is no minting at the edge. A replay finds the existing row and reuses its `turn_seq`, so there are no collisions after crashes or after checkpoint GC.
- **`ordinal`.** Deterministic from structure: `query = 0`; for the k-th HITL exchange `prompt_k = 2k−1`, `reply_k = 2k`; the pending prompt or `final_summary` takes the next slot; `turn_closed` is last. Because replies answer prompts in order, a missing prompt row leaves a detectable hole that reconciliation can fill (§6).
- **First-checkpoint rule.** A turn exists only once a checkpoint with a completed super-step (`metadata.step ≥ 0`) contains its tagged query **[verify step semantics]**.

### 4.3 Idempotency, two levels

The brief asks for `(thread_id, checkpoint_id)`. It is used, but it cannot be the *only* key, for this reason: a row first projectable at checkpoint 5 may, after a crash, be projected from checkpoint 9. If checkpoint id were the row's identity, the replay would insert a duplicate. So:

1. **Batch level, `(thread_id, checkpoint_id)`.** The projector no-ops if `snapshot.checkpoint_id == session.last_checkpoint_id`. Every row records `source_checkpoint_id`. The session watermark and the rows move in **one transaction**.
2. **Item level, `UNIQUE (session_id, source_key)`.** `source_key` comes from stable identity (LangChain `message.id`, interrupt id) and is therefore invariant across which checkpoint the item is projected from. Inserts are `ON CONFLICT DO NOTHING`.

Both must hold. Batch level makes the common replay free; item level makes the uncommon replay correct.

---

## 5. Write envelope for one turn

### 5.1 Boundaries (the only moments the transcript is written)

| Trigger | When | What is written |
| --- | --- | --- |
| **T0** | First durable checkpoint after start (observed after `gateway` ends) | Session upsert (first time only), `query` row. Status `active`. |
| **T1** | Interrupt detected via `aget_state` | Prompt row (`clarification_prompt` or `plan_prompt`). Status `clarifying`. Tokens for the segment. |
| **T1′** | Resume received and applied | Reply/decision row on the next projection (status `resumed`), then T1 or T2 again. |
| **T2** | Run reaches END | `final_summary` row, token totals, tool-call records. Status `completed`, or `failed` if `errors` is terminal. |
| **T3** | WS drop or exception (`finally`, best effort) | **[v2]** Run the projector over a fresh `aget_state` and nothing else. If the derived status is terminal, the closing row comes out of that projection like any other row. An exception is never evidence, so no row is written because of it. |

Between boundaries, **nothing** is written. Streamed deltas are UI-only and never persisted.

Each boundary is one transaction: session upsert/lock, row inserts (`ON CONFLICT DO NOTHING`), session cache update (`status`, `last_checkpoint_*`, `turn_count`, `title` if null, `updated_at`). That is 2–4 small inserts per turn.

### 5.2 Not blocking the WS loop

- T1, T2 and T3 run **after** the corresponding WS frame has been emitted, under a hard `asyncio.timeout` (default 750 ms, configurable). The user never waits on them.
- T0 runs as a tracked background task whose reference is held by the recorder, and it is awaited at the next boundary so ordering is preserved and exceptions are absorbed there. [v2] It is also awaited in the finally block (T3), so a disconnect before T0 completes leaves no dangling task.
- **Project-before-resume.** Before issuing `Command(resume=...)`, the recorder must have projected the pause. It retries once, and on continued failure proceeds (fail-open) and registers a reconcile hint. The pause is recoverable later from `aget_state_history` (public API).

### 5.3 State machine (derived, not stored by the loop)

The closed derive table [v3 — matched against the cross product of {interrupts} × {aliveness} × {checkpoint age}]:

| Derived condition (from snapshot) | Status / outcome |
| --- | --- |
| [v2] Tagged query present, no pause, `next` non-empty, and `aliveness` is true (fresh lock heartbeat) | `active` (or `resumed` if `[v3]` the turn already has a reply row in `turn_rows`) |
| **[v3] Tagged query present, no pause, `next` non-empty, `aliveness` is false, last checkpoint ts within the grace period** | **NoOp — retain the session's cached status. Write no rows, move no watermark.** |
| `state.tasks` has interrupts, checkpoint ts within 24h | `clarifying` |
| Interrupts present, ts older than 24h | `abandoned_expired` |
| `next` empty and `final_report` present | `completed` |
| `next` empty, no `final_report` (terminal errors) | `failed` |
| [v2] `next` non-empty, no interrupt, `aliveness` is false, last checkpoint ts older than the grace period (default 15 min) | `failed` (orphaned, see D1) |

Notes:

- **[v2]** `session.status` is a cache that changes only when a projection applies, so between projections it can be stale. The beat sweep is what moves it for orphaned runs. Read this table as the function `derive_status`, not as continuous truth.
- **[v3]** The NoOp row is what keeps the sweep safe on live equipment: a sweep over a healthy but slow turn — heartbeat stale yet the last checkpoint inside the grace window — writes nothing, moves no watermark, and concludes nothing. The exclusive evidence for orphaned failure is *grace passed with no heartbeat* — "conclusively dead longer than grace," never merely "not fresh right now."

Transitions by branch:

- **Happy path:** `active → completed`
- **Clarification:** `active → clarifying → resumed → completed` (the loop can repeat `clarifying → resumed` for several exchanges)
- **Never answered:** `clarifying → abandoned_expired` (a `turn_closed` system row is appended)
- **Crash or drop:** `active/resumed → failed` after the grace period

### 5.4 What the rows look like

| Moment | Rows for turn N | Session `status` |
| --- | --- | --- |
| Streaming | `query` only. No assistant row exists yet, so none can be false. | `active` |
| Paused at `interrupt()` | `query`, `*_prompt` (`extra_metadata.interrupt{id, checkpoint_id}`) | `clarifying` |
| After resume, running | + `clarification_reply` or `plan_decision` (`extra_metadata.hitl.answers_prompt`) | `resumed` |
| Completed | + `final_summary` (summary, `tokens_used`, `model`, usage and tool-call metadata) | `completed` |
| Abandoned | Prompt row unchanged, + `turn_closed{outcome: abandoned_expired}` | `abandoned_expired` |

Rows are never updated to flip a state, except the one-time fill of null token fields (§6).

### 5.5 Thread lock and aliveness [v2, v3]

One Redis key per thread does three jobs.

- **Mutual exclusion.** The service loop acquires `SET NX` with a TTL (60 s) on connect, renews it on heartbeat, and releases it on disconnect. A second connection on the same thread is refused with `THREAD_BUSY`. It is held for connection lifetime, not just the resume moment, because two concurrent `Command(resume=...)` calls on one paused checkpoint cannot be recovered cleanly (both target the same checkpoint; one wins and the other forks).
- **[v3] Heartbeat cadence.** Renewal is **time-driven, independent of WS traffic**: a periodic task inside the service loop renews the lock every **TTL/3 (20 s at TTL 60 s)**. A long silent LLM turn must never let the lock lapse mid-run. If a renewal fails, the recorder reports `aliveness=false` for subsequent projections and attempts best-effort release. Disconnect release is a courtesy; **TTL expiry is the fence** and the only correctness-critical crash detection.
- **Aliveness input.** A fresh heartbeat means a live connection. The recorder passes `aliveness = True` to `project`; the sweep passes `False` unless the heartbeat is fresh. A snapshot alone cannot tell a slow LLM turn from a dead run.
- **Orphan witness.** TTL expiry is evidence the process died. It feeds the orphan rule as an input and is never a transcript fact on its own.

The lock is acquired before reconcile and before any resume. **[v2] If Redis is unavailable the connection is refused with a retryable error (fail closed).** **[v3] Consequence, made explicit:** aliveness, hints, and exclusion all ride on Redis, so a Redis outage freezes the feature — connections are refused and the sweep cannot judge liveness, therefore **derives nothing** (see §6 and §9). This is a deliberate new hard dependency; the idempotency guard already uses Redis.

---

## 6. Reconciliation

**Rule:** the checkpoint wins. Reconcile = run the projector over a fresh `aget_state` snapshot. It is idempotent, so it can run at any time and any number of times.

**Triggers:**

1. WS connect on an existing thread, **before** any resume or new start.
2. T3 (`finally`) after an abnormal end.
3. A beat sweep over non-terminal sessions (bounded batch), as the backstop for lost hints. [v2] The sweep passes aliveness = false unless the lock heartbeat is fresh, and it is what moves session.status for orphaned runs. **[v3] When Redis is unavailable, the sweep skips status derivation entirely: it cannot judge aliveness, so it derives no verdicts, writes no rows, and marks no terminal closes; it retries on the next beat after Redis returns.**
4. A Redis hint set (`saul:reconcile`) written best-effort whenever a transcript write fails or times out.

**Cases:**

| Situation | Resolution |
| --- | --- |
| Crash mid-turn | **[v2]** Checkpoint holds state up to the last completed super-step. On reconnect, reconcile projects it. The turn stays `active` while the lock heartbeat is fresh; once `aliveness` is false and the grace period has passed it derives `failed` (orphaned). No auto-resume of crashed runs (D1). |
| WS drop before finalization | **[v2]** Same as a crash. T3 runs the projector over a fresh `aget_state`; if the derived status is terminal, the closing row comes out of that projection like any other row. |
| Resume after process death | **[v2]** Connect, acquire the thread lock (`THREAD_BUSY` if held), pass the `hydrate_state` gate, check identity, reconcile, check expiry from the fresh snapshot, re-emit the HITL frame **re-derived from `state.tasks`**, await the reply, then send `Command(resume=...)`. The transcript shows a `*_reply` row appended at the next projection, linked to the prompt it answers. |
| **[v3] Redis outage** | Connections refused (§5.5); sweeps skip derivation (no verdicts, no orphan closes); statuses freeze and unfreeze on the next beat after Redis returns. In-flight WS turns were refused at connect; in-flight graph runs keep running, and their turn status is derived when their next projection (after recovery) fires. |
| Transcript behind checkpoint | Next trigger projects the gap. Order is by `(turn_seq, ordinal)`, so late inserts display correctly. |
| Missing prompt row (hole in ordinals) | Recover from `aget_state_history`; fill by `source_key`. |
| Transcript ahead of checkpoint | **[v2]** Impossible, with no exceptions. Every row is derived from a snapshot; the v1 `turn_closed(failed)` write on a loop exception is deleted. |
| Two projectors race | Session row lock plus unique `source_key`; the loser no-ops. |

**Tokens fill-once.** Usage arrives as a side-input from the live recorder. On conflict the upsert does `SET tokens_used = COALESCE(chat_messages.tokens_used, EXCLUDED.tokens_used)`, so it fills a null once and never overwrites. If the reconciler wins the race, usage is lost. That under-bills, which is the safe direction, and it never double-bills.

---

## 7. Token accounting

- A callback handler attached to the run config aggregates `usage_metadata` from every LLM call (not the model's own final message). It also collects `tool_calls` (name, node, ok, ms).
- Per **segment** (start→pause, resume→pause, resume→end), totals attach to that segment's last assistant row. A turn's total is `SUM(tokens_used) WHERE turn_seq = N`.
- Replay-safe because the totals ride on a row guarded by `(session_id, source_key)` and the fill-once rule.
- Loss of usage data is fail-open (analytics, not state). Billing today is plan-based (Razorpay); no ledger table is built now, and `tokens_used` keeps the door open for metering.

---

## 8. Retention, lifecycle, export, erasure

**Pause window.** 24 h, derived: `now − snapshot.checkpoint_ts > 24h` at a resume attempt ⇒ frame `SESSION_PAUSE_EXPIRED`, reconcile appends `turn_closed(abandoned_expired)`. Only the same `user_id` may resume, from any device. Authorization needs both `session.user_id` and `state.user_id` to match the authenticated user (§9).

**[v2] Expiry authority.** Expiry is evaluated from the fresh snapshot's `checkpoint_ts` at the resume attempt, because the projection watermark (`last_checkpoint_ts` on the session) can lag. The watermark is used only if the snapshot read fails. The closing row is emitted by the projector: an interrupt older than 24 h derives `abandoned_expired`.

**New start frame on a paused thread (D2).** Within the window: reject `SESSION_PAUSED_AWAITING_RESUME`. After expiry: close the old turn, then start the new one.

**Checkpoint GC.** A daily beat task `chat_transcript_gc`:

1. Non-terminal sessions past their window: reconcile; any closing row comes out of that projection.
2. Terminal sessions whose last checkpoint is older than the retention period (proposed 7 days; D5): verify transcript catch-up, then `adelete_thread(thread_id)`.

**[v2] GC audit.** When `adelete_thread` runs, stamp `extra_metadata.gc_swept_at` and the deleted checkpoint id on the session. It is the only proof that the catch-up verification happened.

The checkpoint contains `document_text`, so GC is also a data-minimization control. After GC a thread is transcript-only. A later start on it begins a clean checkpoint lineage, and `turn_seq` continues from the transcript.

**Transcript retention.** Indefinite until the user deletes or an erasure request is processed.

**Export.** `GET /agent-saul/sessions/{id}/export` reads the transcript only and streams via an async generator (`StreamingResponse`). Redacted rows export as redacted. Reads are bounded (500 rows/page) and walk a keyset cursor — `(turn_seq, ordinal)` the previous page ended on — so a long transcript pages in stable display order without rereading, rows appended mid-export cannot duplicate or skip a page, a read failure yields a terminal `{error, message}` line rather than an incomplete file presented as complete.

**Deletion and erasure** (designed for, **not a V1 build item**). One transaction:

- message rows: `content = NULL`, `redacted_at = now()`, `extra_metadata` reduced to an allowlist (ids, kind, usage, status);
- session: `title = NULL`, `extra_metadata.erased_at`, row retained as a tombstone;
- token and billing metadata survive;
- then `adelete_thread`.

The purge does **not** reach cognee memory, Graphiti/Neo4j, or S3 documents. cognee has no turnkey per-user erasure, so end-to-end DPDP compliance is a flagged follow-up. The transcript purge alone does not make the system compliant.

---

## 9. Failure matrix

| Failure | Policy | Behavior |
| --- | --- | --- |
| Transcript write fails or times out, checkpoint OK | **Open** | Run continues; hint to Redis; sweep and next trigger heal. UI may be stale. |
| Projector bug or unknown item shape | **Open** | Log with ids (no content), alert, no exception crosses the boundary. |
| Checkpoint write fails | **Closed (turn)** | **[v2]** Graph errors and the run fails. The transcript keeps what was projected earlier. The next projection over a fresh snapshot derives the terminal status, and any closing row comes from that projection, not from the exception. |
| Persisted graph-state version unknown (`StateSchemaVersionError`) | **Closed** | Session refused (`STATE_SCHEMA_INCOMPATIBLE`); no projection. Transcript reads (list/export) still work. |
| Transcript `schema_version` unknown | **Closed** | `Failure(TranscriptSchemaError)` at the read boundary naming both versions; never silently upgraded. |
| Identity mismatch (body thread_id, `session.user_id`, `state.user_id`) | **Closed** | Rejected; responds like not-found to prevent enumeration. |
| Memory read fails | **Open** (existing) | Degraded context. |
| Memory write fails | **Open** (existing) | `COGNEE_WRITE_FAILED`; run completes. |
| Token aggregate lost | **Open** | Under-bill only. |
| Redis hint lost | **Open** | Sweep is the backstop. |
| **[v3] Redis unavailable (whole)** | **Closed (connections) / Open (derivation)** | **[v3]** Connections are refused until Redis returns (§5.5). The sweep cannot judge aliveness and therefore **derives nothing**: no orphan closes, no status moves — statuses freeze rather than decay. In-flight graph runs keep running and reconcile after recovery; reconcile hints fail silently. A Redis outage halts new sessions but never fabricates transcript facts. |
| **[v2]** Per-thread connection lock already held | **Closed** | **[v2]** Connection refused: `THREAD_BUSY`. Double-resume is worse than refusal (§5.5). |

**The asymmetry, stated once:** if failure degrades presentation or analytics, continue. If it would make the system act on state it can't read, or serve the wrong person, stop loudly.

---

## 10. Repository and service integration (thin)

```
src/app/features/agent_saul/
  transcript/
    projection.py     # snapshot_to_view, derive_status, project  (pure functions)
    repository.py     # ChatTranscriptRepository
    recorder.py       # TurnRecorder: per-run stateful (thread_id, user_id, usage collector, lock heartbeat)
    errors.py         # typed errors used as Failure payloads
```

- **Result idiom.** Follow the repo's existing convention (billing `PlanResult`): `type TranscriptResult[T] = Success[T] | Failure[TranscriptError]`. Typed errors (details/source/operation) are the *payload*; Results are the *interface*; no exceptions cross repo, service, or graph boundaries. Driver exceptions are mapped to sanitized typed errors at the repo edge **[verify exact convention]**.
- **[v2]** Result types come from the `returns` library (`from returns.result import Failure, Success`). Define `TranscriptResult[T]` in `transcript/errors.py`, mirroring where `PlanResult` is aliased in the billing exceptions module, never in `repository.py`; importing the alias from the repository invites circular imports and hand-inlined unions.
- **Repository** (a class holding an injected session factory; it keeps no per-run state): `apply_projection(projection, usage) -> TranscriptResult[ApplyOutcome]`, `list_sessions(user_id, cursor)`, `read_transcript(session_id, user_id)`, `get_session_meta(...)`. Reads check `schema_version` first. **[v3] `close_turn()` does not exist.** Every terminal row — including every `turn_closed` row — is written through `apply_projection` from a snapshot-derived projection. No repository method and no service call may synthesize a terminal row outside `apply_projection` (chain-of-custody rule, §1).
- **Service hook.** `run_session` builds one `TurnRecorder` per connection and calls `recorder.on_first_checkpoint()`, `on_pause()`, `on_end()`, `on_abort()`: once each, at the matching boundary. Each calls `aget_state` → `snapshot_to_view` → `project` → `repo.apply_projection`, and returns a Result that the loop logs and ignores, except for the project-before-resume retry. `on_abort()` takes no error code: it awaits T0, then projects a fresh snapshot, nothing else.
- **[v3] Lock helper.** A small `ThreadLock` helper (acquire with TTL, periodic renew every TTL/3, best-effort release; §5.5) lives in `features/agent_saul/thread_lock.py` and is owned by the service loop; the recorder reads its heartbeat. Renewal is scheduled by the loop, not by WS traffic.
- **Logging.** `logger.bind(session_id, turn_seq, checkpoint_id)`. Never log content, DSNs, or raw driver messages (they can embed parameters). Errors carry a code and operation only.

---

## 11. Migration plan (to execute only after this doc is accepted)

One revision after the current head:

1. **Precondition:** `SELECT count(*)` on both tables; if either has rows, abort with a clear message.
2. Add columns with server defaults; alter `created_at`/`updated_at` to `server_default=func.now()`; keep the column type as it is **[verify tz-awareness from 0014]**.
3. Create `chat_session_status` with the project's DO-block enum pattern. `message_role` is untouched.
4. Add the `kind` CHECK, the unique constraint, and the indexes (tables are empty, so non-concurrent is fine).
5. Reversible downgrade.
6. Never reference LangGraph's `checkpoint*` tables. **[verify]** that `alembic/env.py` excludes them from autogenerate; otherwise autogenerate will propose dropping them.

**[v2]** If `content` is NOT NULL today, the same revision relaxes it, since erasure stores NULL (exports and `kind` counts then distinguish "no content" from "empty content").

---

## 12. Definition-of-done answers

1. **Where does the one canonical copy of turn N's user message live, and who writes it first?** The canonical content is the `HumanMessage` in the checkpoint (`messages`, stable `message.id`), written first by the checkpointer from the service's initial state. The transcript row is a derived copy, written by the projector at T0 and unique on `(session_id, source_key)`. There is exactly one transcript row per message, no matter how many times it is replayed.
2. **What does the row look like while paused, streaming, completed, abandoned?** See §5.4. Streaming: `query` only. Paused: `query` + prompt row, status `clarifying`. Completed: + `final_summary`. Abandoned: + `turn_closed(abandoned_expired)`; earlier rows unchanged.
3. **Transcript write fails but checkpoint succeeds, and vice versa?** Transcript fails: the run is unaffected, the transcript lags, and reconcile heals it (fail-open). Checkpoint fails: the run fails (fail-closed for the turn); the transcript contains only earlier projections and no closing row from an exception (any closing row comes from the next projection over a fresh snapshot) **[v3]**. The transcript can never be ahead of the checkpoint.
4. **How are tokens derived without trusting the model's final message?** Summed from `usage_metadata` on every LLM call by a callback handler, attached per segment to the segment's last assistant row, filled once, replay-safe.
5. **Same thread resumes after server restart?** Connect → acquire thread lock (THREAD_BUSY if held) → state-version gate → identity check → reconcile (projects the pause) → expiry check → re-emit the prompt from `state.tasks` → `Command(resume=...)`. The transcript shows the reply linked to its prompt, status `resumed`, then `completed`. Past 24 h: `abandoned_expired` instead.

---

## 13. Tests

- **Unit, projector (pure):** snapshot fixtures for fresh, paused, resumed, completed, orphaned, expired, unknown-version, and a missing-prompt hole. Assert rows, ordinals, derived status.
- **Unit, repository** (repo's fake-session idioms): apply twice ⇒ identical rows; apply from a **later** checkpoint ⇒ no duplicates; fill-once tokens; schema-version refusal; sanitized errors.
- **Integration (real Postgres checkpointer):** (a) crash mid-turn (cancel the task or raise in a node) → new loop instance → reconcile → no duplicates, status `failed` after grace; (b) pause → restart the app → resume → rows and statuses correct; (c) frozen clock > 24 h → `abandoned_expired`; (d) GC → `adelete_thread` → transcript intact; (e) transcript write failure injected → run completes → reconcile heals.

**[v2]** Additional cases:

- Projector unit tests include a fixture round-tripped through the real checkpoint serializer (JsonPlusSerializer), asserting that `additional_kwargs.saul_kind` survives. If it did not, turn-boundary detection would fail silently; this is the most load-bearing [verify] in the doc (D4).
- A second connection on a held thread is refused with `THREAD_BUSY`.
- A lapsed heartbeat yields orphaned `failed` only after the grace period and only when `aliveness` is false — **[v3]** and within the grace period it yields the **NoOp** (no rows, unchanged status, unchanged watermark), not a skipped-but-assumed-active projection.
- B1 regression: an exception in the loop alone never produces a closing row. Reconnect within the grace period shows the turn still `active`.

**[v3]** Additional cases:

- **Totality sweep:** combinatorial snapshot fixtures — {interrupts} × {aliveness} × {within/past grace, within/past 24h} × {next empty/non-empty} — every fixture maps to exactly one table row or the NoOp. This is `derive_status`'s test #0.
- **Redis outage:** a beat sweep under outage moves no statuses and writes no closing rows; after Redis returns, the next sweep resumes normal derivation.
- **Redis outage refusal:** a connection attempt while Redis is unavailable fails with the retryable error (fail closed, §9).
- **Sweep closing row origin:** a session past grace + orphaned derives `failed` and the `turn_closed` row appears **only** as an `apply_projection` output (assert no other write path touched the table).
- **`plan_decision` content:** both an `approved` and a `modified` decision render the terse label, never the full text.

---

## 14. Decisions for sign-off and items to verify

**Decisions**

- **D1.** Crashed (orphaned) runs are not auto-resumed. Nodes may have side effects (LLM spend, Graphiti writes) whose idempotency is unknown. After a 15-minute grace they become `failed` and the user re-asks. Opt-in resume can come later.
- **D2.** Start frame on a paused thread: reject within the window; close-and-proceed after expiry.
- **D3.** Index sessions by `(user_id, updated_at DESC)` and read transcripts by `(session_id, turn_seq, ordinal)`, a deliberate deviation from the brief.
- **D4.** Projection needs a one-line tag on the seeded query message at the edge (and optionally on the qna reply). Confirm that is acceptable.
- **D5.** Retention numbers: pause 24 h (given), terminal checkpoint GC 7 days (proposed), transcript indefinite until deletion.
- **D6.** A per-thread connection lock (Redis `SET NX` with TTL) is a precondition for safe resume. [v2] Resolved: the lock does not exist in the repo. Build it, held for connection lifetime (§5.5).

**[v2]** D1–D5 are accepted. D4 carries one condition: the `additional_kwargs` survival assertion in §13.

**Verify against code before building**

1. Where assistant-side content lives (`final_report` field names; whether any `AIMessage` enters `messages`).
2. `Interrupt.id` exists in the pinned LangGraph; shape of `state.tasks[].interrupts`.
3. `additional_kwargs` survives checkpoint serialization; `metadata.step` semantics at the first checkpoint.
4. Where the plan approve/modify decision is persisted in state.
5. `adelete_thread` signature in the pinned `langgraph-checkpoint-postgres` (≥ 3.0.1).
6. Result-type convention in `billing/plans/repository.py`.
7. Alembic env excludes `checkpoint*`; timestamp tz-awareness from 0014.
8. Checkpointer pool flags (kept as-is, with the correct reasons): `autocommit=True` so `.setup()` migrations persist; `prepare_threshold=0` to avoid prepared-statement conflicts on pooled connections (upstream later moved to `None` for PgBouncer-style poolers); `dict_row` because the saver does `row["column"]` access.
9. **[v3] Does the plan resume handler persist the modified plan text into `state.plan`** (derive-able from the snapshot), or only the `plan_approved` boolean? Determines whether `plan_decision.extra_metadata` gains a `{modified: true}` pointer or anything richer. The row's content stays a terse label either way (§3.4).
10. **[v3] Terminal-error classification:** `AgentError` is resolved (below); confirm whether "terminal" for the `failed` derivation means `retryable: False` on the last error, an explicit terminal `code`, or something in the nodes' diagnostics conventions — implementer reads `graph/nodes.py` and the error-taxonomy module; do not invent.

**[v2] Resolved by review against the repo.**

- Assistant-side content is `LegalAgentState.final_report: FinalReport | None` (`graph/state.py:465`, a Pydantic model); read its fields for the summary. `plan_approved: bool` is at `graph/state.py:447` and drives the `plan_decision` row. Items 1 and 4 above are resolved.
- Paths: state is `graph/state.py`; interrupts are raised in `graph/nodes.py`.
- Result convention is the `returns` library, with `PlanResult` aliased in the billing exceptions module (item 6 resolved).
- No lock exists anywhere in `service.py`, `router.py`, or `shared/langgraph_layer/` (D6).

**[v3] Resolved by review against the repo (continued).**

- The `errors` state channel is real: `LegalAgentState.errors: Annotated[list[AgentError], operator.add]` (`graph/state.py:476`), an append-only audit log with a legacy default of `[]` (`state.py:381`). `SnapshotView.errors` is grounded exactly as designed.
- **`AgentError` shape** (`graph/state.py:327`): frozen Pydantic model with fields `node: str`, `code: str`, `message: str`, `retryable: bool = True`. `extra_metadata.errors` records `[{node, code}]` only — `message` is content (content-free rule, §3.4) and `retryable` belongs to the *derivation* rule (item 10), not to a stored flag.
- `_build_initial_state` is at `service.py:428` (called at `:135`) and is where the D4 tag goes; the resume loop is `service.py:316/408` (`Command(resume=resume_payload)`).
