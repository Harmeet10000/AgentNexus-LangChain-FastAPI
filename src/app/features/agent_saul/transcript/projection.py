"""Pure transcript projector — the only derivation of transcript rows (doc §4).

Three pure functions, in dependency order:

- ``snapshot_to_view`` — the ONLY code that reads LangGraph ``StateSnapshot``
  internals (``values``, ``next``, ``tasks``). Everything downstream sees a
  ``SnapshotView``, so interrupt detection cannot fork between the service loop
  and the projector.
- ``derive_status`` — total: every snapshot maps to exactly one outcome, a
  derived status or the explicit :class:`NoOp` (§5.3 closed table + NoOp row).
- ``project`` — pure: rows to insert plus a derived status, or a NoOp.

Purity contract: no imports from the service, no I/O, no wall-clock lookups —
``now`` and ``aliveness`` are *inputs* (§4.1). Knowledge that is
snapshot-independent (reply-row presence, the turn's already-projected rows,
the pause-time plan the user was shown) enters through the turn context
(:class:`TurnRows`), never by enriching ``SnapshotView`` (§4.1 [v3] input
discipline).

Chain-of-custody rule (§1): every row this module emits is derived from a
checkpoint snapshot; exceptions, the clock, and connection state are inputs
to derivation, never evidence in themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum

# Runtime-required: pydantic resolves `dict[str, Any]` field annotations at
# class-definition time — do NOT move to TYPE_CHECKING.
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import StateSnapshot
from pydantic import BaseModel, ConfigDict, Field

from app.features.agent_saul.graph.state import (
    STATE_SCHEMA_VERSION,
    HITLInterruptType,
    StateSchemaVersionError,
)

# Doc §3.2 — the databases (0022 CHECK, models.py). Mirror this tuple; a
# mismatch between the CHECK and this set is a bug in whichever copy was
# edited without the other.
MESSAGE_KINDS: tuple[str, ...] = (
    "query",
    "clarification_prompt",
    "clarification_reply",
    "plan_prompt",
    "plan_decision",
    "final_summary",
    "turn_closed",
)
# §6 recovery vocabulary: keyed ROW FACTS vs derived VERDICT OUTCOMES. The
# hole-fill replay may recover facts only; verdicts are the primary
# projection's job (a historical pause judged at today's time plants a false
# expired outcome otherwise).
VERDICT_ROW_KINDS: frozenset[str] = frozenset({"turn_closed", "final_summary"})

TRANSCRIPT_SCHEMA_VERSION = 1

# Policy constants (D5 / §5.3 / §8) — defined exactly once here; sweep code,
# the repository, and the tests import these values.
ORPHAN_GRACE = timedelta(minutes=15)  # aliveness False + this elapsed ⇒ failed (D1)
PAUSE_WINDOW = timedelta(hours=24)  # interrupts older ⇒ abandoned_expired (§8)

_SAUL_KIND_QUERY = "query"


class SessionStatus(StrEnum):
    """Doc §3.1 / migration 0022 ``chat_session_status`` values."""

    ACTIVE = "active"
    CLARIFYING = "clarifying"
    RESUMED = "resumed"
    COMPLETED = "completed"
    ABANDONED_EXPIRED = "abandoned_expired"
    FAILED = "failed"


class ContentRole(StrEnum):
    """``chat_messages.role`` values (migration 0014 native enum)."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    @classmethod
    def for_kind(cls, kind: str) -> ContentRole:
        match kind:
            case "query" | "clarification_reply" | "plan_decision":
                return cls.USER
            case "clarification_prompt" | "plan_prompt" | "final_summary":
                return cls.ASSISTANT
            case "turn_closed":
                return cls.SYSTEM
            case kind as unreachable:
                msg = f"transcript projector refuses unknown row kind {unreachable!r}"
                raise StateSchemaVersionError(  # fail closed: never store an unknown kind
                    msg
                )


class NoOp(BaseModel):
    """The explicit keep-whatever-is-cached outcome (§5.3 [v3] totality row).

    Type-level contract: a caller matching on ``Projection`` sees a real
    alternative for "derive nothing, write nothing, move no watermark" rather
    than three coincidentally-empty optionals.
    """

    model_config = ConfigDict(frozen=True)

    reason: str
    detail: str | None = None


class ErrorMarker(BaseModel):
    """Content-free view of ``AgentError``: node + code only (doc §3.4 [v3])."""

    model_config = ConfigDict(frozen=True)

    node: str
    code: str


class InterruptView(BaseModel):
    """One pending ``interrupt()`` — stable id plus content-bearing payload."""

    model_config = ConfigDict(frozen=True)

    id: str
    value: dict[str, Any]


class SnapshotView(BaseModel):
    """A frozen fact set derived from one checkpoint snapshot (§4.1)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    user_id: str
    checkpoint_id: str
    checkpoint_ts: datetime | None = None
    step: int | None = None
    next_nodes: tuple[str, ...] = ()
    interrupts: tuple[InterruptView, ...] = ()
    # LangChain messages stay unvalidated raw objects (mixed dynamic data,
    # P5a); ``snapshot_to_view`` filters HumanMessages at the boundary.
    messages: tuple[Any, ...] = ()
    # Long-term memory refs the graph CONFIRMED writing (persist_memory_node
    # appends only on a completed store): the terminal row copies a ref from
    # here instead of fabricating one from doc_id.
    long_term_refs: tuple[str, ...] = ()
    final_report_summary: str | None = None
    errors: tuple[ErrorMarker, ...] = ()
    # §§3.4: the plan_decision row is derived from post-resume state
    # (plan_approved, plan) — these two carry exactly that, as plain data.
    plan: tuple[dict[str, Any], ...] = ()
    plan_approved: bool = False
    # doc_id feeds `report_ref` (`legal::{user_id}::reports#{doc_id}`).
    doc_id: str | None = None
    schema_version: int = STATE_SCHEMA_VERSION


class TurnRow(BaseModel):
    """One already-projected transcript row of the turn under projection."""

    model_config = ConfigDict(frozen=True)

    kind: str
    source_key: str
    ordinal: int


class TurnRows(BaseModel):
    """The turn's already-projected rows in display order (§4.2), plus the
    pause-time plan facts the recorder captured while the turn was paused.

    This is the only door for snapshot-independent knowledge (§4.1 [v3]): the
    turn's reply rows and the plan the user was shown live here, never as
    ``SnapshotView`` enrichment.
    """

    model_config = ConfigDict(frozen=True)

    rows: tuple[TurnRow, ...] = ()
    paused_plan: tuple[dict[str, Any], ...] = ()

    @property
    def query_key(self) -> str | None:
        return next((row.source_key for row in self.rows if row.kind == "query"), None)

    @property
    def has_reply(self) -> bool:
        return any(row.kind in _REPLY_KINDS for row in self.rows)

    @property
    def max_ordinal(self) -> int:
        return max((row.ordinal for row in self.rows), default=0)

    @property
    def last_plan_prompt(self) -> TurnRow | None:
        prompts = [row for row in self.rows if row.kind == "plan_prompt"]
        return prompts[-1] if prompts else None

    def nth_clarification_prompt(self, n: int) -> TurnRow | None:
        prompts = [row for row in self.rows if row.kind == "clarification_prompt"]
        return prompts[n - 1] if 0 < n <= len(prompts) else None


class ProjectedRow(BaseModel):
    """One row the repository will insert (``ON CONFLICT DO NOTHING``)."""

    model_config = ConfigDict(frozen=True)

    kind: str
    role: ContentRole
    ordinal: int
    source_key: str
    content: str | None = None
    model: str | None = None
    tokens_used: int | None = None
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class Projection(BaseModel):
    """Output of ``project``: rows + derived status + watermark, or a NoOp."""

    model_config = ConfigDict(frozen=True)

    session_status: SessionStatus | None = None
    no_op: NoOp | None = None
    watermark_checkpoint_id: str | None = None
    watermark_ts: datetime | None = None
    query_source_key: str | None = None
    rows: tuple[ProjectedRow, ...] = ()

    @property
    def is_no_op(self) -> bool:
        return self.no_op is not None


def _no_op_projection(no_op: NoOp) -> Projection:
    return Projection(session_status=None, no_op=no_op)


def _turn_row_of(row: ProjectedRow) -> TurnRow:
    return TurnRow(kind=row.kind, source_key=row.source_key, ordinal=row.ordinal)


def _saul_kind(message: HumanMessage) -> str | None:
    kwargs = getattr(message, "additional_kwargs", None)
    kind: object = kwargs.get("saul_kind") if isinstance(kwargs, dict) else None
    return kind if isinstance(kind, str) else None


def _message_text(message: HumanMessage) -> str | None:
    content = getattr(message, "content", None)
    return content if isinstance(content, str) and content else None


# ---------------------------------------------------------------------------
# snapshot_to_view — the single LangGraph evidence boundary
# ---------------------------------------------------------------------------


def snapshot_to_view(snapshot: StateSnapshot) -> SnapshotView:
    """Read one ``StateSnapshot`` into a frozen view; the designed chokepoint.

    Unknown persisted state schema versions fail closed here (§9): the caller
    maps the typed error to a ``STATE_SCHEMA_INCOMPATIBLE`` refusal and never
    projects. Legacy shapes (version 0 / missing key) arrive already
    normalized by ``hydrate_state``, which the doc requires to have run.
    """
    values = snapshot.values if isinstance(snapshot.values, dict) else {}

    recorded_version = values.get("schema_version")
    if recorded_version not in {None, 0, STATE_SCHEMA_VERSION}:
        msg = (
            f"refusing snapshot: schema_version={recorded_version!r} "
            f"cannot be projected by this runtime (expected {STATE_SCHEMA_VERSION!r})"
        )
        raise StateSchemaVersionError(msg)

    metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
    step_value = metadata.get("step")
    configurable = (snapshot.config or {}).get("configurable") or {}
    created = _coerce_checkpoint_ts(snapshot.created_at)

    doc_id = values.get("doc_id")
    return SnapshotView(
        user_id=str(values.get("user_id") or ""),
        checkpoint_id=str(configurable.get("checkpoint_id") or ""),
        checkpoint_ts=created,
        step=step_value if isinstance(step_value, int) else None,
        next_nodes=tuple(str(node) for node in snapshot.next or ()),
        interrupts=_interrupt_views_of(snapshot),
        messages=_messages_of(values),
        final_report_summary=_final_report_summary(values.get("final_report")),
        long_term_refs=tuple(str(ref) for ref in values.get("long_term_refs") or () if ref),
        errors=_error_markers_of(values),
        plan=_plan_of(values),
        plan_approved=bool(values.get("plan_approved", False)),
        doc_id=str(doc_id) if doc_id else None,
        schema_version=STATE_SCHEMA_VERSION,
    )


def _messages_of(values: Mapping[str, Any]) -> tuple[HumanMessage, ...]:
    """Tagged-opener window: only HumanMessages matter to the projector."""
    raw_messages = values.get("messages")
    return (
        tuple(raw_message for raw_message in raw_messages if isinstance(raw_message, HumanMessage))
        if isinstance(raw_messages, list)
        else ()
    )


def _error_markers_of(values: Mapping[str, Any]) -> tuple[ErrorMarker, ...]:
    raw_errors = values.get("errors")
    return (
        tuple(
            ErrorMarker(node=str(raw_error.node), code=str(raw_error.code))
            for raw_error in raw_errors
            if getattr(raw_error, "node", None) is not None
        )
        if isinstance(raw_errors, list)
        else ()
    )


def _plan_step_dict(raw_step: object) -> dict[str, Any]:
    dump = getattr(raw_step, "model_dump", None)
    if callable(dump):
        dumped = dump()
        if isinstance(dumped, Mapping):
            return {str(key): value for key, value in dumped.items()}
        return {}
    if isinstance(raw_step, Mapping):
        return {str(key): value for key, value in raw_step.items()}
    return {}


def _plan_of(values: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_plan = values.get("plan")
    return (
        tuple(_plan_step_dict(raw_step) for raw_step in raw_plan)
        if isinstance(raw_plan, list)
        else ()
    )


def _interrupt_views_of(snapshot: StateSnapshot) -> tuple[InterruptView, ...]:
    task_interrupts: list[object] = []
    for task in snapshot.tasks:
        # Task appears as a Mapped TypedDict (checkpoint payload) or a live
        # task object (aget_state result) — genuinely dynamic data (P5a).
        raw = (
            task.get("interrupts")
            if isinstance(task, Mapping)
            else getattr(task, "interrupts", None)
        )
        if isinstance(raw, list | tuple):
            task_interrupts.extend(raw)
    return tuple(
        InterruptView(
            id=str(getattr(interrupt, "id", "") or ""),
            value=_interrupt_value(getattr(interrupt, "value", None)),
        )
        for interrupt in task_interrupts
    )


def _final_report_summary(final_report: object) -> str | None:
    summary = getattr(final_report, "summary", None)
    if summary is None and isinstance(final_report, dict):
        summary = final_report.get("summary")
    return summary if isinstance(summary, str) and summary else None


def _coerce_checkpoint_ts(raw: object) -> datetime | None:
    """LangGraph hands ``created_at`` back as an ISO string; normalize once."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed
    return None


def _interrupt_value(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    return {"type": type(raw).__name__}


# ---------------------------------------------------------------------------
# derive_status — the total function (§5.3 closed table; sweep test #0)
# ---------------------------------------------------------------------------

type StatusOrNoOp = SessionStatus | NoOp


def derive_status(
    view: SnapshotView,
    turn_rows: TurnRows,
    now: datetime,
    aliveness: bool,
) -> StatusOrNoOp:
    """Every snapshot maps to exactly one derived status or the NoOp.

    Reading order is the closed table §5.3; each branch below cites its row:
    interrupts → next-empty → running (with the [v3] NoOp inside it).
    """
    checkpoint_ts = view.checkpoint_ts
    age = now - checkpoint_ts if checkpoint_ts is not None else None

    # Rows 3-4: pending interrupts (aliveness irrelevant per the table).
    if view.interrupts:
        if age is None or age <= PAUSE_WINDOW:
            return SessionStatus.CLARIFYING
        return SessionStatus.ABANDONED_EXPIRED

    # Rows 5-6: run reached a terminal point.
    if not view.next_nodes:
        if view.final_report_summary is not None:
            return SessionStatus.COMPLETED
        return SessionStatus.FAILED

    # Rows 1-2 (+NoOp): running; aliveness and grace decide.
    if aliveness:
        return SessionStatus.RESUMED if turn_rows.has_reply else SessionStatus.ACTIVE
    if age is None or age <= ORPHAN_GRACE:
        return NoOp(reason="heartbeat_stale_within_grace")
    return SessionStatus.FAILED  # orphaned past grace (D1)


_REPLY_KINDS = frozenset({"clarification_reply", "plan_decision"})
_TERMINAL_STATUSES = frozenset((SessionStatus.FAILED, SessionStatus.ABANDONED_EXPIRED))


# ---------------------------------------------------------------------------
# project — rows from structure only
# ---------------------------------------------------------------------------


def _reply_rows(
    view: SnapshotView,
    *,
    query_index: int,
    existing_keys: frozenset[str],
    turn_rows: TurnRows,
) -> tuple[ProjectedRow, ...]:
    """Untagged HumanMessages after the CURRENT query are HITL replies (§4.2).

    The scan starts AFTER the latest tagged query — earlier messages belong
    to previous turns whose rows already carry their own keys; projecting
    "everything after the oldest query" would re-attribute later turns to
    the opener of turn one (each reply dedupes by id, but slot arithmetic
    would drift). The reply answers the j-th clarification prompt of this
    turn; a missing prompt row leaves a detectable hole that reconciliation
    fills by source_key, so the reply keys to its prompt when it exists and
    falls back to the structural slot formula (2*j) when the prompt row is
    still a hole.
    """
    replies: list[ProjectedRow] = []
    reply_index = 0
    for message in view.messages[query_index + 1 :]:
        if _saul_kind(message) is not None:
            continue
        reply_index += 1
        reply_id = getattr(message, "id", None)
        if not reply_id or reply_id in existing_keys:
            continue
        prompt_row = turn_rows.nth_clarification_prompt(reply_index)
        replies.append(
            ProjectedRow(
                kind="clarification_reply",
                role=ContentRole.USER,
                ordinal=prompt_row.ordinal + 1 if prompt_row else 2 * reply_index,
                source_key=str(reply_id),
                content=_message_text(message),
                extra_metadata={
                    "hitl": {"answers_prompt": (prompt_row.source_key if prompt_row else None)}
                },
            )
        )
    return tuple(replies)


def _pending_prompt_rows(
    view: SnapshotView,
    *,
    existing_keys: frozenset[str],
    next_slot: int,
) -> tuple[ProjectedRow, ...]:
    """One prompt row per interrupt not yet projected (§4.1), slot-escalated."""
    rows: list[ProjectedRow] = []
    for interrupt_view in view.interrupts:
        source_key = f"interrupt:{interrupt_view.id}"
        if source_key in existing_keys:
            continue
        hitl_type = interrupt_view.value.get("type")
        updated_type = (
            str(hitl_type) if isinstance(hitl_type, str) else HITLInterruptType.PLAN_APPROVAL
        )
        rows.append(
            ProjectedRow(
                kind=(
                    "plan_prompt"
                    if updated_type == str(HITLInterruptType.PLAN_APPROVAL)
                    else "clarification_prompt"
                ),
                role=ContentRole.ASSISTANT,
                ordinal=next_slot + len(rows) + 1,
                source_key=source_key,
                content=_prompt_text(interrupt_view.value),
                extra_metadata={
                    "interrupt": {
                        "id": interrupt_view.id,
                        "checkpoint_id": view.checkpoint_id,
                    },
                    "hitl": {"type": updated_type},
                },
            )
        )
    return tuple(rows)


def _decision_row(
    view: SnapshotView,
    *,
    existing_keys: frozenset[str],
    turn_rows: TurnRows,
) -> ProjectedRow | None:
    """Plan decision from post-resume state (§3.4); ``None`` when absent."""
    last_prompt = turn_rows.last_plan_prompt
    if not view.plan_approved or last_prompt is None:
        return None
    decision_key = f"decision:{last_prompt.source_key.removeprefix('interrupt:')}"
    if decision_key in existing_keys:
        return None
    was_modified = bool(turn_rows.paused_plan) and (tuple(turn_rows.paused_plan) != view.plan)
    return ProjectedRow(
        kind="plan_decision",
        role=ContentRole.USER,
        ordinal=last_prompt.ordinal + 1,
        source_key=decision_key,
        # Terse derived label; never the plan text (§3.4 [v3]).
        content="modified" if was_modified else "approved",
        extra_metadata={
            "hitl": {"answers_prompt": last_prompt.source_key},
            "modified": was_modified,
        },
    )


def _terminal_rows(
    view: SnapshotView,
    *,
    query_id: str,
    existing_keys: frozenset[str],
    next_slot: int,
    status: SessionStatus,
) -> tuple[ProjectedRow, ...]:
    """Final summary (COMPLETED) and turn_closed (non-completion terminal)."""
    rows: list[ProjectedRow] = []

    # 5. Final summary — summary/title only, never the report JSON.
    if status == SessionStatus.COMPLETED and view.final_report_summary:
        final_key = f"final:{query_id}"
        if final_key not in existing_keys:
            extra_metadata: dict[str, Any] = {}
            # Chain-of-custody: the ref is COPIED from what the graph
            # actually confirmed persisting (persist_memory_node appends to
            # ``long_term_refs`` only after a completed store). A skipped or
            # failed storage leaves no ref → the export claims nothing (§3.4:
            # metadata must be facts, never identity-derived placeholders).
            if view.long_term_refs:
                extra_metadata["report_ref"] = view.long_term_refs[-1]
            rows.append(
                ProjectedRow(
                    kind="final_summary",
                    role=ContentRole.ASSISTANT,
                    ordinal=next_slot + 1,
                    source_key=final_key,
                    content=view.final_report_summary,
                    model=None,  # model identity rides the usage side input, not the snapshot
                    extra_metadata=extra_metadata,
                )
            )

    # 6. Terminal non-completion closes the turn from within this projection.
    if status in _TERMINAL_STATUSES:
        closed_key = f"closed:{query_id}"
        if closed_key not in existing_keys:
            rows.append(
                ProjectedRow(
                    kind="turn_closed",
                    role=ContentRole.SYSTEM,
                    ordinal=next_slot + 1,
                    source_key=closed_key,
                    content=None,
                    extra_metadata={
                        "outcome": str(status),
                        "errors": [
                            {"node": marker.node, "code": marker.code} for marker in view.errors
                        ],
                    },
                )
            )
    return tuple(rows)


def project(
    view: SnapshotView,
    existing_keys: frozenset[str],
    turn_rows: TurnRows,
    now: datetime,
    aliveness: bool,
) -> Projection:
    """Derive rows + status from the snapshot and turn context, purely.

    Ordinals are structural (§4.2): the tagged query is 0; every projected /
    pending item takes the next slot after the turn's rows, so replaying from
    any later checkpoint reproduces the same ordering without insert-order
    dependence.
    """
    # First-checkpoint rule (§4.2): no checkpoint with a completed super-step
    # yet — the turn never existed, nothing may exist.
    if view.step is not None and view.step < 0:
        return _no_op_projection(NoOp(reason="pre_first_super_step"))

    # The LATEST tagged query is the current turn's opener: `messages`
    # accumulates across turns (add_messages), and the OLDEST match would
    # re-attach a later turn's rows to turn one's key.
    query_index: int | None = None
    for candidate in range(len(view.messages) - 1, -1, -1):
        if _saul_kind(view.messages[candidate]) == _SAUL_KIND_QUERY:
            query_index = candidate
            break
    if query_index is None:
        # Chain-of-custody: without the tagged opener this snapshot cannot
        # prove which turn it belongs to — no facts, no rows (fail-open for an
        # ingest anomaly, §1).
        return _no_op_projection(NoOp(reason="missing_tagged_query"))
    query = view.messages[query_index]
    query_id = str(getattr(query, "id", "") or "")
    if not query_id:
        return _no_op_projection(NoOp(reason="missing_tagged_query"))

    rows: list[ProjectedRow] = []
    next_slot = turn_rows.max_ordinal  # the query row occupies 0

    def add_row(row: ProjectedRow) -> None:
        nonlocal next_slot
        rows.append(row)
        next_slot = max(next_slot, row.ordinal)

    # 1. Query row — the turn's opener, ordinal 0.
    if query_id not in existing_keys:
        add_row(
            ProjectedRow(
                kind="query",
                role=ContentRole.USER,
                ordinal=0,
                source_key=query_id,
                content=_message_text(query),
                extra_metadata={"hitl": {"type": _SAUL_KIND_QUERY}},
            )
        )

    # 2-4: replies, plan decision, pending prompts — each keyed structurally
    # (§4.2); the shared add_row keeps slot monotonicity. The decision is
    # resolved BEFORE the prompts it may share a projection with: its slot
    # (last_prompt.ordinal + 1) must be reserved first, or a follow-up prompt
    # would take that slot and exports would show it before the approval
    # that preceded it.
    replies = _reply_rows(
        view, query_index=query_index, existing_keys=existing_keys, turn_rows=turn_rows
    )
    decision = _decision_row(view, existing_keys=existing_keys, turn_rows=turn_rows)
    occupied = [
        turn_rows.max_ordinal,
        *(row.ordinal for row in replies),
        *((decision.ordinal,) if decision is not None else ()),
    ]
    prompt_rows = _pending_prompt_rows(
        view,
        existing_keys=existing_keys,
        next_slot=max(occupied),
    )
    candidates = (*replies, *prompt_rows, *(() if decision is None else (decision,)))
    for row in candidates:
        add_row(row)

    # Derive the status with the turn context INCLUDING the rows this very
    # projection is about to apply: without it, the T1' resume — which creates
    # the reply/decision row — would derive `active` instead of `resumed`
    # (§5.3 branch 1 reads reply knowledge from turn_rows). Enrichment is a
    # pure, replay-stable operation: the same rows re-candidate on replay.
    enhanced_rows = TurnRows(
        rows=(
            *turn_rows.rows,
            *(_turn_row_of(row) for row in rows),
        ),
        paused_plan=turn_rows.paused_plan,
    )
    status = derive_status(view, enhanced_rows, now, aliveness)
    if isinstance(status, NoOp):
        return _no_op_projection(status)

    watermark = view.checkpoint_id or None

    # 5-6: final summary and turn_closed ride the derived status.
    for row in _terminal_rows(
        view,
        query_id=query_id,
        existing_keys=existing_keys,
        next_slot=next_slot,
        status=status,
    ):
        add_row(row)

    return Projection(
        session_status=status,
        no_op=None,
        watermark_checkpoint_id=watermark,
        watermark_ts=view.checkpoint_ts,
        query_source_key=str(query_id),
        rows=tuple(sorted(rows, key=lambda row: row.ordinal)),
    )


def _prompt_text(payload: dict[str, Any]) -> str | None:
    # The clarification interrupt carries the actual question in ``question``
    # while ``message`` stays a generic notice; export shows what the user's
    # reply answers, so prefer the question text when the interrupt has one.
    for key in ("question", "message"):
        text = payload.get(key)
        if isinstance(text, str) and text:
            return text
    return None
