"""
ReconciliationGraph: background entity deduplication + conflict resolution.

Triggered by Celery (src/tasks/reconciliation_task.py), NOT part of agent_saul.

Pipeline:
  fetch_existing_node:  query entities table for recently added entities
                        + find similar entities by normalized_name fuzzy match
  reconcile_node:       LLM with RECONCILE_PROMPT → {merge: [], update: [], ignore: []}
  apply_changes_node:   execute DB mutations (merge = update from_entity refs, delete dup)
  write_versions_node:  INSERT into memory_versions (before/after snapshots)

RECONCILE_PROMPT uses loss aversion bias (Section 16.4):
  "NEVER delete without justification" + "Prefer recent + higher confidence"
  → lower hallucination, higher abstention correctness
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict
from uuid import uuid4

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.runnables import Runnable
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = structlog.get_logger(__name__)

_RECONCILE_PROMPT = """
You are a memory reconciliation system for a legal knowledge graph.

Given:
- new_entities: recently extracted entities (last 24 hours)
- existing_entities: similar entities already in the database

Tasks:
1. Detect duplicates: same party/clause by normalized name across different extractions
2. Resolve conflicts: contradicting confidence scores or metadata for same entity
3. Merge entities: combine if they represent the same real-world entity
4. Update confidence: prefer higher confidence when merging

Rules (CRITICAL):
- Prefer RECENT data over old data when both are valid
- Prefer HIGHER confidence scores
- NEVER delete an entity without explicit justification in the reason field
- When uncertain: IGNORE (do not merge or update)
- Normalized names must match to be merge candidates

Output ONLY this JSON, no prose:
{
  "merge": [
    {
      "keep_id": "uuid-to-keep",
      "discard_id": "uuid-to-discard",
      "reason": "..."
    }
  ],
  "update": [
    {
      "entity_id": "uuid",
      "fields": {"confidence": 0.95, "normalized_name": "..."},
      "reason": "..."
    }
  ],
  "ignore": [
    {
      "entity_id": "uuid",
      "reason": "..."
    }
  ]
}
"""


class ReconciliationState(TypedDict, total=False):
    user_id: str
    run_id: str
    lookback_hours: int

    new_entities: list[dict[str, Any]]
    existing_entities: list[dict[str, Any]]
    fetch_error: str | None

    reconciliation_decision: dict[str, Any]
    reconcile_error: str | None

    merged_count: int
    updated_count: int
    versions_written: int
    apply_error: str | None

    completed: bool


def make_fetch_existing_node(db_engine: AsyncEngine) -> Any:
    async def fetch_existing_node(state: ReconciliationState) -> ReconciliationState:
        log = logger.bind(node="reconcile_fetch", user_id=state.get("user_id"))
        lookback_hours = state.get("lookback_hours", 24)

        # Fetch recently added entities for this user
        recent_query = text("""
            SELECT id, entity_type, name, normalized_name, confidence, metadata,
                   created_at, doc_id
            FROM entities
            WHERE user_id = :user_id
              AND created_at > NOW() - INTERVAL ':hours hours'
            ORDER BY created_at DESC
            LIMIT 200
        """)

        # Find similar existing entities by normalized_name prefix match
        similar_query = text("""
            SELECT DISTINCT e2.id, e2.entity_type, e2.name, e2.normalized_name,
                   e2.confidence, e2.metadata, e2.created_at, e2.doc_id
            FROM entities e1
            JOIN entities e2
              ON e2.normalized_name LIKE LEFT(e1.normalized_name, 10) || '%'
             AND e2.entity_type = e1.entity_type
             AND e2.id != e1.id
             AND e2.user_id = :user_id
            WHERE e1.user_id = :user_id
              AND e1.created_at > NOW() - INTERVAL ':hours hours'
            LIMIT 100
        """)

        try:
            async with AsyncSession(db_engine) as session:
                new_rows = (await session.execute(
                    text("""
                        SELECT id::text, entity_type, name, normalized_name,
                               confidence, metadata::text, doc_id
                        FROM entities
                        WHERE user_id = :user_id
                          AND created_at > NOW() - :hours * INTERVAL '1 hour'
                        ORDER BY created_at DESC LIMIT 200
                    """),
                    {"user_id": state.get("user_id"), "hours": lookback_hours},
                )).fetchall()

                similar_rows = (await session.execute(
                    text("""
                        SELECT DISTINCT e2.id::text, e2.entity_type, e2.name,
                               e2.normalized_name, e2.confidence, e2.metadata::text, e2.doc_id
                        FROM entities e1
                        JOIN entities e2
                          ON LEFT(e2.normalized_name, 10) = LEFT(e1.normalized_name, 10)
                         AND e2.entity_type = e1.entity_type
                         AND e2.id != e1.id
                         AND e2.user_id = :user_id
                        WHERE e1.user_id = :user_id
                          AND e1.created_at > NOW() - :hours * INTERVAL '1 hour'
                        LIMIT 100
                    """),
                    {"user_id": state.get("user_id"), "hours": lookback_hours},
                )).fetchall()

            def _row_to_dict(row: Any) -> dict[str, Any]:
                return {
                    "id": row[0],
                    "entity_type": row[1],
                    "name": row[2],
                    "normalized_name": row[3],
                    "confidence": float(row[4]),
                    "metadata": json.loads(row[5]) if row[5] else {},
                    "doc_id": row[6],
                }

            new_entities = [_row_to_dict(r) for r in new_rows]
            existing_entities = [_row_to_dict(r) for r in similar_rows]

            log.info("fetch_done", new=len(new_entities), existing=len(existing_entities))
            return {
                "new_entities": new_entities,
                "existing_entities": existing_entities,
                "fetch_error": None,
            }
        except Exception as exc:  # noqa: BLE001
            log.error("fetch_failed", error=str(exc))
            return {"fetch_error": str(exc), "new_entities": [], "existing_entities": []}

    return fetch_existing_node


def make_reconcile_node(reconcile_llm: Runnable[list[Any], Any]) -> Any:
    async def reconcile_node(state: ReconciliationState) -> ReconciliationState:
        log = logger.bind(node="reconcile_llm", user_id=state.get("user_id"))

        new_entities = state.get("new_entities", [])
        existing_entities = state.get("existing_entities", [])

        if not new_entities:
            log.info("no_entities_to_reconcile")
            return {"reconciliation_decision": {"merge": [], "update": [], "ignore": []}}

        context = json.dumps(
            {"new_entities": new_entities[:50], "existing_entities": existing_entities[:50]},
            indent=2,
            default=str,
        )

        try:
            result = await reconcile_llm.ainvoke([
                SystemMessage(content=_RECONCILE_PROMPT),
                HumanMessage(content=context),
            ])
            content = result.content if hasattr(result, "content") else str(result)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            decision = json.loads(content)
            log.info(
                "reconcile_done",
                merges=len(decision.get("merge", [])),
                updates=len(decision.get("update", [])),
                ignores=len(decision.get("ignore", [])),
            )
            return {"reconciliation_decision": decision, "reconcile_error": None}
        except Exception as exc:  # noqa: BLE001
            log.error("reconcile_failed", error=str(exc))
            return {
                "reconciliation_decision": {"merge": [], "update": [], "ignore": []},
                "reconcile_error": str(exc),
            }

    return reconcile_node


def make_apply_changes_node(db_engine: AsyncEngine) -> Any:
    async def apply_changes_node(state: ReconciliationState) -> ReconciliationState:
        log = logger.bind(node="reconcile_apply", user_id=state.get("user_id"))
        decision = state.get("reconciliation_decision", {})
        merges = decision.get("merge", [])
        updates = decision.get("update", [])
        merged_count = 0
        updated_count = 0

        try:
            async with AsyncSession(db_engine) as session:
                async with session.begin():
                    for merge in merges:
                        keep_id = merge.get("keep_id")
                        discard_id = merge.get("discard_id")
                        if not keep_id or not discard_id:
                            continue
                        # Redirect all relationship foreign keys from discard → keep
                        await session.execute(text(
                            "UPDATE relationships SET from_entity_id = :keep WHERE from_entity_id = :discard"
                        ), {"keep": keep_id, "discard": discard_id})
                        await session.execute(text(
                            "UPDATE relationships SET to_entity_id = :keep WHERE to_entity_id = :discard"
                        ), {"keep": keep_id, "discard": discard_id})
                        # Delete duplicate entity
                        await session.execute(text(
                            "DELETE FROM entities WHERE id = :discard"
                        ), {"discard": discard_id})
                        merged_count += 1

                    for update in updates:
                        entity_id = update.get("entity_id")
                        fields = update.get("fields", {})
                        if not entity_id or not fields:
                            continue
                        set_clauses = ", ".join(f"{k} = :{k}" for k in fields)
                        await session.execute(
                            text(f"UPDATE entities SET {set_clauses} WHERE id = :entity_id"),
                            {**fields, "entity_id": entity_id},
                        )
                        updated_count += 1

            log.info("apply_done", merged=merged_count, updated=updated_count)
            return {"merged_count": merged_count, "updated_count": updated_count}
        except Exception as exc:  # noqa: BLE001
            log.error("apply_failed", error=str(exc))
            return {"apply_error": str(exc), "merged_count": 0, "updated_count": 0}

    return apply_changes_node


def make_write_versions_node(db_engine: AsyncEngine) -> Any:
    async def write_versions_node(state: ReconciliationState) -> ReconciliationState:
        log = logger.bind(node="reconcile_versions", user_id=state.get("user_id"))
        decision = state.get("reconciliation_decision", {})
        versions_written = 0

        merges = decision.get("merge", [])
        updates = decision.get("update", [])
        run_id = state.get("run_id", str(uuid4()))

        try:
            async with AsyncSession(db_engine) as session:
                async with session.begin():
                    # Get current max version per entity
                    for item in [*merges, *updates]:
                        entity_id = item.get("keep_id") or item.get("entity_id")
                        if not entity_id:
                            continue

                        # Fetch current entity state for snapshot
                        row = (await session.execute(
                            text("SELECT * FROM entities WHERE id = :id"), {"id": entity_id}
                        )).fetchone()
                        if not row:
                            continue

                        # Get next version number
                        max_ver_row = (await session.execute(
                            text("SELECT COALESCE(MAX(version), 0) FROM memory_versions WHERE entity_id = :id"),
                            {"id": entity_id},
                        )).fetchone()
                        next_version = (max_ver_row[0] if max_ver_row else 0) + 1

                        change_type = "merge" if item.get("keep_id") else "update"
                        await session.execute(text("""
                            INSERT INTO memory_versions
                                (id, entity_id, version, data, change_type, source)
                            VALUES
                                (:id, :entity_id, :version, :data::jsonb, :change_type, :source)
                        """), {
                            "id": str(uuid4()),
                            "entity_id": entity_id,
                            "version": next_version,
                            "data": json.dumps({
                                "snapshot": dict(row._mapping) if hasattr(row, "_mapping") else {},
                                "reason": item.get("reason", ""),
                                "run_id": run_id,
                            }, default=str),
                            "change_type": change_type,
                            "source": f"reconciliation_agent:{run_id}",
                        })
                        versions_written += 1

            log.info("versions_written", count=versions_written)
            return {"versions_written": versions_written, "completed": True}
        except Exception as exc:  # noqa: BLE001
            log.error("write_versions_failed", error=str(exc))
            return {"versions_written": 0, "completed": True}

    return write_versions_node


def build_reconciliation_graph(
    reconcile_llm: Runnable[list[Any], Any],
    db_engine: AsyncEngine,
) -> CompiledStateGraph:
    """Build ReconciliationGraph. Called once at lifespan, stored in app.state.

    No checkpointer — reconciliation is idempotent; if it fails, re-run.
    """
    graph = StateGraph(ReconciliationState)
    graph.add_node("fetch_existing", make_fetch_existing_node(db_engine))
    graph.add_node("reconcile", make_reconcile_node(reconcile_llm))
    graph.add_node("apply_changes", make_apply_changes_node(db_engine))
    graph.add_node("write_versions", make_write_versions_node(db_engine))

    graph.set_entry_point("fetch_existing")
    graph.add_edge("fetch_existing", "reconcile")
    graph.add_edge("reconcile", "apply_changes")
    graph.add_edge("apply_changes", "write_versions")
    graph.add_edge("write_versions", END)

    return graph.compile()
