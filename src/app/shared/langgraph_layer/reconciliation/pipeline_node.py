"""
Reconciliation graph nodes for background entity deduplication and conflict resolution.

Triggered by Celery task helpers in `src/tasks/memory_decay_reconcilation_tasks.py`,
not by the main agent workflow.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils import logger

from .prompt import reconcile_prompt
from .state import (
    ReconciliationDecision,
    ReconciliationEntityRecord,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

    from .state import ReconciliationRunnable, ReconciliationState


def make_fetch_existing_node(
    db_engine: AsyncEngine,
) -> Callable[[ReconciliationState], Awaitable[dict[str, object]]]:
    async def fetch_existing_node(state: ReconciliationState) -> dict[str, object]:
        log = logger.bind(node="reconcile_fetch", user_id=state.user_id)
        lookback_hours = state.lookback_hours

        try:
            async with AsyncSession(db_engine) as session:
                new_rows = (
                    await session.execute(
                        text(
                            """
                            SELECT id::text, entity_type, name, normalized_name,
                                   confidence, metadata::text, doc_id
                            FROM entities
                            WHERE user_id = :user_id
                              AND created_at > NOW() - :hours * INTERVAL '1 hour'
                            ORDER BY created_at DESC
                            LIMIT 200
                            """
                        ),
                        {"user_id": state.user_id, "hours": lookback_hours},
                    )
                ).fetchall()

                similar_rows = (
                    await session.execute(
                        text(
                            """
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
                            """
                        ),
                        {"user_id": state.user_id, "hours": lookback_hours},
                    )
                ).fetchall()

            new_entities = [_row_to_record(row) for row in new_rows]
            existing_entities = [_row_to_record(row) for row in similar_rows]
            log.info("fetch_done", new=len(new_entities), existing=len(existing_entities))
        except Exception as exc:  # noqa: BLE001
            log.bind(error=str(exc)).exception("fetch_failed")
            return {"fetch_error": str(exc), "new_entities": [], "existing_entities": []}
        else:
            return {
                "new_entities": new_entities,
                "existing_entities": existing_entities,
                "fetch_error": None,
            }

    return fetch_existing_node


def make_reconcile_node(
    reconcile_llm: ReconciliationRunnable,
) -> Callable[[ReconciliationState], Awaitable[dict[str, object]]]:
    async def reconcile_node(state: ReconciliationState) -> dict[str, object]:
        log = logger.bind(node="reconcile_llm", user_id=state.user_id)

        new_entities = state.new_entities
        existing_entities = state.existing_entities

        if not new_entities:
            log.info("no_entities_to_reconcile")
            return {"reconciliation_decision": ReconciliationDecision()}

        context = json.dumps(
            {
                "new_entities": [entity.model_dump() for entity in new_entities[:50]],
                "existing_entities": [
                    entity.model_dump() for entity in existing_entities[:50]
                ],
            },
            indent=2,
            default=str,
        )

        try:
            result = await reconcile_llm.ainvoke(
                [
                    SystemMessage(content=reconcile_prompt),
                    HumanMessage(content=context),
                ]
            )
            decision = _parse_reconciliation_decision(result)
            log.info(
                "reconcile_done",
                merges=len(decision.merge),
                updates=len(decision.update),
                ignores=len(decision.ignore),
            )
        except (ValidationError, json.JSONDecodeError) as exc:
            log.bind(error=str(exc)).warning("reconcile_invalid_payload")
            return {
                "reconciliation_decision": ReconciliationDecision(),
                "reconcile_error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            log.bind(error=str(exc)).exception("reconcile_failed")
            return {
                "reconciliation_decision": ReconciliationDecision(),
                "reconcile_error": str(exc),
            }
        else:
            return {"reconciliation_decision": decision, "reconcile_error": None}

    return reconcile_node


def make_apply_changes_node(
    db_engine: AsyncEngine,
) -> Callable[[ReconciliationState], Awaitable[dict[str, object]]]:
    async def apply_changes_node(state: ReconciliationState) -> dict[str, object]:
        log = logger.bind(node="reconcile_apply", user_id=state.user_id)
        merges = state.reconciliation_decision.merge
        updates = state.reconciliation_decision.update
        merged_count = 0
        updated_count = 0

        try:
            async with AsyncSession(db_engine) as session, session.begin():
                for merge in merges:
                    await session.execute(
                        text(
                            "UPDATE relationships SET from_entity_id = :keep "
                            "WHERE from_entity_id = :discard"
                        ),
                        {"keep": merge.keep_id, "discard": merge.discard_id},
                    )
                    await session.execute(
                        text(
                            "UPDATE relationships SET to_entity_id = :keep "
                            "WHERE to_entity_id = :discard"
                        ),
                        {"keep": merge.keep_id, "discard": merge.discard_id},
                    )
                    await session.execute(
                        text("DELETE FROM entities WHERE id = :discard"),
                        {"discard": merge.discard_id},
                    )
                    merged_count += 1

                for update in updates:
                    field_updates = _filter_supported_entity_updates(update.fields)
                    if not field_updates:
                        continue

                    for field_name, field_value in field_updates.items():
                        statement_value = (
                            json.dumps(field_value)
                            if field_name == "metadata"
                            else field_value
                        )
                        await session.execute(
                            text(_entity_update_statement(field_name)),
                            {"value": statement_value, "entity_id": update.entity_id},
                        )
                    updated_count += 1

            log.info("apply_done", merged=merged_count, updated=updated_count)
        except Exception as exc:  # noqa: BLE001
            log.bind(error=str(exc)).exception("apply_failed")
            return {"apply_error": str(exc), "merged_count": 0, "updated_count": 0}
        else:
            return {"merged_count": merged_count, "updated_count": updated_count}

    return apply_changes_node


def make_write_versions_node(
    db_engine: AsyncEngine,
) -> Callable[[ReconciliationState], Awaitable[dict[str, object]]]:
    async def write_versions_node(state: ReconciliationState) -> dict[str, object]:
        log = logger.bind(node="reconcile_versions", user_id=state.user_id)
        versions_written = 0

        run_id = state.run_id or str(uuid4())

        try:
            async with AsyncSession(db_engine) as session, session.begin():
                for item in [
                    *state.reconciliation_decision.merge,
                    *state.reconciliation_decision.update,
                ]:
                    entity_id = _version_target_entity_id(item)
                    row = (
                        await session.execute(
                            text("SELECT * FROM entities WHERE id = :id"),
                            {"id": entity_id},
                        )
                    ).fetchone()
                    if not row:
                        continue

                    max_ver_row = (
                        await session.execute(
                            text(
                                "SELECT COALESCE(MAX(version), 0) "
                                "FROM memory_versions WHERE entity_id = :id"
                            ),
                            {"id": entity_id},
                        )
                    ).fetchone()
                    next_version = (max_ver_row[0] if max_ver_row else 0) + 1

                    await session.execute(
                        text(
                            """
                            INSERT INTO memory_versions
                                (id, entity_id, version, data, change_type, source)
                            VALUES
                                (:id, :entity_id, :version, CAST(:data AS JSONB), :change_type, :source)
                            """
                        ),
                        {
                            "id": str(uuid4()),
                            "entity_id": entity_id,
                            "version": next_version,
                            "data": json.dumps(
                                {
                                    "snapshot": dict(getattr(row, "_mapping", {})),
                                    "reason": item.reason,
                                    "run_id": run_id,
                                },
                                default=str,
                            ),
                            "change_type": _change_type(item),
                            "source": f"reconciliation_agent:{run_id}",
                        },
                    )
                    versions_written += 1

            log.info("versions_written", count=versions_written)
        except Exception as exc:  # noqa: BLE001
            log.bind(error=str(exc)).exception("write_versions_failed")
            return {"versions_written": 0, "completed": True}
        else:
            return {"versions_written": versions_written, "completed": True}

    return write_versions_node


def _row_to_record(row: object) -> ReconciliationEntityRecord:
    values = cast("tuple[object, ...]", row)
    return ReconciliationEntityRecord(
        id=str(values[0]),
        entity_type=str(values[1]),
        name=str(values[2]),
        normalized_name=str(values[3]),
        confidence=float(values[4]),
        metadata=json.loads(str(values[5])) if values[5] else {},
        doc_id=str(values[6]) if values[6] is not None else None,
    )


def _parse_reconciliation_decision(result: object) -> ReconciliationDecision:
    content = _read_llm_content(result)
    cleaned_content = _strip_markdown_fences(content)
    return ReconciliationDecision.model_validate_json(cleaned_content)


def _read_llm_content(result: object) -> str:
    content = getattr(result, "content", result)
    return content if isinstance(content, str) else str(content)


def _strip_markdown_fences(content: str) -> str:
    cleaned = content.strip()
    if not cleaned.startswith("```"):
        return cleaned

    segments = cleaned.split("```")
    if len(segments) < 2:
        return cleaned

    fenced_content = segments[1].strip()
    if fenced_content.startswith("json"):
        return fenced_content[4:].strip()
    return fenced_content


def _version_target_entity_id(item: object) -> str:
    keep_id = getattr(item, "keep_id", None)
    if isinstance(keep_id, str):
        return keep_id
    entity_id = getattr(item, "entity_id", None)
    if isinstance(entity_id, str):
        return entity_id
    msg = "reconciliation version item is missing a target entity id"
    raise ValueError(msg)


def _change_type(item: object) -> str:
    return "merge" if hasattr(item, "keep_id") else "update"


def _filter_supported_entity_updates(fields: dict[str, object]) -> dict[str, object]:
    supported_fields = {
        "confidence",
        "entity_type",
        "metadata",
        "name",
        "normalized_name",
    }
    return {
        field_name: field_value
        for field_name, field_value in fields.items()
        if field_name in supported_fields
    }


def _entity_update_statement(field_name: str) -> str:
    statements = {
        "confidence": "UPDATE entities SET confidence = :value WHERE id = :entity_id",
        "entity_type": "UPDATE entities SET entity_type = :value WHERE id = :entity_id",
        "metadata": (
            "UPDATE entities SET metadata = CAST(:value AS JSONB) "
            "WHERE id = :entity_id"
        ),
        "name": "UPDATE entities SET name = :value WHERE id = :entity_id",
        "normalized_name": (
            "UPDATE entities SET normalized_name = :value WHERE id = :entity_id"
        ),
    }
    statement = statements.get(field_name)
    if statement is None:
        msg = "unsupported reconciliation entity update field"
        raise ValueError(msg)
    return statement
