"""SQLAlchemy implementation of the audit repository port."""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.features.audit.domain.errors import AuditInfrastructureError
from app.kernel.result import add_database_error_note
from app.observability.logging import logger, trace_layer

from .mappers import to_entity, to_model
from .models import AuditLog

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.features.audit.domain.entities.audit_entry import AuditEntry
    from app.features.audit.domain.errors import AuditResult


class PostgresAuditLogRepository:
    """Append-only audit repository bound to a caller-owned AsyncSession."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @trace_layer("repository")
    async def create(self, entry: AuditEntry) -> AuditResult[AuditEntry]:
        try:
            # This compatibility branch keeps untouched pre-Phase-1 consumers
            # working while the application port receives only domain entities.
            row = entry if isinstance(entry, AuditLog) else to_model(entry)
            self.session.add(row)
            await self.session.flush()
            return Success(entry)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="audit_logs")
            await self.session.rollback()
            logger.bind(
                operation="create", entity_type=entry.entity_type, error=str(exc)
            ).exception("audit_repository_failed")
            return Failure(
                _database_error("create", {"entity_type": entry.entity_type, "error": str(exc)})
            )

    @trace_layer("repository")
    async def find_by_entity(
        self,
        entity_type: str,
        entity_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> AuditResult[list[AuditEntry]]:
        try:
            statement = (
                select(AuditLog)
                .where(AuditLog.entity_type == entity_type, AuditLog.entity_id == entity_id)
                .order_by(AuditLog.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(statement)
            return Success([to_entity(row) for row in result.scalars().all()])
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="audit_logs")
            await self.session.rollback()
            logger.bind(
                operation="find_by_entity",
                entity_type=entity_type,
                entity_id=entity_id,
                error=str(exc),
            ).exception("audit_repository_failed")
            return Failure(
                _database_error(
                    "find_by_entity",
                    {"entity_type": entity_type, "entity_id": entity_id, "error": str(exc)},
                )
            )

    @trace_layer("repository")
    async def query(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        action: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AuditResult[tuple[list[AuditEntry], int]]:
        try:
            conditions = []
            if entity_type is not None:
                conditions.append(AuditLog.entity_type == entity_type)
            if entity_id is not None:
                conditions.append(AuditLog.entity_id == entity_id)
            if action is not None:
                conditions.append(AuditLog.action == action)
            if date_from is not None:
                conditions.append(AuditLog.created_at >= date_from)
            if date_to is not None:
                conditions.append(AuditLog.created_at <= date_to)

            total = (
                await self.session.execute(
                    select(func.count()).select_from(AuditLog).where(*conditions)
                )
            ).scalar_one()
            statement = (
                select(AuditLog)
                .where(*conditions)
                .order_by(AuditLog.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(statement)
            return Success(([to_entity(row) for row in result.scalars().all()], int(total)))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="audit_logs")
            await self.session.rollback()
            logger.bind(operation="query", error=str(exc)).exception("audit_repository_failed")
            return Failure(_database_error("query", {"error": str(exc)}))


def _database_error(operation: str, details: dict[str, object]) -> AuditInfrastructureError:
    message = (
        "Database error while creating audit log entry"
        if operation == "create"
        else "Database error while querying audit logs"
    )
    return AuditInfrastructureError(
        message=message,
        details=details,
        source="audit_repository",
        operation=operation,
    )


AuditLogRepository = PostgresAuditLogRepository
