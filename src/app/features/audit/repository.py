"""Immutable audit log persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.shared.result import InfrastructureAppError
from app.utils.codes import ErrorCode

from .model import AuditLog

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.selectable import Select

    from app.shared.result import AppResult


class AuditLogRepository:
    """Append-only audit trail repository. No update/delete paths exist."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, entry: AuditLog) -> AppResult[AuditLog]:
        try:
            self.session.add(entry)
            await self.session.flush()
            return Success(entry)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while creating audit log entry",
                    details={"entity_type": entry.entity_type, "error": str(exc)},
                    source="audit_repository",
                )
            )

    async def find_by_entity(
        self,
        entity_type: str,
        entity_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> AppResult[list[AuditLog]]:
        try:
            statement: Select[tuple[AuditLog]] = (
                select(AuditLog)
                .where(AuditLog.entity_type == entity_type, AuditLog.entity_id == entity_id)
                .order_by(AuditLog.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while querying audit logs",
                    details={"entity_type": entity_type, "entity_id": entity_id, "error": str(exc)},
                    source="audit_repository",
                )
            )

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
    ) -> AppResult[tuple[list[AuditLog], int]]:
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

            statement: Select[tuple[AuditLog]] = (
                select(AuditLog)
                .where(*conditions)
                .order_by(AuditLog.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(statement)
            return Success((list(result.scalars().all()), int(total)))
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while querying audit logs",
                    details={"error": str(exc)},
                    source="audit_repository",
                )
            )
