"""Plan persistence operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.shared.result import (
    ConflictAppError,
    InfrastructureAppError,
    NotFoundAppError,
)
from app.utils import ErrorCode

from .model import Plan

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.selectable import Select

    from app.shared.result import (
        AppResult,
    )


class PlanRepository:
    """Repository for billing plan lifecycle."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, plan: Plan) -> AppResult[Plan]:
        try:
            self.session.add(plan)
            await self.session.flush()
            return Success(plan)
        except IntegrityError as exc:
            return Failure(
                ConflictAppError(
                    code=ErrorCode.CONFLICT,
                    message="Plan creation failed due to a constraint violation",
                    details={"name": plan.name, "error": str(exc)},
                    source="plan_repository",
                )
            )
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    message="Database error while creating plan",
                    details={"error": str(exc)},
                    source="plan_repository",
                )
            )

    async def find_by_id(self, plan_id: str | UUID) -> AppResult[Plan | None]:
        try:
            statement: Select[tuple[Plan]] = select(Plan).where(Plan.id == plan_id)
            result = await self.session.execute(statement)
            plan = result.scalar_one_or_none()
            if plan is None:
                return Failure(
                    NotFoundAppError(
                        code=ErrorCode.NOT_FOUND,
                        message="Plan not found",
                        details={"plan_id": str(plan_id)},
                        source="plan_repository",
                    )
                )
            return Success(plan)
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    message="Database error while fetching plan",
                    details={"plan_id": str(plan_id), "error": str(exc)},
                    source="plan_repository",
                )
            )

    async def find_by_name(self, name: str) -> AppResult[Plan | None]:
        """Find an ACTIVE plan by name (Requirement 1.6: unique among active plans)."""
        try:
            statement: Select[tuple[Plan]] = select(Plan).where(
                Plan.name == name, Plan.is_active.is_(True)
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    message="Database error while finding plan by name",
                    details={"name": name, "error": str(exc)},
                    source="plan_repository",
                )
            )

    async def list_active(self) -> AppResult[list[Plan]]:
        try:
            statement: Select[tuple[Plan]] = (
                select(Plan).where(Plan.is_active.is_(True)).order_by(Plan.created_at)
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    message="Database error while listing plans",
                    details={"error": str(exc)},
                    source="plan_repository",
                )
            )

    async def archive(self, plan_id: str | UUID) -> AppResult[Plan | None]:
        """Soft-delete a plan. Existing subscriptions keep their plan version."""
        try:
            statement = (
                update(Plan)
                .where(Plan.id == plan_id)
                .values(is_active=False, updated_at=datetime.now(tz=UTC))
                .returning(Plan)
            )
            result = await self.session.execute(statement)
            plan = result.scalar_one_or_none()
            if plan is None:
                return Failure(
                    NotFoundAppError(
                        code=ErrorCode.NOT_FOUND,
                        message="Plan not found",
                        details={"plan_id": str(plan_id)},
                        source="plan_repository",
                    )
                )
            return Success(plan)
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    message="Database error while archiving plan",
                    details={"plan_id": str(plan_id), "error": str(exc)},
                    source="plan_repository",
                )
            )

    async def update(self, plan: Plan, *, values: dict[str, object]) -> AppResult[Plan]:
        try:
            statement = (
                update(Plan)
                .where(Plan.id == plan.id)
                .values(**values, updated_at=datetime.now(tz=UTC))
                .returning(Plan)
            )
            result = await self.session.execute(statement)
            updated = result.scalar_one_or_none()
            if updated is None:
                return Failure(
                    NotFoundAppError(
                        code=ErrorCode.NOT_FOUND,
                        message="Plan not found",
                        details={"plan_id": str(plan.id)},
                        source="plan_repository",
                    )
                )
            return Success(updated)
        except IntegrityError as exc:
            return Failure(
                ConflictAppError(
                    code=ErrorCode.CONFLICT,
                    message="Plan update failed due to a constraint violation",
                    details={"plan_id": str(plan.id), "error": str(exc)},
                    source="plan_repository",
                )
            )
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    message="Database error while updating plan",
                    details={"plan_id": str(plan.id), "error": str(exc)},
                    source="plan_repository",
                )
            )
