"""Plan persistence operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.utils import logger

from .errors import PlanConflictError, PlanInfrastructureError, PlanNotFoundError
from .model import Plan

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.selectable import Select

    from .errors import PlanResult


class PlanRepository:
    """Repository for billing plan lifecycle."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, plan: Plan) -> PlanResult[Plan]:
        try:
            self.session.add(plan)
            await self.session.flush()
            return Success(plan)
        except IntegrityError as exc:
            await self.session.rollback()
            logger.bind(operation="create", plan_name=plan.name).error(
                "plan_repository_failed", error=str(exc)
            )
            return Failure(
                PlanConflictError(
                    message="Plan creation failed due to a constraint violation",
                    details={"name": plan.name, "error": str(exc)},
                    source="plan_repository",
                )
            )
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.bind(operation="create").error("plan_repository_failed", error=str(exc))
            return Failure(
                PlanInfrastructureError(
                    message="Database error while creating plan",
                    details={"error": str(exc)},
                    source="plan_repository",
                    operation="create",
                )
            )

    async def find_by_id(self, plan_id: str | UUID) -> PlanResult[Plan | None]:
        try:
            statement: Select[tuple[Plan]] = select(Plan).where(Plan.id == plan_id)
            result = await self.session.execute(statement)
            plan = result.scalar_one_or_none()
            if plan is None:
                return Failure(
                    PlanNotFoundError(
                        message="Plan not found",
                        details={"plan_id": str(plan_id)},
                        source="plan_repository",
                        plan_id=str(plan_id),
                    )
                )
            return Success(plan)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.bind(operation="find_by_id", plan_id=str(plan_id)).error(
                "plan_repository_failed", error=str(exc)
            )
            return Failure(
                PlanInfrastructureError(
                    message="Database error while fetching plan",
                    details={"plan_id": str(plan_id), "error": str(exc)},
                    source="plan_repository",
                    operation="find_by_id",
                )
            )

    async def find_by_name(self, name: str) -> PlanResult[Plan | None]:
        """Find an ACTIVE plan by name (Requirement 1.6: unique among active plans)."""
        try:
            statement: Select[tuple[Plan]] = select(Plan).where(
                Plan.name == name, Plan.is_active.is_(True)
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.bind(operation="find_by_name", plan_name=name).error(
                "plan_repository_failed", error=str(exc)
            )
            return Failure(
                PlanInfrastructureError(
                    message="Database error while finding plan by name",
                    details={"name": name, "error": str(exc)},
                    source="plan_repository",
                    operation="find_by_name",
                )
            )

    async def list_active(self) -> PlanResult[list[Plan]]:
        try:
            statement: Select[tuple[Plan]] = (
                select(Plan).where(Plan.is_active.is_(True)).order_by(Plan.created_at)
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.bind(operation="list_active").error("plan_repository_failed", error=str(exc))
            return Failure(
                PlanInfrastructureError(
                    message="Database error while listing plans",
                    details={"error": str(exc)},
                    source="plan_repository",
                    operation="list_active",
                )
            )

    async def archive(self, plan_id: str | UUID) -> PlanResult[Plan | None]:
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
                    PlanNotFoundError(
                        message="Plan not found",
                        details={"plan_id": str(plan_id)},
                        source="plan_repository",
                        plan_id=str(plan_id),
                    )
                )
            return Success(plan)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.bind(operation="archive", plan_id=str(plan_id)).error(
                "plan_repository_failed", error=str(exc)
            )
            return Failure(
                PlanInfrastructureError(
                    message="Database error while archiving plan",
                    details={"plan_id": str(plan_id), "error": str(exc)},
                    source="plan_repository",
                    operation="archive",
                )
            )

    async def update(self, plan: Plan, *, values: dict[str, object]) -> PlanResult[Plan]:
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
                    PlanNotFoundError(
                        message="Plan not found",
                        details={"plan_id": str(plan.id)},
                        source="plan_repository",
                        plan_id=str(plan.id),
                    )
                )
            return Success(updated)
        except IntegrityError as exc:
            await self.session.rollback()
            logger.bind(operation="update", plan_id=str(plan.id)).error(
                "plan_repository_failed", error=str(exc)
            )
            return Failure(
                PlanConflictError(
                    message="Plan update failed due to a constraint violation",
                    details={"plan_id": str(plan.id), "error": str(exc)},
                    source="plan_repository",
                )
            )
        except SQLAlchemyError as exc:
            await self.session.rollback()
            logger.bind(operation="update", plan_id=str(plan.id)).error(
                "plan_repository_failed", error=str(exc)
            )
            return Failure(
                PlanInfrastructureError(
                    message="Database error while updating plan",
                    details={"plan_id": str(plan.id), "error": str(exc)},
                    source="plan_repository",
                    operation="update",
                )
            )
