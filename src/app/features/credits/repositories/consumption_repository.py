"""CreditConsumption repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.features.credits.models.consumption import CreditConsumption
from app.shared.result import ConflictAppError, InfrastructureAppError

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.shared.result import AppResult


class ConsumptionRepository:
    """Persistence operations for CreditConsumption."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, consumption: CreditConsumption) -> AppResult[CreditConsumption]:
        try:
            self.session.add(consumption)
            await self.session.flush()
            return Success(consumption)
        except IntegrityError as exc:
            return Failure(
                ConflictAppError(
                    code="CONSUMPTION_CONFLICT",
                    message="Credit consumption creation failed due to a constraint violation",
                    details={"credit_id": str(consumption.credit_id), "error": str(exc)},
                    source="consumption_repository",
                )
            )
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while creating credit consumption",
                    details={"error": str(exc)},
                    source="consumption_repository",
                )
            )

    async def find_by_user(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> AppResult[tuple[list[CreditConsumption], int]]:
        try:
            conditions = [CreditConsumption.user_id == user_id]

            total = (
                await self.session.execute(
                    select(func.count()).select_from(CreditConsumption).where(*conditions)
                )
            ).scalar_one()

            statement = (
                select(CreditConsumption)
                .where(*conditions)
                .order_by(CreditConsumption.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(statement)
            return Success((list(result.scalars().all()), int(total)))
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while listing credit consumptions",
                    details={"user_id": user_id, "error": str(exc)},
                    source="consumption_repository",
                )
            )

    async def find_by_invoice_id(
        self,
        invoice_id: UUID,
    ) -> AppResult[CreditConsumption | None]:
        try:
            statement = select(CreditConsumption).where(CreditConsumption.invoice_id == invoice_id)
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while fetching consumption by invoice",
                    details={"invoice_id": str(invoice_id), "error": str(exc)},
                    source="consumption_repository",
                )
            )

    async def find_by_credit_id(self, credit_id: UUID) -> AppResult[list[CreditConsumption]]:
        """Find all consumption records for a credit."""
        try:
            statement = (
                select(CreditConsumption)
                .where(CreditConsumption.credit_id == credit_id)
                .order_by(CreditConsumption.created_at.desc())
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while fetching consumptions by credit",
                    details={"credit_id": str(credit_id), "error": str(exc)},
                    source="consumption_repository",
                )
            )

    async def get_total_consumed(self, credit_id: UUID) -> AppResult[int]:
        """Get total consumed amount for a credit (in paisa)."""
        try:
            statement = select(func.coalesce(func.sum(CreditConsumption.consumed_amount), 0)).where(
                CreditConsumption.credit_id == credit_id
            )
            result = await self.session.execute(statement)
            return Success(int(result.scalar_one()))
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while calculating total consumed",
                    details={"credit_id": str(credit_id), "error": str(exc)},
                    source="consumption_repository",
                )
            )
