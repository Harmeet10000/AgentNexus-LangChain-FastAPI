"""UserCredit repository with dual-method pattern."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.features.credits.models.credit import CreditStatus, UserCredit
from app.shared.result import ConflictAppError, InfrastructureAppError, NotFoundAppError

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.shared.result import AppResult


class CreditRepository:
    """Persistence operations for UserCredit."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, credit: UserCredit) -> AppResult[UserCredit]:
        try:
            self.session.add(credit)
            await self.session.flush()
            return Success(credit)
        except IntegrityError as exc:
            return Failure(
                ConflictAppError(
                    code="CREDIT_CONFLICT",
                    message="Credit creation failed due to a constraint violation",
                    details={"user_id": credit.user_id, "error": str(exc)},
                    source="credit_repository",
                )
            )
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while creating credit",
                    details={"error": str(exc)},
                    source="credit_repository",
                )
            )

    async def find_by_id(self, credit_id: UUID) -> AppResult[UserCredit | None]:
        try:
            statement = select(UserCredit).where(UserCredit.id == credit_id)
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while fetching credit",
                    details={"credit_id": str(credit_id), "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def find_by_user(
        self,
        user_id: str,
        *,
        status: CreditStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> AppResult[tuple[list[UserCredit], int]]:
        try:
            conditions = [
                UserCredit.user_id == user_id,
                UserCredit.deleted_at.is_(None),
            ]
            if status is not None:
                conditions.append(UserCredit.status == status.value)

            total = (
                await self.session.execute(
                    select(func.count()).select_from(UserCredit).where(*conditions)
                )
            ).scalar_one()

            statement = (
                select(UserCredit)
                .where(*conditions)
                .order_by(UserCredit.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(statement)
            return Success((list(result.scalars().all()), int(total)))
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while listing credits",
                    details={"user_id": user_id, "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def find_available_for_consumption(
        self,
        user_id: str,
        *,
        limit: int = 100,
    ) -> AppResult[list[UserCredit]]:
        """Find ACTIVE, non-expired credits ordered by soonest valid_until, then oldest created_at.

        Credits with no expiry (valid_until IS NULL) are consumed last.
        """
        try:
            now = datetime.now(tz=UTC)
            statement = (
                select(UserCredit)
                .where(
                    UserCredit.user_id == user_id,
                    UserCredit.status == CreditStatus.ACTIVE.value,
                    UserCredit.deleted_at.is_(None),
                    ((UserCredit.valid_until.is_(None)) | (UserCredit.valid_until > now)),
                )
                .order_by(
                    UserCredit.valid_until.asc().nullslast(),
                    UserCredit.created_at.asc(),
                )
                .limit(limit)
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while finding available credits",
                    details={"user_id": user_id, "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def get_active_balance(self, user_id: str) -> AppResult[int]:
        """Sum remaining_balance across ACTIVE, non-expired credits (paisa)."""
        try:
            now = datetime.now(tz=UTC)
            statement = select(func.coalesce(func.sum(UserCredit.remaining_balance), 0)).where(
                UserCredit.user_id == user_id,
                UserCredit.status == CreditStatus.ACTIVE.value,
                UserCredit.deleted_at.is_(None),
                ((UserCredit.valid_until.is_(None)) | (UserCredit.valid_until > now)),
            )
            result = await self.session.execute(statement)
            return Success(int(result.scalar_one()))
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while calculating credit balance",
                    details={"user_id": user_id, "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def update_balance(
        self,
        credit: UserCredit,
        *,
        new_remaining_balance: int,
        new_status: CreditStatus | None = None,
        consumed_at: datetime | None = None,
    ) -> AppResult[UserCredit]:
        try:
            values: dict[str, object] = {
                "remaining_balance": new_remaining_balance,
            }
            if new_status is not None:
                values["status"] = new_status.value
            if consumed_at is not None:
                values["consumed_at"] = consumed_at

            statement = (
                update(UserCredit)
                .where(UserCredit.id == credit.id)
                .values(**values)
                .returning(UserCredit)
            )
            result = await self.session.execute(statement)
            updated = result.scalar_one_or_none()
            if updated is None:
                return Failure(
                    NotFoundAppError(
                        code="CREDIT_NOT_FOUND",
                        message="Credit not found during balance update",
                        details={"credit_id": str(credit.id)},
                        source="credit_repository",
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while updating credit balance",
                    details={"credit_id": str(credit.id), "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def expire_credits_past_date(self, cutoff: datetime) -> AppResult[list[UserCredit]]:
        """Find ACTIVE credits past their valid_until and mark them EXPIRED."""
        try:
            statement = (
                select(UserCredit)
                .where(
                    UserCredit.status == CreditStatus.ACTIVE.value,
                    UserCredit.valid_until.isnot(None),
                    UserCredit.valid_until < cutoff,
                    UserCredit.deleted_at.is_(None),
                )
                .order_by(UserCredit.valid_until.asc())
            )
            result = await self.session.execute(statement)
            credit_rows = list(result.scalars().all())

            if not credit_rows:
                return Success([])

            credit_ids = [c.id for c in credit_rows]

            update_stmt = (
                update(UserCredit)
                .where(UserCredit.id.in_(credit_ids))
                .values(
                    status=CreditStatus.EXPIRED.value,
                    consumed_at=cutoff,
                )
            )
            await self.session.execute(update_stmt)

            for credit in credit_rows:
                credit.status = CreditStatus.EXPIRED.value
                credit.consumed_at = cutoff

            return Success(credit_rows)
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while expiring credits",
                    details={"error": str(exc)},
                    source="credit_repository",
                )
            )
