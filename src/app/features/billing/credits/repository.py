# ── from consumption_repository.py ──
"""CreditConsumption repository."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.features.billing.credits.models import (
    CreditConsumption,
    CreditNote,
    CreditNoteApplication,
    CreditStatus,
    UserCredit,
)
from app.kernel.result import add_database_error_note

from .exceptions import (
    CreditConflictError,
    CreditConsumptionConflictError,
    CreditInfrastructureError,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy import Select
    from sqlalchemy.ext.asyncio import AsyncSession

    from .exceptions import CreditResult


class ConsumptionRepository:
    """Persistence operations for CreditConsumption."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, consumption: CreditConsumption) -> CreditResult[CreditConsumption]:
        try:
            self.session.add(consumption)
            await self.session.flush()
            return Success(inner_value=consumption)
        except IntegrityError as exc:
            add_database_error_note(exc, table="credit_consumptions")
            await self.session.rollback()
            return Failure(
                inner_value=CreditConsumptionConflictError(
                    message="Credit consumption creation failed due to a constraint violation",
                    details={"credit_id": str(consumption.credit_id), "error": str(exc)},
                    source="consumption_repository",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="credit_consumptions")
            await self.session.rollback()
            return Failure(
                inner_value=CreditInfrastructureError(
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
    ) -> CreditResult[tuple[list[CreditConsumption], int]]:
        try:
            conditions = [CreditConsumption.user_id == user_id]

            total = (
                await self.session.execute(
                    select(func.count()).select_from(CreditConsumption).where(*conditions)
                )
            ).scalar_one()

            statement: Select[tuple[CreditConsumption]] = (
                select(CreditConsumption)
                .where(*conditions)
                .order_by(CreditConsumption.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(statement)
            return Success(inner_value=(list(result.scalars().all()), int(total)))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="credit_consumptions")
            await self.session.rollback()
            return Failure(
                inner_value=CreditInfrastructureError(
                    message="Database error while listing credit consumptions",
                    details={"user_id": user_id, "error": str(exc)},
                    source="consumption_repository",
                )
            )

    async def find_by_invoice_id(
        self,
        invoice_id: UUID,
    ) -> CreditResult[CreditConsumption | None]:
        try:
            statement: Select[tuple[CreditConsumption]] = select(CreditConsumption).where(
                CreditConsumption.invoice_id == invoice_id
            )
            result = await self.session.execute(statement)
            return Success(inner_value=result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="credit_consumptions")
            await self.session.rollback()
            return Failure(
                inner_value=CreditInfrastructureError(
                    message="Database error while fetching consumption by invoice",
                    details={"invoice_id": str(invoice_id), "error": str(exc)},
                    source="consumption_repository",
                )
            )

    async def find_all_by_invoice_id(
        self, invoice_id: UUID
    ) -> CreditResult[list[CreditConsumption]]:
        """Return every consumption for an invoice, not just the first row."""
        try:
            statement: Select[tuple[CreditConsumption]] = (
                select(CreditConsumption)
                .where(CreditConsumption.invoice_id == invoice_id)
                .order_by(CreditConsumption.created_at.asc())
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="credit_consumptions")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while fetching invoice consumptions",
                    details={"invoice_id": str(invoice_id), "error": str(exc)},
                    source="consumption_repository",
                )
            )

    async def find_by_credit_id(self, credit_id: UUID) -> CreditResult[list[CreditConsumption]]:
        """Find all consumption records for a credit."""
        try:
            statement: Select[tuple[CreditConsumption]] = (
                select(CreditConsumption)
                .where(CreditConsumption.credit_id == credit_id)
                .order_by(CreditConsumption.created_at.desc())
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="credit_consumptions")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while fetching consumptions by credit",
                    details={"credit_id": str(credit_id), "error": str(exc)},
                    source="consumption_repository",
                )
            )

    async def get_total_consumed(self, credit_id: UUID) -> CreditResult[int]:
        """Get total consumed amount for a credit (in paisa)."""
        try:
            statement: Select[tuple[int]] = select(
                func.coalesce(func.sum(CreditConsumption.consumed_amount), 0)
            ).where(CreditConsumption.credit_id == credit_id)
            result = await self.session.execute(statement)
            return Success(inner_value=int(result.scalar_one()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="credit_consumptions")
            await self.session.rollback()
            return Failure(
                inner_value=CreditInfrastructureError(
                    message="Database error while calculating total consumed",
                    details={"credit_id": str(credit_id), "error": str(exc)},
                    source="consumption_repository",
                )
            )


# ── from credit_repository.py ──
"""UserCredit repository with dual-method pattern."""


if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy import Select, Update
    from sqlalchemy.ext.asyncio import AsyncSession

    from .exceptions import CreditResult


class CreditRepository:
    """Persistence operations for UserCredit."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, credit: UserCredit) -> CreditResult[UserCredit]:
        try:
            self.session.add(credit)
            await self.session.flush()
            return Success(credit)
        except IntegrityError as exc:
            add_database_error_note(exc, table="user_credits")
            await self.session.rollback()
            return Failure(
                CreditConflictError(
                    message="Credit creation failed due to a constraint violation",
                    details={"user_id": credit.user_id, "error": str(exc)},
                    source="credit_repository",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="user_credits")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while creating credit",
                    details={"error": str(exc)},
                    source="credit_repository",
                )
            )

    async def find_credit_note_by_operation_key(
        self, operation_key: str
    ) -> CreditResult[CreditNote | None]:
        try:
            statement: Select[tuple[CreditNote]] = (
                select(CreditNote)
                .where(CreditNote.operation_key == operation_key)
                .with_for_update()
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="credit_notes")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while finding credit note",
                    details={"operation_key": operation_key, "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def find_credit_notes_for_consumption(
        self,
        user_id: str,
        *,
        limit: int | None = None,
    ) -> CreditResult[list[CreditNote]]:
        """Lock all eligible authoritative notes in deterministic order.

        ``limit=None`` is intentional: a user may have more than one hundred
        notes and truncating the economic ledger can leave cash due when a
        later note could have covered the invoice.
        """
        try:
            now = datetime.now(tz=UTC)
            statement: Select[tuple[CreditNote]] = (
                select(CreditNote)
                .where(
                    CreditNote.user_id == user_id,
                    CreditNote.status == "available",
                    CreditNote.remaining_paisa > 0,
                    CreditNote.applies_at <= now,
                )
                .order_by(CreditNote.applies_at.asc(), CreditNote.created_at.asc())
                .with_for_update()
            )
            if limit is not None:
                statement = statement.limit(limit)
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="credit_notes")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while finding credit notes",
                    details={"user_id": user_id, "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def find_by_operation_key(self, operation_key: str) -> CreditResult[UserCredit | None]:
        """Direct indexed JSON lookup for a downgrade grant; never scan a page."""
        try:
            statement: Select[tuple[UserCredit]] = select(UserCredit).where(
                UserCredit.metadata_["operation_key"].as_string() == operation_key,
                UserCredit.deleted_at.is_(None),
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="user_credits")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while finding credit by operation key",
                    details={"operation_key": operation_key, "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def create_credit_note(self, note: CreditNote) -> CreditResult[CreditNote]:
        try:
            self.session.add(note)
            await self.session.flush()
            return Success(note)
        except IntegrityError as exc:
            add_database_error_note(exc, table="credit_notes")
            await self.session.rollback()
            existing = await self.find_credit_note_by_operation_key(note.operation_key)
            if isinstance(existing, Success) and existing.unwrap() is not None:
                return existing
            return Failure(
                CreditConflictError(
                    message="Credit note operation has already been recorded",
                    details={"operation_key": note.operation_key},
                    source="credit_repository",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="credit_notes")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while creating credit note",
                    details={"operation_key": note.operation_key, "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def find_by_id(self, credit_id: UUID) -> CreditResult[UserCredit | None]:
        try:
            statement: Select[tuple[UserCredit]] = (
                select(UserCredit).where(UserCredit.id == credit_id).with_for_update()
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="user_credits")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
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
    ) -> CreditResult[tuple[list[UserCredit], int]]:
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

            statement: Select[tuple[UserCredit]] = (
                select(UserCredit)
                .where(*conditions)
                .order_by(UserCredit.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(statement)
            return Success((list(result.scalars().all()), int(total)))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="user_credits")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while listing credits",
                    details={"user_id": user_id, "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def find_available_for_consumption(
        self,
        user_id: str,
        *,
        limit: int | None = None,
    ) -> CreditResult[list[UserCredit]]:
        """Find ACTIVE, non-expired credits ordered by soonest valid_until, then oldest created_at.

        Credits with no expiry (valid_until IS NULL) are consumed last.
        """
        try:
            now = datetime.now(tz=UTC)
            statement: Select[tuple[UserCredit]] = (
                select(UserCredit)
                .where(
                    UserCredit.user_id == user_id,
                    UserCredit.status == CreditStatus.ACTIVE.value,
                    UserCredit.deleted_at.is_(None),
                    UserCredit.valid_from <= now,
                    ((UserCredit.valid_until.is_(None)) | (UserCredit.valid_until > now)),
                )
                .order_by(
                    UserCredit.valid_until.asc().nullslast(),
                    UserCredit.created_at.asc(),
                )
                .with_for_update()
            )
            if limit is not None:
                statement = statement.limit(limit)
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="user_credits")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while finding available credits",
                    details={"user_id": user_id, "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def get_active_balance(self, user_id: str) -> CreditResult[int]:
        """Sum remaining_balance across ACTIVE, non-expired credits (paisa)."""
        try:
            now = datetime.now(tz=UTC)
            statement: Select[tuple[int]] = select(
                func.coalesce(func.sum(UserCredit.remaining_balance), 0)
            ).where(
                UserCredit.user_id == user_id,
                UserCredit.status == CreditStatus.ACTIVE.value,
                UserCredit.deleted_at.is_(None),
                ((UserCredit.valid_until.is_(None)) | (UserCredit.valid_until > now)),
            )
            result = await self.session.execute(statement)
            return Success(int(result.scalar_one()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="user_credits")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
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
    ) -> CreditResult[UserCredit]:
        try:
            values: dict[str, object] = {
                "remaining_balance": new_remaining_balance,
            }
            if new_status is not None:
                values["status"] = new_status.value
            if consumed_at is not None:
                values["consumed_at"] = consumed_at

            # The row is normally locked by the caller.  The compare-and-set
            # predicate is still required: it protects callers that enter this
            # repository method directly and turns a lost update into a typed
            # conflict instead of silently overwriting a concurrent spend.
            conditions = [
                UserCredit.id == credit.id,
                UserCredit.remaining_balance == credit.remaining_balance,
                UserCredit.remaining_balance >= new_remaining_balance,
            ]
            statement: Update = (
                update(UserCredit).where(*conditions).values(**values).returning(UserCredit)
            )
            result = await self.session.execute(statement)
            updated = result.scalar_one_or_none()
            if updated is None:
                return Failure(
                    CreditConflictError(
                        message="Credit balance changed concurrently",
                        details={"credit_id": str(credit.id)},
                        source="credit_repository",
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="user_credits")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while updating credit balance",
                    details={"credit_id": str(credit.id), "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def update_credit_note_balance(
        self,
        note: CreditNote,
        *,
        new_remaining_paisa: int,
    ) -> CreditResult[CreditNote]:
        """Conditionally advance the authoritative note balance."""
        try:
            status = "applied" if new_remaining_paisa == 0 else "available"
            statement: Update = (
                update(CreditNote)
                .where(
                    CreditNote.id == note.id,
                    CreditNote.remaining_paisa == note.remaining_paisa,
                    CreditNote.remaining_paisa >= new_remaining_paisa,
                    CreditNote.status == "available",
                )
                .values(remaining_paisa=new_remaining_paisa, status=status)
                .returning(CreditNote)
            )
            result = await self.session.execute(statement)
            updated = result.scalar_one_or_none()
            if updated is None:
                return Failure(
                    CreditConflictError(
                        message="Credit note balance changed concurrently",
                        details={"credit_note_id": str(note.id)},
                        source="credit_repository",
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="credit_notes")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while updating credit note balance",
                    details={"credit_note_id": str(note.id), "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def create_note_application(
        self, application: CreditNoteApplication
    ) -> CreditResult[CreditNoteApplication]:
        """Persist the unique invoice/note application boundary."""

        async def insert() -> None:
            self.session.add(application)
            await self.session.flush()

        try:
            # A duplicate application is an expected race when two payment
            # notifications for the same invoice arrive together.  A
            # savepoint contains the failed INSERT so the caller's outer
            # invoice transaction remains usable.  The small fallback keeps
            # this repository's lightweight test doubles compatible.
            if hasattr(self.session, "begin_nested"):
                async with self.session.begin_nested():
                    await insert()
            else:
                await insert()
            return Success(application)
        except IntegrityError as exc:
            add_database_error_note(exc, table="credit_note_applications")
            if not hasattr(self.session, "begin_nested"):
                # Legacy/test sessions have no savepoint API; they must be
                # reset before exposing a failed flush to their caller.
                await self.session.rollback()
            else:
                # The begin_nested context has already rolled back its
                # savepoint.  Looking up the winner makes the operation
                # idempotent without rolling back unrelated outer work.
                try:
                    result = await self.session.execute(
                        select(CreditNoteApplication).where(
                            CreditNoteApplication.credit_note_id == application.credit_note_id,
                            CreditNoteApplication.invoice_id == application.invoice_id,
                        )
                    )
                    existing = result.scalar_one_or_none()
                except SQLAlchemyError as lookup_exc:
                    add_database_error_note(
                        lookup_exc,
                        table="credit_note_applications",
                        operation="lookup_duplicate",
                    )
                    await self.session.rollback()
                    return Failure(
                        CreditInfrastructureError(
                            message="Database error while checking credit note application",
                            details={"invoice_id": str(application.invoice_id)},
                            source="credit_repository",
                        )
                    )
                if existing is not None:
                    return Success(existing)
            return Failure(
                CreditConsumptionConflictError(
                    message="Credit note has already been applied to this invoice",
                    details={
                        "credit_note_id": str(application.credit_note_id),
                        "invoice_id": str(application.invoice_id),
                        "error": str(exc),
                    },
                    source="credit_repository",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="credit_note_applications")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while creating credit note application",
                    details={"invoice_id": str(application.invoice_id), "error": str(exc)},
                    source="credit_repository",
                )
            )

    async def expire_credits_past_date(self, cutoff: datetime) -> CreditResult[list[UserCredit]]:
        """Find ACTIVE credits past their valid_until and mark them EXPIRED."""
        try:
            statement: Select[tuple[UserCredit]] = (
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

            update_stmt: Update = (
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
            add_database_error_note(exc, table="user_credits")
            await self.session.rollback()
            return Failure(
                CreditInfrastructureError(
                    message="Database error while expiring credits",
                    details={"error": str(exc)},
                    source="credit_repository",
                )
            )
