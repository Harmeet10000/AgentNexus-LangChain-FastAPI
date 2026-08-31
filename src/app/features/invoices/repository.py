"""Invoice persistence operations with sequential numbering."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from returns.result import Failure, Success
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.shared.result import (
    ConflictAppError,
    InfrastructureAppError,
    NotFoundAppError,
)
from app.utils.codes import ErrorCode

from .model import Invoice

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.selectable import Select

    from app.shared.result import AppResult


class InvoiceRepository:
    """Repository for invoice lifecycle."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, invoice: Invoice) -> AppResult[Invoice]:
        try:
            self.session.add(invoice)
            await self.session.flush()
            return Success(invoice)
        except IntegrityError as exc:
            await self.session.rollback()
            return Failure(
                ConflictAppError(
                    code="INVOICE_CONFLICT",
                    message="Invoice creation failed due to a constraint violation",
                    details={"invoice_number": invoice.invoice_number, "error": str(exc)},
                    source="invoice_repository",
                )
            )
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while creating invoice",
                    details={"error": str(exc)},
                    source="invoice_repository",
                )
            )

    async def find_by_id(self, invoice_id: str | UUID) -> AppResult[Invoice | None]:
        try:
            statement: Select[tuple[Invoice]] = select(Invoice).where(Invoice.id == invoice_id)
            result = await self.session.execute(statement)
            invoice = result.scalar_one_or_none()
            if invoice is None:
                return Failure(
                    NotFoundAppError(
                        code="INVOICE_NOT_FOUND",
                        message="Invoice not found",
                        details={"invoice_id": str(invoice_id)},
                        source="invoice_repository",
                    )
                )
            return Success(invoice)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while fetching invoice",
                    details={"invoice_id": str(invoice_id), "error": str(exc)},
                    source="invoice_repository",
                )
            )

    async def find_by_payment_id(self, payment_id: str | UUID) -> AppResult[Invoice | None]:
        try:
            statement: Select[tuple[Invoice]] = select(Invoice).where(
                Invoice.payment_id == payment_id
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while fetching invoice by payment",
                    details={"payment_id": str(payment_id), "error": str(exc)},
                    source="invoice_repository",
                )
            )

    async def list_by_user(
        self,
        user_id: str,
        *,
        subscription_id: str | UUID | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> AppResult[list[Invoice]]:
        """List issued/paid invoices for a user (Requirement 13)."""
        try:
            conditions = [
                Invoice.user_id == user_id,
                Invoice.status.in_(["issued", "paid"]),
            ]
            if subscription_id is not None:
                conditions.append(Invoice.subscription_id == subscription_id)
            if status is not None:
                conditions.append(Invoice.status == status)
            statement: Select[tuple[Invoice]] = (
                select(Invoice)
                .where(*conditions)
                .order_by(Invoice.issued_at.desc())
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
                    message="Database error while listing invoices",
                    details={"user_id": user_id, "error": str(exc)},
                    source="invoice_repository",
                )
            )

    async def list_by_subscription(self, subscription_id: str | UUID) -> AppResult[list[Invoice]]:
        try:
            statement: Select[tuple[Invoice]] = (
                select(Invoice)
                .where(Invoice.subscription_id == subscription_id)
                .order_by(Invoice.created_at.desc())
            )
            result = await self.session.execute(statement)
            return Success(list(result.scalars().all()))
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while listing invoices for subscription",
                    details={"subscription_id": str(subscription_id), "error": str(exc)},
                    source="invoice_repository",
                )
            )

    async def generate_invoice_number(self, *, prefix: str, year: int) -> AppResult[str]:
        """Sequential invoice number via a PostgreSQL sequence (Requirement 12.1)."""
        try:
            result = await self.session.execute(
                text("SELECT nextval('billing_invoice_number_seq')")
            )
            sequence = int(result.scalar_one())
            return Success(f"{prefix}-{year}-{sequence:04d}")
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while generating invoice number",
                    details={"error": str(exc)},
                    source="invoice_repository",
                )
            )

    async def generate_receipt_number(self, *, prefix: str, year: int) -> AppResult[str]:
        """Sequential receipt number via a PostgreSQL sequence (Requirement 36.3)."""
        try:
            result = await self.session.execute(
                text("SELECT nextval('billing_receipt_number_seq')")
            )
            sequence = int(result.scalar_one())
            return Success(f"{prefix}-{year}-{sequence:04d}")
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while generating receipt number",
                    details={"error": str(exc)},
                    source="invoice_repository",
                )
            )

    async def update_status(
        self,
        invoice: Invoice,
        *,
        status: str,
        extra_values: dict[str, object] | None = None,
    ) -> AppResult[Invoice]:
        try:
            values: dict[str, object] = {"status": status}
            if extra_values:
                values.update(extra_values)
            statement = (
                update(Invoice)
                .where(Invoice.id == invoice.id)
                .values(**values, updated_at=datetime.now(tz=UTC))
                .returning(Invoice)
            )
            result = await self.session.execute(statement)
            updated = result.scalar_one_or_none()
            if updated is None:
                return Failure(
                    NotFoundAppError(
                        code="INVOICE_NOT_FOUND",
                        message="Invoice not found",
                        details={"invoice_id": str(invoice.id)},
                        source="invoice_repository",
                    )
                )
            return Success(updated)
        except SQLAlchemyError as exc:
            await self.session.rollback()
            return Failure(
                InfrastructureAppError(
                    code=ErrorCode.DATABASE_ERROR,
                    retryable=False,
                    message="Database error while updating invoice status",
                    details={"invoice_id": str(invoice.id), "error": str(exc)},
                    source="invoice_repository",
                )
            )
