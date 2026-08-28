"""Scheduled credit jobs: daily expiration, weekly reconciliation."""

from __future__ import annotations

import asyncio

from returns.result import Failure
from sqlalchemy import select

from app.connections.celery import CeleryTaskRegistry, NoKwargsPayload, ResilientTask, celery_app
from app.connections.celery_task_names import CREDITS_EXPIRE, CREDITS_RECONCILE
from app.connections.postgres import init_db
from app.features.audit.model import AuditLog
from app.features.audit.repository import AuditLogRepository
from app.features.credits.models.credit import UserCredit
from app.features.credits.repositories.consumption_repository import ConsumptionRepository
from app.features.credits.repositories.credit_repository import CreditRepository
from app.features.credits.services.credit_service import CreditService
from app.utils import logger


async def _expire_credits_job() -> dict[str, int]:
    engine, session_local = await init_db()
    try:
        async with session_local() as session:
            service = CreditService(
                session,
                CreditRepository(session),
                ConsumptionRepository(session),
                AuditLogRepository(session),
            )
            try:
                count = await service.expire_credits()
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return {"expired": count}
    finally:
        await engine.dispose()


async def _reconcile_credits_job() -> dict[str, int]:
    """Verify ledger integrity: credit_amount == remaining_balance + SUM(consumed_amount)."""
    engine, session_local = await init_db()
    try:
        async with session_local() as session:
            consumption_repo = ConsumptionRepository(session)
            audit = AuditLogRepository(session)

            statement = select(UserCredit).where(UserCredit.deleted_at.is_(None))
            result = await session.execute(statement)
            all_credits = list(result.scalars().all())

            discrepancies: list[dict[str, object]] = []
            for credit in all_credits:
                total_consumed_result = await consumption_repo.get_total_consumed(credit.id)
                if isinstance(total_consumed_result, Failure):
                    logger.bind(
                        operation="credits.reconcile",
                        credit_id=str(credit.id),
                    ).warning(
                        "Failed to get consumed total",
                        error=total_consumed_result.failure().message,
                    )
                    continue
                total_consumed = total_consumed_result.unwrap()
                expected_remaining = credit.credit_amount - total_consumed
                if credit.remaining_balance != expected_remaining:
                    discrepancies.append(
                        {
                            "credit_id": str(credit.id),
                            "user_id": credit.user_id,
                            "credit_amount": credit.credit_amount,
                            "remaining_balance": credit.remaining_balance,
                            "total_consumed": total_consumed,
                            "expected_remaining": expected_remaining,
                        }
                    )

            if discrepancies:
                logger.bind(
                    operation="credits.reconcile",
                    discrepancy_count=len(discrepancies),
                ).error("Ledger discrepancies detected")
                await audit.create(
                    AuditLog(
                        entity_type="system",
                        entity_id="credits",
                        action="credit.reconciliation_discrepancy",
                        changes={"discrepancies": discrepancies, "total_checked": len(all_credits)},
                    )
                )
            else:
                logger.bind(
                    operation="credits.reconcile",
                    total_checked=len(all_credits),
                ).info("Credit ledger reconciliation passed")

            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise

            return {"checked": len(all_credits), "discrepancies": len(discrepancies)}
    finally:
        await engine.dispose()


# Scheduler-dispatched with no keyword arguments; see the equivalent note in the
# billing jobs for why the empty contract is registered rather than left absent.
CeleryTaskRegistry.register(CREDITS_EXPIRE, NoKwargsPayload)
CeleryTaskRegistry.register(CREDITS_RECONCILE, NoKwargsPayload)


@celery_app.task(name=CREDITS_EXPIRE, base=ResilientTask)
def credits_expire() -> dict[str, int]:
    """Daily job to expire past-due credits (Requirement 51)."""
    logger.bind(operation="credits.expire").info("Starting credit expiration job")
    result = asyncio.run(_expire_credits_job())
    logger.bind(operation="credits.expire", expired=result["expired"]).info(
        "Credit expiration job completed"
    )
    return result


@celery_app.task(name=CREDITS_RECONCILE, base=ResilientTask)
def credits_reconcile() -> dict[str, int]:
    """Weekly job to verify credit ledger integrity (Requirement 53.1)."""
    logger.bind(operation="credits.reconcile").info("Starting credit reconciliation job")
    result = asyncio.run(_reconcile_credits_job())
    logger.bind(
        operation="credits.reconcile",
        checked=result["checked"],
        discrepancies=result["discrepancies"],
    ).info("Credit reconciliation job completed")
    return result
