"""Async, transactional, idempotent seeders — run via `uv run seed` or lifespan (dev)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import get_settings
from app.features.plans.model import Plan
from app.utils import logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# ponytail: registry is one list, not scattered calls — add a new seeder here.
Seeder = Callable[["AsyncSession"], Awaitable[int]]


async def _seed_plans(session: AsyncSession) -> int:
    """Seed default billing plans — idempotent ON CONFLICT DO NOTHING."""

    # Minimal seed so `alembic check` + fresh DB can boot without manual SQL.
    # Real plans come from admin API; this is just the free tier / fallback.
    default_plans = [
        {
            "name": "Free",
            "description": "Free tier — community support",
            "amount": 0,
            "currency": "INR",
            "interval": "monthly",
            "interval_count": 1,
            "is_active": True,
            "features": {"max_documents": 5},
        },
    ]
    inserted = 0
    for plan in default_plans:
        # Plan's unique key is partial: uq_plans_active_name WHERE is_active
        stmt = (
            pg_insert(Plan)
            .values(**plan)
            .on_conflict_do_nothing(
                index_elements=["name"],
                index_where=Plan.__table__.c.is_active,  # type: ignore[attr-defined]
            )
        )
        result = await session.execute(stmt)
        inserted += result.rowcount or 0
    logger.info("seed_plans", inserted=inserted)
    return inserted


_SEEDERS: tuple[Seeder, ...] = (_seed_plans,)


async def run_all_seeders(session: AsyncSession | None = None) -> dict[str, int]:
    """Run all seeders in a single transaction — safe to call twice.

    Args:
        session: Existing session (e.g. from lifespan). If None, creates one.

    Returns:
        Mapping seeder_name → rows inserted.
    """
    from app.connections import init_db  # noqa: PLC0415 — avoid circular import at module load

    if session is not None:
        return await _run_in_transaction(session)
    _, session_factory = await init_db()
    async with session_factory() as inner_session:
        return await _run_in_transaction(inner_session)


async def _run_in_transaction(session: AsyncSession) -> dict[str, int]:
    results: dict[str, int] = {}
    # Single transaction — all or nothing, idempotent so re-runnable.
    async with session.begin():
        for seeder in _SEEDERS:
            name = getattr(seeder, "__name__", str(seeder))
            try:
                count = await seeder(session)
                results[name] = count
            except Exception as exc:
                exc.add_note(f"seeder={name}")
                logger.error("seeder_failed", seeder=name, error=str(exc))
                raise
    logger.info("seeding_completed", results=results)
    return results


def main() -> None:
    """CLI entrypoint: `uv run seed`."""
    settings = get_settings()
    if settings.ENVIRONMENT == "production":
        logger.warning("seed_skip_production", environment=settings.ENVIRONMENT)
        return
    asyncio.run(run_all_seeders())
    logger.info("seed_done")


if __name__ == "__main__":
    main()
