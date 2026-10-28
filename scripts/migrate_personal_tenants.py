"""Backfill Mongo users into deterministic SQL personal tenants.

Examples::

    uv run python scripts/migrate_personal_tenants.py --dry-run
    uv run python scripts/migrate_personal_tenants.py
    uv run python scripts/migrate_personal_tenants.py --check
    uv run python scripts/migrate_personal_tenants.py --reconcile --limit 250

The command performs no destructive operation.  A non-personal existing
``tenant_id`` is reported as a conflict and skipped for manual resolution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.config import get_settings
from app.connections.mongodb import close_mongo_client, create_mongo_client
from app.connections.postgres import close_db_engine, init_db
from app.features.auth.account_store import BeanieAuthAccountStore
from app.features.auth.models import User
from app.features.auth.tenant_migration import (
    PersonalTenantMigration,
    check_tenant_readiness,
)
from app.features.auth.tenant_repository import TenantMembershipService


async def _run(*, dry_run: bool, check: bool, reconcile: bool, limit: int) -> dict[str, object]:
    settings = get_settings()
    mongo_client, _ = await create_mongo_client(
        settings.MONGODB_URI,
        settings.MONGODB_DB_NAME,
        [User],
    )
    engine, session_factory = await init_db()
    try:
        if check:
            return (await check_tenant_readiness(session_factory)).as_dict()
        if reconcile:
            service = TenantMembershipService(session_factory, BeanieAuthAccountStore())
            reconciliation = await service.reconcile_pending_mirrors(limit=limit)
            return dict(reconciliation)
        report = await PersonalTenantMigration(session_factory, dry_run=dry_run).run()
        return report.as_dict()
    finally:
        await close_db_engine(engine)
        close_mongo_client(mongo_client)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and report changes without writing Mongo or PostgreSQL",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify every active Mongo account has a personal plus valid SQL memberships and no pending/mismatched mirror; exits 1 when not ready",
    )
    mode.add_argument(
        "--reconcile",
        action="store_true",
        help="retry SQL-committed Mongo mirror writes recorded as pending",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="maximum pending mirror rows to reconcile (default: 100)",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    report = asyncio.run(
        _run(
            dry_run=args.dry_run,
            check=args.check,
            reconcile=args.reconcile,
            limit=args.limit,
        )
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.check and report.get("ready") is not True:
        raise SystemExit(1)
    if args.reconcile and report.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
