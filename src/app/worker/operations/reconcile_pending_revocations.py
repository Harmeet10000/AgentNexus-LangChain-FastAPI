"""Bounded operator command for pending session-revocation recovery.

Examples:
  uv run python -m app.worker.operations.reconcile_pending_revocations --dry-run
  uv run python -m app.worker.operations.reconcile_pending_revocations --apply --force
  uv run python -m app.worker.operations.reconcile_pending_revocations --apply --user-id ID
"""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING

from app.config import get_settings
from app.connections.redis import close_redis_client, create_redis_client
from app.features.auth.application.revocation.pending import (
    build_production_reconciler,
    close_production_resources,
)
from app.observability.logging import logger

if TYPE_CHECKING:
    from argparse import Namespace


async def reconcile(args: Namespace) -> dict[str, object]:
    settings = get_settings()
    redis = create_redis_client(settings.REDIS_URL)
    try:
        reconciler = await build_production_reconciler(redis, settings)
        return await reconciler.run(
            user_id=args.user_id,
            dry_run=not args.apply,
            force=args.force,
        )
    finally:
        await close_redis_client(redis)
        close_production_resources()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="revoke and clear markers")
    parser.add_argument("--dry-run", action="store_true", help="report only (default)")
    parser.add_argument("--force", action="store_true", help="retry deferred/poisoned records")
    parser.add_argument("--user-id", help="restrict the bounded pass to one user")
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive")
    report = asyncio.run(reconcile(args))
    logger.bind(**report).info("auth_pending_revocation_manual_reconciliation_finished")


if __name__ == "__main__":
    main()
