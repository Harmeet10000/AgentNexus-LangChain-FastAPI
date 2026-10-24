"""Repair Redis auth session indexes in bounded, resumable batches.

Dry-run is the default. Apply changes explicitly with ``--apply``. The command
never deletes a session record. Apply mode may restore a valid orphan
membership, remove stale memberships, and extend an existing index to the
longest live session.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from redis.asyncio import Redis

from app.config import get_settings
from app.features.auth.session_index_repair import SessionIndexRepair


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="mutate indexes (default is dry-run)")
    parser.add_argument("--cursor", type=int, default=0, help="index SCAN cursor from a previous run")
    parser.add_argument("--orphan-cursor", type=int, default=0, help="session SCAN cursor from a previous run")
    parser.add_argument("--batch-limit", type=int, default=100)
    parser.add_argument("--member-limit", type=int, default=1_000)
    parser.add_argument("--scan-count", type=int, default=100)
    parser.add_argument("--orphan-limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=2.0, dest="operation_timeout")
    parser.add_argument(
        "--key-prefix",
        default="",
        help="exact application namespace prefix (for example tenant-a:); never scan without the auth pattern",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    redis = Redis.from_url(
        settings.REDIS_URL,
        db=settings.REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=args.operation_timeout,
        socket_timeout=args.operation_timeout,
        retry_on_timeout=False,
    )
    try:
        repair = SessionIndexRepair(
            redis,
            key_prefix=args.key_prefix,
            batch_limit=args.batch_limit,
            member_limit=args.member_limit,
            scan_count=args.scan_count,
            operation_timeout=args.operation_timeout,
        )
        report = await repair.run(
            cursor=args.cursor,
            orphan_cursor=args.orphan_cursor,
            orphan_limit=args.orphan_limit,
            dry_run=not args.apply,
        )
        print(json.dumps(report.as_dict(), sort_keys=True))  # noqa: T201
    finally:
        await redis.aclose()


def main() -> None:
    asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    main()
