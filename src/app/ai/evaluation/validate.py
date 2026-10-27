"""Command-line validation for evaluation gold-set provenance."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from returns.result import Failure

from .schema import load_validated_golden_set


async def _validate(path: Path) -> int:
    result = await load_validated_golden_set(path)
    if isinstance(result, Failure):
        sys.stderr.write(f"{result.failure().message}\n")
        return 1
    sys.stdout.write(f"validated {path} ({len(result.unwrap().queries)} items)\n")
    return 0


def main() -> None:
    """Validate one or more JSONL gold-set files and return non-zero on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    status = max(asyncio.run(_validate(path)) for path in args.paths)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
