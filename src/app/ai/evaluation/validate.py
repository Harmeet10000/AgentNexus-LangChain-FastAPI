"""Command-line validation for evaluation gold-set provenance."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from returns.result import Failure

from .schema import (
    INGESTION_FIDELITY_SET_VERSION,
    load_ingestion_fidelity_set,
    load_validated_golden_set,
)


def _set_version(path: Path) -> str | None:
    """Peek the version string from a gold-set metadata line, or None."""
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0]
        value = json.loads(first).get("version")
        return value if isinstance(value, str) else None
    except (OSError, IndexError, json.JSONDecodeError, AttributeError):
        return None


async def _validate(path: Path) -> int:
    if _set_version(path) == INGESTION_FIDELITY_SET_VERSION:
        result = await load_ingestion_fidelity_set(path)
        if isinstance(result, Failure):
            sys.stderr.write(f"{result.failure().message}\n")
            return 1
        sys.stdout.write(f"validated {path} ({len(result.unwrap().rows)} rows)\n")
        return 0
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
