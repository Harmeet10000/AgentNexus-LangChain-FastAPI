"""Warn when staged files exceed the approximate token budget."""

from __future__ import annotations

import subprocess
import sys

MAX_BYTES = 20_000


def main() -> int:
    oversized: list[tuple[str, int]] = []

    for filename in sys.argv[1:]:
        staged_file = subprocess.run(
            ["git", "show", f":{filename}"],
            check=False,
            capture_output=True,
        )
        if staged_file.returncode != 0:
            continue

        byte_count = len(staged_file.stdout)
        if byte_count > MAX_BYTES:
            oversized.append((filename, byte_count))

    if oversized:
        print("WARNING: staged files exceed 5,000 approximate tokens (bytes / 4):")
        for filename, byte_count in oversized:
            print(f"  {filename}: {byte_count / 4:,.0f} tokens ({byte_count:,} bytes)")
        print(
            "Split by responsibility: move cohesive classes or functions into "
            "focused modules, keep shared contracts in a small common module, "
            "and preserve existing imports with re-exports where needed. "
            "Move or update tests alongside the code."
        )
        print("This check is warning-only; it will not block the commit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
