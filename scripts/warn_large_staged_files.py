"""Warn when staged files exceed the approximate token budget."""

from __future__ import annotations

import subprocess  # noqa: S404 — invokes the local Git executable only
import sys

MAX_BYTES = 20_000


def main() -> int:
    oversized: list[tuple[str, int]] = []

    for filename in sys.argv[1:]:
        staged_file = subprocess.run(  # noqa: S603 — fixed local command; filename is a Git ref
            ["git", "show", f":{filename}"],  # noqa: S607 — Git is intentionally resolved from PATH
            check=False,
            capture_output=True,
        )
        if staged_file.returncode != 0:
            continue

        byte_count = len(staged_file.stdout)
        if byte_count > MAX_BYTES:
            oversized.append((filename, byte_count))

    if oversized:
        sys.stdout.write("WARNING: staged files exceed 5,000 approximate tokens (bytes / 4):\n")
        for filename, byte_count in oversized:
            sys.stdout.write(
                f"  {filename}: {byte_count / 4:,.0f} tokens ({byte_count:,} bytes)\n"
            )
        sys.stdout.write(
            "Split by responsibility: move cohesive classes or functions into "
            "focused modules, keep shared contracts in a small common module, "
            "and preserve existing imports with re-exports where needed. "
            "Move or update tests alongside the code.\n"
        )
        sys.stdout.write("This check is warning-only; it will not block the commit.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
