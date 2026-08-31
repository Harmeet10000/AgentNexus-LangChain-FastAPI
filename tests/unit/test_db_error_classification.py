"""7.3 pin 49/7 split — relational dead (500) vs auth retryable (503)."""

import pathlib


def _count(pattern: str, paths: list[str]) -> int:
    import re

    total = 0
    for p in paths:
        text = pathlib.Path(p).read_text()
        total += len(re.findall(pattern, text))
    return total


def test_db_error_split():
    relational = [
        "src/app/features/audit/repository.py",
        "src/app/features/credits/repositories/credit_repository.py",
        "src/app/features/credits/repositories/consumption_repository.py",
        "src/app/features/documents/repository.py",
        "src/app/features/invoices/repository.py",
        "src/app/features/payments/repository.py",
        "src/app/features/subscriptions/repository.py",
        "src/app/features/webhooks/repository.py",
        "src/app/features/plans/repository.py",
    ]
    auth = ["src/app/features/auth/repository.py"]
    # Count ErrorCode.DATABASE_ERROR with retryable=False (relational dead)
    rel_dead = _count(r"(?s)DATABASE_ERROR.*?retryable=False", relational)  # type: ignore[arg-type]
    # Alternative: count all DATABASE_ERROR in relational (49 + 6 for plans already converted)
    rel_total = _count(r"DATABASE_ERROR", relational)
    auth_total = _count(r"DATABASE_ERROR", auth)
    # 49 original + 6 plans already had enum = 55 total relational; 7 auth
    assert rel_total == 55, f"relational {rel_total} != 55 (49+6 plans)"
    assert auth_total == 7, f"auth {auth_total} != 7"
    # All relational should be retryable=False
    assert rel_dead == 55, f"relational dead {rel_dead} != 55"
    # Auth should have 0 retryable=False (all retryable)
    auth_dead = _count(r"(?s)DATABASE_ERROR.*?retryable=False", auth)
    assert auth_dead == 0, f"auth dead {auth_dead} != 0"


def test_no_db_error_string_literal_remains():
    import pathlib

    # No literal "DB_ERROR" should remain in relational or auth (replaced with enum)
    text = pathlib.Path("src/app/features").as_posix()
    # Use python count instead of rg to avoid exit code
    import subprocess

    try:
        out = subprocess.check_output(
            ["rg", "-c", '"DB_ERROR"', "src/app/features/"], text=True
        )
        total = sum(int(line.split(":")[-1]) for line in out.strip().splitlines() if line)
    except subprocess.CalledProcessError:
        total = 0
    assert total == 0, f"DB_ERROR literal remains {total}"
