"""Fixture loading with fail-closed checksum verification for the ingestion gate."""

from __future__ import annotations

import hashlib
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CHECKSUMS_FILE = FIXTURES_DIR / "CHECKSUMS.sha256"

FORGED_TENANT = "personal:victim-tenant-999"


class FixtureCorruptionError(RuntimeError):
    """A fixture file does not match its recorded checksum."""


def _verify_checksums() -> dict[str, str]:
    """Verify every recorded fixture checksum; fail closed on any mismatch."""
    if not CHECKSUMS_FILE.exists():
        message = f"missing fixture checksum manifest: {CHECKSUMS_FILE}"
        raise FixtureCorruptionError(message)
    verified: dict[str, str] = {}
    for raw_line in CHECKSUMS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, _, filename = line.partition("  ")
        path = FIXTURES_DIR / filename.strip()
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash.strip():
            message = f"fixture checksum mismatch: {filename} (corrupted?)"
            raise FixtureCorruptionError(message)
        verified[filename.strip()] = path.read_text(encoding="utf-8")
    return verified
