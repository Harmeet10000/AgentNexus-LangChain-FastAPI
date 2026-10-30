"""Pytest fixtures for the ingestion gate: checksum-verified corpus fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from evals.ingestion.fixtures_loader import _verify_checksums

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture(scope="session")
def fixtures() -> dict[str, Any]:
    """All gate fixtures, verified against CHECKSUMS.sha256 (fail closed)."""
    return _verify_checksums()
