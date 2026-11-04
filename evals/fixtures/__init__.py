"""Test fixtures for evaluation infrastructure testing.

This module provides fixture loading utilities and sample data for testing
ingestion evaluation, deep research metrics, and signal ingestion pipeline.

Fixture Categories:
- Golden Sets: Sample golden set rows for ingestion and retrieval testing
- Tickets: Sample support tickets with various failure patterns
- Research Outputs: Sample deep research reports with varying quality
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

FIXTURES_DIR = Path(__file__).parent


def load_golden_set_fixture(fixture_name: str) -> list[dict[str, Any]]:
    """Load a golden set fixture from the golden_sets directory.

    Args:
        fixture_name: Name of the fixture file (without .jsonl extension)

    Returns:
        List of golden set row dictionaries

    Example:
        >>> rows = load_golden_set_fixture("ingestion_basic")
        >>> len(rows)
        5
    """
    fixture_path = FIXTURES_DIR / "golden_sets" / f"{fixture_name}.jsonl"
    if not fixture_path.exists():
        msg = f"Golden set fixture not found: {fixture_name}"
        raise FileNotFoundError(msg)

    rows = []
    with fixture_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_ticket_fixture(fixture_name: str) -> list[dict[str, Any]]:
    """Load a support ticket fixture from the tickets directory.

    Args:
        fixture_name: Name of the fixture file (without .json extension)

    Returns:
        List of support ticket dictionaries

    Example:
        >>> tickets = load_ticket_fixture("retrieval_failures")
        >>> len(tickets)
        10
    """
    fixture_path = FIXTURES_DIR / "tickets" / f"{fixture_name}.json"
    if not fixture_path.exists():
        msg = f"Ticket fixture not found: {fixture_name}"
        raise FileNotFoundError(msg)

    with fixture_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_research_output_fixture(fixture_name: str) -> dict[str, Any]:
    """Load a deep research output fixture from the research_outputs directory.

    Args:
        fixture_name: Name of the fixture file (without .json extension)

    Returns:
        Deep research output dictionary

    Example:
        >>> output = load_research_output_fixture("complete_report")
        >>> output["report_sections"]
        [...]
    """
    fixture_path = FIXTURES_DIR / "research_outputs" / f"{fixture_name}.json"
    if not fixture_path.exists():
        msg = f"Research output fixture not found: {fixture_name}"
        raise FileNotFoundError(msg)

    with fixture_path.open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "FIXTURES_DIR",
    "load_golden_set_fixture",
    "load_research_output_fixture",
    "load_ticket_fixture",
]
