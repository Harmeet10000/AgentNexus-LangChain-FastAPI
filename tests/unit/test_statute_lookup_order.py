"""The statute point lookup prefers dated rows over unversioned rows."""

from pathlib import Path


def test_statute_lookup_orders_null_years_last() -> None:
    source = Path(
        "src/app/shared/langchain_layer/agents/tools/retrieve_statute_section.py"
    ).read_text(encoding="utf-8")
    assert "ORDER BY instrument_year DESC NULLS LAST" in source
