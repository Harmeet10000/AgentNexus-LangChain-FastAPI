"""Tests for fixture loading utilities.

This module validates that all test fixtures can be loaded correctly
and contain the expected data structure.
"""

from __future__ import annotations

import pytest

from evals.fixtures import (
    FIXTURES_DIR,
    load_golden_set_fixture,
    load_research_output_fixture,
    load_ticket_fixture,
)


class TestGoldenSetFixtures:
    """Test golden set fixture loading."""

    def test_load_ingestion_basic_fixture(self) -> None:
        """Test loading basic ingestion golden set fixture."""
        rows = load_golden_set_fixture("ingestion_basic")

        assert len(rows) == 5
        assert all("case_id" in row for row in rows)
        assert all("query" in row for row in rows)
        assert all("case_type" in row for row in rows)
        assert all("expected_ingestion_states" in row for row in rows)

    def test_load_coverage_gap_examples_fixture(self) -> None:
        """Test loading coverage gap examples fixture."""
        rows = load_golden_set_fixture("coverage_gap_examples")

        assert len(rows) == 3
        assert all("case_id" in row for row in rows)
        assert all(row["quality_tier"] == "candidate" for row in rows)

    def test_load_nonexistent_golden_set_fixture_raises_error(self) -> None:
        """Test that loading a nonexistent fixture raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Golden set fixture not found"):
            load_golden_set_fixture("nonexistent_fixture")


class TestTicketFixtures:
    """Test support ticket fixture loading."""

    def test_load_retrieval_failures_fixture(self) -> None:
        """Test loading retrieval failures ticket fixture."""
        tickets = load_ticket_fixture("retrieval_failures")

        assert len(tickets) == 10
        assert all("ticket_id" in ticket for ticket in tickets)
        assert all("customer_id" in ticket for ticket in tickets)
        assert all("issue_category" in ticket for ticket in tickets)
        assert all("symptom_description" in ticket for ticket in tickets)
        assert all(ticket["issue_category"] == "retrieval_failure" for ticket in tickets)

    def test_load_clustered_failures_fixture(self) -> None:
        """Test loading clustered failures ticket fixture."""
        tickets = load_ticket_fixture("clustered_failures")

        assert len(tickets) == 6
        # All tickets should have similar symptom patterns for clustering tests
        assert all("indemn" in ticket["symptom_description"].lower() for ticket in tickets)

    def test_load_nonexistent_ticket_fixture_raises_error(self) -> None:
        """Test that loading a nonexistent fixture raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Ticket fixture not found"):
            load_ticket_fixture("nonexistent_fixture")


class TestResearchOutputFixtures:
    """Test deep research output fixture loading."""

    def test_load_complete_report_fixture(self) -> None:
        """Test loading complete report fixture."""
        output = load_research_output_fixture("complete_report")

        assert "operation_id" in output
        assert "initial_query" in output
        assert "report_sections" in output
        assert "retrieved_documents" in output
        assert "process_metrics" in output
        assert len(output["report_sections"]) == 3
        # Complete report should have citations
        assert any(len(section.get("citations", [])) > 0 for section in output["report_sections"])

    def test_load_missing_citations_report_fixture(self) -> None:
        """Test loading report with missing citations."""
        output = load_research_output_fixture("missing_citations_report")

        assert "report_sections" in output
        # Should have sections without citations
        sections_without_citations = [
            section
            for section in output["report_sections"]
            if len(section.get("citations", [])) == 0
        ]
        assert len(sections_without_citations) > 0

    def test_load_low_quality_report_fixture(self) -> None:
        """Test loading low quality report fixture."""
        output = load_research_output_fixture("low_quality_report")

        assert "process_metrics" in output
        # Low quality report should have minimal follow-up queries
        assert output["process_metrics"]["follow_up_queries"] == 0

    def test_load_contradictory_claims_report_fixture(self) -> None:
        """Test loading report with contradictory claims."""
        output = load_research_output_fixture("contradictory_claims_report")

        assert "report_sections" in output
        assert len(output["report_sections"]) >= 2
        # Should have multiple sections that could contain contradictions
        section_titles = [section["title"] for section in output["report_sections"]]
        assert "Limited Warranty Provisions" in section_titles
        assert "Warranty Disclaimers" in section_titles

    def test_load_nonexistent_research_output_fixture_raises_error(self) -> None:
        """Test that loading a nonexistent fixture raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Research output fixture not found"):
            load_research_output_fixture("nonexistent_fixture")


class TestFixturesDirectory:
    """Test fixtures directory structure."""

    def test_fixtures_dir_exists(self) -> None:
        """Test that fixtures directory exists."""
        assert FIXTURES_DIR.exists()
        assert FIXTURES_DIR.is_dir()

    def test_fixtures_subdirectories_exist(self) -> None:
        """Test that all fixture subdirectories exist."""
        assert (FIXTURES_DIR / "golden_sets").exists()
        assert (FIXTURES_DIR / "tickets").exists()
        assert (FIXTURES_DIR / "research_outputs").exists()

    def test_readme_exists(self) -> None:
        """Test that fixtures README exists."""
        readme_path = FIXTURES_DIR / "README.md"
        assert readme_path.exists()
