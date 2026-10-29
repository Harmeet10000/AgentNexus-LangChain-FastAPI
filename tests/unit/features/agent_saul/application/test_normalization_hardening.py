"""Regression coverage for incomplete normalization model output."""

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from app.features.agent_saul.domain.state import (
    DocumentSection,
    NormalizedDocument,
    WorkflowStatus,
)
from app.features.agent_saul.graph.nodes import make_normalization_node, make_segmentation_node
from app.features.agent_saul.graph.prompts import _NORMALIZATION_SYSTEM_PROMPT

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

    from app.features.agent_saul.domain.state import LegalAgentState


def test_document_section_infers_omitted_level_only_from_numeric_hierarchy() -> None:
    top_level = DocumentSection.model_validate(
        {"section_id": "3", "title": "Rent", "content": "Monthly rent."}
    )
    nested = DocumentSection.model_validate(
        {"section_id": "3.1", "title": "Monthly Rent", "content": "Pay monthly."}
    )
    deeper = DocumentSection.model_validate(
        {"section_id": "3.1.2", "title": "Late Fee", "content": "Fee applies."}
    )

    assert [top_level.level, nested.level, deeper.level] == [1, 2, 3]


def test_document_section_does_not_guess_level_for_prose_identifier() -> None:
    with pytest.raises(ValidationError, match="level"):
        DocumentSection.model_validate(
            {"section_id": "Schedule A", "title": "Schedule", "content": "Terms."}
        )


def test_normalized_document_discards_empty_artifact_but_requires_real_sections() -> None:
    normalized = NormalizedDocument.model_validate(
        {
            "document_id": "doc-1",
            "sections": [
                {"section_id": "3", "title": "Rent", "content": "Monthly rent.", "level": 1},
                {},
            ],
        }
    )
    assert len(normalized.sections) == 1

    with pytest.raises(ValidationError):
        NormalizedDocument.model_validate({"document_id": "doc-1", "sections": [{}]})

    with pytest.raises(ValidationError, match="title"):
        NormalizedDocument.model_validate(
            {"document_id": "doc-1", "sections": [{"section_id": "3"}]}
        )


def test_normalization_parse_failure_returns_graph_error_state() -> None:
    class InvalidOutputRunnable:
        async def ainvoke(self, input: Any, config: Any = None) -> NormalizedDocument:
            message = "model omitted required section fields"
            raise OutputParserException(message)

    node = make_normalization_node(
        cast("Runnable[list[Any], NormalizedDocument]", InvalidOutputRunnable())
    )
    state = cast(
        "LegalAgentState",
        {"doc_id": "doc-1", "document_text": "Section 3.1: Pay monthly."},
    )

    result = asyncio.run(node(state))

    assert result["status"] == WorkflowStatus.FAILED
    errors = result["errors"]
    assert errors[0].node == "normalization"
    assert errors[0].code == "INVALID_NORMALIZED_DOCUMENT"
    assert errors[0].retryable is True


def test_normalization_prompt_requires_level_for_every_section() -> None:
    assert "Every section must include section_id, title, content, and level" in (
        _NORMALIZATION_SYSTEM_PROMPT
    )
    assert "1-based hierarchy depth" in _NORMALIZATION_SYSTEM_PROMPT


def test_segmentation_parse_failure_returns_graph_error_state() -> None:
    class InvalidOutputRunnable:
        async def ainvoke(self, input: Any, config: Any = None) -> Any:
            message = "model omitted required segment fields"
            raise OutputParserException(message)

    node = make_segmentation_node(cast("Runnable[list[Any], Any]", InvalidOutputRunnable()))
    state = cast(
        "LegalAgentState",
        {
            "doc_id": "doc-1",
            "normalized_document": NormalizedDocument(
                document_id="doc-1",
                sections=[
                    DocumentSection(section_id="1", title="Rent", content="Pay monthly.", level=1)
                ],
            ),
            "messages": [],
        },
    )

    result = asyncio.run(node(state))

    assert result["status"] == WorkflowStatus.FAILED
    assert result["errors"][0].node == "segmentation"
    assert result["errors"][0].code == "INVALID_CLAUSE_SEGMENTATION"
    assert result["errors"][0].retryable is True
