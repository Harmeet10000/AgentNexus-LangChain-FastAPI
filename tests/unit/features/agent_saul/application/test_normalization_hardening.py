"""Regression coverage for document-structure model shapes.

Task 6.1 (harden-agent-saul-evidence-pipeline) removed the LLM normalization
and segmentation paths: structure comes exclusively from the deterministic
persisted-chunk adapter (covered by ``test_canonical_chunks.py``). What remains
model-produced is the structured-output *schema*, so the shape contracts below
keep guarding what a provider payload must satisfy when the schema types are
still accepted across checkpoint hydration.
"""

import pytest
from pydantic import ValidationError

from app.features.agent_saul.domain.state import DocumentSection, NormalizedDocument


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
                {"section_id": "3", "title": "Rent", "content": "Pay monthly.", "level": 1},
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
