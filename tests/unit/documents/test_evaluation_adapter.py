from app.features.documents.classification import classify_document
from app.features.documents.evaluation import stored_document_kind, stored_jurisdiction


def test_golden_document_kinds_map_to_ingestion_kinds() -> None:
    assert stored_document_kind("contracts") == "legal_contract"
    assert stored_document_kind("statutes") == "generic"
    assert stored_document_kind("judgments") == "generic"
    assert stored_document_kind("filings") == "generic"


def test_statute_mapping_matches_the_ingestion_classifier() -> None:
    classified = classify_document(
        filename="Indian Contract Act.pdf",
        markdown="Section 73 provides compensation for loss caused by breach of contract.",
    )

    assert classified.document_kind == "generic"
    assert stored_document_kind("statutes") == classified.document_kind
    assert stored_jurisdiction("statutes", "India") == classified.jurisdiction


def test_contract_evaluation_retains_the_ingested_jurisdiction() -> None:
    assert stored_jurisdiction("contracts", "India") == "India"
