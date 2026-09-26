from app.features.documents.evaluation import stored_document_kind


def test_golden_document_kinds_map_to_ingestion_kinds() -> None:
    assert stored_document_kind("contracts") == "legal_contract"
    assert stored_document_kind("statutes") == "legal_policy"
    assert stored_document_kind("judgments") == "generic"
    assert stored_document_kind("filings") == "generic"
