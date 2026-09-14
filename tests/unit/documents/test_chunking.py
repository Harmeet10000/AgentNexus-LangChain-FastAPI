"""The documents feature exposes only policy-driven structural chunking."""

from app.features.documents.chunking import resolve_chunk_policy
from app.features.documents.classification import ClassifiedDocument, ParsedDocument, segment_chunks


async def test_generic_ingestion_reaches_structure_aware_chunking() -> None:
    chunks, _warnings = await segment_chunks(
        parsed=ParsedDocument(
            title="Unicode notes",
            markdown="# Résumé\n\n## Café\n\nnaïve coöperatief über cool",
            page_count=1,
        ),
        classified=ClassifiedDocument(document_kind="generic", parties=[]),
    )

    assert chunks
    assert "café" in chunks[0].preamble.lower()
    assert "naïve coöperatief über cool" in chunks[0].content
    assert chunks[0].metadata_["chunk_policy"] == resolve_chunk_policy("generic").name


def test_every_structure_aware_policy_has_no_overlap() -> None:
    assert all(
        resolve_chunk_policy(kind).overlap == 0
        for kind in ("contracts", "statutes", "judgments", "filings", "generic")
    )
