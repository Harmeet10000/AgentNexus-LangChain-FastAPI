"""Kind dispatch and clause-locus recovery (ingestion-chunking 3.x / 4.x)."""

from __future__ import annotations

from app.features.documents.chunking import resolve_chunk_policy
from app.features.documents.classification import (
    PreparedChunk,
    recover_clause_numbering,
)


def test_four_legal_families_resolve_distinct_policies() -> None:
    names = {
        resolve_chunk_policy("contracts").name,
        resolve_chunk_policy("statutes").name,
        resolve_chunk_policy("judgments").name,
        resolve_chunk_policy("filings").name,
    }
    assert names == {"contract", "statute", "judgment", "filing"}
    assert len(names) == 4


def test_unclassifiable_kind_resolves_default_without_raising() -> None:
    policy = resolve_chunk_policy("totally-unknown-kind")
    assert policy.name == "default"


def test_structure_aware_policies_have_zero_overlap() -> None:
    for kind in ("contracts", "statutes", "judgments", "filings", "generic"):
        assert resolve_chunk_policy(kind).overlap == 0


def test_recover_clause_numbering_from_inline_bold_fixture() -> None:
    chunks = [
        PreparedChunk(
            chunk_index=0,
            chunk_kind="legal_contract",
            # Matches `_CLAUSE_START_RE`'s numbered form (`12.3)` + body).
            content="12.3) The parties shall indemnify each other against claims.",
        ),
        PreparedChunk(
            chunk_index=1,
            chunk_kind="legal_contract",
            content="Section 4 Termination. Either party may terminate on notice.",
        ),
    ]

    recovered = recover_clause_numbering(chunks)

    assert recovered[0].locus == "12.3)"
    assert recovered[1].locus.lower().startswith("section 4")


def test_recover_clause_numbering_keeps_absent_locus_when_unnumbered() -> None:
    chunks = [
        PreparedChunk(
            chunk_index=0,
            chunk_kind="generic",
            content="This preamble has no clause number at all.",
        )
    ]

    recovered = recover_clause_numbering(chunks)

    assert recovered[0].locus is None
