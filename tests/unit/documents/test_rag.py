from app.features.documents.fusion import RankedChunk
from app.features.documents.rag import SearchChunkRecord, assemble_rag_context


def test_assemble_rag_context_groups_by_document_and_merges_adjacent_chunks() -> None:
    ranked_chunks = [
        RankedChunk(chunk_id="c2", score=0.9, rank=1),
        RankedChunk(chunk_id="c1", score=0.8, rank=2),
        RankedChunk(chunk_id="c3", score=0.7, rank=3),
    ]
    chunk_lookup = {
        "c1": SearchChunkRecord(
            document_id="doc-1",
            chunk_index=0,
            content="alpha",
            title="Doc 1",
            chunk_metadata={"heading": "Intro"},
        ),
        "c2": SearchChunkRecord(
            document_id="doc-1",
            chunk_index=1,
            content="beta",
            title="Doc 1",
            chunk_metadata={"heading": "Intro"},
        ),
        "c3": SearchChunkRecord(
            document_id="doc-2",
            chunk_index=0,
            content="gamma",
            title="Doc 2",
            chunk_metadata={"heading": "Elsewhere"},
        ),
    }

    sections = assemble_rag_context(
        ranked_chunks, chunk_lookup, max_tokens=100, count_tokens=lambda text: len(text.split())
    )

    assert [section.document_id for section in sections] == ["doc-1", "doc-2"]
    assert sections[0].content == "alpha\n\nbeta"
    assert sections[0].chunk_indices == [0, 1]
    assert sections[1].content == "gamma"
    assert sections[0].model_dump()["document_id"] == "doc-1"


def test_assemble_rag_context_drops_sections_at_the_token_boundary() -> None:
    from app.features.documents.fusion import RankedChunk
    from app.features.documents.rag import SearchChunkRecord, assemble_rag_context

    def ten_words(index: int) -> str:
        return " ".join(f"w{i}" for i in range(index * 10, index * 10 + 10))

    ranked_chunks = [
        RankedChunk(chunk_id="c1", score=0.9, rank=1),
        RankedChunk(chunk_id="c2", score=0.8, rank=2),
    ]
    chunk_lookup = {
        "c1": SearchChunkRecord(
            document_id="doc-1",
            chunk_index=0,
            content=ten_words(0),
            title="Doc 1",
            chunk_metadata={},
        ),
        "c2": SearchChunkRecord(
            document_id="doc-2",
            chunk_index=0,
            content=ten_words(1),
            title="Doc 2",
            chunk_metadata={},
        ),
    }

    # Token count diverges from word count by 200%: 10 words count as 30
    # tokens. A word-count budget of 45 would keep both sections; the token
    # budget keeps the first (30 <= 45) and drops the second (60 > 45).
    sections = assemble_rag_context(
        ranked_chunks,
        chunk_lookup,
        max_tokens=45,
        count_tokens=lambda text: 3 * len(text.split()),
    )

    assert [section.document_id for section in sections] == ["doc-1"]
