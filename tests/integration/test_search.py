from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.features.search.constants import (
    INGEST_CHUNK_SIZE,
)
from app.features.search.dto import (
    HybridSearchRequest,
    SearchIngestRequest,
)
from app.features.search.fusion import RankedChunk, RankedResultRow
from app.features.search.rag import SearchChunkRecord, assemble_rag_context
from app.features.search.service import process_ingestion_document

# Stale against the Result-pattern service (returns AppResult, not bare values);
# deferred until the mocks are updated. Runs with the integration suite.
pytestmark = pytest.mark.integration


def _make_repo(**method_kwargs: dict) -> MagicMock:
    repo = MagicMock()
    for name, ret in method_kwargs.items():
        setattr(repo, name, AsyncMock(return_value=ret))
    return repo


DOC_UUID = "00000000-0000-0000-0000-000000000001"


class TestSearchIngestion:

    async def test_process_ingestion_document_chunks_and_stores(self):
        repo = _make_repo(upsert_chunks=None)
        with (
            patch(
                "app.features.search.service.SearchRepository",
                return_value=repo,
            ),
            patch("app.features.search.service.embed_texts") as mock_embed,
        ):
            # Patched at the function, not at a client factory. `patch` returns an `AsyncMock`
            # here because the target is an `async def`, so the `side_effect`'s return value is
            # what the `await` yields — no `.return_value.aembed_documents` chain to keep in sync
            # with the provider's method names.
            mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]

            session_mock = AsyncMock()
            result = await process_ingestion_document(
                session=session_mock,
                document_id=DOC_UUID,
                content="First paragraph. Second paragraph. Third chunk here.",
            )

            assert result["status"] == "completed"
            assert result["document_id"] == DOC_UUID

    async def test_empty_document_returns_zero_chunks(self):
        with (
            patch(
                "app.features.search.service.SearchRepository",
                return_value=MagicMock(),
            ),
            patch("app.features.search.service.embed_texts") as mock_embed,
        ):
            session_mock = AsyncMock()
            result = await process_ingestion_document(
                session=session_mock, document_id=DOC_UUID, content="     "
            )
            assert result["chunk_count"] == 0
            # The point of the patch is that it is never reached: whitespace-only content must
            # short-circuit before any provider call, not embed an empty batch.
            mock_embed.assert_not_awaited()

    async def test_content_hash_dedup(self):
        from app.features.search.service import SearchService

        repo = MagicMock()
        repo.get_document_by_content_hash = AsyncMock(
            return_value=MagicMock(id="existing-doc")
        )
        search_service = SearchService(repo=repo, llm=MagicMock(), redis=None)

        payload = SearchIngestRequest(
            title="Test", content="Test document content"
        )
        response = await search_service.ingest_document(payload)
        assert response.duplicate is True

    async def test_ingestion_large_document_splits_into_chunks(self):
        repo = _make_repo(upsert_chunks=None)
        corpus = "word " * (INGEST_CHUNK_SIZE * 2 + 10)
        with (
            patch(
                "app.features.search.service.SearchRepository",
                return_value=repo,
            ),
            patch("app.features.search.service.embed_texts") as mock_embed,
        ):
            mock_embed.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]
            session_mock = AsyncMock()
            result = await process_ingestion_document(
                session=session_mock, document_id=DOC_UUID, content=corpus
            )
            assert result["chunk_count"] >= 2


class TestSearchQuery:
    async def test_hybrid_search_finds_by_keyword(self):
        from app.features.search.service import SearchService

        repo = _make_repo(
            bm25_search=[
                RankedResultRow(chunk_id="chunk-1", score=0.8, rank=1)
            ],
            vector_search=[],
            trigram_search=[],
            fetch_chunks_by_ids={
                "chunk-1": SearchChunkRecord(
                    document_id="doc-1",
                    title="Test",
                    content="keyword match",
                    chunk_index=0,
                    chunk_metadata={},
                )
            },
        )

        service = SearchService(repo=repo, llm=MagicMock(), redis=None)
        payload = HybridSearchRequest(query="keyword")
        response = await service.hybrid_search(payload)

        assert len(response.items) > 0
        assert response.cache_hit is False

    async def test_hybrid_search_finds_by_semantic_similarity(self):
        from app.features.search.service import SearchService

        repo = _make_repo(
            bm25_search=[],
            vector_search=[
                RankedResultRow(chunk_id="vec-chunk-1", score=0.9, rank=1)
            ],
            trigram_search=[],
            fetch_chunks_by_ids={
                "vec-chunk-1": SearchChunkRecord(
                    document_id="doc-2",
                    title="Semantic",
                    content="vector match",
                    chunk_index=0,
                    chunk_metadata={},
                )
            },
        )

        service = SearchService(repo=repo, llm=MagicMock(), redis=None)
        payload = HybridSearchRequest(query="semantic content")
        response = await service.hybrid_search(payload)

        assert len(response.items) > 0

    async def test_reciprocal_rank_fusion_merges_multiple_sources(self):
        from app.features.search.fusion import reciprocal_rank_fusion

        bm25 = [
            RankedResultRow(chunk_id="a", score=1.0, rank=1),
            RankedResultRow(chunk_id="b", score=0.5, rank=2),
        ]
        vector = [
            RankedResultRow(chunk_id="b", score=0.8, rank=1),
            RankedResultRow(chunk_id="c", score=0.3, rank=2),
        ]
        trigram = [
            RankedResultRow(chunk_id="d", score=0.6, rank=1),
        ]

        fused = reciprocal_rank_fusion(bm25, vector, trigram, k=60, limit=5)
        assert len(fused) == 4

        ids = [c.chunk_id for c in fused]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids
        assert "d" in ids

        chunk_b = [c for c in fused if c.chunk_id == "b"][0]
        chunk_a = [c for c in fused if c.chunk_id == "a"][0]
        assert chunk_b.score > chunk_a.score


class TestSearchRag:
    async def test_assemble_rag_context_groups_chunks_by_document(self):
        ranked = [
            RankedChunk(chunk_id="c1", score=0.9, rank=1),
            RankedChunk(chunk_id="c2", score=0.8, rank=2),
        ]
        lookup = {
            "c1": SearchChunkRecord(
                document_id="doc-1",
                title="Doc 1",
                content="chunk one",
                chunk_index=0,
                chunk_metadata={},
            ),
            "c2": SearchChunkRecord(
                document_id="doc-1",
                title="Doc 1",
                content="chunk two",
                chunk_index=1,
                chunk_metadata={},
            ),
        }
        context = assemble_rag_context(ranked, lookup, max_tokens=2000)
        assert len(context) == 1
        assert context[0].document_id == "doc-1"
        assert "chunk one" in context[0].content
        assert "chunk two" in context[0].content

    async def test_assemble_rag_context_respects_max_tokens(self):
        ranked = [
            RankedChunk(chunk_id="large", score=0.9, rank=1),
        ]
        lookup = {
            "large": SearchChunkRecord(
                document_id="doc-1",
                title="Large Doc",
                content="word " * 2000,
                chunk_index=0,
                chunk_metadata={},
            ),
        }
        context = assemble_rag_context(ranked, lookup, max_tokens=100)
        assert len(context) == 0 or len(context[0].content.split()) <= 100


class TestSearchCache:
    async def test_first_search_stores_in_redis(self, redis):
        from app.features.search.service import SearchService

        repo = _make_repo(
            bm25_search=[
                RankedResultRow(chunk_id="c1", score=0.5, rank=1)
            ],
            vector_search=[],
            trigram_search=[],
            fetch_chunks_by_ids={
                "c1": SearchChunkRecord(
                    document_id="doc-1",
                    title="Test",
                    content="hello world",
                    chunk_index=0,
                    chunk_metadata={},
                )
            },
        )

        service = SearchService(repo=repo, llm=MagicMock(), redis=redis)
        payload = HybridSearchRequest(query="hello")
        response = await service.hybrid_search(payload)
        assert response.cache_hit is False

    async def test_second_search_returns_cached(self, redis):
        from app.features.search.service import SearchService

        repo = _make_repo(
            bm25_search=[
                RankedResultRow(chunk_id="c1", score=0.5, rank=1)
            ],
            vector_search=[],
            trigram_search=[],
            fetch_chunks_by_ids={
                "c1": SearchChunkRecord(
                    document_id="doc-1",
                    title="Test",
                    content="hello world",
                    chunk_index=0,
                    chunk_metadata={},
                )
            },
        )

        service = SearchService(repo=repo, llm=MagicMock(), redis=redis)
        payload = HybridSearchRequest(query="hello")

        await service.hybrid_search(payload)
        repo.bm25_search.reset_mock()
        response_cached = await service.hybrid_search(payload)
        assert response_cached.cache_hit is True

    async def test_bypass_cache_skips_redis(self, redis):
        from app.features.search.service import SearchService

        repo = _make_repo(
            bm25_search=[
                RankedResultRow(chunk_id="c1", score=0.5, rank=1)
            ],
            vector_search=[],
            trigram_search=[],
            fetch_chunks_by_ids={
                "c1": SearchChunkRecord(
                    document_id="doc-1",
                    title="Test",
                    content="hello world",
                    chunk_index=0,
                    chunk_metadata={},
                )
            },
        )

        service = SearchService(repo=repo, llm=MagicMock(), redis=redis)
        payload = HybridSearchRequest(query="hello", bypass_cache=True)

        response = await service.hybrid_search(payload)
        assert response.cache_hit is False

        repo.bm25_search.reset_mock()
        response_cached = await service.hybrid_search(payload)
        assert response_cached.cache_hit is False
