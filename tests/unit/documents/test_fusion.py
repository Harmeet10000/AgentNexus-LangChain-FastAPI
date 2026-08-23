from app.features.documents.fusion import RankedResultRow, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_merges_duplicate_results() -> None:
    fused = reciprocal_rank_fusion(
        [
            RankedResultRow(chunk_id="a", score=0.0, rank=1),
            RankedResultRow(chunk_id="b", score=0.0, rank=2),
        ],
        [
            RankedResultRow(chunk_id="b", score=0.0, rank=1),
            RankedResultRow(chunk_id="c", score=0.0, rank=2),
        ],
        k=60,
        limit=3,
    )

    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].score > fused[1].score > fused[2].score
    assert fused[0].model_dump()["chunk_id"] == "b"


def test_reciprocal_rank_fusion_truncates_to_requested_limit() -> None:
    fused = reciprocal_rank_fusion(
        [RankedResultRow(chunk_id="a", score=0.0, rank=1)],
        [RankedResultRow(chunk_id="b", score=0.0, rank=1)],
        k=60,
        limit=1,
    )

    assert len(fused) == 1
    assert fused[0].model_dump()["rank"] == 1


def test_reciprocal_rank_fusion_supports_trigram_branch() -> None:
    fused = reciprocal_rank_fusion(
        [
            RankedResultRow(chunk_id="a", score=0.0, rank=1),
            RankedResultRow(chunk_id="c", score=0.0, rank=2),
        ],
        [
            RankedResultRow(chunk_id="b", score=0.0, rank=1),
            RankedResultRow(chunk_id="c", score=0.0, rank=2),
        ],
        [
            RankedResultRow(chunk_id="c", score=0.0, rank=1),
            RankedResultRow(chunk_id="d", score=0.0, rank=2),
        ],
        k=60,
        limit=4,
    )

    assert [item.chunk_id for item in fused] == ["c", "a", "b", "d"]


def test_reciprocal_rank_fusion_with_three_sources() -> None:
    bm25 = [
        RankedResultRow(chunk_id="x", score=0.0, rank=1),
        RankedResultRow(chunk_id="y", score=0.0, rank=2),
        RankedResultRow(chunk_id="z", score=0.0, rank=3),
    ]
    vector = [
        RankedResultRow(chunk_id="y", score=0.0, rank=1),
        RankedResultRow(chunk_id="z", score=0.0, rank=2),
    ]
    trigram = [
        RankedResultRow(chunk_id="z", score=0.0, rank=1),
    ]

    fused = reciprocal_rank_fusion(bm25, vector, trigram, k=60, limit=5)

    assert len(fused) == 3
    assert fused[0].chunk_id == "z"
    scores = {c.chunk_id: c.score for c in fused}
    assert scores["z"] > scores["y"] > scores["x"]


def test_reciprocal_rank_fusion_empty_input() -> None:
    fused = reciprocal_rank_fusion(k=60, limit=10)
    assert fused == []
