from app.features.search.chunking import chunk_text


def test_chunk_text_preserves_order_and_overlap() -> None:
    text = " ".join(f"token-{index}" for index in range(12))

    chunks = chunk_text(text, chunk_size=5, chunk_overlap=2)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert chunks[0].content == "token-0 token-1 token-2 token-3 token-4"
    assert chunks[1].content == "token-3 token-4 token-5 token-6 token-7"
    assert chunks[2].content == "token-6 token-7 token-8 token-9 token-10"
    assert chunks[3].content == "token-9 token-10 token-11"
    assert chunks[0].model_dump()["token_count"] == 5


def test_chunk_text_returns_empty_list_for_blank_input() -> None:
    assert chunk_text("   ", chunk_size=5, chunk_overlap=1) == []
