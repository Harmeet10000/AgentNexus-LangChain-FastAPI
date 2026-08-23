from app.features.documents.chunking import chunk_text


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


def test_chunk_text_overlap_behavior() -> None:
    text = "one two three four five six seven eight"
    chunks = chunk_text(text, chunk_size=4, chunk_overlap=2)

    assert len(chunks) >= 2
    assert chunks[0].content == "one two three four"
    assert chunks[1].content == "three four five six"
    assert chunks[0].content != chunks[1].content
    assert "three four" in chunks[0].content
    assert "three four" in chunks[1].content


def test_chunk_text_unicode_content() -> None:
    text = "café résumé naïve coöperatief über cool"
    chunks = chunk_text(text, chunk_size=3, chunk_overlap=0)

    assert len(chunks) >= 2
    assert "café" in chunks[0].content
    assert all(len(c.content) > 0 for c in chunks)
