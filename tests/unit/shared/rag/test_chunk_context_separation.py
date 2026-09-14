"""Contract tests for contextualized chunk storage and fallback provenance."""

from __future__ import annotations

from types import SimpleNamespace

from docling_core.types.doc import DocItemLabel, DoclingDocument
from transformers import PreTrainedTokenizerBase

from app.shared.rag.docling.chunker import (
    _hybrid_chunk_documents,
    _simple_fallback_chunk,
    chunk_document,
    create_hybrid_chunker,
)
from app.shared.rag.docling.models import ChunkRequest, IngestionConfig


class _Tokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class _HybridTokenizer(PreTrainedTokenizerBase):
    def __len__(self) -> int:
        return 1000

    def encode(self, text, **_kwargs):
        return list(range(len(text.split()) + 2))

    def tokenize(self, text, **_kwargs):
        return text.split()


class _HybridChunker:
    def chunk(self, *, dl_doc: object):
        del dl_doc
        return [SimpleNamespace(text="The supplier shall indemnify the buyer.")]

    def contextualize(self, *, chunk: object) -> str:
        return f"Master Agreement\nIndemnity\n\n{chunk.text}"


def test_hybrid_chunk_separates_context_without_changing_search_text() -> None:
    old_search_text = "Master Agreement\nIndemnity\n\nThe supplier shall indemnify the buyer."

    chunks = _hybrid_chunk_documents(
        _HybridChunker(), object(), _Tokenizer(), {"document_kind": "contracts"}
    )

    assert len(chunks) == 1
    assert chunks[0].content == "The supplier shall indemnify the buyer."
    assert chunks[0].preamble == "Master Agreement\nIndemnity"
    assert chunks[0].search_text == old_search_text
    assert chunks[0].token_count == len(_Tokenizer().encode(old_search_text))


def test_hybrid_contextualized_text_respects_configured_token_bound() -> None:
    bound = 16
    chunks = _hybrid_chunk_documents(
        _HybridChunker(), object(), _Tokenizer(), {"document_kind": "contracts"}
    )

    assert all(len(_Tokenizer().encode(chunk.search_text)) <= bound for chunk in chunks)


async def test_real_hybrid_chunks_bound_the_contextualized_embedding_text() -> None:
    config = IngestionConfig(max_tokens=32)
    tokenizer = _HybridTokenizer()
    hybrid = create_hybrid_chunker(tokenizer, config)
    document = DoclingDocument(name="agreement")
    document.add_heading(text="Master Agreement", level=1)
    for index in range(8):
        document.add_text(
            label=DocItemLabel.TEXT,
            text=f"Section {index}. Supplier obligation with bounded clause text.",
        )

    chunks = await chunk_document(
        ChunkRequest(
            content="fallback",
            title="agreement",
            source="fixture.pdf",
            config=config,
            metadata={"document_kind": "contracts"},
        ),
        tokenizer,
        hybrid_chunker=hybrid,
        docling_doc=document,
    )

    assert chunks
    assert all(len(tokenizer.encode(chunk.search_text)) <= config.max_tokens for chunk in chunks)


def test_fallback_chunk_exposes_impurity_and_overlap_to_consumer() -> None:
    chunks = _simple_fallback_chunk(
        "a" * 220,
        {"document_kind": "generic"},
        IngestionConfig(chunk_size=100, chunk_overlap=10, min_chunk_size=1),
        _Tokenizer(),
    )

    assert chunks
    restored = type(chunks[0]).model_validate_json(chunks[0].model_dump_json())
    assert restored.metadata["impure_split"] is True
    assert restored.metadata["overlap"] == 10
