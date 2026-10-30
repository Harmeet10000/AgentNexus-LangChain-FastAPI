"""Pytest fixtures for the ingestion gate: checksum-verified corpus fixtures.

The gate is hermetic by construction: an autouse fixture replaces the
HuggingFace tokenizer used by the real chunker and the fidelity measurement
with a deterministic offline stub, so no CI run ever downloads model assets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from transformers import PreTrainedTokenizerBase

from app.features.documents.ingestion.chunking import classification, fidelity
from evals.ingestion.fixtures_loader import _verify_checksums

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture(scope="session")
def fixtures() -> dict[str, Any]:
    """All gate fixtures, verified against CHECKSUMS.sha256 (fail closed)."""
    return _verify_checksums()


class _OfflineTokenizer(PreTrainedTokenizerBase):
    """Deterministic ~1-token-per-whitespace-word counter.

    Satisfies the real ``HybridChunker`` (``count_tokens`` via ``tokenize``,
    ``semchunk`` via ``encode``) without any model download.  WordPiece maps a
    word onto slightly more tokens than whitespace splitting, so this stub
    under-counts modestly — the safe direction for a lower bound, and the
    plausibility bounds stay deterministic.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def tokenize(self, text: str, **_kwargs: object) -> list[str]:
        return str(text).split()

    def encode(
        self,
        text: str,
        *_args: object,
        **_kwargs: object,
    ) -> list[int]:
        return list(range(1, len(str(text).split()) + 1))

    def _tokenize(self, text: str, **_kwargs: object) -> list[str]:
        return str(text).split()

    def get_vocab(self) -> dict[str, int]:
        return {}

    @property
    def vocab_size(self) -> int:
        return 0

    def save_vocabulary(self, *_args: object, **_kwargs: object) -> tuple[str, None]:
        return ("", None)


@pytest.fixture(autouse=True)
def hermetic_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the chunker and fidelity token counters to the offline stub."""
    monkeypatch.setattr(classification, "get_tokenizer", _OfflineTokenizer)
    monkeypatch.setattr(fidelity, "get_tokenizer", _OfflineTokenizer)
