"""Cross-encoder reranking adapter for legal retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncer
from sentence_transformers import CrossEncoder

from app.utils import logger

if TYPE_CHECKING:
    from .state import RetrievedChunk

_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_FALLBACK_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Lazy sentence-transformers cross-encoder wrapper.

    CPU-bound: move this behind Celery in V2 if query latency becomes visible.
    """

    def __init__(self, model_name: str = _DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model: object | None = None

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        def _sync_rerank() -> list[RetrievedChunk]:
            model = self._load_model()
            pairs = [(query, f"{chunk.preamble}\n\n{chunk.chunk_text}") for chunk in chunks]
            scores = model.predict(pairs)  # type: ignore[attr-defined]
            ranked = sorted(
                zip(chunks, scores, strict=False),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            return [
                chunk.model_copy(update={"score": float(score)}) for chunk, score in ranked[:limit]
            ]

        try:
            return await asyncer.asyncify(_sync_rerank)()
        except Exception as exc:  # noqa: BLE001 - retrieval can continue without reranking.
            logger.bind(error=str(exc)).warning("cross_encoder_rerank_failed")
            return chunks[:limit]

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model

        try:
            self._model = CrossEncoder(self.model_name)
        except Exception:  # noqa: BLE001 - configured fallback model is intentional.
            logger.bind(model=self.model_name).warning("default_reranker_load_failed")
            self._model = CrossEncoder(_FALLBACK_RERANKER_MODEL)
        return self._model
