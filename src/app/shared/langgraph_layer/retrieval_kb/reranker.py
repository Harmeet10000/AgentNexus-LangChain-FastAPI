"""Cross-encoder reranking adapter for legal retrieval."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import asyncer
from sentence_transformers import CrossEncoder

from app.utils import logger

if TYPE_CHECKING:
    from .state import RetrievedChunk

_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_FALLBACK_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_shared_reranker() -> CrossEncoderReranker:
    """Return the process-lifetime shared re-ranker (default model).

    Every retrieval path re-ranks through this accessor rather than
    constructing per call: the cross-encoder weights load once per process and
    are reused. `lru_cache` (rather than a module global) is what makes "once"
    atomic under concurrent first requests. An explicit instance may still be
    injected (tests, alternate models) — the accessor is the default, not the
    only construction path.
    """
    return CrossEncoderReranker()


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
            scores = model.predict(pairs)  # ty: ignore[unresolved-attribute]
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
        except (OSError, ValueError, RuntimeError) as exc:
            exc.add_note(f"model={self.model_name}, operation=rerank")
            logger.bind(error=str(exc)).warning("cross_encoder_rerank_failed")
            return chunks[:limit]

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model

        try:
            self._model = CrossEncoder(self.model_name)
        except (OSError, ValueError) as exc:
            exc.add_note(f"model={self.model_name}, operation=load_model")
            logger.bind(model=self.model_name).warning("default_reranker_load_failed")
            self._model = CrossEncoder(_FALLBACK_RERANKER_MODEL)
        return self._model
