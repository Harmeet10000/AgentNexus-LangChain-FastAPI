"""Embedding utility functions."""

from __future__ import annotations

from app.utils.logger import logger


def normalize_embedding(embedding: list[float], expected_dim: int | None = None) -> list[float]:
    """Normalise an embedding vector to the expected dimension.

    Logs a warning if the actual dimension differs from the expected dimension.
    """
    from app.config import get_settings  # noqa: PLC0415 — lazy import to avoid circular dependency

    if expected_dim is None:
        expected_dim = get_settings().EMBEDDING_DIMENSION

    actual_dim = len(embedding)
    if actual_dim == expected_dim:
        return embedding

    logger.warning(
        "embedding_dimension_mismatch",
        actual=actual_dim,
        expected=expected_dim,
        delta=actual_dim - expected_dim,
    )

    if actual_dim > expected_dim:
        return embedding[:expected_dim]
    return [*embedding, *([0.0] * (expected_dim - actual_dim))]
