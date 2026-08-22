"""Embedding utility functions."""

from __future__ import annotations

from app.utils.exceptions import InfrastructureException
from app.utils.logger import logger

# Imported from the leaf module rather than `app.utils`, matching 319c698: this
# module is itself imported from `app/utils/__init__.py:35`, so a package-level
# import here would re-enter a partially initialised package. `exceptions.py`
# imports only `typing`, `fastapi`, and `.codes`, so it is safe as a leaf.


def stored_width_mismatch(stored_dim: int) -> tuple[int, int] | None:
    """Return ``(stored, configured)`` when a relation's vector width disagrees with config.

    Returns ``None`` when they agree. This is the pure half of the dual-method
    pattern: it neither raises nor wraps, so a ``Result``-returning repository can
    turn the mismatch into a ``Failure`` while an offline batch path raises. See
    ``assert_stored_width_matches_configured`` for the raising half.

    Why this is a *stored*-width check and not the provider-width check in
    ``rag/document_processing/embedder._validated_width``: that one catches a model
    returning the wrong shape right now, which is recoverable by fixing the model
    id. This one catches a column whose width was fixed at migration time against
    a setting that has since changed — nothing about that is recoverable at
    runtime, because the vectors already in the column were produced by the old
    model and no conversion exists. The only remedy is re-embedding.
    """
    from app.config import get_settings  # noqa: PLC0415 — lazy import to avoid circular dependency

    expected = get_settings().EMBEDDING_DIMENSION
    if stored_dim == expected:
        return None
    return (stored_dim, expected)


def width_mismatch_detail(stored_dim: int, expected_dim: int, *, relation: str) -> str:
    """Phrase the one diagnostic both halves of the guard report.

    Shared so the ``Result`` path and the raising path cannot drift into saying
    different things about the same condition, and so the re-embedding remedy is
    stated exactly once.
    """
    return (
        f"{relation} stores {stored_dim}-dimensional vectors but EMBEDDING_DIMENSION "
        f"is {expected_dim}; writes are refused. Re-embedding the corpus at "
        f"{expected_dim} dimensions is required — changing the setting does not "
        f"convert vectors already stored, and a mixed-width column ranks nothing "
        f"correctly."
    )


def assert_stored_width_matches_configured(stored_dim: int, *, relation: str) -> None:
    """Raise unless ``relation``'s declared vector width equals the configured one.

    ``retryable=False`` is the load-bearing argument: a width disagreement is a
    deployment inconsistency, and a retry loop around it would spin forever
    against a condition no amount of waiting changes.

    Raises:
        InfrastructureException: the widths disagree.
    """
    mismatch = stored_width_mismatch(stored_dim)
    if mismatch is None:
        return

    stored, expected = mismatch
    msg = width_mismatch_detail(stored, expected, relation=relation)
    raise InfrastructureException(
        detail=msg,
        retryable=False,
        data={"relation": relation, "stored_dim": stored, "configured_dim": expected},
    )


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
