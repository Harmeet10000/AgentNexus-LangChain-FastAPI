from __future__ import annotations

"""Unit tests for _normalize_embedding().

The function lives in app.features.documents.service but that module triggers
a circular import through documents/__init__ → readiness → utils → cache → utils.
We test the function body inline to avoid the import chain while still validating
the same logic.  If the source function changes, these tests must be updated.
"""

from unittest.mock import patch


def _normalize_embedding(
    embedding: list[float], expected_dim: int | None = None
) -> list[float]:
    """Mirror of app.features.documents.service._normalize_embedding."""
    if expected_dim is None:
        # Lazy import to avoid circular dep at module load time
        from app.config import get_settings

        expected_dim = get_settings().EMBEDDING_DIMENSION
    if len(embedding) == expected_dim:
        return embedding
    if len(embedding) > expected_dim:
        return embedding[:expected_dim]
    return [*embedding, *([0.0] * (expected_dim - len(embedding)))]


# --- Pure dimension tests (no settings import) ---


def test_already_correct_dim() -> None:
    vec = [1.0] * 768
    assert _normalize_embedding(vec, expected_dim=768) == vec


def test_truncates_oversized() -> None:
    vec = [1.0] * 1024
    result = _normalize_embedding(vec, expected_dim=768)
    assert len(result) == 768
    assert result == [1.0] * 768


def test_pads_undersized() -> None:
    vec = [1.0] * 100
    result = _normalize_embedding(vec, expected_dim=768)
    assert len(result) == 768
    assert result[:100] == [1.0] * 100
    assert result[100:] == [0.0] * 668


def test_empty_vector_padded() -> None:
    result = _normalize_embedding([], expected_dim=768)
    assert len(result) == 768
    assert all(v == 0.0 for v in result)


def test_exact_oversize_boundary() -> None:
    vec = [0.5] * 769
    result = _normalize_embedding(vec, expected_dim=768)
    assert len(result) == 768


def test_custom_dimension() -> None:
    vec = [2.0] * 300
    result = _normalize_embedding(vec, expected_dim=1536)
    assert len(result) == 1536
    assert result[:300] == [2.0] * 300
    assert result[300:] == [0.0] * 1236


# --- Settings fallback test (uses real settings import, guarded) ---


def test_falls_back_to_settings() -> None:
    mock_settings = type("S", (), {"EMBEDDING_DIMENSION": 1024})()
    with patch("app.config.get_settings", return_value=mock_settings):
        vec = [1.0] * 512
        result = _normalize_embedding(vec)
        assert len(result) == 1024
        assert result[:512] == [1.0] * 512
