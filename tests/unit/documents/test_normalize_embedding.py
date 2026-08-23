"""Unit tests for normalize_embedding() in app.utils.embedding."""

from __future__ import annotations

from unittest.mock import patch

from app.utils.embedding import normalize_embedding

# --- Pure dimension tests (no settings import) ---


def test_already_correct_dim() -> None:
    vec = [1.0] * 768
    assert normalize_embedding(vec, expected_dim=768) == vec


def test_truncates_oversized() -> None:
    vec = [1.0] * 1024
    result = normalize_embedding(vec, expected_dim=768)
    assert len(result) == 768
    assert result == [1.0] * 768


def test_pads_undersized() -> None:
    vec = [1.0] * 100
    result = normalize_embedding(vec, expected_dim=768)
    assert len(result) == 768
    assert result[:100] == [1.0] * 100
    assert result[100:] == [0.0] * 668


def test_empty_vector_padded() -> None:
    result = normalize_embedding([], expected_dim=768)
    assert len(result) == 768
    # Padding writes the literal 0.0, so list identity is exact by construction.
    assert result == [0.0] * 768


def test_exact_oversize_boundary() -> None:
    vec = [0.5] * 769
    result = normalize_embedding(vec, expected_dim=768)
    assert len(result) == 768


def test_custom_dimension() -> None:
    vec = [2.0] * 300
    result = normalize_embedding(vec, expected_dim=1536)
    assert len(result) == 1536
    assert result[:300] == [2.0] * 300
    assert result[300:] == [0.0] * 1236


# --- Settings fallback test (uses real settings import, guarded) ---


def test_falls_back_to_settings() -> None:
    mock_settings = type("S", (), {"EMBEDDING_DIMENSION": 1024})()
    with patch("app.config.get_settings", return_value=mock_settings):
        vec = [1.0] * 512
        result = normalize_embedding(vec)
        assert len(result) == 1024
        assert result[:512] == [1.0] * 512
