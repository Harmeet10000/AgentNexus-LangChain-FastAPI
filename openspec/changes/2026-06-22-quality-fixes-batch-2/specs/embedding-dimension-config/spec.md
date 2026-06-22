# Embedding Dimension Configurability

## Scope

- `src/app/features/documents/service.py` — `_normalize_embedding()` (line 804)
- `src/app/shared/langgraph_layer/retrieval_kb/nodes.py` — `_normalize_embedding()` (line 364)
- `src/app/shared/langgraph_layer/ingestion_kb/nodes.py` — `_normalize_embedding()` (line 738)
- `src/app/utils/embedding.py` — new consolidation module
- `src/app/config/settings.py` — `EMBEDDING_DIMENSION` (line 184)

## Problem

The same `_normalize_embedding()` function is copy-pasted in three files with identical logic:

```python
def _normalize_embedding(embedding: list[float], expected_dim: int | None = None) -> list[float]:
    if expected_dim is None:
        expected_dim = get_settings().EMBEDDING_DIMENSION
    if len(embedding) == expected_dim:
        return embedding
    if len(embedding) > expected_dim:
        return embedding[:expected_dim]
    return [*embedding, *([0.0] * (expected_dim - len(embedding)))]
```

If a developer adds a new call site and imports from the wrong file, they get the wrong function (or a hidden import cycle). Additionally, there is no warning when the runtime embedding dimension differs from `settings.EMBEDDING_DIMENSION`.

## Solution

### 1. Consolidate into `src/app/utils/embedding.py`

```python
"""Embedding utility functions."""

from __future__ import annotations

from app.utils import logger


def normalize_embedding(embedding: list[float], expected_dim: int | None = None) -> list[float]:
    """Normalise an embedding vector to the expected dimension.

    Logs a warning if the actual dimension differs from the expected dimension.
    """
    from app.config import get_settings  # avoid circular import

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
```

### 2. Replace all 3 call sites

Each of the three files imports `_normalize_embedding` → change to import `normalize_embedding` from `app.utils.embedding`.

### 3. Settings validation (optional non-breaking addition)

Add a `model_validator` to the settings class that compares `EMBEDDING_DIMENSION` against a known-good value for the configured embedding model:

```python
# In settings.py Settings class:
from pydantic import model_validator

_EMBEDDING_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-004": 768,       # Gemini
    "text-embedding-3-small": 1536,  # OpenAI
    "text-embedding-3-large": 3072,  # OpenAI
}

@model_validator(mode="before")
def validate_embedding_dimension(cls, values: dict[str, object]) -> dict[str, object]:
    dim = values.get("EMBEDDING_DIMENSION")
    model = values.get("EMBEDDINGS_MODEL", "")
    if isinstance(model, str) and model in _EMBEDDING_MODEL_DIMENSIONS:
        expected = _EMBEDDING_MODEL_DIMENSIONS[model]
        if dim is not None and dim != expected:
            import warnings
            warnings.warn(f"EMBEDDING_DIMENSION={dim} but {model} expects {expected}")
    return values
```

This is optional — the warning log in `normalize_embedding` already catches mismatches at runtime. Include if the settings file already has model-level config.

## Verification

1. Unit tests pass: `pytest tests/unit/documents/test_normalize_embedding.py` (update import path)
2. Manual: set `EMBEDDING_DIMENSION=512` in `.env.development`, run search, check logs for "embedding_dimension_mismatch"
3. `rg "_normalize_embedding" src/` returns 0 results after consolidation
