"""Shared embedding contract: the persisted vector dimension.

This is the *stored* width — the dimension `chunks.embedding` was migrated
with. It is read from settings so the write guard, the Cognee dimension pin,
and the mapped column all compare against one definition. Moving it here
changes no persisted shape: the value is identical, only the owner changed.
"""

from app.config import get_settings

CHUNK_EMBEDDING_DIM = get_settings().EMBEDDING_DIMENSION
