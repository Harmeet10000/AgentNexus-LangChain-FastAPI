"""Pin graphiti's embedder to one item per request.

``gemini-embedding-2`` aggregates a multi-item ``contents`` list into a single
vector, while graphiti's node deduplication zips embeddings to nodes with
``strict=True``. Any configured model that is not ``gemini-embedding-001``
therefore has to be driven one query at a time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from app.connections import graphiti as graphiti_connection

if TYPE_CHECKING:
    from typing import Any

pytestmark = pytest.mark.unit


class _RecordingGraphiti:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


async def test_setup_graphiti_drives_the_embedder_one_query_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graphiti_connection, "Graphiti", _RecordingGraphiti)

    created = await graphiti_connection.setup_graphiti(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database="neo4j",
    )

    embedder = created.kwargs["embedder"]
    assert embedder.batch_size == 1
    assert embedder.config.embedding_dim == graphiti_connection.settings.EMBEDDING_DIMENSION
