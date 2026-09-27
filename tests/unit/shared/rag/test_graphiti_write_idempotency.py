"""Deterministic Graphiti episode identity across ingestion retries."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest

from app.shared.rag.graphiti.client import write_clause_episode
from app.shared.rag.graphiti.schemas import ClauseEpisodeMetadata

if TYPE_CHECKING:
    from typing import Any


@pytest.mark.asyncio
async def test_clause_retry_reuses_the_same_graphiti_episode_uuid() -> None:
    add_episode = AsyncMock(return_value=SimpleNamespace())
    graphiti = cast("Any", SimpleNamespace(add_episode=add_episode))
    metadata = ClauseEpisodeMetadata(
        doc_id="doc-1",
        clause_id="4.1",
        clause_type="payment",
        jurisdiction="India",
        document_type="legal_contract",
        user_id="user-1",
        thread_id="ingestion:doc-1",
    )

    first = await write_clause_episode(graphiti, "Payment is due.", metadata)
    second = await write_clause_episode(graphiti, "Payment is due.", metadata)

    assert first == second
    assert add_episode.await_args_list[0].kwargs["uuid"] == first
    assert add_episode.await_args_list[1].kwargs["uuid"] == first
