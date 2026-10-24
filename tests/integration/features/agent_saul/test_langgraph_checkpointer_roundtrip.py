"""LangGraph 3.2 saver round-trip against an isolated Postgres test database.

The test is opt-in because ``AsyncPostgresSaver.setup()`` applies checkpoint DDL. CI
should provide ``LANGGRAPH_POSTGRES_TEST_DSN`` pointing at a disposable database; a
developer checkout without that variable skips rather than touching the application's
managed database or its placeholder host.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.connections.checkpointer import setup_langgraph_checkpointer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]


@pytest.fixture
async def saver() -> AsyncIterator[tuple[AsyncPostgresSaver, str]]:
    dsn = os.getenv("LANGGRAPH_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("LANGGRAPH_POSTGRES_TEST_DSN is required for the saver integration test")

    checkpointer = await setup_langgraph_checkpointer(dsn)
    thread_id = f"langgraph-upgrade-{uuid4()}"
    try:
        yield checkpointer, thread_id
    finally:
        await checkpointer.adelete_thread(thread_id)
        await checkpointer.conn.close()


async def test_application_pool_saver_writes_reads_and_resumes_state(
    saver: tuple[AsyncPostgresSaver, str],
) -> None:
    """Exercise setup, a checkpoint write, a read, and a parent-linked resume."""
    checkpointer, thread_id = saver
    first = empty_checkpoint()
    first["channel_values"] = {"state": "before-resume"}
    first["channel_versions"] = {"state": 1}

    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
        }
    }
    first_config = await checkpointer.aput(
        config,
        first,
        {"source": "input", "step": 1, "parents": {}},
        {"state": 1},
    )

    first_state = await checkpointer.aget_tuple(first_config)
    assert first_state is not None
    assert first_state.checkpoint["channel_values"]["state"] == "before-resume"

    resumed = empty_checkpoint()
    resumed["channel_values"] = {"state": "after-resume"}
    resumed["channel_versions"] = {"state": 2}
    resumed_config = await checkpointer.aput(
        first_config,
        resumed,
        {"source": "loop", "step": 2, "parents": {}},
        {"state": 2},
    )

    resumed_state = await checkpointer.aget_tuple(resumed_config)
    assert resumed_state is not None
    assert resumed_state.checkpoint["channel_values"]["state"] == "after-resume"
    assert resumed_state.parent_config == first_config


async def test_application_pool_saver_keeps_threads_isolated(
    saver: tuple[AsyncPostgresSaver, str],
) -> None:
    checkpointer, thread_id = saver
    other_thread_id = f"langgraph-isolation-{uuid4()}"
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"state": "only-first-thread"}
    checkpoint["channel_versions"] = {"state": 1}
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    other_config = {"configurable": {"thread_id": other_thread_id, "checkpoint_ns": ""}}

    try:
        await checkpointer.aput(
            config,
            checkpoint,
            {"source": "input", "step": 1, "parents": {}},
            {"state": 1},
        )

        assert await checkpointer.aget_tuple(config) is not None
        assert await checkpointer.aget_tuple(other_config) is None
    finally:
        await checkpointer.adelete_thread(other_thread_id)
