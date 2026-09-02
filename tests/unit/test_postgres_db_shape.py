"""7.6 pin get_postgres_db shape — no code change."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.connections.postgres import get_postgres_db


@pytest.mark.asyncio
async def test_get_postgres_db_commits_on_clean_exit():
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()
    # session_local() returns context manager yielding session
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_session
    mock_cm.__aexit__.return_value = None
    mock_session_local = MagicMock(return_value=mock_cm)

    mock_connection = MagicMock()
    mock_connection.app.state.db_session_local = mock_session_local

    gen = get_postgres_db(mock_connection)
    session = await anext(gen)
    assert session is mock_session
    # clean exit: commit, not rollback
    await gen.aclose()  # triggers finally + commit path
    # Actually get_postgres_db uses try/yield/commit — need to drive generator correctly
    # Use async generator protocol: send None and close
    # Simpler: iterate via async for
    mock_session2 = AsyncMock()
    mock_session2.commit = AsyncMock()
    mock_session2.rollback = AsyncMock()
    mock_session2.close = AsyncMock()
    mock_cm2 = AsyncMock()
    mock_cm2.__aenter__.return_value = mock_session2
    mock_cm2.__aexit__.return_value = None
    mock_session_local2 = MagicMock(return_value=mock_cm2)
    mock_connection2 = MagicMock()
    mock_connection2.app.state.db_session_local = mock_session_local2

    async def use_clean():
        async for sess in get_postgres_db(mock_connection2):
            assert sess is mock_session2
            # no exception -> should commit on exit

    await use_clean()
    mock_session2.commit.assert_awaited_once()
    mock_session2.rollback.assert_not_awaited()
    mock_session2.close.assert_awaited_once()


def test_get_postgres_db_rolls_back_on_exception():
    import inspect

    src = inspect.getsource(get_postgres_db)
    # Must rollback on exception, not commit
    assert "except Exception:" in src
    assert "await session.rollback()" in src
    # Must re-raise after rollback
    assert "raise" in src.split("await session.rollback()")[1].split("finally")[0]


def test_get_postgres_db_cannot_see_result():
    import inspect

    src = inspect.getsource(get_postgres_db)
    # Must not import Result or handle Failure — it is a pure session boundary
    assert "Result" not in src
    assert "Failure" not in src
    assert "Success" not in src
    # Must have commit on clean, rollback on exception, close in finally
    assert "await session.commit()" in src
    assert "await session.rollback()" in src
    assert "await session.close()" in src
