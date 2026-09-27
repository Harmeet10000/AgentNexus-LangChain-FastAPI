"""Celery child-process resource lifetime proofs for document ingestion."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.documents.service import (
    _document_ingestion_lock_key,
    _release_document_ingestion_lock,
    run_document_ingestion_task,
)
from app.lifecycle import document_worker
from app.shared.langchain_layer.agents.tools.idempotency import IdempotencyGuard

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_worker_state() -> Iterator[None]:
    document_worker._WORKER_RESOURCES = None
    document_worker._WORKER_RUNNER = None
    yield
    runner = document_worker._WORKER_RUNNER
    document_worker._WORKER_RESOURCES = None
    document_worker._WORKER_RUNNER = None
    if runner is not None:
        runner.close()


def test_worker_hooks_construct_once_across_two_invocations_and_release_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {"construct": 0, "release": 0, "invoke": 0}
    resources = cast(
        "Any",
        SimpleNamespace(
            engine=object(),
            session_local=object(),
            graphiti=object(),
            ingestion_graph=object(),
        ),
    )

    async def provision() -> Any:
        counts["construct"] += 1
        return resources

    async def release(received: Any) -> None:
        assert received is resources
        counts["release"] += 1

    async def invoke() -> int:
        counts["invoke"] += 1
        return counts["invoke"]

    monkeypatch.setattr(document_worker, "_provision_document_worker", provision)
    monkeypatch.setattr(document_worker, "_release_document_worker", release)

    document_worker.initialize_document_worker()
    assert document_worker.get_document_worker_resources() is resources
    assert document_worker.run_on_document_worker_loop(invoke) == 1
    assert document_worker.run_on_document_worker_loop(invoke) == 2
    document_worker.shutdown_document_worker()

    assert counts == {"construct": 1, "release": 1, "invoke": 2}


def test_shutdown_without_successful_initialization_releases_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases: list[object] = []

    async def release(resources: object) -> None:
        releases.append(resources)

    monkeypatch.setattr(document_worker, "_release_document_worker", release)

    document_worker.shutdown_document_worker()

    assert releases == []


def test_failed_initialization_does_not_leave_a_runner_or_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail() -> Any:
        message = "provisioning failed"
        raise RuntimeError(message)

    monkeypatch.setattr(document_worker, "_provision_document_worker", fail)

    document_worker.initialize_document_worker()

    assert document_worker._WORKER_RESOURCES is None
    assert document_worker._WORKER_RUNNER is None


def test_non_ingestion_worker_skips_document_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provision = AsyncMock()
    monkeypatch.setattr(document_worker, "_provision_document_worker", provision)
    monkeypatch.setattr(document_worker.sys, "argv", ["celery", "worker", "-Q", "default"])
    monkeypatch.setattr(
        document_worker,
        "get_settings",
        lambda: SimpleNamespace(CELERY_INGESTION_QUEUE="ingestion"),
    )

    document_worker.initialize_document_worker()

    provision.assert_not_awaited()
    assert document_worker._WORKER_RUNNER is None
    assert document_worker._WORKER_RESOURCES is None


@pytest.mark.asyncio
async def test_release_attempts_every_resource_after_close_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_error = RuntimeError("graph close failed")
    redis_error = RuntimeError("redis close failed")
    close_graph = AsyncMock(side_effect=graph_error)
    redis = SimpleNamespace(aclose=AsyncMock(side_effect=redis_error))
    engine = SimpleNamespace(dispose=AsyncMock())
    monkeypatch.setattr(document_worker, "close_graphiti", close_graph)
    resources = cast(
        "Any",
        SimpleNamespace(graphiti=object(), redis=redis, engine=engine),
    )

    with pytest.raises(ExceptionGroup) as raised:
        await document_worker._release_document_worker(resources)

    assert raised.value.exceptions == (graph_error, redis_error)
    close_graph.assert_awaited_once()
    redis.aclose.assert_awaited_once()
    engine.dispose.assert_awaited_once()


def test_worker_compiles_once_and_two_service_invocations_reuse_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {
        "bound_session": 0,
        "commit": 0,
        "compile": 0,
        "invoke": 0,
        "lock": 0,
        "unlock": 0,
    }
    active_lock_connections: list[object] = []

    class Transaction:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: object) -> None:
            return None

    class SessionContext:
        def __init__(self, **local_kw: object) -> None:
            assert local_kw.get("bind") is active_lock_connections[-1]
            counts["bound_session"] += 1
            self.session = MagicMock(spec=AsyncSession)
            self.session.begin.return_value = Transaction()

        async def __aenter__(self) -> AsyncSession:
            return self.session

        async def __aexit__(self, *_args: object) -> None:
            return None

    class Graph:
        async def ainvoke(self, _state: object, config: dict[str, object]) -> dict[str, object]:
            assert counts["lock"] == counts["invoke"] + 1
            assert counts["commit"] == counts["invoke"] * 2 + 1
            counts["invoke"] += 1
            configurable = cast("dict[str, object]", config["configurable"])
            assert "document_repository" in configurable
            assert isinstance(configurable.get("document_idempotency"), IdempotencyGuard)
            return {"status": "completed"}

    class LockConnection:
        async def __aenter__(self) -> LockConnection:
            active_lock_connections.append(self)
            return self

        async def __aexit__(self, *_args: object) -> None:
            assert active_lock_connections.pop() is self

        async def scalar(self, *_args: object, **_kwargs: object) -> bool:
            statement = str(_args[0])
            if "pg_try_advisory_lock" in statement:
                counts["lock"] += 1
            else:
                counts["unlock"] += 1
            return True

        async def commit(self) -> None:
            counts["commit"] += 1

        async def invalidate(self) -> None:
            pytest.fail("A successful advisory unlock must not invalidate the connection")

    engine = SimpleNamespace(connect=LockConnection, dispose=AsyncMock())
    graphiti = object()
    compiled_graph = Graph()

    async def init_db() -> tuple[object, object]:
        return engine, SessionContext

    async def setup_graphiti(**_kwargs: object) -> object:
        return graphiti

    async def setup_indices(_graphiti: object) -> None:
        return None

    def provide_graph(**_kwargs: object) -> Graph:
        counts["compile"] += 1
        idempotency = _kwargs["idempotency"]
        assert isinstance(idempotency, IdempotencyGuard)
        return compiled_graph

    settings = SimpleNamespace(
        NEO4J_URI="bolt://unused",
        NEO4J_USERNAME="unused",
        NEO4J_PASSWORD=SimpleNamespace(get_secret_value=lambda: "unused"),
    )
    monkeypatch.setattr(document_worker, "get_settings", lambda: settings)
    monkeypatch.setattr(document_worker, "init_db", init_db)
    monkeypatch.setattr(document_worker, "setup_graphiti", setup_graphiti)
    monkeypatch.setattr(document_worker, "setup_graphiti_indices", setup_indices)
    monkeypatch.setattr(document_worker.StorageService, "from_settings", lambda **_kwargs: object())
    monkeypatch.setattr(document_worker, "provide_document_ingestion_graph", provide_graph)

    document_worker.initialize_document_worker()
    resources = document_worker.get_document_worker_resources()
    for document_id in ("doc-1", "doc-2"):
        result = document_worker.run_on_document_worker_loop(
            lambda document_id=document_id: run_document_ingestion_task(
                document_id=document_id,
                user_id="user-1",
                filename="fixture.txt",
                content_type="text/plain",
                object_uri="s3://bucket/fixture.txt",
                graph=resources.ingestion_graph,
                engine=resources.engine,
                session_local=resources.session_local,
                idempotency=resources.idempotency,
            )
        )
        assert result == {"status": "completed"}

    assert counts == {
        "bound_session": 2,
        "commit": 4,
        "compile": 1,
        "invoke": 2,
        "lock": 2,
        "unlock": 2,
    }


def test_document_ingestion_advisory_lock_key_is_stable_and_tenant_scoped() -> None:
    first = _document_ingestion_lock_key("user-1", "doc-1")

    assert first == _document_ingestion_lock_key("user-1", "doc-1")
    assert first != _document_ingestion_lock_key("user-2", "doc-1")
    assert -(2**63) <= first < 2**63


@pytest.mark.asyncio
async def test_busy_document_advisory_lock_skips_without_waiting_or_ingesting() -> None:
    commits = 0

    class BusyConnection:
        async def __aenter__(self) -> BusyConnection:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def scalar(self, *_args: object, **_kwargs: object) -> bool:
            return False

        async def commit(self) -> None:
            nonlocal commits
            commits += 1

        async def execute(self, *_args: object, **_kwargs: object) -> None:
            pytest.fail("A lock not owned by this task must not be unlocked")

    class Graph:
        async def ainvoke(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            pytest.fail("A duplicate delivery must not enter the ingestion graph")

    def session_local(**_local_kw: object) -> object:
        pytest.fail("A duplicate delivery must not open an ingestion transaction")

    result = await run_document_ingestion_task(
        document_id="doc-1",
        user_id="user-1",
        filename="fixture.txt",
        content_type="text/plain",
        object_uri="s3://bucket/fixture.txt",
        graph=cast("Any", Graph()),
        engine=cast("Any", SimpleNamespace(connect=BusyConnection)),
        session_local=cast("Any", session_local),
        idempotency=cast("Any", object()),
    )

    assert result == {"status": "skipped", "document_id": "doc-1"}
    assert commits == 1


@pytest.mark.asyncio
async def test_failed_advisory_unlock_invalidates_connection() -> None:
    connection = SimpleNamespace(
        scalar=AsyncMock(return_value=False),
        commit=AsyncMock(),
        invalidate=AsyncMock(),
    )

    with pytest.raises(RuntimeError, match="advisory lock release failed"):
        await _release_document_ingestion_lock(
            connection=cast("Any", connection),
            lock_key=42,
            primary_error=None,
        )

    connection.invalidate.assert_awaited_once()


@pytest.mark.asyncio
async def test_advisory_unlock_failure_does_not_replace_primary_error() -> None:
    cleanup_error = RuntimeError("database connection lost")
    connection = SimpleNamespace(
        scalar=AsyncMock(side_effect=cleanup_error),
        commit=AsyncMock(),
        invalidate=AsyncMock(),
    )
    primary_error = ValueError("ingestion failed")

    await _release_document_ingestion_lock(
        connection=cast("Any", connection),
        lock_key=42,
        primary_error=primary_error,
    )

    connection.invalidate.assert_awaited_once()
    assert primary_error.__notes__ == [
        "Advisory-lock cleanup failed: RuntimeError('database connection lost')"
    ]
