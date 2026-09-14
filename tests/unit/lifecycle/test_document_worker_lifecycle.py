"""Celery child-process resource lifetime proofs for document ingestion."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.documents.service import run_document_ingestion_task
from app.lifecycle import document_worker

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


def test_worker_compiles_once_and_two_service_invocations_reuse_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {"compile": 0, "invoke": 0}

    class Transaction:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *_args: object) -> None:
            return None

    class SessionContext:
        def __init__(self) -> None:
            self.session = MagicMock(spec=AsyncSession)
            self.session.begin.return_value = Transaction()

        async def __aenter__(self) -> AsyncSession:
            return self.session

        async def __aexit__(self, *_args: object) -> None:
            return None

    class Graph:
        async def ainvoke(self, _state: object, config: dict[str, object]) -> dict[str, object]:
            counts["invoke"] += 1
            configurable = cast("dict[str, object]", config["configurable"])
            assert "document_repository" in configurable
            return {"status": "completed"}

    engine = SimpleNamespace(dispose=AsyncMock())
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
                session_local=resources.session_local,
            )
        )
        assert result == {"status": "completed"}

    assert counts == {"compile": 1, "invoke": 2}
