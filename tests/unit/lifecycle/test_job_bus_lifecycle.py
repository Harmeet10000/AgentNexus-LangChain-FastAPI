"""Composition-root tests for the production direct JobBus seam."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

import app.lifecycle.lifespan as lifespan_module

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_celery_setup_constructs_one_shared_adapter_and_job_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = object()
    monkeypatch.setattr(lifespan_module, "celery_app", producer)
    monkeypatch.setattr(lifespan_module, "setup_celery", lambda: producer)

    app = FastAPI()
    await lifespan_module._setup_celery_state(app, SimpleNamespace())

    adapter = app.state.direct_publication_adapter
    assert app.state.celery is producer
    assert adapter._celery is producer
    assert app.state.job_bus._direct_adapter is adapter


@pytest.mark.asyncio
async def test_broker_health_failure_keeps_producer_for_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = object()
    monkeypatch.setattr(lifespan_module, "celery_app", producer)
    monkeypatch.setattr(lifespan_module, "setup_celery", lambda: None)

    app = FastAPI()
    await lifespan_module._setup_celery_state(app, SimpleNamespace())

    assert app.state.celery is producer
    assert app.state.direct_publication_adapter._celery is producer
    assert app.state.job_bus is not None


@pytest.mark.asyncio
async def test_dev_without_broker_publishes_through_in_memory_job_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dev runs uvicorn-only; with the broker down the bus falls back to memory.

    Production never takes this branch: a job only recorded in RAM is a job
    silently lost, so this check also pins the environment distinction.
    """
    producer = object()
    monkeypatch.setattr(lifespan_module, "celery_app", producer)
    monkeypatch.setattr(lifespan_module, "setup_celery", lambda: None)
    monkeypatch.setattr(
        lifespan_module, "get_settings", lambda: SimpleNamespace(ENVIRONMENT="development")
    )
    monkeypatch.setattr(lifespan_module, "_dev_environment", lambda _settings: True)

    app = FastAPI()
    await lifespan_module._setup_celery_state(app, SimpleNamespace(ENVIRONMENT="development"))

    from app.shared.jobs.in_memory import InMemoryJobBus

    assert isinstance(app.state.job_bus, InMemoryJobBus)
    # The real transport stays on app.state for the outbox relay, which keeps
    # retrying durable events until RabbitMQ returns.
    assert app.state.direct_publication_adapter._celery is producer


@pytest.mark.asyncio
async def test_production_without_broker_keeps_the_real_job_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never publish into process memory outside development."""
    producer = object()
    monkeypatch.setattr(lifespan_module, "celery_app", producer)
    monkeypatch.setattr(lifespan_module, "setup_celery", lambda: None)
    monkeypatch.setattr(lifespan_module, "_dev_environment", lambda _settings: False)

    app = FastAPI()
    await lifespan_module._setup_celery_state(app, SimpleNamespace(ENVIRONMENT="production"))

    adapter = app.state.direct_publication_adapter
    assert isinstance(app.state.job_bus, lifespan_module.JobBus)
    assert app.state.job_bus._direct_adapter is adapter


@pytest.mark.asyncio
async def test_broker_health_failure_records_safe_degradation_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = object()
    failure = lifespan_module.ServiceUnavailableException("broker unavailable")
    monkeypatch.setattr(lifespan_module, "celery_app", producer)

    def fail_setup() -> object:
        raise failure

    monkeypatch.setattr(lifespan_module, "setup_celery", fail_setup)

    app = FastAPI()
    await lifespan_module._setup_celery_state(app, SimpleNamespace())

    assert app.state.celery is producer
    assert "component=celery, operation=setup_celery_state" in failure.__notes__


@pytest.mark.asyncio
async def test_relay_receives_the_exact_lifespan_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _finished() -> None:
        return None

    adapter = object()
    relay = SimpleNamespace(run_listener=_finished)
    captured: dict[str, object] = {}

    async def create(publisher: object, _session_factory: object) -> object:
        captured["publisher"] = publisher
        return relay

    listener_tasks: list[asyncio.Task[None]] = []
    real_create_task = asyncio.create_task

    def create_task(coro: object) -> asyncio.Task[None]:
        task = real_create_task(coro)  # type: ignore[arg-type]
        listener_tasks.append(task)
        return task

    app = FastAPI()
    app.state.direct_publication_adapter = adapter
    app.state.db_session_local = object()
    monkeypatch.setattr(lifespan_module, "create_outbox_relay", create)
    monkeypatch.setattr(lifespan_module.asyncio, "create_task", create_task)

    await lifespan_module._setup_outbox_relay_state(app, SimpleNamespace())
    await asyncio.gather(*listener_tasks)

    assert captured["publisher"] is adapter
    assert app.state.outbox_relay is relay


@pytest.mark.asyncio
async def test_relay_state_degrades_without_an_adapter() -> None:
    app = FastAPI()
    await lifespan_module._setup_outbox_relay_state(app, SimpleNamespace())
    assert app.state.outbox_relay is None


def test_lifespan_state_uses_no_second_celery_app() -> None:
    # Importing the lifecycle module exposes the configured producer object;
    # composition constructs adapters around it rather than calling Celery().
    assert lifespan_module.celery_app is not None
