"""Real-process imports and loop ownership, independent of suite package mocks."""

from __future__ import annotations

import asyncio
import subprocess  # noqa: S404 — trusted local interpreter import smoke tests
import sys
from pathlib import Path

import pytest

from app.worker import runtime
from app.worker.reliability import run_redis_call
from app.worker.task_modules import TASK_MODULES


def test_producer_validation_imports_no_consumers_or_network_resources() -> None:
    script = """
import socket
import sys
def no_connection(*args, **kwargs):
    raise AssertionError("Producer import attempted network I/O")
socket.socket.connect = no_connection
from app.connections.celery_producer import celery_app
from app.shared.jobs.names import DOCUMENTS_INGEST
from app.shared.jobs.registry import CeleryTaskRegistry, TaskPayloadValidationError
assert len(CeleryTaskRegistry.registered_names()) == 13
try:
    CeleryTaskRegistry.validate(DOCUMENTS_INGEST, {})
except TaskPayloadValidationError:
    pass
else:
    raise AssertionError("Invalid payload was accepted")
assert not celery_app.conf.include
assert not celery_app.conf.imports
assert not any(n == "app.worker" or n.startswith("app.worker.") for n in sys.modules)
assert not any(n.startswith("app.features.") and n.endswith(".tasks") for n in sys.modules)
"""
    result = subprocess.run(  # noqa: S603 — literal script with trusted interpreter
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=45, check=False
    )
    assert result.returncode == 0, result.stderr


def test_fresh_worker_registers_all_wire_names_without_provisioning() -> None:
    script = """
import inspect
import socket
def no_connection(*args, **kwargs):
    raise AssertionError("Worker import attempted network I/O")
socket.socket.connect = no_connection
from app.worker.app import celery_app
from app.worker.task_modules import TASK_DECLARING_MODULES
from app.shared.jobs.registry import CeleryTaskRegistry
celery_app.loader.import_default_modules()
assert set(TASK_DECLARING_MODULES) == CeleryTaskRegistry.registered_names()
assert set(TASK_DECLARING_MODULES) <= set(celery_app.tasks)
for entry in celery_app.conf.beat_schedule.values():
    task = celery_app.tasks[entry["task"]]
    inspect.signature(task.run).bind(*entry.get("args", ()), **entry.get("kwargs", {}))
from app.worker import crawler, documents, runtime
from app.worker.task import ResilientTask
assert crawler._WORKER_RESOURCES is None
assert documents._WORKER_RESOURCES is None
assert runtime._runner is None
assert ResilientTask._redis_client is None
"""
    result = subprocess.run(  # noqa: S603 — literal script with trusted interpreter
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=45, check=False
    )
    assert result.returncode == 0, result.stderr


def test_every_feature_task_module_is_explicitly_registered() -> None:
    src = Path(__file__).resolve().parents[3] / "src"
    discovered = {
        ".".join(path.relative_to(src).with_suffix("").parts)
        for path in (src / "app/worker/tasks").glob("*.py")
        if path.name != "__init__.py"
    }
    assert discovered == set(TASK_MODULES)
    assert len(TASK_MODULES) == len(set(TASK_MODULES))


@pytest.fixture
def clean_runtime():
    runtime.close()
    yield
    runtime.close()


def test_repeated_async_reliability_operations_use_one_live_loop(clean_runtime) -> None:
    loops = []

    async def operation():
        loop = asyncio.get_running_loop()
        loops.append(loop)
        await asyncio.sleep(0)
        return len(loops)

    assert run_redis_call(operation()) == 1
    assert run_redis_call(operation()) == 2
    assert loops[0] is loops[1]
    assert not loops[0].is_closed()
    runtime.close()
    runtime.close()
    assert loops[0].is_closed()


def test_child_reset_discards_parent_runner(clean_runtime) -> None:
    async def loop():
        return asyncio.get_running_loop()

    parent_loop = runtime.run(loop())
    parent_runner = runtime._runner
    runtime.reset_after_fork()
    try:
        child_loop = runtime.run(loop())
        assert child_loop is not parent_loop
        assert not parent_loop.is_closed()
    finally:
        parent_runner.close()


def test_worker_reliability_pool_is_shared_and_closed_on_its_loop(
    clean_runtime, monkeypatch
) -> None:
    from app.worker import signals
    from app.worker.task import ResilientTask

    loops = []

    class Client:
        async def get(self):
            loops.append(asyncio.get_running_loop())
            return "ok"

        async def aclose(self):
            loops.append(asyncio.get_running_loop())

    client = Client()
    monkeypatch.setattr("app.worker.task.create_redis_client", lambda _url: client)
    signals.initialize_worker()

    class FirstTask(ResilientTask):
        pass

    class SecondTask(ResilientTask):
        pass

    assert FirstTask.get_redis_client() is SecondTask.get_redis_client()
    assert run_redis_call(client.get()) == "ok"
    signals.shutdown_worker()
    signals.shutdown_worker()
    assert len(loops) == 2
    assert loops[0] is loops[1]
    assert loops[0].is_closed()


@pytest.mark.asyncio
async def test_async_idempotency_context_uses_the_callers_loop() -> None:
    import json

    from fakeredis.aioredis import FakeRedis

    from app.worker.reliability import build_idempotency_key, idempotency_manager

    client = FakeRedis(decode_responses=True)
    try:
        async with idempotency_manager(client, "operation-1"):
            record = json.loads(await client.get(build_idempotency_key("operation-1")))
            assert record["status"] == "processing"
        record = json.loads(await client.get(build_idempotency_key("operation-1")))
        assert record["status"] == "completed"
    finally:
        await client.aclose()
