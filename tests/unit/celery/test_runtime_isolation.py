"""Real-process imports and loop ownership, independent of suite package mocks."""

from __future__ import annotations

import asyncio
import os
import subprocess  # noqa: S404 — trusted local interpreter import smoke tests
import sys
from pathlib import Path

import pytest

from app.worker.celery.registration import TASK_MODULES
from app.worker.execution import runtime
from app.worker.execution.reliability import run_redis_call


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
assert len(CeleryTaskRegistry.registered_names()) == 16
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


@pytest.mark.parametrize("role", ["default", "ingestion", "crawler"])
def test_fresh_role_process_loads_only_its_wire_names_without_provisioning(role: str) -> None:
    """Each supported role loads only its catalog subset in a fresh process."""
    from app.worker.celery.profiles import profile_for_role

    script = f"""
import socket
import sys
def no_connection(*args, **kwargs):
    raise AssertionError("Worker import attempted network I/O")
socket.socket.connect = no_connection
from app.worker.celery.app import create_worker_celery_app
from app.worker.celery.profiles import profile_for_role
from app.worker.execution import runtime

app = create_worker_celery_app({role!r})
profile = profile_for_role({role!r})
assert tuple(app.conf.include) == profile.task_modules
app.loader.import_default_modules()
loaded = {{name for name in sys.modules if name.startswith("app.worker.jobs.")}}
expected_modules = set(profile.task_modules)
assert expected_modules <= loaded
assert loaded <= expected_modules
assert profile.task_names <= set(app.tasks)
assert runtime._runner is None
print("TASK_NAMES=" + ",".join(sorted(profile.task_names)))
"""
    result = subprocess.run(  # noqa: S603 — literal script with trusted interpreter
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    marker = next(line for line in result.stdout.splitlines() if line.startswith("TASK_NAMES="))
    actual = set(marker.removeprefix("TASK_NAMES=").split(","))
    assert actual == profile_for_role(role).task_names


def test_role_profiles_union_matches_catalog() -> None:
    from app.shared.jobs.catalog import JOB_CATALOG
    from app.worker.celery.profiles import profile_for_role

    registered = set().union(
        *(profile_for_role(role).task_names for role in ("default", "ingestion", "crawler"))
    )
    assert registered == set(JOB_CATALOG)


def test_fresh_beat_validates_catalog_without_importing_consumers() -> None:
    script = """
import sys
from app.worker.celery.app import CatalogValidatingBeat, create_worker_celery_app
from app.worker.celery.completeness import ensure_beat_catalog_completeness

app = create_worker_celery_app(beat=True)
assert tuple(app.conf.include) == ()
beat = CatalogValidatingBeat(app=app)
beat.init_loader()
ensure_beat_catalog_completeness(app)
assert not any(name.startswith("app.worker.jobs.") for name in sys.modules)
assert not any(name.startswith("app.worker.capabilities.") for name in sys.modules)
print("BEAT_OK")
"""
    result = subprocess.run(  # noqa: S603 — literal script with trusted interpreter
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=45, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "BEAT_OK" in result.stdout


def test_every_feature_task_module_is_explicitly_registered() -> None:
    src = Path(__file__).resolve().parents[3] / "src"
    discovered = {
        ".".join(path.relative_to(src).with_suffix("").parts)
        for path in (src / "app/worker/jobs").glob("*.py")
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
    import app.worker.celery.task as task_module
    from app.worker.celery import signals
    from app.worker.celery.task import ResilientTask

    # The real-celery fixture reloads worker modules between test modules; bind
    # the signal implementation to this test's canonical task class so the
    # assertion exercises the shutdown contract rather than module identity.
    monkeypatch.setattr(signals, "ResilientTask", ResilientTask)

    loops = []

    class Client:
        async def get(self):
            loops.append(asyncio.get_running_loop())
            return "ok"

        async def aclose(self):
            loops.append(asyncio.get_running_loop())

    client = Client()
    monkeypatch.setattr(task_module, "create_redis_client", lambda _url: client)
    signals.initialize_worker()
    # Keep the class-level pool explicit when another test module has reloaded
    # the canonical task module after this test's imports.
    ResilientTask._redis_client = client
    ResilientTask._redis_pid = os.getpid()

    class FirstTask(ResilientTask):
        pass

    class SecondTask(ResilientTask):
        pass

    assert FirstTask.get_redis_client() is SecondTask.get_redis_client()
    assert signals.ResilientTask._redis_client is client
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

    from app.worker.execution.reliability import build_idempotency_key, idempotency_manager

    client = FakeRedis(decode_responses=True)
    try:
        async with idempotency_manager(client, "operation-1"):
            record = json.loads(await client.get(build_idempotency_key("operation-1")))
            assert record["status"] == "processing"
        record = json.loads(await client.get(build_idempotency_key("operation-1")))
        assert record["status"] == "completed"
    finally:
        await client.aclose()
