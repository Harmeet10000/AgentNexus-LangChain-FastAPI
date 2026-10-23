"""Session discovery bootstrap and pre-stream ownership ordering."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from returns.result import Failure, Success

import app.features.agent_saul.service as service_module
from app.features.agent_saul.service import AgentSaulService
from app.features.agent_saul.transcript.errors import TranscriptInfrastructureError
from app.features.agent_saul.transcript.recorder import TurnRecorder

if TYPE_CHECKING:
    from typing import Any

pytestmark = pytest.mark.unit


class _Log:
    def bind(self, **_kwargs: Any) -> _Log:
        return self

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _BootstrapRepo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def ensure_session(self, *, session_id: str, user_id: str) -> Success[bool]:
        self.calls.append((session_id, user_id))
        return Success(True)


class _GraphMustNotBeRead:
    def __getattr__(self, name: str) -> Any:
        msg = f"bootstrap unexpectedly accessed graph.{name}"
        raise AssertionError(msg)


async def test_recorder_bootstrap_uses_only_repository_identity_seam() -> None:
    repo = _BootstrapRepo()
    recorder = TurnRecorder(
        repo=repo,  # type: ignore[arg-type] — minimal structural fake
        graph=_GraphMustNotBeRead(),  # type: ignore[arg-type]
        config={},
        thread_id="thread-1",
        user_id="user-1",
        redis=object(),  # type: ignore[arg-type]
        lock=object(),  # type: ignore[arg-type]
    )

    result = await recorder.bootstrap_session()

    assert isinstance(result, Success)
    assert result.unwrap() is True
    assert repo.calls == [("thread-1", "user-1")]


class _Lock:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def acquire(self) -> bool:
        self._events.append("lock_acquired")
        return True

    async def release(self) -> None:
        self._events.append("lock_released")


class _Recorder:
    def __init__(self, events: list[str], bootstrap_result: Any) -> None:
        self._events = events
        self._bootstrap_result = bootstrap_result
        self.usage_handler = object()

    async def bootstrap_session(self) -> Any:
        self._events.append("session_ensured")
        return self._bootstrap_result

    async def on_abort(self) -> None:
        self._events.append("abort")


class _HarnessService(AgentSaulService):
    def __init__(self, events: list[str], *, gate_result: bool = True) -> None:
        self.events = events
        self.gate_result = gate_result
        self.frames: list[dict[str, Any]] = []
        self._redis = object()  # type: ignore[assignment]
        self._session_factory = object()  # type: ignore[assignment]
        self._graph = object()  # type: ignore[assignment]
        self._log = _Log()  # type: ignore[assignment]

    @staticmethod
    def _build_config(_thread_id: str) -> dict[str, Any]:
        return {}

    async def _gate_and_reconcile(self, **_kwargs: Any) -> bool:
        self.events.extend(("owner_checked", "reconciled"))
        return self.gate_result

    async def _session_first_input(self, **_kwargs: Any) -> tuple[dict[str, Any], bool]:
        self.events.append("input_prepared")
        return {}, False

    async def _stream_round_guarded(self, **_kwargs: Any) -> bool:
        self.events.append("astream_entered")
        return True

    async def _send_json(self, _ws: Any, payload: dict[str, Any]) -> None:
        self.frames.append(payload)


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    lock: _Lock,
    recorder: _Recorder,
) -> None:
    monkeypatch.setattr(service_module, "ThreadLock", lambda *_args, **_kwargs: lock)
    monkeypatch.setattr(service_module, "ChatTranscriptRepository", lambda *_args: object())
    monkeypatch.setattr(service_module, "TurnRecorder", lambda **_kwargs: recorder)
    monkeypatch.setattr(
        service_module,
        "get_settings",
        lambda: SimpleNamespace(SAUL_PROJECTION_TIMEOUT_MS=750),
    )


async def test_run_bootstraps_before_input_side_effects_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    lock = _Lock(events)
    recorder = _Recorder(events, Success(True))
    _patch_runtime(monkeypatch, lock=lock, recorder=recorder)
    service = _HarnessService(events)

    await service.run_session(None, object(), "thread-1", "user-1")  # type: ignore[arg-type]

    assert events == [
        "lock_acquired",
        "owner_checked",
        "reconciled",
        "session_ensured",
        "input_prepared",
        "astream_entered",
        "abort",
        "lock_released",
    ]


async def test_bootstrap_failure_prevents_input_stream_and_abort_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    lock = _Lock(events)
    failure = Failure(
        TranscriptInfrastructureError(
            message="database unavailable",
            source="test",
            operation="ensure_session",
        )
    )
    recorder = _Recorder(events, failure)
    _patch_runtime(monkeypatch, lock=lock, recorder=recorder)
    service = _HarnessService(events)

    await service.run_session(None, object(), "thread-1", "user-1")  # type: ignore[arg-type]

    assert events == [
        "lock_acquired",
        "owner_checked",
        "reconciled",
        "session_ensured",
        "lock_released",
    ]
    assert service.frames[-1]["code"] == "SESSION_BOOT_FAILED"


async def test_owner_gate_refusal_skips_bootstrap_stream_and_abort_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    lock = _Lock(events)
    recorder = _Recorder(events, Success(True))
    _patch_runtime(monkeypatch, lock=lock, recorder=recorder)
    service = _HarnessService(events, gate_result=False)

    await service.run_session(None, object(), "thread-1", "user-1")  # type: ignore[arg-type]

    assert events == [
        "lock_acquired",
        "owner_checked",
        "reconciled",
        "lock_released",
    ]


class _OwnerSnapshotGraph:
    async def aget_state(self, _config: object) -> SimpleNamespace:
        return SimpleNamespace(values={"user_id": "other-user"}, tasks=())


class _NeverReconcile:
    async def reconcile(self) -> None:
        pytest.fail("foreign checkpoint was reconciled before owner refusal")


async def test_foreign_checkpoint_owner_is_refused_before_reconcile() -> None:
    service = _HarnessService([])
    service._graph = _OwnerSnapshotGraph()  # type: ignore[assignment]

    allowed = await AgentSaulService._gate_and_reconcile(
        service,
        ws=None,
        recorder=_NeverReconcile(),  # type: ignore[arg-type]
        config={},
        thread_id="thread-1",
        user_id="user-1",
    )

    assert allowed is False
    assert service.frames[-1]["code"] == "THREAD_NOT_OWNED"
