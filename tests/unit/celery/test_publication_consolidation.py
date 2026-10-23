"""Deletion-test for the single-publication-entrypoint consolidation.

Job publication routes through :class:`app.shared.jobs.JobBus` — the direct
``typed_send``/``dispatch`` entrypath was deliberately removed. This module
sweeps ``src/app`` and fails if any direct-send seam creeps back. It greps raw
file contents (deterministic, milliseconds) instead of relaunching the suite.

What each assertion locks out:

* ``typed_send`` — the deleted legacy direct sender, anywhere in ``app``.
* ``send_task`` — the raw broker call; only the transport adapter
  (``connections/celery_producer.py``), which backs ``JobBus.enqueue``, may
  make one.
* ``celery_producer`` imports inside ``features`` — publication call sites in
  the domain layer must reach the shared bus, not the transport.
* ``publish_sync``/``publish_sync_result`` — helpers that returned Celery's
  native result for the deleted facade; their reappearance reintroduces a
  second publication algorithm.
* A runtime check that the registry is validate-only: it exposes no publish
  methods, while ``JobBus.enqueue`` exists as the single entrypoint.
"""

from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src" / "app"
_ALLOWED_SEND_TASK = "src/app/connections/celery_producer.py"
_HARD_BYPASS_TOKENS = ("typed_send", "publish_sync", "publish_sync_result")


def _candidate_files() -> list[Path]:
    if not _SRC.is_dir():
        message = f"Expected source tree to exist at {_SRC}"
        raise AssertionError(message)
    return list(_SRC.rglob("*.py"))


def test_deleted_direct_send_symbols_stay_deleted() -> None:
    offenders = [
        f"{path.relative_to(_SRC.parents[1])}: deleted direct-send symbol {token!r}"
        for path in _candidate_files()
        if (
            token := next(
                (
                    t
                    for t in _HARD_BYPASS_TOKENS
                    if t in path.read_text(encoding="utf-8", errors="replace")
                ),
                None,
            )
        )
    ]

    assert offenders == [], "\n".join(offenders)


def test_broker_send_task_lives_only_in_the_job_bus_transport() -> None:
    offenders = [
        str(path.relative_to(_SRC.parents[1]))
        for path in _candidate_files()
        if "send_task" in path.read_text(encoding="utf-8", errors="replace")
        and str(path.relative_to(_SRC.parents[1])) != _ALLOWED_SEND_TASK
    ]

    assert offenders == [], "\n".join(offenders)


def test_feature_layer_never_imports_the_celery_transport() -> None:
    offenders = [
        str(path.relative_to(_SRC.parents[1]))
        for path in (_SRC / "features").rglob("*.py")
        if "celery_producer" in path.read_text(encoding="utf-8", errors="replace")
    ]

    assert offenders == [], "\n".join(offenders)


def test_registry_is_validate_only_and_the_bus_is_the_entrypoint() -> None:
    from app.shared.jobs import JobBus
    from app.shared.jobs.registry import CeleryTaskRegistry

    # The registry publishes nothing — no method may bypass the bus.
    leaked_publish_methods = [
        name for name in dir(CeleryTaskRegistry) if name.startswith(("typed_", "publish", "send"))
    ]
    assert leaked_publish_methods == []

    # The single publication entrypoint is the bus, callable from producers.
    assert hasattr(JobBus, "enqueue")
