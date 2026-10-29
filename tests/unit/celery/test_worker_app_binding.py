"""CatalogValidatingWorker must bind the registry's active app.

Celery's click path constructs ``app.Worker(...)`` without an ``app=`` kwarg;
the Worker class default is None and only the process-current app could fill
it — which this app never is under ``worker_main``.  Without the binding the
worker crashes in ``Worker.__init__`` on ``self.app.loader`` before any queue
is consumed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.worker.celery import app as worker_app_module
from app.worker.celery.app import CatalogValidatingWorker

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


class _FakeApp:
    def __init__(self) -> None:
        self.conf: dict[str, Any] = {}


def _capture_init(captured: dict[str, Any]) -> Callable[..., None]:
    """Replace Worker.__init__ with a stub that records kwargs and binds app.

    The stub emulates celery's real fallback: with no explicit ``app=``, the
    Worker binds the process-current app (a fresh fake here).
    """

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        self.app = kwargs.get("app") or _FakeApp()
        captured.update(kwargs)

    return init


def _noop(*args: Any, **kwargs: Any) -> None:
    """Absorb the completeness validators under test-injection."""


def _stub_completeness(monkeypatch: Any) -> None:
    """The unit under test is the app binding, not catalog validation."""
    monkeypatch.setattr(
        worker_app_module, "ensure_worker_catalog_completeness", _noop
    )
    monkeypatch.setattr(
        worker_app_module, "ensure_worker_profile_completeness", _noop
    )


def test_worker_binds_active_app_when_celery_passes_none(
    monkeypatch: Any,
) -> None:
    """No ``app=`` kwarg ⇒ the registry's active app is injected."""
    fake_app = _FakeApp()
    monkeypatch.setattr(worker_app_module._apps, "active", fake_app)
    _stub_completeness(monkeypatch)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        worker_app_module.Worker,
        "__init__",
        _capture_init(captured),
    )

    CatalogValidatingWorker(pool_cls="solo", concurrency=1)

    assert captured.get("app") is fake_app


def test_worker_preserves_explicit_app_binding(monkeypatch: Any) -> None:
    """An explicit ``app=`` kwarg passes through untouched."""
    explicit = _FakeApp()
    monkeypatch.setattr(worker_app_module._apps, "active", _FakeApp())
    _stub_completeness(monkeypatch)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        worker_app_module.Worker,
        "__init__",
        _capture_init(captured),
    )

    CatalogValidatingWorker(app=explicit)

    assert captured.get("app") is explicit


def test_worker_leaves_binding_to_celery_when_registry_empty(
    monkeypatch: Any,
) -> None:
    """No active app in the registry ⇒ no injection (celery default behavior)."""
    monkeypatch.setattr(worker_app_module._apps, "active", None)
    _stub_completeness(monkeypatch)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        worker_app_module.Worker,
        "__init__",
        _capture_init(captured),
    )

    CatalogValidatingWorker()

    assert "app" not in captured
