"""F8: the crawl task must never issue a sync owner operation on the running loop.

The 2026-11-15 smoke recorded every crawl job dying deterministically in ~226 ms
with "Capability owner synchronous operations must run at a task boundary" and
zero pages attempted: the job coroutine — which runs ON the owner's loop via
``run_on_crawler_worker_loop`` — called the synchronous
``get_crawler_worker_resources()`` accessor, and the owner correctly refused.
The fix provisions at the task boundary and hands the browser into the
coroutine; these tests pin both the wiring and the boundary invariant it relies
on.
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from app.config import Settings
from app.utils import ServiceUnavailableException
from app.worker.capabilities.owner import CapabilityUsageError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

pytestmark = pytest.mark.unit


def _run_at_task_boundary(
    factory: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    """Run the coroutine factory exactly like the Celery task boundary does."""

    async def _invoke() -> None:
        await factory()

    asyncio.run(_invoke())


def _current_crawler_jobs():
    """Return the module instance left by worker-isolation reload tests."""
    return importlib.import_module("app.features.crawler.adapters.inbound.worker.tasks")


def test_crawl_job_hands_the_boundary_provisioned_browser_into_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The browser is captured before the loop runs and passed into the coroutine.

    The smoke failure was the coroutine itself re-resolving resources through the
    synchronous accessor. After the fix the task body resolves once at the
    boundary; the coroutine receives the browser and never touches the owner's
    sync interface.
    """
    jobs = _current_crawler_jobs()
    resources = SimpleNamespace(browser=object())
    monkeypatch.setattr(jobs, "get_crawler_worker_resources", lambda: resources)
    monkeypatch.setattr(jobs, "run_on_crawler_worker_loop", _run_at_task_boundary)

    seen: dict[str, Any] = {}

    async def _stub(
        crawl_id: str,
        request_payload: dict[str, object],
        *,
        final_attempt: bool,
        browser: Any,
    ) -> None:
        seen["crawl_id"] = crawl_id
        seen["final_attempt"] = final_attempt
        seen["browser"] = browser

    monkeypatch.setattr(jobs, "_run_crawl_job", _stub)

    # `crawl_job` is a bound Celery task (PromiseProxy) with autoretry wrapping;
    # `_orig_run` is the underlying bound function. Outside a task context
    # `self.request.retries` defaults to 0, so `final_attempt` is False.
    result = jobs.crawl_job._orig_run(
        crawl_id="crawl-f8",
        request_payload={"url": "https://example.com"},
    )

    assert result == {"crawl_id": "crawl-f8", "status": "finished"}
    assert seen["crawl_id"] == "crawl-f8"
    assert seen["final_attempt"] is False
    assert seen["browser"] is resources.browser


def test_a_running_owner_loop_refuses_the_sync_resource_accessor() -> None:
    """The boundary invariant the wiring depends on, pinned on the real owner.

    Provisioning inside the coroutine (the defect shape) must fail immediately
    with the capability-usage error — which is why the browser has to travel in
    from the task boundary instead.
    """
    from app.features.crawler.adapters.inbound.worker import capabilities as capability

    async def _miswired() -> None:
        with pytest.raises(CapabilityUsageError, match="task boundary"):
            capability.get_crawler_worker_resources()

    asyncio.run(_miswired())


def test_terminal_preflight_failure_marks_the_job_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provisioning failure on the last retry must not strand PENDING state."""
    jobs = _current_crawler_jobs()
    settings = Settings(CELERY_RETRY_MAX_RETRIES=0)
    recorded: list[tuple[str, str]] = []

    def unavailable() -> None:
        raise ServiceUnavailableException(detail="crawler egress unavailable")

    monkeypatch.setattr(jobs, "get_settings", lambda: settings)
    monkeypatch.setattr(jobs, "get_crawler_worker_resources", unavailable)
    monkeypatch.setattr(
        jobs,
        "_mark_failed_sync",
        lambda crawl_id, message: recorded.append((crawl_id, message)),
    )

    with pytest.raises(ServiceUnavailableException):
        jobs.crawl_job._orig_run(
            crawl_id="crawl-preflight-failure",
            request_payload={"url": "https://example.com"},
        )

    assert len(recorded) == 1
    assert recorded[0][0] == "crawl-preflight-failure"
    assert "crawler egress unavailable" in recorded[0][1]
