"""Shutdown ordering for durable relay delivery."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from fastapi import FastAPI

lifespan_module = importlib.import_module("app.lifecycle.lifespan")

if TYPE_CHECKING:
    from typing import Any

pytestmark = pytest.mark.unit


async def test_outbox_drains_before_listener_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Relay:
        async def drain(self) -> bool:
            events.append("drain")
            assert not listener.cancelled()
            return True

    async def listener_body() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            events.append("cancel")

    async def no_op_async(*_args: object, **_kwargs: object) -> None:
        return None

    listener = asyncio.create_task(listener_body())
    await asyncio.sleep(0)
    app = FastAPI()
    app.state.outbox_relay = Relay()
    app.state.outbox_relay_task = listener
    app.state.websocket_security = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(lifespan_module, "shutdown_otel", lambda: None)
    monkeypatch.setattr(lifespan_module, "teardown_langgraph_checkpointer", no_op_async)

    await lifespan_module._shutdown_resources(cast("Any", app))

    assert events == ["drain", "cancel"]
