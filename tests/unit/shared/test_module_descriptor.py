"""Contract tests for the per-slice Module descriptor."""

from __future__ import annotations

from app.shared.module import Module


def test_module_defaults_are_empty_and_immutable() -> None:
    module = Module(name="audit")

    assert module.routes == ()
    assert module.consumers == ()
    assert module.schedules == ()


def test_module_holds_zero_argument_factories() -> None:
    calls: list[str] = []

    def routes() -> tuple[str, ...]:
        calls.append("routes")
        return ("r",)

    module = Module(name="audit", routes=(routes,))

    assert calls == []
    assert module.routes[0]() == ("r",)
    assert calls == ["routes"]


def test_module_is_hashable_and_equatable() -> None:
    module = Module(name="audit")

    assert module == Module(name="audit")
    assert module != Module(name="health")
