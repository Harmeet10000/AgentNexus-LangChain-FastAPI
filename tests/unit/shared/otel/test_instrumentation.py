from __future__ import annotations


def test_instrumentation_failures_are_exposed_with_exception_details(monkeypatch) -> None:
    from app.shared.otel import instrument

    instrument.reset_instrumentation_diagnostics()

    class BrokenInstrumentor:
        def instrument(self) -> None:
            error = "collector unavailable"
            raise RuntimeError(error)

    monkeypatch.setattr(
        "opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor",
        BrokenInstrumentor,
    )
    instrument._instrument_path(
        "httpx", "opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"
    )

    diagnostics = instrument.get_instrumentation_diagnostics()
    failure = next(item for item in diagnostics if item["component"] == "httpx")
    assert failure["status"] == "failed"
    assert failure["error"] == "collector unavailable"


def test_otel_diagnostics_are_available_through_public_boundary() -> None:
    from app.shared.otel import get_otel_diagnostics

    assert isinstance(get_otel_diagnostics(), tuple)
