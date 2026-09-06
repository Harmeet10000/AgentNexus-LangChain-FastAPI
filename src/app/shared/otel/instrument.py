from app.utils import logger


def _setup_auto_instrumentation() -> None:
    try:
        from opentelemetry.instrumentation.sqlalchemy import (
            SQLAlchemyInstrumentor,
        )

        SQLAlchemyInstrumentor().instrument()
    except Exception:  # noqa: BLE001 — optional instrumentation must not crash app
        logger.bind(component="sqlalchemy").warning(
            "SQLAlchemy auto-instrumentation failed, continuing"
        )

    try:
        from opentelemetry.instrumentation.redis import (
            RedisInstrumentor,
        )

        RedisInstrumentor().instrument()
    except Exception:  # noqa: BLE001 — optional instrumentation must not crash app
        logger.bind(component="redis").warning("Redis auto-instrumentation failed, continuing")

    try:
        from opentelemetry.instrumentation.httpx import (
            HTTPXClientInstrumentor,
        )

        HTTPXClientInstrumentor().instrument()
    except Exception:  # noqa: BLE001 — optional instrumentation must not crash app
        logger.bind(component="httpx").warning("HTTPX auto-instrumentation failed, continuing")

    try:
        from opentelemetry.instrumentation.celery import (
            CeleryInstrumentor,
        )

        CeleryInstrumentor().instrument()
    except Exception:  # noqa: BLE001 — optional instrumentation must not crash app
        logger.bind(component="celery").warning("Celery auto-instrumentation failed, continuing")

    # ASGI instrumentor NOT called here — deferred to per-app OpenTelemetryMiddleware
