from app.utils import logger


def _setup_auto_instrumentation() -> None:
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor  # noqa: PLC0415

        SQLAlchemyInstrumentor().instrument()
    except Exception:  # noqa: BLE001
        logger.warning("SQLAlchemy auto-instrumentation failed — continuing")

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor  # noqa: PLC0415

        RedisInstrumentor().instrument()
    except Exception:  # noqa: BLE001
        logger.warning("Redis auto-instrumentation failed — continuing")

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor  # noqa: PLC0415

        HTTPXClientInstrumentor().instrument()
    except Exception:  # noqa: BLE001
        logger.warning("HTTPX auto-instrumentation failed — continuing")

    try:
        from opentelemetry.instrumentation.celery import CeleryInstrumentor  # noqa: PLC0415

        CeleryInstrumentor().instrument()
    except Exception:  # noqa: BLE001
        logger.warning("Celery auto-instrumentation failed — continuing")

    # ASGI instrumentor NOT called here — deferred to per-app OpenTelemetryMiddleware
