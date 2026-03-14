"""Signal handling guidance for the application process lifecycle."""

from app.utils import logger


def setup_signal_handlers() -> None:
    """Defer SIGINT and SIGTERM handling to the ASGI server."""
    logger.bind(layer="lifecycle", signal_owner="uvicorn").debug(
        "Custom signal handlers are disabled; the ASGI server manages graceful shutdown"
    )
