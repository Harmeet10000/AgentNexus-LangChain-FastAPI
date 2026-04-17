from typing import TYPE_CHECKING

import uvicorn

from app.config import get_settings
from app.utils import logger

if TYPE_CHECKING:
    from app.config import Settings


def main() -> None:
    settings: Settings = get_settings()

    logger.bind(environment=settings.ENVIRONMENT).info("Starting server")

    uvicorn.run(
        app="app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT != "production",
        reload_dirs=["src"] if settings.ENVIRONMENT != "production" else None,
        reload_excludes=[".venv/*", "*.pyc", "__pycache__/*"],
        log_config=None,  # Use custom logging
        access_log=False,  # Custom access logging via middleware
        workers=4 if settings.ENVIRONMENT == "production" else 1,
        ws_max_size=settings.UVICORN_WS_MAX_SIZE,
        ws_max_queue=settings.UVICORN_WS_MAX_QUEUE,
        ws_ping_interval=settings.UVICORN_WS_PING_INTERVAL,
        ws_ping_timeout=settings.UVICORN_WS_PING_TIMEOUT,
    )


if __name__ == "__main__":
    main()
