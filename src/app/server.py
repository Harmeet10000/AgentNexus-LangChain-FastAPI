import uvicorn

from app.config.settings import get_settings
from app.lifecycle.signals import setup_signal_handlers
from app.main import app
from app.utils.logger import logger


def main() -> None:
    setup_signal_handlers()
    settings = get_settings()

    logger.info(f"Starting server in {settings.ENVIRONMENT} mode...")

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
    )


if __name__ == "__main__":
    main()
