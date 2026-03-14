import uvicorn

from app.config.settings import get_settings
from app.utils.logger import logger


def main() -> None:
    settings = get_settings()

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
    )


if __name__ == "__main__":
    main()
