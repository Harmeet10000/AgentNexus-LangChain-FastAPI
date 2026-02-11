import uvicorn

from app.config.settings import get_settings
from app.lifecycle.signals import setup_signal_handlers
from app.main import app
from app.utils.logger import logger


def main():
    setup_signal_handlers()
    settings = get_settings()

    logger.info(f"Starting server in {settings.ENVIRONMENT} mode...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        reload=settings.ENVIRONMENT != "production",
        log_config=None,  # Use custom logging
        access_log=False,  # Custom access logging via middleware
        # workers=4 if settings.ENVIRONMENT == "production" else 1,
    )


if __name__ == "__main__":
    main()
