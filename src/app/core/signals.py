"""Signal handlers for graceful shutdown."""

import signal
import sys
from types import FrameType

from app.utils.logger import logger


def setup_signal_handlers() -> None:
    """Setup graceful shutdown handlers."""

    def graceful_shutdown(sig_name: str) -> None:
        """Handle graceful shutdown."""
        logger.info(f"Received {sig_name}, shutting down gracefully...")
        sys.exit(0)

    def handle_sigterm(signum: int, frame: FrameType | None) -> None:
        graceful_shutdown("SIGTERM")

    def handle_sigint(signum: int, frame: FrameType | None) -> None:
        graceful_shutdown("SIGINT")

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigint)
