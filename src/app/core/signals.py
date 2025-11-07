"""Signal handlers for graceful shutdown."""

import signal
import sys
from types import FrameType

from app.utils.logger import logger


def setup_signal_handlers() -> None:
    """Setup graceful shutdown handlers."""
    logger.debug("Setting up signal handlers", signals=["SIGTERM", "SIGINT"])

    def graceful_shutdown(sig_name: str, signum: int) -> None:
        """Handle graceful shutdown."""
        logger.warning(f"Received {sig_name}, shutting down gracefully", signal=sig_name, signal_number=signum)
        sys.exit(0)

    def handle_sigterm(signum: int, frame: FrameType | None) -> None:
        graceful_shutdown("SIGTERM", signum)

    def handle_sigint(signum: int, frame: FrameType | None) -> None:
        graceful_shutdown("SIGINT", signum)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigint)
