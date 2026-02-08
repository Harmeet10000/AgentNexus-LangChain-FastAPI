"""Signal handlers for graceful shutdown."""

import signal
import sys

from app.utils.logger import logger


def setup_signal_handlers() -> None:
    """Setup graceful shutdown handlers for SIGTERM and SIGINT."""

    def graceful_shutdown(signum: int, frame) -> None:
        """Handle graceful shutdown."""
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}, shutting down gracefully")
        sys.exit(0)

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

