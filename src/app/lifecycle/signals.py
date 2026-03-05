"""Signal handlers for graceful shutdown."""

import signal
import sys

from app.utils import logger


def setup_signal_handlers() -> None:
    """Setup graceful shutdown handlers for SIGTERM and SIGINT."""

    def graceful_shutdown(signum: int, frame) -> None:
        """Handle graceful shutdown."""
        sig_name = signal.Signals(value=signum).name
        logger.warning(f"Received {sig_name}, shutting down gracefully")
        sys.exit(0)

    signal.signal(signalnum=signal.SIGTERM, handler=graceful_shutdown)
    signal.signal(signalnum=signal.SIGINT, handler=graceful_shutdown)
