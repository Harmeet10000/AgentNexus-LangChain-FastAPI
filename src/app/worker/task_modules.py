"""Compatibility view of derived worker registration metadata.

The canonical map is built from the producer-safe catalog in
``app.worker.celery.registration``.  This module preserves historical imports
used by deployment tooling while keeping one declaration source.
"""

from app.worker.celery.registration import TASK_DECLARING_MODULES, TASK_MODULES

__all__ = ["TASK_DECLARING_MODULES", "TASK_MODULES"]
