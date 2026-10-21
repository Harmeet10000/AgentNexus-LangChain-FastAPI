"""Canonical task-base interface; implementation remains a compatibility seam."""

from app.worker.task import ResilientTask, TypedCeleryTask

__all__ = ["ResilientTask", "TypedCeleryTask"]
