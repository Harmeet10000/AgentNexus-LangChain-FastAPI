"""One derived registration map for worker task modules."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from app.shared.jobs.catalog import JOB_CATALOG

if TYPE_CHECKING:
    from typing import Final


def _jobs_module(module: str) -> str:
    return module


TASK_DECLARING_MODULES: Final = MappingProxyType(
    {name: _jobs_module(definition.declaring_module) for name, definition in JOB_CATALOG.items()}
)
TASK_MODULES: Final = tuple(dict.fromkeys(TASK_DECLARING_MODULES.values()))

__all__ = [
    "TASK_DECLARING_MODULES",
    "TASK_MODULES",
]
