"""Host and process health probes."""

from __future__ import annotations

import os
import platform
import time
from typing import TYPE_CHECKING, cast

import psutil

from ....application.dto.result import ComponentCheck

if TYPE_CHECKING:
    from collections.abc import Callable


class MemoryProbe:
    """Sample host and process memory."""

    name = "memory"

    @staticmethod
    async def check() -> ComponentCheck:
        memory = psutil.virtual_memory()
        process_memory = psutil.Process().memory_info()
        details = {
            "system": {
                "total_mb": round(memory.total / 1024 / 1024, 2),
                "available_mb": round(memory.available / 1024 / 1024, 2),
                "used_mb": round(memory.used / 1024 / 1024, 2),
                "percent": round(memory.percent, 1),
            },
            "process": {
                "rss_mb": round(process_memory.rss / 1024 / 1024, 2),
                "vms_mb": round(process_memory.vms / 1024 / 1024, 2),
            },
        }
        if memory.percent >= 90:
            return ComponentCheck.warning(state="high_usage", **details)
        return ComponentCheck.healthy(state="sampled", **details)


class DiskProbe:
    """Sample disk capacity for the process working directory."""

    name = "disk"

    @staticmethod
    async def check() -> ComponentCheck:
        try:
            disk = psutil.disk_usage(".")
            details = {
                "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
                "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
                "percent": round(disk.percent, 1),
            }
        except (FileNotFoundError, PermissionError, OSError) as exc:
            exc.add_note("component=disk, operation=check_disk")
            return ComponentCheck.unhealthy(str(exc), state="inaccessible")
        if disk.percent >= 90:
            return ComponentCheck.warning(state="high_usage", **details)
        return ComponentCheck.healthy(state="accessible", **details)


class SystemProbe:
    """Return the legacy application/system metadata fields."""

    name = "runtime"

    def __init__(self, environment: str, start_time: float) -> None:
        self._environment = environment
        self._start_time = start_time

    async def check(self) -> ComponentCheck:
        process = psutil.Process()
        memory_info = process.memory_info()
        application = {
            "environment": self._environment,
            "uptime": f"{time.time() - self._start_time:.2f} seconds",
            "memoryUsage": {
                "rss": f"{memory_info.rss / 1024 / 1024:.2f} MB",
                "vms": f"{memory_info.vms / 1024 / 1024:.2f} MB",
            },
            "pid": os.getpid(),
        }
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        try:
            get_load_average = cast(
                "Callable[[], tuple[float, float, float]]", psutil.__dict__["getloadavg"]
            )
            load_average = list(get_load_average())
        except (AttributeError, KeyError, OSError):
            load_average = [0.0, 0.0, 0.0]
        system = {
            "cpuUsage": load_average,
            "cpuUsagePercent": f"{cpu_percent:.2f}%",
            "totalMemory": f"{memory.total / 1024 / 1024:.2f} MB",
            "freeMemory": f"{memory.available / 1024 / 1024:.2f} MB",
            "platform": platform.system(),
            "arch": platform.machine(),
        }
        return ComponentCheck.healthy(state="sampled", application=application, system=system)
