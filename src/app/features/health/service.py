"""Health check service functions."""

import os
import time
from typing import Any

import psutil
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis

# Store startup time
_start_time = time.time()


def get_system_health() -> dict[str, Any]:
    """Get system health metrics."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    return {
        "cpuUsage": list(psutil.getloadavg()),
        "cpuUsagePercent": f"{cpu_percent:.2f}%",
        "totalMemory": f"{memory.total / 1024 / 1024:.2f} MB",
        "freeMemory": f"{memory.available / 1024 / 1024:.2f} MB",
        "platform": os.uname().sysname,
        "arch": os.uname().machine,
    }


def get_application_health() -> dict[str, Any]:
    """Get application health metrics."""
    process = psutil.Process()
    memory_info = process.memory_info()
    uptime = time.time() - _start_time

    return {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "uptime": f"{uptime:.2f} seconds",
        "memoryUsage": {
            "rss": f"{memory_info.rss / 1024 / 1024:.2f} MB",
            "vms": f"{memory_info.vms / 1024 / 1024:.2f} MB",
        },
        "pid": os.getpid(),
        # "pythonVersion": f"{os.sys.version.split()[0]}",
    }


async def check_database(mongo_client: AsyncIOMotorClient) -> dict[str, Any]:
    """Check MongoDB database health."""
    try:
        start = time.perf_counter()
        await mongo_client.admin.command("ping")
        response_time = (time.perf_counter() - start) * 1000

        # Get server info
        server_info = await mongo_client.server_info()

        return {
            "status": "healthy",
            "state": "connected",
            "responseTime": f"{response_time:.2f}ms",
            "version": server_info.get("version", "unknown"),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "state": "disconnected",
            "error": str(e),
        }


async def check_redis(redis_client: Redis) -> dict[str, Any]:
    """Check Redis health."""
    try:
        start = time.perf_counter()
        await redis_client.ping()
        response_time = (time.perf_counter() - start) * 1000

        # Get Redis info
        info = await redis_client.info()

        return {
            "status": "healthy",
            "state": "connected",
            "responseTime": f"{response_time:.2f}ms",
            "version": info.get("redis_version", "unknown"),
            "connectedClients": info.get("connected_clients", 0),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "state": "disconnected",
            "error": str(e),
        }


def check_memory() -> dict[str, Any]:
    """Check memory usage."""
    memory = psutil.virtual_memory()
    process = psutil.Process()
    process_memory = process.memory_info()

    return {
        "status": "healthy" if memory.percent < 90 else "warning",
        "system": {
            "total": f"{memory.total / 1024 / 1024:.2f} MB",
            "available": f"{memory.available / 1024 / 1024:.2f} MB",
            "used": f"{memory.used / 1024 / 1024:.2f} MB",
            "percent": f"{memory.percent:.1f}%",
        },
        "process": {
            "rss": f"{process_memory.rss / 1024 / 1024:.2f} MB",
            "vms": f"{process_memory.vms / 1024 / 1024:.2f} MB",
        },
    }


def check_disk() -> dict[str, Any]:
    """Check disk health."""
    try:
        disk = psutil.disk_usage(".")

        return {
            "status": "healthy" if disk.percent < 90 else "warning",
            "accessible": True,
            "total": f"{disk.total / 1024 / 1024 / 1024:.2f} GB",
            "used": f"{disk.used / 1024 / 1024 / 1024:.2f} GB",
            "free": f"{disk.free / 1024 / 1024 / 1024:.2f} GB",
            "percent": f"{disk.percent:.1f}%",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "accessible": False,
            "error": str(e),
        }
