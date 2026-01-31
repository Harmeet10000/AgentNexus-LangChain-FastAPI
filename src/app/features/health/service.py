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
        "cpuUsage": psutil.getloadavg(),
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
        "uptime": f"{uptime:.2f} Seconds",
        "memoryUsage": {
            "heapTotal": f"{memory_info.rss / 1024 / 1024:.2f} MB",
            "heapUsed": f"{memory_info.rss / 1024 / 1024:.2f} MB",
        },
        "pid": os.getpid(),
        "version": f"Python {os.sys.version.split()[0]}",
    }


async def check_database(db_client: AsyncIOMotorClient) -> dict[str, Any]:
    """Check MongoDB database health."""
    try:
        start = time.time()
        await db_client.admin.command("ping")
        response_time = time.time() - start

        return {
            "status": "healthy",
            "state": "connected",
            "responseTime": f"{response_time * 1000:.2f}ms",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def check_redis(redis_client: Redis) -> dict[str, Any]:
    """Check Redis health."""
    try:
        start = time.time()
        await redis_client.ping()
        response_time = (time.time() - start) * 1000

        return {
            "status": "healthy",
            "responseTime": f"{response_time:.2f}ms",
            "connection": "connected",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connection": "disconnected",
        }


def check_memory() -> dict[str, Any]:
    """Check memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    total_mb = memory_info.rss / 1024 / 1024
    used_mb = memory_info.rss / 1024 / 1024
    usage_percent = psutil.virtual_memory().percent

    return {
        "status": "healthy" if usage_percent < 90 else "warning",
        "totalMB": round(total_mb),
        "usedMB": round(used_mb),
        "usagePercent": round(usage_percent),
    }


def check_disk() -> dict[str, Any]:
    """Check disk health."""
    try:
        disk = psutil.disk_usage(".")

        return {
            "status": "healthy",
            "accessible": True,
            "total": f"{disk.total / 1024 / 1024 / 1024:.2f} GB",
            "used": f"{disk.used / 1024 / 1024 / 1024:.2f} GB",
            "free": f"{disk.free / 1024 / 1024 / 1024:.2f} GB",
            "percent": f"{disk.percent}%",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
