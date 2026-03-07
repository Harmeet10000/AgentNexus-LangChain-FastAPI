"""Health service layer."""

import os
import time
from typing import Any

import psutil
from celery import Celery
from motor.motor_asyncio import AsyncIOMotorClient
from neo4j import AsyncDriver
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.utils import logger

from .dto import HealthChecksDTO, HealthDataDTO, HealthResultDTO, SelfInfoDTO


class HealthService:
    """Service for system and dependency health checks."""

    def __init__(
        self,
        mongo_client: AsyncIOMotorClient | None,
        redis_client: Redis | None,
        postgres_session_factory: async_sessionmaker[AsyncSession] | None,
        neo4j_driver: AsyncDriver | None,
        celery_app: Celery | None,
    ) -> None:
        self.mongo_client = mongo_client
        self.redis_client = redis_client
        self.postgres_session_factory = postgres_session_factory
        self.neo4j_driver = neo4j_driver
        self.celery_app = celery_app
        self.start_time = time.time()

    async def get_self_info(
        self,
        server_name: str,
        server_version: str,
        client_host: str,
    ) -> SelfInfoDTO:
        """Return basic service metadata."""
        return SelfInfoDTO(
            server=server_name,
            version=server_version,
            client=client_host,
            timestamp=time.time(),
        )

    async def get_health(self) -> HealthResultDTO:
        """Run all health checks and return aggregated status."""
        database_check = await self._check_mongodb() if self.mongo_client else self._not_configured()
        redis_check = await self._check_redis() if self.redis_client else self._not_configured()
        postgres_check = await self._check_postgres() if self.postgres_session_factory else self._not_configured()
        neo4j_check = await self._check_neo4j() if self.neo4j_driver else self._not_configured()
        celery_check = self._check_celery()
        memory_check = self._check_memory()
        disk_check = self._check_disk()

        checks = HealthChecksDTO(
            database=database_check,
            redis=redis_check,
            postgres=postgres_check,
            neo4j=neo4j_check,
            celery=celery_check,
            memory=memory_check,
            disk=disk_check,
        )

        overall_status = self._compute_overall_status(checks=checks)
        status_code = 200 if overall_status == "healthy" else 503

        data = HealthDataDTO(
            status=overall_status,
            timestamp=time.time(),
            application=self._get_application_health(),
            system=self._get_system_health(),
            checks=checks,
        )

        logger.bind(status=overall_status, status_code=status_code).info("Health check evaluated")
        return HealthResultDTO(
            message=f"Health check: {overall_status}",
            status_code=status_code,
            data=data,
        )

    async def _check_mongodb(self) -> dict[str, Any]:
        client = self.mongo_client
        if client is None:
            return self._not_configured()
        try:
            start = time.perf_counter()
            await client.admin.command("ping")
            response_time = (time.perf_counter() - start) * 1000
            server_info = await client.server_info()
            return {
                "status": "healthy",
                "state": "connected",
                "responseTime": f"{response_time:.2f}ms",
                "version": server_info.get("version", "unknown"),
            }
        except Exception as exc:
            logger.bind(error=str(exc)).warning("MongoDB health check failed")
            return {"status": "unhealthy", "state": "disconnected", "error": str(exc)}

    async def _check_redis(self) -> dict[str, Any]:
        redis_client = self.redis_client
        if redis_client is None:
            return self._not_configured()
        try:
            start = time.perf_counter()
            await redis_client.ping()
            response_time = (time.perf_counter() - start) * 1000
            info = await redis_client.info()
            return {
                "status": "healthy",
                "state": "connected",
                "responseTime": f"{response_time:.2f}ms",
                "version": info.get("redis_version", "unknown"),
                "connectedClients": info.get("connected_clients", 0),
            }
        except Exception as exc:
            logger.bind(error=str(exc)).warning("Redis health check failed")
            return {"status": "unhealthy", "state": "disconnected", "error": str(exc)}

    async def _check_postgres(self) -> dict[str, Any]:
        session_factory = self.postgres_session_factory
        if session_factory is None:
            return self._not_configured()
        try:
            start = time.perf_counter()
            async with session_factory() as session:
                await session.execute(text("SELECT 1"))
                version_result = await session.execute(text("SELECT version()"))
                version = version_result.scalar() or "unknown"
            response_time = (time.perf_counter() - start) * 1000
            return {
                "status": "healthy",
                "state": "connected",
                "responseTime": f"{response_time:.2f}ms",
                "version": str(version),
            }
        except Exception as exc:
            logger.bind(error=str(exc)).warning("Postgres health check failed")
            return {"status": "unhealthy", "state": "disconnected", "error": str(exc)}

    async def _check_neo4j(self) -> dict[str, Any]:
        driver = self.neo4j_driver
        if driver is None:
            return self._not_configured()
        try:
            start = time.perf_counter()
            async with driver.session() as session:
                result = await session.run("RETURN 1 AS ok")
                record = await result.single()
            response_time = (time.perf_counter() - start) * 1000
            return {
                "status": "healthy",
                "state": "connected",
                "responseTime": f"{response_time:.2f}ms",
                "ok": bool(record and record.get("ok") == 1),
            }
        except Exception as exc:
            logger.bind(error=str(exc)).warning("Neo4j health check failed")
            return {"status": "unhealthy", "state": "disconnected", "error": str(exc)}

    def _check_celery(self) -> dict[str, Any]:
        if self.celery_app is None:
            return self._not_configured()
        try:
            start = time.perf_counter()
            conn = self.celery_app.connection()
            conn.ensure_connection(max_retries=1, timeout=2)
            conn.release()
            response_time = (time.perf_counter() - start) * 1000
        except Exception as exc:
            logger.bind(error=str(exc)).warning("Celery health check failed")
            return {"status": "unhealthy", "state": "disconnected", "error": str(exc)}
        else:
            return {
                "status": "healthy",
                "state": "connected",
                "responseTime": f"{response_time:.2f}ms",
            }

    @staticmethod
    def _check_memory() -> dict[str, Any]:
        memory = psutil.virtual_memory()
        process_memory = psutil.Process().memory_info()
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

    @staticmethod
    def _check_disk() -> dict[str, Any]:
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
        except Exception as exc:
            return {"status": "unhealthy", "accessible": False, "error": str(exc)}

    @staticmethod
    def _not_configured() -> dict[str, Any]:
        return {"status": "unknown", "state": "not_configured"}

    @staticmethod
    def _compute_overall_status(checks: HealthChecksDTO) -> str:
        all_checks = [
            checks.database,
            checks.redis,
            checks.postgres,
            checks.neo4j,
            checks.celery,
            checks.memory,
            checks.disk,
        ]
        if any(check.get("status") == "unhealthy" for check in all_checks):
            return "unhealthy"
        if any(check.get("status") == "warning" for check in all_checks):
            return "degraded"
        return "healthy"

    def _get_application_health(self) -> dict[str, Any]:
        process = psutil.Process()
        memory_info = process.memory_info()
        uptime = time.time() - self.start_time
        return {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "uptime": f"{uptime:.2f} seconds",
            "memoryUsage": {
                "rss": f"{memory_info.rss / 1024 / 1024:.2f} MB",
                "vms": f"{memory_info.vms / 1024 / 1024:.2f} MB",
            },
            "pid": os.getpid(),
        }

    @staticmethod
    def _get_system_health() -> dict[str, Any]:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        try:
            load_avg = list(os.getloadavg())
        except (AttributeError, OSError):
            load_avg = [0.0, 0.0, 0.0]
        return {
            "cpuUsage": load_avg,
            "cpuUsagePercent": f"{cpu_percent:.2f}%",
            "totalMemory": f"{memory.total / 1024 / 1024:.2f} MB",
            "freeMemory": f"{memory.available / 1024 / 1024:.2f} MB",
            "platform": os.uname().sysname,
            "arch": os.uname().machine,
        }
