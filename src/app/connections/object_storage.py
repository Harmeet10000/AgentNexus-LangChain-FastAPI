"""Object-storage connection establishment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure

from app.shared.services.storage import StorageService
from app.utils import logger

if TYPE_CHECKING:
    from app.config.settings import Settings
    from app.shared.services.errors import StorageUnavailableError, StorageValidationError


async def create_object_store(settings: Settings) -> StorageService | None:
    """Build the object store from settings and verify bucket access.

    Returns None when unconfigured or when access verification fails, so
    callers degrade without object storage instead of failing boot.
    """
    if not settings.S3_BUCKET_NAME:
        logger.info("Object storage not configured, skipping")
        return None
    object_store = StorageService.from_settings(settings)
    result = await object_store.verify_access()
    if isinstance(result, Failure):
        error: StorageValidationError | StorageUnavailableError = result.failure()
        logger.warning(
            "Object storage access verification failed",
            error=error.message,
            details=error.details,
        )
        return None
    logger.bind(bucket=settings.S3_BUCKET_NAME).info("Object storage initialized")
    return object_store
