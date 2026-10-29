"""Async, Result-native handle over one bucket's object store.

The handle owns configuration (bucket, optional public base URL) and the
thread-hop/Result translation; the raw S3 calls themselves live in
``app.utils.storage.s3_func`` and the client is formed in
``app.connections.object_storage``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, PrivateAttr
from returns.result import Failure

from app.observability.logging import logger

from . import s3_func
from .protocols import ObjectStoreClient

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from app.config.settings import Settings

    from .errors import StorageResult
    from .models import (
        MultipartUploadPlan,
        PresignedUploadURL,
        S3CompleteMultipartUploadResponse,
        S3ListObjectsResponse,
        S3ListPartsResponse,
        S3UploadPartResponse,
        UploadRequest,
    )


class StorageService(BaseModel):  # noqa: PLR0904 — one method per S3 operation is the point
    """One bucket, one shared client, every operation returning ``StorageResult``."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    bucket: str
    public_url: str | None = None
    presigned_get_ttl_seconds: int = 18_000
    _client: ObjectStoreClient = PrivateAttr()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: ObjectStoreClient | None = None,
    ) -> StorageService:
        """Bind settings to the process-wide client (created on first use)."""
        if client is None:
            # Imported here so ``app.utils.storage`` never imports the transport
            # adapter at module import time.
            from app.connections.object_storage import get_object_store  # noqa: PLC0415

            client = get_object_store(settings)
        service = cls(
            bucket=settings.S3_BUCKET_NAME,
            public_url=settings.S3_PUBLIC_URL.rstrip("/") or None,
            presigned_get_ttl_seconds=settings.S3_PRESIGNED_GET_TTL_SECONDS,
        )
        service._client = client
        return service

    @property
    def client(self) -> ObjectStoreClient:
        """The underlying client, for callers driving ``s3_func`` directly."""
        return self._client

    async def put_object(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str],
    ) -> StorageResult[str]:
        return await s3_func.put_object(
            self._client,
            key=key,
            data=data,
            content_type=content_type,
            metadata=metadata,
        )

    async def get_object(self, *, key: str) -> StorageResult[bytes]:
        return await s3_func.get_object(self._client, key=key)

    async def delete_object(self, *, key: str) -> StorageResult[None]:
        return await s3_func.delete_object(self._client, key=key)

    async def get_by_uri(self, *, uri: str) -> StorageResult[bytes]:
        return await s3_func.get_by_uri(self._client, uri=uri)

    async def delete_by_uri(self, *, uri: str) -> StorageResult[None]:
        return await s3_func.delete_by_uri(self._client, uri=uri)

    async def verify_access(self) -> StorageResult[None]:
        return await s3_func.verify_access(self._client)

    async def object_exists(self, *, key: str) -> StorageResult[bool]:
        return await s3_func.object_exists(self._client, key=key)

    async def list_objects(
        self,
        *,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> StorageResult[S3ListObjectsResponse]:
        return await s3_func.list_objects(self._client, prefix=prefix, max_keys=max_keys)

    async def copy_object(
        self,
        *,
        source_key: str,
        destination_key: str,
    ) -> StorageResult[None]:
        return await s3_func.copy_object(
            self._client,
            source_key=source_key,
            destination_key=destination_key,
        )

    async def create_multipart_upload(
        self,
        *,
        key: str,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> StorageResult[dict[str, Any]]:
        return await s3_func.create_multipart_upload(
            self._client,
            key=key,
            content_type=content_type,
            metadata=metadata,
        )

    async def upload_part(
        self,
        *,
        key: str,
        upload_id: str,
        part_number: int,
        body: bytes,
    ) -> StorageResult[S3UploadPartResponse]:
        return await s3_func.upload_part(
            self._client,
            key=key,
            upload_id=upload_id,
            part_number=part_number,
            body=body,
        )

    async def complete_multipart_upload(
        self,
        *,
        key: str,
        upload_id: str,
        parts: list[dict[str, Any]],
    ) -> StorageResult[S3CompleteMultipartUploadResponse]:
        return await s3_func.complete_multipart_upload(
            self._client,
            key=key,
            upload_id=upload_id,
            parts=parts,
        )

    async def abort_multipart_upload(
        self,
        *,
        key: str,
        upload_id: str,
    ) -> StorageResult[None]:
        return await s3_func.abort_multipart_upload(
            self._client,
            key=key,
            upload_id=upload_id,
        )

    async def list_multipart_upload_parts(
        self,
        *,
        key: str,
        upload_id: str,
    ) -> StorageResult[S3ListPartsResponse]:
        return await s3_func.list_multipart_upload_parts(
            self._client,
            key=key,
            upload_id=upload_id,
        )

    async def get_signed_get_url(
        self,
        *,
        key: str,
        expires_in: int | None = None,
    ) -> StorageResult[str]:
        return await s3_func.get_signed_get_url(
            self._client,
            key=key,
            expires_in=expires_in or self.presigned_get_ttl_seconds,
        )

    async def get_signed_put_url(
        self,
        *,
        key: str,
        content_type: str,
        expires_in: int = 900,
    ) -> StorageResult[str]:
        return await s3_func.get_signed_put_url(
            self._client,
            key=key,
            content_type=content_type,
            expires_in=expires_in,
        )

    async def generate_presigned_upload_urls(
        self,
        *,
        files: Sequence[UploadRequest] = (),
        destination: str = "uploads",
        expires_in: int = 900,
    ) -> StorageResult[list[PresignedUploadURL]]:
        return await s3_func.generate_presigned_upload_urls(
            self._client,
            files=files,
            destination=destination,
            expires_in=expires_in,
        )

    async def create_multipart_upload_plan(
        self,
        *,
        filename: str,
        content_type: str,
        file_size: int,
        destination: str = "uploads",
        part_size: int = 5 * 1024 * 1024,
        expires_in: int = 3600,
        metadata: dict[str, str] | None = None,
    ) -> StorageResult[MultipartUploadPlan]:
        return await s3_func.create_multipart_upload_plan(
            self._client,
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            destination=destination,
            part_size=part_size,
            expires_in=expires_in,
            metadata=metadata,
        )

    async def get_signed_multipart_part_url(
        self,
        *,
        key: str,
        upload_id: str,
        part_number: int,
        expires_in: int = 3600,
    ) -> StorageResult[str]:
        return await s3_func.get_signed_multipart_part_url(
            self._client,
            key=key,
            upload_id=upload_id,
            part_number=part_number,
            expires_in=expires_in,
        )


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
        error = result.failure()
        logger.warning(
            "Object storage access verification failed",
            error=error.message,
            details=error.details,
        )
        return None
    logger.bind(bucket=settings.S3_BUCKET_NAME).info("Object storage initialized")
    return object_store
