"""Object-storage client formation.

Only transport concerns live here: the botocore configuration, the
process-wide boto3 ``S3Client`` singleton, and the thin synchronous wrapper
that the CRUD layer drives from worker threads. Object operations themselves
live in ``app.utils.storage``.
"""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from mypy_boto3_s3 import S3Client
from pydantic import BaseModel, ConfigDict, PrivateAttr

from app.config import get_settings
from app.observability.logging import logger

if TYPE_CHECKING:
    from typing import Any

    from app.config.settings import Settings


class S3ClientWrapper(BaseModel):
    """Thin synchronous wrapper around the process-wide boto3 S3 client.

    Owns the raw S3 API calls and returns boto3-native responses; every
    Result/logging/thread-hop concern belongs to ``app.utils.storage``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    bucket: str
    _client: S3Client = PrivateAttr()

    def put_object(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata=metadata,
        )

    def get_object(self, *, key: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        return bytes(response["Body"].read())

    def delete_object(self, *, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def head_object(self, *, key: str) -> dict[str, Any]:
        return self._client.head_object(Bucket=self.bucket, Key=key)  # ty: ignore[invalid-return-type]

    def head_bucket(self) -> None:
        self._client.head_bucket(Bucket=self.bucket)

    def list_objects(self, *, prefix: str, max_keys: int) -> dict[str, Any]:
        return self._client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
            MaxKeys=max_keys,
        )  # ty: ignore[invalid-return-type]

    def copy_object(self, *, source_key: str, destination_key: str) -> None:
        self._client.copy_object(
            Bucket=self.bucket,
            CopySource=f"{self.bucket}/{source_key}",
            Key=destination_key,
        )

    def create_multipart_upload(
        self,
        *,
        key: str,
        content_type: str,
        metadata: dict[str, str],
    ) -> dict[str, Any]:
        return self._client.create_multipart_upload(  # ty: ignore[invalid-return-type]
            Bucket=self.bucket,
            Key=key,
            ContentType=content_type,
            Metadata=metadata,
        )

    def upload_part(
        self,
        *,
        key: str,
        upload_id: str,
        part_number: int,
        body: bytes,
    ) -> dict[str, Any]:
        return self._client.upload_part(  # ty: ignore[invalid-return-type]
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
            PartNumber=part_number,
            Body=body,
        )

    def complete_multipart_upload(
        self,
        *,
        key: str,
        upload_id: str,
        parts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._client.complete_multipart_upload(  # ty: ignore[invalid-return-type]
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},  # ty: ignore[invalid-argument-type]
        )

    def abort_multipart_upload(self, *, key: str, upload_id: str) -> None:
        self._client.abort_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
        )

    def list_parts(self, *, key: str, upload_id: str) -> dict[str, Any]:
        return self._client.list_parts(  # ty: ignore[invalid-return-type]
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
        )

    def generate_presigned_url(
        self,
        operation: str,
        params: dict[str, Any],
        expires_in: int,
    ) -> str:
        return self._client.generate_presigned_url(operation, Params=params, ExpiresIn=expires_in)

    @classmethod
    def from_boto_client(cls, *, bucket: str, client: S3Client) -> S3ClientWrapper:
        wrapper = cls(bucket=bucket)
        wrapper._client = client
        return wrapper


# ── Process-wide client singleton ──────────────────────────────────────────
#
# One boto3 client serves every bucket this process touches: it carries the
# endpoint, credentials, timeouts, retries, and (critically) one urllib3
# connection pool. Rebuilding it per request threw that pool away each time.


class _ClientState:
    """Mutable process-wide holder; avoids ``global`` rewrites of module state."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.client: S3Client | None = None
        self.stores: dict[str, S3ClientWrapper] = {}


_state = _ClientState()


def _boto_config(settings: Settings) -> Config:
    """Timeout/retry/pool policy — the defaults are wrong for a hot path."""
    return Config(
        signature_version="s3v4",
        s3={"addressing_style": "path" if settings.S3_FORCE_PATH_STYLE else "virtual"},
        retries={"max_attempts": 4, "mode": "standard"},
        connect_timeout=5,
        read_timeout=30,
        max_pool_connections=32,
        # botocore >= 1.36 otherwise adds a CRC32 header that browsers cannot
        # reproduce, which breaks every presigned PUT it signs.
        request_checksum_calculation="when_required",
    )


def get_s3_client(settings: Settings | None = None) -> S3Client:
    """Return the process-wide S3 client, creating it on first use.

    The first caller wins: pass ``settings`` only to override the process-wide
    ``get_settings()`` result at boot. Later calls ignore their argument and
    return the cached client — call ``reset_object_storage()`` to rebind it
    (credential rotation, tests).
    """
    if _state.client is not None:
        return _state.client
    with _state.lock:
        client = _state.client
        if client is None:
            resolved = settings or get_settings()
            client = boto3.client(
                "s3",
                endpoint_url=resolved.S3_ENDPOINT_URL,
                aws_access_key_id=resolved.S3_ACCESS_KEY_ID.get_secret_value(),
                aws_secret_access_key=resolved.S3_SECRET_ACCESS_KEY.get_secret_value(),
                region_name=resolved.S3_REGION,
                config=_boto_config(resolved),
            )
            _state.client = client
            logger.bind(
                bucket=resolved.S3_BUCKET_NAME,
                endpoint=resolved.S3_ENDPOINT_URL,
                region=resolved.S3_REGION,
            ).info("S3 client created")
    return client


def get_object_store(
    settings: Settings | None = None,
    *,
    bucket: str | None = None,
) -> S3ClientWrapper:
    """Return the cached bucket-bound wrapper over the singleton client."""
    resolved = settings or get_settings()
    bucket_name = bucket or resolved.S3_BUCKET_NAME
    if not bucket_name:
        message = "Object storage bucket is not configured (S3_BUCKET_NAME is empty)"
        raise ValueError(message)
    cached = _state.stores.get(bucket_name)
    if cached is not None:
        return cached
    with _state.lock:
        cached = _state.stores.get(bucket_name)
        if cached is None:
            cached = S3ClientWrapper.from_boto_client(
                bucket=bucket_name,
                client=get_s3_client(resolved),
            )
            _state.stores[bucket_name] = cached
    return cached


def reset_object_storage() -> None:
    """Drop the singleton client and its wrappers (tests, credential rotation)."""
    with _state.lock:
        _state.client = None
        _state.stores.clear()


__all__ = [
    "S3ClientWrapper",
    "get_object_store",
    "get_s3_client",
    "reset_object_storage",
]
