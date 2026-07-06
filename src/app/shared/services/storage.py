"""S3-compatible object storage utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

import asyncer
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from app.config import Settings
from app.utils import ServiceUnavailableException, ValidationException, logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from typing import Any

    from mypy_boto3_s3 import S3Client

    from app.config import Settings


class S3ObjectLocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return build_s3_uri(bucket=self.bucket, key=self.key)


class PresignedUploadURL(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    signed_url: str
    key: str


class MultipartPartURL(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    part_number: int
    presigned_url: str


class MultipartUploadPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    upload_id: str
    key: str
    parts: list[MultipartPartURL] = Field(default_factory=list)
    part_size: int


class ObjectMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content_type: str | None = None
    content_length: int | None = None
    last_modified: datetime | None = None
    e_tag: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class S3UploadPartResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    e_tag: str


class S3CompleteMultipartUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    location: str
    bucket: str
    key: str
    e_tag: str


class S3PartInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    part_number: int
    e_tag: str
    size: int
    last_modified: datetime


class S3ListPartsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parts: list[S3PartInfo] = Field(default_factory=list)


class S3ListObjectsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    keys: list[str] = Field(default_factory=list)
    prefixes: list[str] = Field(default_factory=list)
    is_truncated: bool = False
    key_count: int = 0


class ObjectStore(Protocol):
    async def put_object(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str],
    ) -> str: ...

    async def get_object(self, *, key: str) -> bytes: ...

    async def delete_object(self, *, key: str) -> None: ...

    async def get_by_uri(self, *, uri: str) -> bytes: ...

    async def delete_by_uri(self, *, uri: str) -> None: ...

    async def verify_access(self) -> None: ...


class S3ClientWrapper(BaseModel):
    """Thin synchronous wrapper around the boto3 S3 client.

    Owns the raw S3 API calls and returns boto3-native dict responses.
    Callers should not depend on the dict shapes exposed by this class.
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
    ) -> str:
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata=metadata,
        )
        return build_s3_uri(bucket=self.bucket, key=key)

    def get_object(self, *, key: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        return bytes(response["Body"].read())

    def delete_object(self, *, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def head_object(self, *, key: str) -> dict[str, Any]:
        return self._client.head_object(Bucket=self.bucket, Key=key)

    def head_bucket(self) -> None:
        self._client.head_bucket(Bucket=self.bucket)

    def list_objects(self, *, prefix: str, max_keys: int) -> dict[str, Any]:
        return self._client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
            MaxKeys=max_keys,
        )

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
        return self._client.create_multipart_upload(
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
        return self._client.upload_part(
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
        return self._client.complete_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )

    def abort_multipart_upload(self, *, key: str, upload_id: str) -> None:
        self._client.abort_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
        )

    def list_parts(self, *, key: str, upload_id: str) -> dict[str, Any]:
        return self._client.list_parts(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
        )

    def generate_presigned_url(
        self, operation: str, params: dict[str, Any], expires_in: int
    ) -> str:
        return self._client.generate_presigned_url(operation, Params=params, ExpiresIn=expires_in)

    @classmethod
    def from_boto_client(cls, *, bucket: str, client: S3Client) -> S3ClientWrapper:
        wrapper = cls(bucket=bucket)
        object.__setattr__(wrapper, "_client", client)
        return wrapper


class StorageService(BaseModel):
    """Async S3/R2 service with presign and multipart helpers."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    bucket: str
    public_url: str | None
    _wrapper: S3ClientWrapper = PrivateAttr()

    @classmethod
    def from_settings(cls, settings: Settings) -> StorageService:
        client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.S3_ACCESS_KEY_ID.get_secret_value(),
            aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY.get_secret_value(),
            region_name=settings.S3_REGION,
        )
        wrapper = S3ClientWrapper.from_boto_client(
            bucket=settings.S3_BUCKET_NAME,
            client=client,
        )
        public_url = settings.S3_PUBLIC_URL.rstrip("/") if settings.S3_PUBLIC_URL else None
        service = cls(bucket=settings.S3_BUCKET_NAME, public_url=public_url)
        object.__setattr__(service, "_wrapper", wrapper)
        return service

    async def put_object(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str],
    ) -> str:
        try:
            uri = await asyncer.asyncify(self._wrapper.put_object)(
                key=key,
                data=data,
                content_type=content_type,
                metadata=metadata,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.bind(bucket=self.bucket, key=key).error("s3_put_failed", error=str(exc))
            raise ServiceUnavailableException(
                detail="Object storage upload failed",
                data={"bucket": self.bucket, "key": key},
            ) from exc
        return uri

    async def get_object(self, *, key: str) -> bytes:
        try:
            return await asyncer.asyncify(self._wrapper.get_object)(key=key)
        except (BotoCoreError, ClientError) as exc:
            logger.bind(bucket=self.bucket, key=key).error("s3_get_failed", error=str(exc))
            raise ServiceUnavailableException(
                detail="Object storage download failed",
                data={"bucket": self.bucket, "key": key},
            ) from exc

    async def delete_object(self, *, key: str) -> None:
        try:
            await asyncer.asyncify(self._wrapper.delete_object)(key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NotFound", "NoSuchKey"}:
                return
            logger.bind(bucket=self.bucket, key=key).error("s3_delete_failed", error=str(exc))
            raise ServiceUnavailableException(
                detail="Object storage delete failed",
                data={"bucket": self.bucket, "key": key},
            ) from exc
        except BotoCoreError as exc:
            logger.bind(bucket=self.bucket, key=key).error("s3_delete_failed", error=str(exc))
            raise ServiceUnavailableException(
                detail="Object storage delete failed",
                data={"bucket": self.bucket, "key": key},
            ) from exc

    async def get_by_uri(self, *, uri: str) -> bytes:
        return await self.get_object(key=key_from_s3_uri(uri))

    async def delete_by_uri(self, *, uri: str) -> None:
        await self.delete_object(key=key_from_s3_uri(uri))

    async def verify_access(self) -> None:
        try:
            await asyncer.asyncify(self._wrapper.head_bucket)()
        except (BotoCoreError, ClientError) as exc:
            logger.bind(bucket=self.bucket).error("s3_verify_access_failed", error=str(exc))
            raise ServiceUnavailableException(
                detail="Object storage bucket access failed",
                data={"bucket": self.bucket},
            ) from exc

    async def object_exists(self, *, key: str) -> bool:
        try:
            await asyncer.asyncify(self._wrapper.head_object)(key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in {"404", "NotFound", "NoSuchKey"}:
                return False
            logger.bind(bucket=self.bucket, key=key).error("s3_head_failed", error=str(exc))
            raise ServiceUnavailableException(
                detail="Object existence check failed",
                data={"bucket": self.bucket, "key": key},
            ) from exc
        except BotoCoreError as exc:
            logger.bind(bucket=self.bucket, key=key).error("s3_head_failed", error=str(exc))
            raise ServiceUnavailableException(
                detail="Object existence check failed",
                data={"bucket": self.bucket, "key": key},
            ) from exc
        return True

    async def list_objects(
        self,
        *,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> S3ListObjectsResponse:
        try:
            response = await asyncer.asyncify(self._wrapper.list_objects)(
                prefix=prefix, max_keys=max_keys
            )
        except (BotoCoreError, ClientError) as exc:
            logger.bind(bucket=self.bucket, prefix=prefix).error("s3_list_failed", error=str(exc))
            raise ServiceUnavailableException(
                detail="Object listing failed",
                data={"bucket": self.bucket, "prefix": prefix},
            ) from exc
        contents = response.get("Contents") or []
        common_prefixes = response.get("CommonPrefixes") or []
        return S3ListObjectsResponse(
            keys=[obj["Key"] for obj in contents if "Key" in obj],
            prefixes=[p["Prefix"] for p in common_prefixes if "Prefix" in p],
            is_truncated=response.get("IsTruncated", False),
            key_count=response.get("KeyCount", 0),
        )

    async def copy_object(self, *, source_key: str, destination_key: str) -> None:
        try:
            await asyncer.asyncify(self._wrapper.copy_object)(
                source_key=source_key,
                destination_key=destination_key,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.bind(
                bucket=self.bucket,
                source_key=source_key,
                destination_key=destination_key,
            ).error("s3_copy_failed", error=str(exc))
            raise ServiceUnavailableException(
                detail="Object copy failed",
                data={
                    "bucket": self.bucket,
                    "source_key": source_key,
                    "destination_key": destination_key,
                },
            ) from exc

    async def create_multipart_upload(
        self,
        *,
        key: str,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            return await asyncer.asyncify(self._wrapper.create_multipart_upload)(
                key=key,
                content_type=content_type,
                metadata=metadata or {},
            )
        except (BotoCoreError, ClientError) as exc:
            logger.bind(bucket=self.bucket, key=key).error(
                "s3_multipart_create_failed", error=str(exc)
            )
            raise ServiceUnavailableException(
                detail="Multipart upload initialization failed",
                data={"bucket": self.bucket, "key": key},
            ) from exc

    async def upload_part(
        self,
        *,
        key: str,
        upload_id: str,
        part_number: int,
        body: bytes,
    ) -> S3UploadPartResponse:
        try:
            response = await asyncer.asyncify(self._wrapper.upload_part)(
                key=key,
                upload_id=upload_id,
                part_number=part_number,
                body=body,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.bind(
                bucket=self.bucket,
                key=key,
                upload_id=upload_id,
                part_number=part_number,
            ).error("s3_upload_part_failed", error=str(exc))
            raise ServiceUnavailableException(
                detail="Multipart part upload failed",
                data={
                    "bucket": self.bucket,
                    "key": key,
                    "upload_id": upload_id,
                    "part_number": part_number,
                },
            ) from exc
        return S3UploadPartResponse(e_tag=str(response.get("ETag", "")))

    async def complete_multipart_upload(
        self,
        *,
        key: str,
        upload_id: str,
        parts: list[dict[str, Any]],
    ) -> S3CompleteMultipartUploadResponse:
        try:
            response = await asyncer.asyncify(self._wrapper.complete_multipart_upload)(
                key=key,
                upload_id=upload_id,
                parts=parts,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.bind(bucket=self.bucket, key=key, upload_id=upload_id).error(
                "s3_multipart_complete_failed",
                error=str(exc),
            )
            raise ServiceUnavailableException(
                detail="Multipart upload completion failed",
                data={"bucket": self.bucket, "key": key, "upload_id": upload_id},
            ) from exc
        return S3CompleteMultipartUploadResponse(
            location=str(response.get("Location", "")),
            bucket=str(response.get("Bucket", "")),
            key=str(response.get("Key", "")),
            e_tag=str(response.get("ETag", "")),
        )

    async def abort_multipart_upload(self, *, key: str, upload_id: str) -> None:
        try:
            await asyncer.asyncify(self._wrapper.abort_multipart_upload)(
                key=key, upload_id=upload_id
            )
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchUpload"}:
                return
            logger.bind(bucket=self.bucket, key=key, upload_id=upload_id).error(
                "s3_multipart_abort_failed",
                error=str(exc),
            )
            raise ServiceUnavailableException(
                detail="Multipart upload abort failed",
                data={"bucket": self.bucket, "key": key, "upload_id": upload_id},
            ) from exc
        except BotoCoreError as exc:
            logger.bind(bucket=self.bucket, key=key, upload_id=upload_id).error(
                "s3_multipart_abort_failed",
                error=str(exc),
            )
            raise ServiceUnavailableException(
                detail="Multipart upload abort failed",
                data={"bucket": self.bucket, "key": key, "upload_id": upload_id},
            ) from exc

    async def list_multipart_upload_parts(
        self,
        *,
        key: str,
        upload_id: str,
    ) -> S3ListPartsResponse:
        try:
            response = await asyncer.asyncify(self._wrapper.list_parts)(
                key=key, upload_id=upload_id
            )
        except (BotoCoreError, ClientError) as exc:
            logger.bind(bucket=self.bucket, key=key, upload_id=upload_id).error(
                "s3_multipart_list_parts_failed",
                error=str(exc),
            )
            raise ServiceUnavailableException(
                detail="Multipart upload part listing failed",
                data={"bucket": self.bucket, "key": key, "upload_id": upload_id},
            ) from exc
        parts_data = response.get("Parts") or []
        return S3ListPartsResponse(
            parts=[
                S3PartInfo(
                    part_number=p["PartNumber"],
                    e_tag=p["ETag"],
                    size=p["Size"],
                    last_modified=p["LastModified"],
                )
                for p in parts_data
            ],
        )

    async def get_signed_get_url(self, *, key: str, expires_in: int = 900) -> str:
        params = {"Bucket": self.bucket, "Key": key}
        url = await asyncer.asyncify(self._wrapper.generate_presigned_url)(
            "get_object",
            params=params,
            expires_in=expires_in,
        )
        logger.bind(bucket=self.bucket, key=key).debug("s3_signed_get_url_generated")
        return url

    async def get_signed_put_url(
        self,
        *,
        key: str,
        content_type: str,
        expires_in: int = 900,
    ) -> str:
        params = {"Bucket": self.bucket, "Key": key, "ContentType": content_type}
        url = await asyncer.asyncify(self._wrapper.generate_presigned_url)(
            "put_object",
            params=params,
            expires_in=expires_in,
        )
        logger.bind(bucket=self.bucket, key=key).debug("s3_signed_put_url_generated")
        return url

    async def generate_presigned_upload_urls(
        self,
        *,
        files: Sequence[UploadRequest] = (),
        destination: str = "uploads",
        expires_in: int = 900,
    ) -> list[PresignedUploadURL]:
        results: list[PresignedUploadURL] = []
        for file in files:
            key = build_s3_key(
                prefix=destination,
                user_id=file.user_id,
                document_id=file.document_id,
                content_hash=file.content_hash,
                filename=file.filename,
            )
            signed_url = await self.get_signed_put_url(
                key=key,
                content_type=file.content_type,
                expires_in=expires_in,
            )
            results.append(PresignedUploadURL(signed_url=signed_url, key=key))
        return results

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
    ) -> MultipartUploadPlan:
        if file_size <= 0:
            message = "file_size must be greater than zero"
            raise ValidationException(message)

        key = build_s3_key(
            prefix=destination,
            user_id="multipart",
            document_id=uuid4().hex,
            content_hash=str(file_size),
            filename=filename,
        )
        init_response = await self.create_multipart_upload(
            key=key,
            content_type=content_type,
            metadata=metadata or {},
        )
        upload_id = str(init_response.get("UploadId") or "")
        if not upload_id:
            message = "S3 did not return an upload id"
            raise ServiceUnavailableException(
                detail=message, data={"bucket": self.bucket, "key": key}
            )

        num_parts = max(1, (file_size + part_size - 1) // part_size)
        parts: list[MultipartPartURL] = []
        for part_number in range(1, num_parts + 1):
            url = await self.get_signed_multipart_part_url(
                key=key,
                upload_id=upload_id,
                part_number=part_number,
                expires_in=expires_in,
            )
            parts.append(MultipartPartURL(part_number=part_number, presigned_url=url))
        return MultipartUploadPlan(upload_id=upload_id, key=key, parts=parts, part_size=part_size)

    async def get_signed_multipart_part_url(
        self,
        *,
        key: str,
        upload_id: str,
        part_number: int,
        expires_in: int = 3600,
    ) -> str:
        params = {
            "Bucket": self.bucket,
            "Key": key,
            "UploadId": upload_id,
            "PartNumber": part_number,
        }
        url = await asyncer.asyncify(self._wrapper.generate_presigned_url)(
            "upload_part",
            params=params,
            expires_in=expires_in,
        )
        logger.bind(
            bucket=self.bucket, key=key, upload_id=upload_id, part_number=part_number
        ).debug("s3_signed_multipart_part_url_generated")
        return url


class UploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    content_type: str
    user_id: str
    document_id: str
    content_hash: str


def build_s3_key(
    *, prefix: str, user_id: str, document_id: str, content_hash: str, filename: str
) -> str:
    extension = _normalized_extension(filename)
    return f"{prefix}/{user_id}/{document_id}/{content_hash}{extension}"


def build_s3_uri(*, bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def key_from_s3_uri(uri: str) -> str:
    prefix = "s3://"
    if not uri.startswith(prefix):
        message = f"Unsupported object URI: {uri}"
        raise ValidationException(message)
    parts = uri[len(prefix) :].split("/", maxsplit=1)
    if len(parts) != 2:
        message = f"Invalid object URI: {uri}"
        raise ValidationException(message)
    return parts[1]


def uri_to_location(uri: str) -> S3ObjectLocation:
    prefix = "s3://"
    if not uri.startswith(prefix):
        message = f"Unsupported object URI: {uri}"
        raise ValidationException(message)
    remainder = uri[len(prefix) :]
    parts = remainder.split("/", maxsplit=1)
    if len(parts) != 2:
        message = f"Invalid object URI: {uri}"
        raise ValidationException(message)
    return S3ObjectLocation(bucket=parts[0], key=parts[1])


def _normalized_extension(filename: str) -> str:
    return Path(filename).suffix.lower() or ".bin"


async def put_object(
    storage: StorageService, *, key: str, data: bytes, content_type: str, metadata: dict[str, str]
) -> str:
    return await storage.put_object(
        key=key, data=data, content_type=content_type, metadata=metadata
    )


async def get_object(storage: StorageService, *, key: str) -> bytes:
    return await storage.get_object(key=key)


async def delete_object(storage: StorageService, *, key: str) -> None:
    await storage.delete_object(key=key)


async def get_by_uri(storage: StorageService, *, uri: str) -> bytes:
    return await storage.get_by_uri(uri=uri)


async def delete_by_uri(storage: StorageService, *, uri: str) -> None:
    await storage.delete_by_uri(uri=uri)


async def verify_access(storage: StorageService) -> None:
    await storage.verify_access()


async def object_exists(storage: StorageService, *, key: str) -> bool:
    return await storage.object_exists(key=key)


async def list_objects(
    storage: StorageService, *, prefix: str = "", max_keys: int = 1000
) -> S3ListObjectsResponse:
    return await storage.list_objects(prefix=prefix, max_keys=max_keys)


async def copy_object(storage: StorageService, *, source_key: str, destination_key: str) -> None:
    await storage.copy_object(source_key=source_key, destination_key=destination_key)


async def create_multipart_upload(
    storage: StorageService, *, key: str, content_type: str, metadata: dict[str, str] | None = None
) -> dict[str, Any]:
    return await storage.create_multipart_upload(
        key=key, content_type=content_type, metadata=metadata
    )


async def upload_part(
    storage: StorageService, *, key: str, upload_id: str, part_number: int, body: bytes
) -> S3UploadPartResponse:
    return await storage.upload_part(
        key=key, upload_id=upload_id, part_number=part_number, body=body
    )


async def complete_multipart_upload(
    storage: StorageService, *, key: str, upload_id: str, parts: list[dict[str, Any]]
) -> S3CompleteMultipartUploadResponse:
    return await storage.complete_multipart_upload(key=key, upload_id=upload_id, parts=parts)


async def abort_multipart_upload(storage: StorageService, *, key: str, upload_id: str) -> None:
    await storage.abort_multipart_upload(key=key, upload_id=upload_id)


async def list_multipart_upload_parts(
    storage: StorageService, *, key: str, upload_id: str
) -> S3ListPartsResponse:
    return await storage.list_multipart_upload_parts(key=key, upload_id=upload_id)


async def get_signed_get_url(storage: StorageService, *, key: str, expires_in: int = 900) -> str:
    return await storage.get_signed_get_url(key=key, expires_in=expires_in)


async def get_signed_put_url(
    storage: StorageService, *, key: str, content_type: str, expires_in: int = 900
) -> str:
    return await storage.get_signed_put_url(
        key=key, content_type=content_type, expires_in=expires_in
    )
