"""Result-native S3 object operations.

Transport failures are translated at this adapter boundary into
``StorageResult`` values: callers branch on ``Failure`` and never catch a
botocore exception to implement expected control flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import asyncer
from botocore.exceptions import BotoCoreError, ClientError
from returns.result import Failure, Success

from app.observability.logging import logger

from .errors import StorageUnavailableError, StorageValidationError
from .keys import build_s3_key, build_s3_uri, key_from_s3_uri
from .models import (
    MultipartPartURL,
    MultipartUploadPlan,
    PresignedUploadURL,
    S3CompleteMultipartUploadResponse,
    S3ListObjectsResponse,
    S3ListPartsResponse,
    S3PartInfo,
    S3UploadPartResponse,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from .errors import StorageResult
    from .models import UploadRequest
    from .protocols import ObjectStoreClient

_NOT_FOUND_CODES = frozenset({"404", "NotFound", "NoSuchKey"})
_NO_SUCH_UPLOAD_CODES = frozenset({"404", "NoSuchUpload"})


def _error_code(exc: BaseException) -> str:
    """Read the S3 error code without narrowing every caller to ``ClientError``."""
    if not isinstance(exc, ClientError):
        return ""
    return exc.response.get("Error", {}).get("Code", "")


def _failure(
    operation: str,
    event: str,
    message: str,
    exc: BaseException,
    *,
    bucket: str,
    **context: object,
) -> StorageUnavailableError:
    """Translate one botocore failure into a typed, logged error value."""
    exc.add_note(
        ", ".join(f"{name}={value}" for name, value in {"bucket": bucket, **context}.items())
        + f", operation={operation}"
    )
    logger.bind(operation=operation, bucket=bucket, error=str(exc), **context).exception(event)
    return StorageUnavailableError(
        message=message,
        details={"bucket": bucket, **context, "notes": list(getattr(exc, "__notes__", []))},
        source="object_storage",
        operation=operation,
    )


async def put_object(
    client: ObjectStoreClient,
    *,
    key: str,
    data: bytes,
    content_type: str,
    metadata: dict[str, str],
) -> StorageResult[str]:
    """Upload bytes and return the ``s3://bucket/key`` handle for the object."""
    try:
        await asyncer.asyncify(client.put_object)(
            key=key,
            data=data,
            content_type=content_type,
            metadata=metadata,
        )
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "put_object",
                "s3_put_failed",
                "Object storage upload failed",
                exc,
                bucket=client.bucket,
                key=key,
            )
        )
    return Success(build_s3_uri(bucket=client.bucket, key=key))


async def get_object(client: ObjectStoreClient, *, key: str) -> StorageResult[bytes]:
    try:
        value = await asyncer.asyncify(client.get_object)(key=key)
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "get_object",
                "s3_get_failed",
                "Object storage download failed",
                exc,
                bucket=client.bucket,
                key=key,
            )
        )
    return Success(value)


async def delete_object(client: ObjectStoreClient, *, key: str) -> StorageResult[None]:
    try:
        await asyncer.asyncify(client.delete_object)(key=key)
    except (BotoCoreError, ClientError) as exc:
        if _error_code(exc) in _NOT_FOUND_CODES:
            return Success(None)
        return Failure(
            _failure(
                "delete_object",
                "s3_delete_failed",
                "Object storage delete failed",
                exc,
                bucket=client.bucket,
                key=key,
            )
        )
    return Success(None)


async def get_by_uri(client: ObjectStoreClient, *, uri: str) -> StorageResult[bytes]:
    key_result = key_from_s3_uri(uri)
    if isinstance(key_result, Failure):
        return key_result
    return await get_object(client, key=key_result.unwrap())


async def delete_by_uri(client: ObjectStoreClient, *, uri: str) -> StorageResult[None]:
    key_result = key_from_s3_uri(uri)
    if isinstance(key_result, Failure):
        return key_result
    return await delete_object(client, key=key_result.unwrap())


async def verify_access(client: ObjectStoreClient) -> StorageResult[None]:
    try:
        await asyncer.asyncify(client.head_bucket)()
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "verify_access",
                "s3_verify_access_failed",
                "Object storage bucket access failed",
                exc,
                bucket=client.bucket,
            )
        )
    return Success(None)


async def object_exists(client: ObjectStoreClient, *, key: str) -> StorageResult[bool]:
    try:
        await asyncer.asyncify(client.head_object)(key=key)
    except (BotoCoreError, ClientError) as exc:
        if _error_code(exc) in _NOT_FOUND_CODES:
            return Success(False)
        return Failure(
            _failure(
                "object_exists",
                "s3_head_failed",
                "Object existence check failed",
                exc,
                bucket=client.bucket,
                key=key,
            )
        )
    return Success(True)


async def list_objects(
    client: ObjectStoreClient,
    *,
    prefix: str = "",
    max_keys: int = 1000,
) -> StorageResult[S3ListObjectsResponse]:
    try:
        response = await asyncer.asyncify(client.list_objects)(prefix=prefix, max_keys=max_keys)
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "list_objects",
                "s3_list_failed",
                "Object listing failed",
                exc,
                bucket=client.bucket,
                prefix=prefix,
            )
        )
    contents = response.get("Contents") or []
    common_prefixes = response.get("CommonPrefixes") or []
    return Success(
        S3ListObjectsResponse(
            keys=[obj["Key"] for obj in contents if "Key" in obj],
            prefixes=[p["Prefix"] for p in common_prefixes if "Prefix" in p],
            is_truncated=response.get("IsTruncated", False),
            key_count=response.get("KeyCount", 0),
        )
    )


async def copy_object(
    client: ObjectStoreClient,
    *,
    source_key: str,
    destination_key: str,
) -> StorageResult[None]:
    try:
        await asyncer.asyncify(client.copy_object)(
            source_key=source_key,
            destination_key=destination_key,
        )
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "copy_object",
                "s3_copy_failed",
                "Object copy failed",
                exc,
                bucket=client.bucket,
                source_key=source_key,
                destination_key=destination_key,
            )
        )
    return Success(None)


async def create_multipart_upload(
    client: ObjectStoreClient,
    *,
    key: str,
    content_type: str,
    metadata: dict[str, str] | None = None,
) -> StorageResult[dict[str, Any]]:
    try:
        response = await asyncer.asyncify(client.create_multipart_upload)(
            key=key,
            content_type=content_type,
            metadata=metadata or {},
        )
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "create_multipart_upload",
                "s3_multipart_create_failed",
                "Multipart upload initialization failed",
                exc,
                bucket=client.bucket,
                key=key,
            )
        )
    return Success(response)


async def upload_part(
    client: ObjectStoreClient,
    *,
    key: str,
    upload_id: str,
    part_number: int,
    body: bytes,
) -> StorageResult[S3UploadPartResponse]:
    try:
        response = await asyncer.asyncify(client.upload_part)(
            key=key,
            upload_id=upload_id,
            part_number=part_number,
            body=body,
        )
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "upload_part",
                "s3_upload_part_failed",
                "Multipart part upload failed",
                exc,
                bucket=client.bucket,
                key=key,
                upload_id=upload_id,
                part_number=part_number,
            )
        )
    return Success(S3UploadPartResponse(e_tag=str(response.get("ETag", ""))))


async def complete_multipart_upload(
    client: ObjectStoreClient,
    *,
    key: str,
    upload_id: str,
    parts: list[dict[str, Any]],
) -> StorageResult[S3CompleteMultipartUploadResponse]:
    try:
        response = await asyncer.asyncify(client.complete_multipart_upload)(
            key=key,
            upload_id=upload_id,
            parts=parts,
        )
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "complete_multipart_upload",
                "s3_multipart_complete_failed",
                "Multipart upload completion failed",
                exc,
                bucket=client.bucket,
                key=key,
                upload_id=upload_id,
            )
        )
    return Success(
        S3CompleteMultipartUploadResponse(
            location=str(response.get("Location", "")),
            bucket=str(response.get("Bucket", "")),
            key=str(response.get("Key", "")),
            e_tag=str(response.get("ETag", "")),
        )
    )


async def abort_multipart_upload(
    client: ObjectStoreClient,
    *,
    key: str,
    upload_id: str,
) -> StorageResult[None]:
    try:
        await asyncer.asyncify(client.abort_multipart_upload)(key=key, upload_id=upload_id)
    except (BotoCoreError, ClientError) as exc:
        # An already-gone upload is the desired end state, not a failure.
        if _error_code(exc) in _NO_SUCH_UPLOAD_CODES:
            return Success(None)
        return Failure(
            _failure(
                "abort_multipart_upload",
                "s3_multipart_abort_failed",
                "Multipart upload abort failed",
                exc,
                bucket=client.bucket,
                key=key,
                upload_id=upload_id,
            )
        )
    return Success(None)


async def list_multipart_upload_parts(
    client: ObjectStoreClient,
    *,
    key: str,
    upload_id: str,
) -> StorageResult[S3ListPartsResponse]:
    try:
        response = await asyncer.asyncify(client.list_parts)(key=key, upload_id=upload_id)
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "list_multipart_upload_parts",
                "s3_multipart_list_parts_failed",
                "Multipart upload part listing failed",
                exc,
                bucket=client.bucket,
                key=key,
                upload_id=upload_id,
            )
        )
    parts_data = response.get("Parts") or []
    return Success(
        S3ListPartsResponse(
            parts=[
                S3PartInfo(
                    part_number=p["PartNumber"],
                    e_tag=p["ETag"],
                    size=p["Size"],
                    last_modified=p["LastModified"],
                )
                for p in parts_data
            ]
        )
    )


async def get_signed_get_url(
    client: ObjectStoreClient,
    *,
    key: str,
    expires_in: int = 900,
) -> StorageResult[str]:
    params = {"Bucket": client.bucket, "Key": key}
    try:
        url = await asyncer.asyncify(client.generate_presigned_url)(
            "get_object",
            params=params,
            expires_in=expires_in,
        )
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "get_signed_get_url",
                "s3_signed_get_url_failed",
                "Signed download URL generation failed",
                exc,
                bucket=client.bucket,
                key=key,
            )
        )
    logger.bind(bucket=client.bucket, key=key).debug("s3_signed_get_url_generated")
    return Success(url)


async def get_signed_put_url(
    client: ObjectStoreClient,
    *,
    key: str,
    content_type: str,
    expires_in: int = 900,
) -> StorageResult[str]:
    params = {"Bucket": client.bucket, "Key": key, "ContentType": content_type}
    try:
        url = await asyncer.asyncify(client.generate_presigned_url)(
            "put_object",
            params=params,
            expires_in=expires_in,
        )
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "get_signed_put_url",
                "s3_signed_put_url_failed",
                "Signed upload URL generation failed",
                exc,
                bucket=client.bucket,
                key=key,
            )
        )
    logger.bind(bucket=client.bucket, key=key).debug("s3_signed_put_url_generated")
    return Success(url)


async def generate_presigned_upload_urls(
    client: ObjectStoreClient,
    *,
    files: Sequence[UploadRequest] = (),
    destination: str = "uploads",
    expires_in: int = 900,
) -> StorageResult[list[PresignedUploadURL]]:
    results: list[PresignedUploadURL] = []
    for file in files:
        key = build_s3_key(
            prefix=destination,
            user_id=file.user_id,
            document_id=file.document_id,
            content_hash=file.content_hash,
            filename=file.filename,
        )
        signed_url_result = await get_signed_put_url(
            client,
            key=key,
            content_type=file.content_type,
            expires_in=expires_in,
        )
        if isinstance(signed_url_result, Failure):
            return signed_url_result
        results.append(PresignedUploadURL(signed_url=signed_url_result.unwrap(), key=key))
    return Success(results)


async def create_multipart_upload_plan(
    client: ObjectStoreClient,
    *,
    filename: str,
    content_type: str,
    file_size: int,
    destination: str = "uploads",
    part_size: int = 5 * 1024 * 1024,
    expires_in: int = 3600,
    metadata: dict[str, str] | None = None,
) -> StorageResult[MultipartUploadPlan]:
    if file_size <= 0:
        return Failure(StorageValidationError(message="file_size must be greater than zero"))

    key = build_s3_key(
        prefix=destination,
        user_id="multipart",
        document_id=uuid4().hex,
        content_hash=str(file_size),
        filename=filename,
    )
    init_response = await create_multipart_upload(
        client,
        key=key,
        content_type=content_type,
        metadata=metadata or {},
    )
    if isinstance(init_response, Failure):
        return init_response
    upload_id = str(init_response.unwrap().get("UploadId") or "")
    if not upload_id:
        return Failure(
            StorageUnavailableError(
                message="S3 did not return an upload id",
                details={"bucket": client.bucket, "key": key},
                operation="create_multipart_upload_plan",
            )
        )

    num_parts = max(1, (file_size + part_size - 1) // part_size)
    parts: list[MultipartPartURL] = []
    for part_number in range(1, num_parts + 1):
        url_result = await get_signed_multipart_part_url(
            client,
            key=key,
            upload_id=upload_id,
            part_number=part_number,
            expires_in=expires_in,
        )
        if isinstance(url_result, Failure):
            # Abandoning a started upload leaves billable parts behind until a
            # bucket lifecycle rule reaps them; abort before returning.
            await abort_multipart_upload(client, key=key, upload_id=upload_id)
            return url_result
        parts.append(MultipartPartURL(part_number=part_number, presigned_url=url_result.unwrap()))
    return Success(
        MultipartUploadPlan(upload_id=upload_id, key=key, parts=parts, part_size=part_size)
    )


async def get_signed_multipart_part_url(
    client: ObjectStoreClient,
    *,
    key: str,
    upload_id: str,
    part_number: int,
    expires_in: int = 3600,
) -> StorageResult[str]:
    params = {
        "Bucket": client.bucket,
        "Key": key,
        "UploadId": upload_id,
        "PartNumber": part_number,
    }
    try:
        url = await asyncer.asyncify(client.generate_presigned_url)(
            "upload_part",
            params=params,
            expires_in=expires_in,
        )
    except (BotoCoreError, ClientError) as exc:
        return Failure(
            _failure(
                "get_signed_multipart_part_url",
                "s3_signed_multipart_part_url_failed",
                "Signed multipart URL generation failed",
                exc,
                bucket=client.bucket,
                key=key,
                upload_id=upload_id,
                part_number=part_number,
            )
        )
    logger.bind(
        bucket=client.bucket,
        key=key,
        upload_id=upload_id,
        part_number=part_number,
    ).debug("s3_signed_multipart_part_url_generated")
    return Success(url)


__all__ = [
    "abort_multipart_upload",
    "complete_multipart_upload",
    "copy_object",
    "create_multipart_upload",
    "create_multipart_upload_plan",
    "delete_by_uri",
    "delete_object",
    "generate_presigned_upload_urls",
    "get_by_uri",
    "get_object",
    "get_signed_get_url",
    "get_signed_multipart_part_url",
    "get_signed_put_url",
    "list_multipart_upload_parts",
    "list_objects",
    "object_exists",
    "put_object",
    "upload_part",
    "verify_access",
]
