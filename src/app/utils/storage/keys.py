"""Object key and ``s3://`` URI helpers (pure, no I/O)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from returns.result import Failure, Success

from .errors import StorageValidationError
from .models import S3ObjectLocation

if TYPE_CHECKING:
    from .errors import StorageResult


def build_s3_key(
    *,
    prefix: str,
    user_id: str,
    document_id: str,
    content_hash: str,
    filename: str,
    tenant_id: str | None = None,
) -> str:
    extension = _normalized_extension(filename)
    owner_prefix = f"{tenant_id}/{user_id}" if tenant_id else user_id
    return f"{prefix}/{owner_prefix}/{document_id}/{content_hash}{extension}"


def build_s3_uri(*, bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def key_from_s3_uri(uri: str) -> StorageResult[str]:
    prefix = "s3://"
    if not uri.startswith(prefix):
        return Failure(StorageValidationError(message=f"Unsupported object URI: {uri}"))
    parts = uri[len(prefix) :].split("/", maxsplit=1)
    if len(parts) != 2:
        return Failure(StorageValidationError(message=f"Invalid object URI: {uri}"))
    return Success(parts[1])


def uri_to_location(uri: str) -> StorageResult[S3ObjectLocation]:
    prefix = "s3://"
    if not uri.startswith(prefix):
        return Failure(StorageValidationError(message=f"Unsupported object URI: {uri}"))
    remainder = uri[len(prefix) :]
    parts = remainder.split("/", maxsplit=1)
    if len(parts) != 2:
        return Failure(StorageValidationError(message=f"Invalid object URI: {uri}"))
    return Success(S3ObjectLocation(bucket=parts[0], key=parts[1]))


def _normalized_extension(filename: str) -> str:
    return Path(filename).suffix.lower() or ".bin"


__all__ = [
    "build_s3_key",
    "build_s3_uri",
    "key_from_s3_uri",
    "uri_to_location",
]
