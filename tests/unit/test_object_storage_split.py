"""Client-formation vs CRUD split: singleton client, cached wrappers, error mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from botocore.exceptions import ClientError
from returns.result import Failure, Success

from app.config.settings import Settings
from app.connections.object_storage import (
    get_object_store,
    get_s3_client,
    reset_object_storage,
)
from app.utils.storage import (
    StorageService,
    StorageUnavailableError,
    build_s3_key,
    build_s3_uri,
    delete_object,
    key_from_s3_uri,
    put_object,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

pytestmark = pytest.mark.unit


def _settings(**overrides: object) -> Settings:
    values: dict[str, Any] = {
        "S3_BUCKET_NAME": "test-bucket",
        "S3_ENDPOINT_URL": "https://account.r2.cloudflarestorage.com",
        "S3_ACCESS_KEY_ID": "access-key",
        "S3_SECRET_ACCESS_KEY": "secret-key",
        "S3_PUBLIC_URL": "https://cdn.example.com/",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(autouse=True)
def _reset_singleton() -> Iterator[None]:
    reset_object_storage()
    yield
    reset_object_storage()


def test_s3_client_is_a_process_wide_singleton() -> None:
    settings = _settings()

    first = get_s3_client(settings)
    second = get_s3_client(settings)

    assert first is second
    assert first.meta.config.retries["mode"] == "standard"


def test_reset_drops_the_singleton() -> None:
    settings = _settings()

    original = get_s3_client(settings)
    reset_object_storage()

    assert get_s3_client(settings) is not original


def test_object_store_is_cached_per_bucket_over_the_same_client() -> None:
    settings = _settings()

    private = get_object_store(settings)
    same = get_object_store(settings)
    other_bucket = get_object_store(settings, bucket="other-bucket")

    assert private is same
    assert private.bucket == "test-bucket"
    assert other_bucket is not private
    assert other_bucket.bucket == "other-bucket"
    assert private._client is other_bucket._client


def test_get_object_store_rejects_an_unconfigured_bucket() -> None:
    with pytest.raises(ValueError, match="S3_BUCKET_NAME"):
        get_object_store(_settings(S3_BUCKET_NAME=""))


def test_service_from_settings_binds_the_singleton_and_normalizes_public_url() -> None:
    settings = _settings()

    service = StorageService.from_settings(settings)

    assert service.bucket == "test-bucket"
    assert service.public_url == "https://cdn.example.com"
    assert service.client is get_object_store(settings)


def test_service_from_settings_falls_back_to_none_without_a_public_url() -> None:
    service = StorageService.from_settings(_settings(S3_PUBLIC_URL=""))

    assert service.public_url is None


def test_key_helpers_round_trip() -> None:
    key = build_s3_key(
        prefix="documents",
        user_id="user-1",
        document_id="doc-1",
        content_hash="abc123",
        filename="Contract.PDF",
        tenant_id="tenant-1",
    )
    uri = build_s3_uri(bucket="test-bucket", key=key)

    assert key == "documents/tenant-1/user-1/doc-1/abc123.pdf"
    assert isinstance(key_from_s3_uri(uri), Success)
    assert key_from_s3_uri(uri).unwrap() == key
    assert isinstance(key_from_s3_uri("https://cdn.example.com/key"), Failure)


@pytest.mark.asyncio
async def test_put_object_returns_the_uri_from_the_crud_layer() -> None:
    class _Client:
        bucket = "test-bucket"

        def put_object(self, **_kwargs: object) -> None:
            return None

    result = await put_object(
        _Client(),  # ty: ignore[invalid-argument-type] — structural client double
        key="documents/a.txt",
        data=b"payload",
        content_type="text/plain",
        metadata={},
    )

    assert result.unwrap() == "s3://test-bucket/documents/a.txt"


@pytest.mark.asyncio
async def test_client_errors_become_typed_retryable_failures() -> None:
    class _Client:
        bucket = "test-bucket"

        def put_object(self, **_kwargs: object) -> None:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")

        def delete_object(self, **_kwargs: object) -> None:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "DeleteObject")

    client = _Client()
    failed = await put_object(
        client,  # ty: ignore[invalid-argument-type] — structural client double
        key="documents/a.txt",
        data=b"payload",
        content_type="text/plain",
        metadata={},
    )
    deleted = await delete_object(client, key="documents/a.txt")  # ty: ignore[invalid-argument-type]

    assert isinstance(failed, Failure)
    error = failed.failure()
    assert isinstance(error, StorageUnavailableError)
    assert error.retryable is True
    assert error.operation == "put_object"
    assert error.details is not None
    assert error.details["key"] == "documents/a.txt"
    # Deleting an absent object is the desired end state, not a failure.
    assert deleted.unwrap() is None
