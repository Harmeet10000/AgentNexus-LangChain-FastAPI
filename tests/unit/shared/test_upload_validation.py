"""Upload validation stays bounded before document parsing or persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from returns.result import Failure, Success

from app.shared.upload_validation import (
    UploadValidationCode,
    read_bounded_upload,
)

if TYPE_CHECKING:
    from fastapi import UploadFile

pytestmark = pytest.mark.unit


class _FakeUpload:
    def __init__(self, *, filename: str | None, content_type: str, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self.data = data
        self.requested_size: int | None = None

    async def read(self, size: int = -1) -> bytes:
        self.requested_size = size
        return self.data[:size]


def _upload(
    *,
    filename: str | None = "contract.pdf",
    content_type: str = "application/pdf",
    data: bytes = b"contract",
) -> _FakeUpload:
    return _FakeUpload(filename=filename, content_type=content_type, data=data)


async def test_oversized_upload_is_rejected_after_bounded_read() -> None:
    upload = _upload(data=b"123456")

    result = await read_bounded_upload(
        cast("UploadFile", upload),
        max_size=5,
        allowed_extensions=["pdf"],
    )

    assert isinstance(result, Failure)
    assert result.failure().code is UploadValidationCode.FILE_TOO_LARGE
    assert upload.requested_size == 6


async def test_disallowed_extension_is_rejected_before_read() -> None:
    upload = _upload(filename="contract.exe", content_type="application/octet-stream")

    result = await read_bounded_upload(
        cast("UploadFile", upload),
        max_size=100,
        allowed_extensions=["pdf"],
    )

    assert isinstance(result, Failure)
    assert result.failure().code is UploadValidationCode.EXTENSION_NOT_ALLOWED
    assert upload.requested_size is None


async def test_contradictory_content_type_is_rejected_before_read() -> None:
    upload = _upload(content_type="text/plain")

    result = await read_bounded_upload(
        cast("UploadFile", upload),
        max_size=100,
        allowed_extensions=["pdf"],
    )

    assert isinstance(result, Failure)
    assert result.failure().code is UploadValidationCode.CONTENT_TYPE_NOT_ALLOWED
    assert upload.requested_size is None


async def test_empty_upload_is_rejected() -> None:
    result = await read_bounded_upload(
        cast("UploadFile", _upload(data=b"")),
        max_size=100,
        allowed_extensions=["pdf"],
    )

    assert isinstance(result, Failure)
    assert result.failure().code is UploadValidationCode.EMPTY_FILE


async def test_valid_upload_is_returned_unchanged() -> None:
    upload = _upload(data=b"%PDF-1.7")

    result = await read_bounded_upload(
        cast("UploadFile", upload),
        max_size=100,
        allowed_extensions=["pdf"],
    )

    assert isinstance(result, Success)
    assert result.unwrap() == b"%PDF-1.7"
    assert upload.requested_size == 101


async def test_missing_filename_is_rejected_before_read() -> None:
    upload = _upload(filename=None)

    result = await read_bounded_upload(
        cast("UploadFile", upload),
        max_size=100,
        allowed_extensions=["pdf"],
    )

    assert isinstance(result, Failure)
    assert result.failure().code is UploadValidationCode.FILENAME_REQUIRED
    assert upload.requested_size is None
