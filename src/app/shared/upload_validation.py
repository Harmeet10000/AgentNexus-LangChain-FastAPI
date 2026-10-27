"""Bounded validation and reading for multipart document uploads."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePath

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success


class UploadValidationCode(StrEnum):
    """Stable machine-readable reasons for rejecting an upload."""

    FILENAME_REQUIRED = "FILENAME_REQUIRED"
    EXTENSION_NOT_ALLOWED = "EXTENSION_NOT_ALLOWED"
    CONTENT_TYPE_NOT_ALLOWED = "CONTENT_TYPE_NOT_ALLOWED"
    EMPTY_FILE = "EMPTY_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"


class UploadValidationIssue(BaseModel):
    """Transport-neutral validation failure translated by each owning feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: UploadValidationCode
    message: str
    details: dict[str, object] = Field(default_factory=dict)


type UploadReadResult = Result[bytes, UploadValidationIssue]


# MIME headers are advisory, but a clearly contradictory non-generic header is
# still useful to reject. ``application/octet-stream`` is accepted for an
# allowed extension because clients and object-store proxies commonly use it
# when they cannot infer a type. The extension remains the authoritative gate.
_CONTENT_TYPES_BY_EXTENSION: dict[str, frozenset[str]] = {
    "pdf": frozenset({"application/pdf"}),
    "txt": frozenset({"text/plain"}),
    "docx": frozenset({"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}),
    "xlsx": frozenset({"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}),
    "pptx": frozenset(
        {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
    ),
    "md": frozenset({"text/markdown", "text/plain"}),
    "html": frozenset({"text/html", "application/xhtml+xml", "text/plain"}),
}
_GENERIC_CONTENT_TYPE = "application/octet-stream"


def _issue(
    code: UploadValidationCode,
    message: str,
    **details: object,
) -> Failure[UploadValidationIssue]:
    return Failure(UploadValidationIssue(code=code, message=message, details=details))


def _normalise_extension(value: str) -> str:
    return value.strip().lower().removeprefix(".")


def _file_extension(filename: str) -> str:
    return _normalise_extension(PurePath(filename).suffix)


def _content_type(value: str | None) -> str:
    return (value or _GENERIC_CONTENT_TYPE).split(";", maxsplit=1)[0].strip().lower()


async def read_bounded_upload(
    file: UploadFile,
    *,
    max_size: int,
    allowed_extensions: list[str],
) -> UploadReadResult:
    """Validate metadata and read at most one byte beyond the configured limit.

    Multipart parsing happens inside Starlette before a route runs, but the
    application must not perform a second unbounded read. Reading ``limit + 1``
    gives an exact oversized decision while bounding application memory and
    ensuring rejected content never reaches a parser or persistence service.
    """
    filename = file.filename
    if not filename:
        return _issue(
            UploadValidationCode.FILENAME_REQUIRED,
            "Filename is required",
        )
    if max_size < 1:
        message = "max_size must be positive"
        raise ValueError(message)

    extension = _file_extension(filename)
    allowed = {_normalise_extension(value) for value in allowed_extensions}
    if not extension or extension not in allowed:
        return _issue(
            UploadValidationCode.EXTENSION_NOT_ALLOWED,
            "File extension is not allowed",
            extension=extension or None,
            allowed_extensions=sorted(allowed),
        )

    content_type = _content_type(file.content_type)
    accepted_types = _CONTENT_TYPES_BY_EXTENSION.get(extension)
    if (
        accepted_types is not None
        and content_type != _GENERIC_CONTENT_TYPE
        and content_type not in accepted_types
    ):
        return _issue(
            UploadValidationCode.CONTENT_TYPE_NOT_ALLOWED,
            "File content type is not allowed for this extension",
            extension=extension,
            content_type=content_type,
        )

    raw_bytes = await file.read(max_size + 1)
    if not raw_bytes:
        return _issue(UploadValidationCode.EMPTY_FILE, "Uploaded file is empty")
    if len(raw_bytes) > max_size:
        return _issue(
            UploadValidationCode.FILE_TOO_LARGE,
            "Uploaded file exceeds the maximum allowed size",
            max_size=max_size,
        )
    return Success(raw_bytes)


def upload_issue_details(issue: UploadValidationIssue) -> dict[str, object]:
    """Build stable, non-sensitive error metadata for the API envelope."""
    return {"validation_code": issue.code.value, **issue.details}


__all__ = [
    "UploadReadResult",
    "UploadValidationCode",
    "UploadValidationIssue",
    "read_bounded_upload",
    "upload_issue_details",
]
