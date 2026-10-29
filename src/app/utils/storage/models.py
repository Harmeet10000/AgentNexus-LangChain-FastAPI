"""Read models and value objects returned by the storage CRUD layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid", frozen=True)


class S3ObjectLocation(BaseModel):
    model_config = _STRICT

    bucket: str
    key: str

    @property
    def uri(self) -> str:
        # Inlined rather than imported from ``keys`` so this module stays the
        # leaf of the package (``keys`` imports these models).
        return f"s3://{self.bucket}/{self.key}"


class PresignedUploadURL(BaseModel):
    model_config = _STRICT

    signed_url: str
    key: str


class MultipartPartURL(BaseModel):
    model_config = _STRICT

    part_number: int
    presigned_url: str


class MultipartUploadPlan(BaseModel):
    model_config = _STRICT

    upload_id: str
    key: str
    parts: list[MultipartPartURL] = Field(default_factory=list)
    part_size: int


class ObjectMetadata(BaseModel):
    model_config = _STRICT

    content_type: str | None = None
    content_length: int | None = None
    last_modified: datetime | None = None
    e_tag: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class S3UploadPartResponse(BaseModel):
    model_config = _STRICT

    e_tag: str


class S3CompleteMultipartUploadResponse(BaseModel):
    model_config = _STRICT

    location: str
    bucket: str
    key: str
    e_tag: str


class S3PartInfo(BaseModel):
    model_config = _STRICT

    part_number: int
    e_tag: str
    size: int
    last_modified: datetime


class S3ListPartsResponse(BaseModel):
    model_config = _STRICT

    parts: list[S3PartInfo] = Field(default_factory=list)


class S3ListObjectsResponse(BaseModel):
    model_config = _STRICT

    keys: list[str] = Field(default_factory=list)
    prefixes: list[str] = Field(default_factory=list)
    is_truncated: bool = False
    key_count: int = 0


class UploadRequest(BaseModel):
    model_config = _STRICT

    filename: str
    content_type: str
    user_id: str
    document_id: str
    content_hash: str
