import mimetypes
from dataclasses import dataclass
from uuid import uuid4

import asyncer
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings
from app.utils import ValidationException, logger

_ALLOWED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_MAX_AVATAR_BYTES = 5 * 1024 * 1024  # 5 MB


@dataclass(frozen=True)
class StorageService:
    """S3/R2-compatible async upload service.

    boto3 is synchronous. We bridge it with asyncer.asyncify so the
    event loop is never blocked. boto3 client is NOT thread-safe for
    concurrent calls — asyncer runs each call in a fresh thread from
    the default executor, which is correct here.
    """

    bucket: str
    public_url: str
    _client: object  # boto3 S3 client — typed as object to avoid boto3 stubs dep

    @classmethod
    def from_settings(cls) -> "StorageService":
        s = get_settings()
        client = boto3.client(
            "s3",
            endpoint_url=s.S3_ENDPOINT_URL,
            aws_access_key_id=s.S3_ACCESS_KEY_ID,
            aws_secret_access_key=s.S3_SECRET_ACCESS_KEY,
            region_name=s.S3_REGION,
        )
        return cls(
            bucket=s.S3_BUCKET_NAME,
            public_url=s.S3_PUBLIC_URL.rstrip("/"),
            _client=client,
        )

    def _sync_put(self, key: str, data: bytes, content_type: str) -> None:
        self._client.put_object(  # type: ignore[attr-defined]
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            CacheControl="public, max-age=31536000, immutable",
        )

    def _sync_delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)  # type: ignore[attr-defined]

    async def upload_avatar(
        self,
        user_id: str,
        data: bytes,
        content_type: str,
    ) -> str:
        """Validate, upload, and return the public CDN URL."""
        if content_type not in _ALLOWED_IMAGE_TYPES:
            raise ValidationException("Invalid file type. Only JPEG, PNG, and WebP are accepted.")
        if len(data) > _MAX_AVATAR_BYTES:
            raise ValidationException("File exceeds 5 MB maximum size.")

        ext = mimetypes.guess_extension(content_type, strict=False) or ".bin"
        # .jpeg → .jpg for cleaner URLs
        if ext == ".jpeg":
            ext = ".jpg"

        key = f"avatars/{user_id}/{uuid4()}{ext}"

        try:
            await asyncer.asyncify(self._sync_put)(key, data, content_type)
        except (BotoCoreError, ClientError) as exc:
            logger.bind(user_id=user_id, key=key).error(f"S3 upload failed: {exc}")
            raise

        return f"{self.public_url}/{key}"

    async def delete_object(self, key: str) -> None:
        """Best-effort deletion. Logs but does not raise on failure."""
        try:
            await asyncer.asyncify(self._sync_delete)(key)
        except (BotoCoreError, ClientError) as exc:
            logger.bind(key=key).warning(f"S3 delete failed (non-fatal): {exc}")
