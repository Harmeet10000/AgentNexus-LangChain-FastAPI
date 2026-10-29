"""Structural contract the storage CRUD layer needs from an S3 client.

Defining the seam here — rather than importing the concrete client — keeps
``app.utils.storage`` free of any dependency on ``app.connections``: the
transport adapter satisfies this protocol structurally, so the CRUD layer can
be driven by a real client, a test double, or a future provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from typing import Any


class ObjectStoreClient(Protocol):
    """Synchronous S3 operations that ``s3_func`` drives from worker threads."""

    bucket: str

    def put_object(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str],
    ) -> None: ...

    def get_object(self, *, key: str) -> bytes: ...

    def delete_object(self, *, key: str) -> None: ...

    def head_object(self, *, key: str) -> dict[str, Any]: ...

    def head_bucket(self) -> None: ...

    def list_objects(self, *, prefix: str, max_keys: int) -> dict[str, Any]: ...

    def copy_object(self, *, source_key: str, destination_key: str) -> None: ...

    def create_multipart_upload(
        self,
        *,
        key: str,
        content_type: str,
        metadata: dict[str, str],
    ) -> dict[str, Any]: ...

    def upload_part(
        self,
        *,
        key: str,
        upload_id: str,
        part_number: int,
        body: bytes,
    ) -> dict[str, Any]: ...

    def complete_multipart_upload(
        self,
        *,
        key: str,
        upload_id: str,
        parts: list[dict[str, Any]],
    ) -> dict[str, Any]: ...

    def abort_multipart_upload(self, *, key: str, upload_id: str) -> None: ...

    def list_parts(self, *, key: str, upload_id: str) -> dict[str, Any]: ...

    def generate_presigned_url(
        self,
        operation: str,
        params: dict[str, Any],
        expires_in: int,
    ) -> str: ...
