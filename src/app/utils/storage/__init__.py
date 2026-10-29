"""Object-storage utilities: typed CRUD over an injected S3 client.

The client itself is formed once in ``app.connections.object_storage``; this
package owns the operations, the key/URI helpers, and the ``StorageResult``
vocabulary, mirroring ``app.utils.cache``.
"""

from .errors import (
    StorageCode,
    StorageError,
    StorageResult,
    StorageUnavailableError,
    StorageValidationError,
)
from .keys import build_s3_key, build_s3_uri, key_from_s3_uri, uri_to_location
from .models import (
    MultipartPartURL,
    MultipartUploadPlan,
    ObjectMetadata,
    PresignedUploadURL,
    S3CompleteMultipartUploadResponse,
    S3ListObjectsResponse,
    S3ListPartsResponse,
    S3ObjectLocation,
    S3PartInfo,
    S3UploadPartResponse,
    UploadRequest,
)
from .protocols import ObjectStoreClient
from .s3_func import (
    abort_multipart_upload,
    complete_multipart_upload,
    copy_object,
    create_multipart_upload,
    create_multipart_upload_plan,
    delete_by_uri,
    delete_object,
    generate_presigned_upload_urls,
    get_by_uri,
    get_object,
    get_signed_get_url,
    get_signed_multipart_part_url,
    get_signed_put_url,
    list_multipart_upload_parts,
    list_objects,
    object_exists,
    put_object,
    upload_part,
    verify_access,
)
from .service import StorageService, create_object_store

__all__ = [
    "MultipartPartURL",
    "MultipartUploadPlan",
    "ObjectMetadata",
    "ObjectStoreClient",
    "PresignedUploadURL",
    "S3CompleteMultipartUploadResponse",
    "S3ListObjectsResponse",
    "S3ListPartsResponse",
    "S3ObjectLocation",
    "S3PartInfo",
    "S3UploadPartResponse",
    "StorageCode",
    "StorageError",
    "StorageResult",
    "StorageService",
    "StorageUnavailableError",
    "StorageValidationError",
    "UploadRequest",
    "abort_multipart_upload",
    "build_s3_key",
    "build_s3_uri",
    "complete_multipart_upload",
    "copy_object",
    "create_multipart_upload",
    "create_multipart_upload_plan",
    "create_object_store",
    "delete_by_uri",
    "delete_object",
    "generate_presigned_upload_urls",
    "get_by_uri",
    "get_object",
    "get_signed_get_url",
    "get_signed_multipart_part_url",
    "get_signed_put_url",
    "key_from_s3_uri",
    "list_multipart_upload_parts",
    "list_objects",
    "object_exists",
    "put_object",
    "upload_part",
    "uri_to_location",
    "verify_access",
]
