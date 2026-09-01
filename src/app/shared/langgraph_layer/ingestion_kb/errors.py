"""Closed error contract for ingestion graph state failures."""

from enum import StrEnum
from typing import ClassVar

from app.shared.result import ErrorKind, FeatureError


class IngestionGraphCode(StrEnum):
    VALIDATION_FAILED = "INGESTION_GRAPH_VALIDATION_FAILED"


class IngestionGraphValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[IngestionGraphCode] = IngestionGraphCode.VALIDATION_FAILED
    doc_id: str = ""


type IngestionGraphError = IngestionGraphValidationError
