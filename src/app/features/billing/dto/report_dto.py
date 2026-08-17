"""Report request/response DTOs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from app.features.billing.models.report import ReportFormat

if TYPE_CHECKING:
    from datetime import datetime

    from app.features.billing.models.report import ReportType


class ReportCreateDTO(BaseModel):
    """Request a report generation."""

    model_config = ConfigDict(extra="forbid")

    report_type: ReportType
    report_name: str = Field(min_length=1, max_length=200)
    date_from: datetime | None = None
    date_to: datetime | None = None
    plan_ids: list[str] | None = None
    output_format: ReportFormat = ReportFormat.CSV


class ReportResponse(BaseModel):
    """Report representation returned by the API."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    report_type: str = Field(serialization_alias="reportType")
    report_name: str = Field(serialization_alias="reportName")
    status: str
    date_from: datetime | None = Field(default=None, serialization_alias="dateFrom")
    date_to: datetime | None = Field(default=None, serialization_alias="dateTo")
    generated_at: datetime | None = Field(default=None, serialization_alias="generatedAt")
    output_format: str = Field(serialization_alias="outputFormat")
    file_url: str | None = Field(default=None, serialization_alias="fileUrl")
    row_count: int | None = Field(default=None, serialization_alias="rowCount")
    created_at: datetime = Field(serialization_alias="createdAt")
