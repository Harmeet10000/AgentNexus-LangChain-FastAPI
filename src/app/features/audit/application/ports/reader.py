"""Public query port for immutable audit projections."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.features.audit.application.dto.list import (
        ListAuditEntriesInput,
        ListAuditEntriesOutput,
    )
    from app.features.audit.domain.errors import AuditResult


class AuditReader(Protocol):
    async def list(
        self,
        query: "ListAuditEntriesInput",
    ) -> "AuditResult[ListAuditEntriesOutput]": ...
