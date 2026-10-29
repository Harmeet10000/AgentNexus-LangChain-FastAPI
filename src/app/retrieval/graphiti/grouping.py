"""Canonical, provider-safe Graphiti group identifiers.

Graphiti passes ``group_id`` to Neo4j as a database/partition identifier.  The
provider accepts only ASCII letters, digits, ``-`` and ``_``; the old
colon-delimited IDs therefore failed at the Graphiti boundary and, worse,
made it easy for a caller to accidentally fall back to a document-only group.

Values are hex encoded so the tuple remains lossless even when document or
tenant identifiers contain a separator.  The format is deliberately boring:
it is stable across processes and has no dependence on URL/path escaping.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from graphiti_core.errors import GroupIdValidationError
from graphiti_core.helpers import validate_group_id
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Iterable

_GROUP_ID_PATTERN = re.compile(
    r"^tenant_(?P<tenant>[0-9a-f]+)_document_(?P<document>[0-9a-f]+)"
    r"_version_(?P<version>[0-9]+)_contract_(?P<contract>[0-9a-f]+|none)$"
)
_EMPTY_DOCUMENT = "document_id must not be empty"
_INVALID_VERSION = "document_version must be >= 1"
_INVALID_CONTRACT = "contract_index must be >= 0"
_NOT_CANONICAL = "group_id is not a canonical Graphiti scope"
_INVALID_SCOPE = "group_id contains an invalid Graphiti scope"


class GraphitiGroupScope(BaseModel):
    """The identity tuple represented by a Graphiti group ID."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    document_id: str
    document_version: int
    contract_index: int | None


class GraphitiDocumentScope(BaseModel):
    """Authorized version plus the exact contracts visible for that version."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    document_id: str
    document_version: int
    contract_indexes: tuple[int | None, ...]
    # ``None`` is legal only for explicitly legacy documents whose metadata
    # proves that no contract index was ever assigned.
    legacy_null_contract: bool = False

    def group_ids(self) -> tuple[str, ...]:
        if not self.tenant_id or (not self.contract_indexes and not self.legacy_null_contract):
            return ()
        indexes = self.contract_indexes or (None,)
        return tuple(
            build_graphiti_group_id(
                tenant_id=self.tenant_id,
                document_id=self.document_id,
                document_version=self.document_version,
                contract_index=index,
            )
            for index in indexes
            if index is not None or self.legacy_null_contract
        )


def fanout_graphiti_group_ids(scopes: Iterable[GraphitiDocumentScope]) -> list[str]:
    """Resolve a read scope without ever silently falling back to contract_none."""
    result: list[str] = []
    for scope in scopes:
        result.extend(scope.group_ids())
    return list(dict.fromkeys(result))


def _encode(value: str) -> str:
    return value.encode("utf-8").hex() or "none"


def _decode(value: str) -> str:
    if value == "none":
        return ""
    return bytes.fromhex(value).decode("utf-8")


def build_graphiti_group_id(
    *,
    tenant_id: str,
    document_id: str,
    document_version: int = 1,
    contract_index: int | None = None,
) -> str:
    """Build the only supported Graphiti partition identifier.

    An empty tenant is encoded as an explicit ``none`` value for legacy
    writers that have no tenancy context.  Reads must never use that escape
    hatch: callers constructing an authorization scope should require a real
    tenant and return no results when it is absent.
    """
    if not document_id:
        raise ValueError(_EMPTY_DOCUMENT)
    if document_version < 1:
        raise ValueError(_INVALID_VERSION)
    if contract_index is not None and contract_index < 0:
        raise ValueError(_INVALID_CONTRACT)
    group_id = (
        f"tenant_{_encode(tenant_id)}_document_{_encode(document_id)}"
        f"_version_{document_version}_contract_"
        f"{_encode(str(contract_index)) if contract_index is not None else 'none'}"
    )
    # Keep this validation coupled to the provider's own contract.  It also
    # protects future edits to the wire format from silently creating IDs that
    # Graphiti rejects only after a write has started.
    validate_group_id(group_id)
    return group_id


def parse_graphiti_group_id(group_id: str) -> GraphitiGroupScope:
    """Decode and validate a canonical group ID (useful for diagnostics/tests)."""
    try:
        validate_group_id(group_id)
    except GroupIdValidationError as exc:
        raise ValueError(_NOT_CANONICAL) from exc
    match = _GROUP_ID_PATTERN.fullmatch(group_id)
    if match is None:
        raise ValueError(_NOT_CANONICAL)
    contract_value = match.group("contract")
    contract_index = None if contract_value == "none" else int(_decode(contract_value))
    scope = GraphitiGroupScope(
        tenant_id=_decode(match.group("tenant")),
        document_id=_decode(match.group("document")),
        document_version=int(match.group("version")),
        contract_index=contract_index,
    )
    if not scope.document_id or scope.document_version < 1:
        raise ValueError(_INVALID_SCOPE)
    return scope
