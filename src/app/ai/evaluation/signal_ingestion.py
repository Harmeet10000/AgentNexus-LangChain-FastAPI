"""Support-ticket signal ingestion models and parsers.

The pipeline normalizes tickets from Jira, Zendesk, and Salesforce into a single
``SupportTicket`` shape so downstream clustering and taxonomy-candidate
generation never branch on the source system.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from typing import Literal, Protocol, TypedDict, assert_never, cast

from pydantic import BaseModel, ConfigDict

from .schema import GoldenCaseType, TaxonomyCandidate  # noqa: TC001

__all__ = [
    "CandidateGenerationError",
    "ClusteringError",
    "GoldenQueryCandidate",
    "JiraTicketParser",
    "SalesforceTicketParser",
    "SignalIngestionError",
    "SupportTicket",
    "TaxonomyCandidate",
    "TaxonomyCandidateGenerator",
    "TaxonomyCandidateRecord",
    "TicketCluster",
    "TicketClusterer",
    "TicketParseError",
    "TicketParser",
    "ZendeskTicketParser",
    "parser_for_system",
]


class SupportTicket(TypedDict):
    """Normalized support ticket shared across ticketing systems."""

    ticket_id: str
    customer_id: str
    issue_category: str
    symptom_description: str
    resolution_note: str | None
    created_at: str
    resolved_at: str | None


class GoldenQueryCandidate(TypedDict):
    """Candidate golden query derived from a cluster of failure tickets."""

    query: str
    source_ticket_ids: list[str]
    failure_count: int
    cluster_id: str | None


class TicketCluster(BaseModel):
    """A cluster of similar failure tickets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cluster_id: str
    ticket_ids: list[str]
    centroid_similarity_score: float
    symptom_variations: list[str]
    failure_count: int
    first_seen: str
    last_seen: str


class TaxonomyCandidateRecord(BaseModel):
    """Persisted taxonomy candidate with approval lifecycle metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    query_template: str
    recommended_case_type: str
    source_cluster_id: str
    symptom_variations: list[str]
    cluster_similarity_score: float
    approval_status: Literal["pending", "approved", "rejected"]
    created_at: str
    approved_at: str | None = None


class SignalIngestionError(Exception):
    """Base class for signal ingestion pipeline errors."""


class TicketParseError(SignalIngestionError):
    """A raw support ticket could not be normalized."""

    def __init__(self, *, system: str, field: str, reason: str) -> None:
        self.system = system
        self.field = field
        self.reason = reason
        super().__init__(f"{system} ticket parse failed: {reason} (field={field!r})")


class ClusteringError(SignalIngestionError):
    """Ticket clustering was given empty input or an unusable timestamp."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"ticket clustering failed: {reason}")


class TicketParser(Protocol):
    """Parse a raw ticketing-system payload into a normalized ticket."""

    async def parse(self, raw_ticket: object) -> SupportTicket:
        """Return a normalized ``SupportTicket`` for ``raw_ticket``."""
        ...


_MISSING = object()


def _required_mapping(raw_ticket: object, *, system: str) -> Mapping[object, object]:
    if not isinstance(raw_ticket, Mapping):
        raise TicketParseError(
            system=system,
            field="<root>",
            reason=f"expected a mapping, got {type(raw_ticket).__name__}",
        )
    return cast("Mapping[object, object]", raw_ticket)


def _lookup(mapping: object, *path: str) -> object:
    current = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return _MISSING
        current = current.get(key, _MISSING)
        if current is _MISSING:
            return _MISSING
    return current


def _require_text(value: object, *, system: str, field: str) -> str:
    if value is _MISSING or value is None:
        raise TicketParseError(system=system, field=field, reason="missing required field")
    if not isinstance(value, str):
        raise TicketParseError(
            system=system,
            field=field,
            reason=f"expected a string, got {type(value).__name__}",
        )
    stripped = value.strip()
    if not stripped:
        raise TicketParseError(system=system, field=field, reason="required field is empty")
    return stripped


def _require_identifier(value: object, *, system: str, field: str) -> str:
    if isinstance(value, bool):
        raise TicketParseError(
            system=system,
            field=field,
            reason="expected a string or integer identifier, got bool",
        )
    if isinstance(value, int):
        return str(value)
    return _require_text(value, system=system, field=field)


def _optional_text(value: object, *, system: str, field: str) -> str | None:
    if value is _MISSING or value is None:
        return None
    if not isinstance(value, str):
        raise TicketParseError(
            system=system,
            field=field,
            reason=f"expected a string, got {type(value).__name__}",
        )
    return value.strip() or None


def _epoch_to_iso(value: int | float, *, system: str, field: str) -> str:
    try:
        timestamp = datetime.fromtimestamp(value, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise TicketParseError(
            system=system,
            field=field,
            reason=f"invalid epoch timestamp {value!r}",
        ) from exc
    return timestamp.isoformat().replace("+00:00", "Z")


def _coerce_timestamp(value: object, *, system: str, field: str) -> str:
    if isinstance(value, bool):
        raise TicketParseError(
            system=system,
            field=field,
            reason="expected an ISO-8601 string or epoch number, got bool",
        )
    if isinstance(value, int | float):
        return _epoch_to_iso(value, system=system, field=field)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise TicketParseError(system=system, field=field, reason="required field is empty")
        if stripped.isdigit():
            return _epoch_to_iso(int(stripped), system=system, field=field)
        return stripped
    if value is _MISSING or value is None:
        raise TicketParseError(system=system, field=field, reason="missing required field")
    raise TicketParseError(
        system=system,
        field=field,
        reason=f"expected an ISO-8601 string or epoch number, got {type(value).__name__}",
    )


def _optional_timestamp(value: object, *, system: str, field: str) -> str | None:
    if value is _MISSING or value is None:
        return None
    return _coerce_timestamp(value, system=system, field=field)


def _first_text(value: object) -> object:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
        return _MISSING
    return value


class JiraTicketParser:
    """Normalize Jira issue payloads."""

    _system = "jira"

    async def parse(self, raw_ticket: object) -> SupportTicket:
        ticket = _required_mapping(raw_ticket, system=self._system)
        return SupportTicket(
            ticket_id=_require_identifier(
                _lookup(ticket, "key"),
                system="jira",
                field="key",
            ),
            customer_id=_require_identifier(
                _lookup(ticket, "reporter", "accountId"),
                system="jira",
                field="reporter.accountId",
            ),
            issue_category=_require_text(
                _lookup(ticket, "fields", "issuetype", "name"),
                system="jira",
                field="fields.issuetype.name",
            ),
            symptom_description=_require_text(
                _lookup(ticket, "fields", "description"),
                system="jira",
                field="fields.description",
            ),
            resolution_note=_optional_text(
                _lookup(ticket, "fields", "resolution", "description"),
                system="jira",
                field="fields.resolution.description",
            ),
            created_at=_coerce_timestamp(
                _lookup(ticket, "fields", "created"),
                system="jira",
                field="fields.created",
            ),
            resolved_at=_optional_timestamp(
                _lookup(ticket, "fields", "resolutiondate"),
                system="jira",
                field="fields.resolutiondate",
            ),
        )


class ZendeskTicketParser:
    """Normalize Zendesk ticket payloads, including ``{"ticket": {...}}`` wrappers."""

    _system = "zendesk"

    async def parse(self, raw_ticket: object) -> SupportTicket:
        root = _required_mapping(raw_ticket, system=self._system)
        wrapped = root.get("ticket")
        ticket = wrapped if isinstance(wrapped, Mapping) else root

        category = _lookup(ticket, "type")
        if category is _MISSING or not category:
            category = _first_text(_lookup(ticket, "tags"))

        resolution = _lookup(ticket, "comment")
        if isinstance(resolution, Mapping):
            resolution = _lookup(resolution, "body")
        if resolution is _MISSING or resolution is None:
            resolution = _lookup(ticket, "notes")

        return SupportTicket(
            ticket_id=_require_identifier(
                _lookup(ticket, "id"),
                system="zendesk",
                field="ticket.id",
            ),
            customer_id=_require_identifier(
                _lookup(ticket, "requester_id"),
                system="zendesk",
                field="requester_id",
            ),
            issue_category=_require_text(category, system="zendesk", field="type"),
            symptom_description=_require_text(
                _lookup(ticket, "description"),
                system="zendesk",
                field="description",
            ),
            resolution_note=_optional_text(
                resolution,
                system="zendesk",
                field="comment/notes",
            ),
            created_at=_coerce_timestamp(
                _lookup(ticket, "created_at"),
                system="zendesk",
                field="created_at",
            ),
            resolved_at=_optional_timestamp(
                _lookup(ticket, "solved_at"),
                system="zendesk",
                field="solved_at",
            ),
        )


class SalesforceTicketParser:
    """Normalize Salesforce Case payloads."""

    _system = "salesforce"

    async def parse(self, raw_ticket: object) -> SupportTicket:
        ticket = _required_mapping(raw_ticket, system=self._system)

        ticket_id = _lookup(ticket, "CaseNumber")
        if ticket_id is _MISSING or not ticket_id:
            ticket_id = _lookup(ticket, "Id")

        category = _lookup(ticket, "Type")
        if category is _MISSING or not category:
            category = _lookup(ticket, "Reason")

        return SupportTicket(
            ticket_id=_require_identifier(
                ticket_id,
                system="salesforce",
                field="CaseNumber/Id",
            ),
            customer_id=_require_identifier(
                _lookup(ticket, "ContactId"),
                system="salesforce",
                field="ContactId",
            ),
            issue_category=_require_text(
                category,
                system="salesforce",
                field="Type/Reason",
            ),
            symptom_description=_require_text(
                _lookup(ticket, "Description"),
                system="salesforce",
                field="Description",
            ),
            resolution_note=_optional_text(
                _lookup(ticket, "Resolution__c"),
                system="salesforce",
                field="Resolution__c",
            ),
            created_at=_coerce_timestamp(
                _lookup(ticket, "CreatedDate"),
                system="salesforce",
                field="CreatedDate",
            ),
            resolved_at=_optional_timestamp(
                _lookup(ticket, "ClosedDate"),
                system="salesforce",
                field="ClosedDate",
            ),
        )


def parser_for_system(system: Literal["jira", "zendesk", "salesforce"]) -> TicketParser:
    """Return the parser for a closed set of supported ticketing systems."""
    match system:
        case "jira":
            return JiraTicketParser()
        case "zendesk":
            return ZendeskTicketParser()
        case "salesforce":
            return SalesforceTicketParser()
        case _:
            assert_never(system)


_CANONICAL_TOKENS: dict[str, str] = {
    "absent": "empty",
    "agreement": "contract",
    "agreements": "contract",
    "cannot": "empty",
    "clause": "clause",
    "clauses": "clause",
    "contract": "contract",
    "contracts": "contract",
    "document": "contract",
    "documents": "contract",
    "empty": "empty",
    "fail": "empty",
    "failing": "empty",
    "fails": "empty",
    "find": "retrieve",
    "finding": "retrieve",
    "finds": "retrieve",
    "found": "retrieve",
    "get": "retrieve",
    "gets": "retrieve",
    "getting": "retrieve",
    "indemnification": "indemnity",
    "indemnifications": "indemnity",
    "indemnified": "indemnity",
    "indemnify": "indemnity",
    "indemnities": "indemnity",
    "indemnity": "indemnity",
    "locate": "retrieve",
    "located": "retrieve",
    "look": "retrieve",
    "looking": "retrieve",
    "match": "empty",
    "matches": "empty",
    "missing": "empty",
    "msa": "contract",
    "msas": "contract",
    "no": "empty",
    "none": "empty",
    "not": "empty",
    "nothing": "empty",
    "obligation": "clause",
    "obligations": "clause",
    "provision": "clause",
    "provisions": "clause",
    "queries": "retrieve",
    "query": "retrieve",
    "result": "empty",
    "results": "empty",
    "retrieve": "retrieve",
    "retrieved": "retrieve",
    "retrieves": "retrieve",
    "retrieving": "retrieve",
    "return": "empty",
    "returned": "empty",
    "returns": "empty",
    "search": "retrieve",
    "searches": "retrieve",
    "searching": "retrieve",
    "section": "clause",
    "sections": "clause",
    "term": "clause",
    "terms": "clause",
    "unable": "empty",
}

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "but",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)

_SIMILARITY_THRESHOLD = 0.8


def _normalize_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in _STOPWORDS:
            continue
        canonical = _CANONICAL_TOKENS.get(word)
        if canonical is None and len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            canonical = word[:-1]
        tokens.append(canonical or word)
    return tuple(sorted(set(tokens)))


def _parse_iso_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        message = f"malformed {field}: {value!r}"
        raise ClusteringError(message) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


class TicketClusterer:
    """Group failure tickets by deterministic lexical similarity."""

    @staticmethod
    def similarity(a: SupportTicket, b: SupportTicket) -> float:
        """Return a similarity score in ``[0.0, 1.0]`` for two tickets.

        Both symptom descriptions are normalized into lowercase alphanumeric
        canonical-concept tokens: a fixed synonym table collapses equivalent
        legal/retrieval vocabulary (for example ``indemnification`` and
        ``indemnity``) and stopwords are dropped. The score is
        ``0.9 * Jaccard(token_sets) + 0.1 * SequenceMatcher(sorted_tokens)``,
        plus a ``0.05`` boost when the non-empty ``issue_category`` matches,
        capped at ``1.0``.
        """
        tokens_a = set(_normalize_tokens(a["symptom_description"]))
        tokens_b = set(_normalize_tokens(b["symptom_description"]))
        union = tokens_a | tokens_b
        jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0
        sequence = SequenceMatcher(
            None,
            " ".join(sorted(tokens_a)),
            " ".join(sorted(tokens_b)),
        ).ratio()
        score = 0.9 * jaccard + 0.1 * sequence
        if a["issue_category"] and a["issue_category"] == b["issue_category"]:
            score += 0.05
        return min(1.0, score)

    def cluster(self, tickets: list[SupportTicket]) -> dict[str, list[SupportTicket]]:
        """Cluster tickets using union-find transitive closure at the 0.8 threshold.

        Two tickets are joined when ``similarity >= 0.8``; union-find then makes
        the relation transitive, so a ticket can belong to a cluster without
        directly matching every other member. Cluster ids are assigned in order
        of each component's first ticket, so equivalent inputs always produce
        identical ids.
        """
        if not tickets:
            message = "cannot cluster an empty ticket list"
            raise ClusteringError(message)

        parent = list(range(len(tickets)))

        def find(index: int) -> int:
            root = index
            while parent[root] != root:
                root = parent[root]
            while parent[index] != root:
                parent[index], index = root, parent[index]
            return root

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for left in range(len(tickets)):
            for right in range(left + 1, len(tickets)):
                if self.similarity(tickets[left], tickets[right]) >= _SIMILARITY_THRESHOLD:
                    union(left, right)

        cluster_ids: dict[int, str] = {}
        clusters: dict[str, list[SupportTicket]] = {}
        for index, ticket in enumerate(tickets):
            root = find(index)
            if root not in cluster_ids:
                cluster_ids[root] = f"cluster-{len(cluster_ids) + 1}"
            clusters.setdefault(cluster_ids[root], []).append(ticket)
        return clusters

    def compute_cluster_centroid(self, ticket_group: list[SupportTicket]) -> dict[str, object]:
        """Compute summary statistics for one homogeneous cluster.

        ``centroid_similarity_score`` is the mean pairwise similarity inside the
        group and is ``1.0`` for a singleton. ``first_seen``/``last_seen`` are
        the original ISO ``created_at`` values with the minimum/maximum instant.
        """
        if not ticket_group:
            message = "cannot compute a centroid for an empty ticket group"
            raise ClusteringError(message)

        timestamps = [
            _parse_iso_timestamp(ticket["created_at"], field="created_at")
            for ticket in ticket_group
        ]
        first_index = min(range(len(ticket_group)), key=timestamps.__getitem__)
        last_index = max(range(len(ticket_group)), key=timestamps.__getitem__)

        if len(ticket_group) == 1:
            centroid_score = 1.0
        else:
            scores = [
                self.similarity(ticket_group[left], ticket_group[right])
                for left in range(len(ticket_group))
                for right in range(left + 1, len(ticket_group))
            ]
            centroid_score = sum(scores) / len(scores)

        variations = list(
            dict.fromkeys(
                ticket["symptom_description"]
                for ticket in ticket_group
                if ticket["symptom_description"].strip()
            )
        )

        cluster_id = next(iter(self.cluster(ticket_group)))
        return {
            "cluster_id": cluster_id,
            "ticket_ids": [ticket["ticket_id"] for ticket in ticket_group],
            "failure_count": len(ticket_group),
            "symptom_variations": variations,
            "centroid_similarity_score": centroid_score,
            "first_seen": ticket_group[first_index]["created_at"],
            "last_seen": ticket_group[last_index]["created_at"],
        }

    def cluster_to_models(self, tickets: list[SupportTicket]) -> list[TicketCluster]:
        """Return validated ``TicketCluster`` models for a ticket collection."""
        models: list[TicketCluster] = []
        for group in self.cluster(tickets).values():
            centroid = self.compute_cluster_centroid(group)
            models.append(
                TicketCluster(
                    cluster_id=cast("str", centroid["cluster_id"]),
                    ticket_ids=cast("list[str]", centroid["ticket_ids"]),
                    centroid_similarity_score=cast("float", centroid["centroid_similarity_score"]),
                    symptom_variations=cast("list[str]", centroid["symptom_variations"]),
                    failure_count=cast("int", centroid["failure_count"]),
                    first_seen=cast("str", centroid["first_seen"]),
                    last_seen=cast("str", centroid["last_seen"]),
                )
            )
        return models


class CandidateGenerationError(SignalIngestionError):
    """Taxonomy candidate generation was given invalid threshold/window arguments."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"taxonomy candidate generation failed: {reason}")


_CASE_TYPE_RULES: tuple[tuple[tuple[str, ...], GoldenCaseType], ...] = (
    (("ingestion_failure", "ingestion failure"), "no_evidence_vs_ingestion_failure"),
    (("index_not_ready", "index not ready", "indexing lag", "index lag"), "index_not_ready"),
    (("conflicting", "contradict", "conflict"), "conflicting_clauses"),
    (("near miss", "near-miss", "similar clause"), "near_miss_clause"),
    (("multi document", "multiple documents", "across documents"), "multi_document"),
    (("multi hop", "multi-hop", "multi step", "multi-step"), "multi_hop"),
    (("definition", "exception"), "definition_exception_trap"),
    (("cross tenant", "cross-tenant", "tenant isolation"), "cross_tenant_isolation"),
)

_QUESTION_REWRITES: tuple[tuple[str, str], ...] = (
    ("cannot ", "Why can't "),
    ("can not ", "Why can't "),
    ("unable to ", "Why am I unable to "),
    ("looking for ", "Where can I find "),
    ("search for ", "Where can I find "),
    ("searching for ", "Where can I find "),
)

_QUESTION_STARTS = ("why ", "what ", "when ", "where ", "which ", "who ", "how ")


def _window_reaches_threshold(
    timestamps: list[datetime],
    *,
    threshold: int,
    window_days: int,
) -> bool:
    ordered = sorted(timestamps)
    for index, start in enumerate(ordered):
        end = start + timedelta(days=window_days)
        count = 0
        for timestamp in ordered[index:]:
            if timestamp > end:
                break
            count += 1
        if count >= threshold:
            return True
    return False


def _symptom_to_question(symptom: str) -> str:
    normalized = " ".join(symptom.split())
    if not normalized:
        return "Why does retrieval fail for this query?"
    if normalized.endswith("?"):
        return normalized
    stripped = normalized.rstrip(".!;:")
    lowered = stripped.lower()
    for prefix, replacement in _QUESTION_REWRITES:
        if lowered.startswith(prefix):
            return f"{replacement}{stripped[len(prefix) :]}?"
    if lowered.startswith(_QUESTION_STARTS):
        return f"{stripped}?"
    return f"Why does the system report: {stripped}?"


def _recommended_case_type(tickets: list[SupportTicket]) -> GoldenCaseType:
    haystack = " ".join(
        f"{ticket['issue_category']} {ticket['symptom_description']}" for ticket in tickets
    ).lower()
    for keywords, case_type in _CASE_TYPE_RULES:
        if any(keyword in haystack for keyword in keywords):
            return case_type
    return "unanswerable"


class TaxonomyCandidateGenerator:
    """Convert homogeneous ticket clusters into golden-set taxonomy candidates."""

    def generate_candidates(  # noqa: PLR6301
        self,
        cluster: dict[str, list[SupportTicket]],
        threshold: int = 5,
        window_days: int = 30,
    ) -> list[TaxonomyCandidate]:
        """Return one candidate per cluster meeting the failure threshold.

        A cluster qualifies when at least ``threshold`` of its tickets fall inside
        one sliding ``window_days`` window measured from a ticket's ``created_at``.
        """
        if threshold < 1:
            message = "threshold must be at least 1"
            raise CandidateGenerationError(message)
        if window_days < 1:
            message = "window_days must be at least 1"
            raise CandidateGenerationError(message)

        clusterer = TicketClusterer()
        candidates: list[TaxonomyCandidate] = []
        for tickets in cluster.values():
            if not tickets:
                continue
            timestamps = [
                _parse_iso_timestamp(ticket["created_at"], field="created_at") for ticket in tickets
            ]
            if not _window_reaches_threshold(
                timestamps,
                threshold=threshold,
                window_days=window_days,
            ):
                continue
            centroid = clusterer.compute_cluster_centroid(tickets)
            representative = next(
                (
                    ticket["symptom_description"]
                    for ticket in tickets
                    if ticket["symptom_description"].strip()
                ),
                "",
            )
            candidates.append(
                TaxonomyCandidate(
                    query_template=_symptom_to_question(representative),
                    recommended_case_type=_recommended_case_type(tickets),
                    source_ticket_ids=[ticket["ticket_id"] for ticket in tickets],
                    cluster_similarity_score=cast("float", centroid["centroid_similarity_score"]),
                    symptom_variations=list(cast("list[str]", centroid["symptom_variations"])),
                )
            )
        return candidates

    def to_record(  # noqa: PLR6301
        self,
        candidate: TaxonomyCandidate,
        *,
        cluster_id: str,
        created_at: str,
    ) -> TaxonomyCandidateRecord:
        """Persist a generated candidate with pending approval status."""
        digest = hashlib.sha256(f"{cluster_id}\0{candidate.query_template}".encode()).hexdigest()[
            :16
        ]
        return TaxonomyCandidateRecord(
            candidate_id=f"candidate-{digest}",
            query_template=candidate.query_template,
            recommended_case_type=candidate.recommended_case_type,
            source_cluster_id=cluster_id,
            symptom_variations=list(candidate.symptom_variations),
            cluster_similarity_score=candidate.cluster_similarity_score,
            approval_status="pending",
            created_at=created_at,
            approved_at=None,
        )
